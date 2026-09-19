"""Model 3 (deep sequential): an LSTM over a window of recent observations.

Unlike the tree model, no lag features are engineered: the network receives the raw
(scaled) sequence and learns the temporal dependence itself through its gated memory
(Hochreiter & Schmidhuber, 1997). LSTM rather than GRU because Santos et al. (2022)
found LSTM the better of the two on this same Milan dataset, and because the two are
variations of one architecture rather than genuinely different models.

Input representation (per time step of the window):
  * the scaled traffic value (min-max, fitted on training data only), plus
  * calendar channels: daily and weekly sine/cosine, weekend and holiday flags.
The window is 144 steps (24 h), the span over which the ACF still carries information.

Output target — two formulations, compared experimentally (see experiments/log.md):
  * "level": predict the scaled value x(t+1) directly.
  * "delta": predict the change x(t+1) - x(t), standardised by its training spread, and add
    it back to x(t). With lag-1 autocorrelation 0.987, the level formulation forces the network
    to reproduce its last input almost exactly before it can add anything useful; the delta
    formulation makes persistence the starting point and learns only the correction.

Training: Adam on MSE, mini-batches, early stopping on the last 10% of the supplied history
(an internal split, so the official validation week is untouched during tuning). That
internal loss proved noisy from epoch to epoch, so patience is generous (10 epochs) to avoid
stopping on a single bad epoch. Seeds are fixed for reproducibility.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import LSTM_WINDOW, SEED
from models.dataset import MinMaxScaler, calendar_features

CALENDAR_CHANNELS = ["day_sin", "day_cos", "week_sin", "week_cos", "is_weekend", "is_holiday"]
TARGETS = ("level", "delta")


class _Net(nn.Module):
    """LSTM encoder whose last hidden state is mapped to a single output."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class LstmForecaster:
    def __init__(self, window: int = LSTM_WINDOW, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.0, learning_rate: float = 1e-3, target: str = "delta",
                 batch_size: int = 128, max_epochs: int = 60, patience: int = 10,
                 seed: int = SEED) -> None:
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.target = target
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.name = (f"LSTM({target}, window {window}, hidden {hidden}, layers {layers}, "
                     f"dropout {dropout}, lr {learning_rate})")
        self.history: dict[str, list[float]] = {"train": [], "valid": []}
        self.epochs_run = 0
        self.best_epoch = 0
        self._scaler = MinMaxScaler()
        self._delta_scale = 1.0
        self._net: _Net | None = None

    # -- data preparation ------------------------------------------------------------
    def _channels(self, series: pd.Series) -> np.ndarray:
        """[n_steps, n_features] array: scaled traffic followed by calendar channels."""
        scaled = self._scaler.transform(series.to_numpy(dtype=np.float32))
        calendar = calendar_features(series.index)[CALENDAR_CHANNELS].to_numpy(dtype=np.float32)
        return np.column_stack([scaled, calendar]).astype(np.float32)

    def _windows(self, series: pd.Series):
        """Sliding windows over the channels.

        Returns X[i] (the `window` steps ending at t), the scaled value at t+1, the scaled
        value at t (the anchor for the delta target) and the timestamps of t+1.
        """
        channels = self._channels(series)
        n = len(series) - self.window
        views = np.lib.stride_tricks.sliding_window_view(channels, self.window, axis=0)
        X = np.ascontiguousarray(views[:n].transpose(0, 2, 1))     # [n, window, features]
        next_value = channels[self.window:, 0]
        last_value = channels[self.window - 1:-1, 0]
        return X, next_value, last_value, series.index[self.window:]

    def _targets(self, next_value: np.ndarray, last_value: np.ndarray) -> np.ndarray:
        if self.target == "level":
            return next_value
        return (next_value - last_value) / self._delta_scale

    # -- training --------------------------------------------------------------------
    def fit(self, history: pd.Series) -> "LstmForecaster":
        torch.set_num_threads(4)                                  # all physical cores
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._scaler.fit(history.to_numpy(dtype=np.float32))      # training range only

        X, next_value, last_value, _ = self._windows(history)
        cut = int(len(X) * 0.9)                                   # last 10% for early stopping
        # The delta spread is also estimated on the training part only.
        self._delta_scale = float(np.std(next_value[:cut] - last_value[:cut])) or 1.0
        y = self._targets(next_value, last_value).astype(np.float32)

        train_X, train_y = torch.from_numpy(X[:cut]), torch.from_numpy(y[:cut])
        valid_X, valid_y = torch.from_numpy(X[cut:]), torch.from_numpy(y[cut:])

        self._net = _Net(X.shape[2], self.hidden, self.layers, self.dropout)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_X, train_y),
            batch_size=self.batch_size, shuffle=True)

        best_loss, best_state, waited = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self._net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimiser.zero_grad()
                loss = loss_fn(self._net(batch_X), batch_y)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch_X)
            self._net.eval()
            with torch.no_grad():
                valid_loss = loss_fn(self._net(valid_X), valid_y).item()
            self.history["train"].append(epoch_loss / len(train_X))
            self.history["valid"].append(valid_loss)
            self.epochs_run = epoch + 1

            if valid_loss < best_loss - 1e-7:
                best_loss, waited, self.best_epoch = valid_loss, 0, epoch + 1
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break                                          # early stopping
        if best_state is not None:
            self._net.load_state_dict(best_state)                 # restore the best epoch
        return self

    # -- forecasting -----------------------------------------------------------------
    def predict(self, series: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
        """One step ahead for each target timestamp, from the true preceding window."""
        if self._net is None:
            raise RuntimeError("fit() must be called first")
        needed = series.loc[:target_index[-1]]
        X, _, last_value, stamps = self._windows(needed)
        wanted = pd.Index(stamps).get_indexer(target_index)
        if (wanted < 0).any():
            raise ValueError("target timestamps are not covered by the supplied series")
        self._net.eval()
        with torch.no_grad():
            output = self._net(torch.from_numpy(X[wanted])).numpy()
        if self.target == "delta":
            output = last_value[wanted] + output * self._delta_scale
        return self._scaler.inverse(output)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._net.parameters())
