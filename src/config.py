"""Single source of truth for the experimental setup.

Every model reads its split, seed and feature definition from here, so the comparison
between models cannot drift apart by accident.
"""

# --- temporal structure of the data -------------------------------------------------
SLOTS_PER_HOUR = 6
SLOTS_PER_DAY = 144
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY          # 1008

# --- chronological split (no shuffling: later data must never train earlier forecasts)
# Train:      Nov 1  – Dec 8   (38 days, 5,472 slots)
# Validation: Dec 9  – Dec 15  (7 days, 1,008 slots)  -> all tuning decisions
# Test:       Dec 16 – Dec 22  (7 days, 1,008 slots)  -> touched once, at the end
TRAIN_START = "2013-11-01"
VAL_START = "2013-12-09"
TEST_START = "2013-12-16"
TEST_END = "2013-12-23"                      # exclusive

# --- reproducibility ----------------------------------------------------------------
SEED = 42

# --- input representation -----------------------------------------------------------
# Lags chosen from the EDA: 1-3 from the PACF (significant at lags 1-2), 6 = one hour,
# 144 = same time yesterday, 1008 = same time last week (both ACF peaks).
FEATURE_LAGS = (1, 2, 3, 6, SLOTS_PER_DAY, SLOTS_PER_WEEK)
ROLLING_WINDOWS = (6, 36)                    # 1 hour and 6 hours
LSTM_WINDOW = SLOTS_PER_DAY                  # one full daily cycle

# Italian public holidays inside the observation period (traffic behaves like a Sunday).
HOLIDAYS = ["2013-11-01",   # All Saints' Day
            "2013-12-07",   # Sant'Ambrogio (Milan's patron saint)
            "2013-12-08",   # Immaculate Conception
            "2013-12-25",   # Christmas Day
            "2013-12-26",   # St Stephen's Day
            "2014-01-01"]   # New Year's Day
