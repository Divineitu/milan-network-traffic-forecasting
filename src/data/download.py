"""Download the Telecom Italia Milan dataset from Harvard Dataverse.

Files are fetched one at a time through the Dataverse native API and streamed
to disk in 1 MB blocks, so memory use stays flat no matter how big a file is.
Files that already exist with the expected size are skipped, so the script can
be re-run safely after an interruption.

The dataset is protected by a Dataverse "guestbook" that requires an email
address before download. The email is passed on the command line (never
stored in the repository) and submitted once per file to obtain a short-lived
signed download URL.

Usage:
    python src/data/download.py --email you@example.com              # telecom data + grid
    python src/data/download.py --email you@example.com --only grid  # grid GeoJSON only
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu"
DATASETS = {
    "telecom": "doi:10.7910/DVN/EGZHFV",  # SMS, Call, Internet - MI (62 daily files)
    "grid": "doi:10.7910/DVN/QJWLFU",     # Milano grid (GeoJSON, 10,000 squares)
}
CHUNK_BYTES = 1 << 20  # 1 MB
RETRIES = 8  # with linear back-off this tolerates network outages of ~3 minutes
# Dataverse rejects Python's default "Python-urllib" user agent with HTTP 403.
HEADERS = {"User-Agent": "milan-traffic-forecasting/1.0 (academic project)"}


def open_url(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def list_files(doi: str) -> list[dict]:
    """Return the Dataverse metadata record of every file in a dataset."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={doi}"
    with open_url(url) as response:
        meta = json.load(response)
    return [f["dataFile"] for f in meta["data"]["latestVersion"]["files"]]


def signed_download_url(file_id: int, email: str) -> str:
    """Submit the guestbook response and return a short-lived signed download URL."""
    body = json.dumps({"guestbookResponse": {"email": email}}).encode()
    request = urllib.request.Request(
        f"{DATAVERSE}/api/access/datafile/{file_id}?signed=true",
        data=body,
        headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["data"]["signedUrl"]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def download_file(meta: dict, out_dir: Path, email: str) -> None:
    dest = out_dir / meta["filename"]
    expected_size = meta["filesize"]
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"skip  {dest.name} (already downloaded)")
        return

    partial = dest.with_name(dest.name + ".part")  # renamed only once complete
    for attempt in range(1, RETRIES + 1):
        try:
            start = time.perf_counter()
            url = signed_download_url(meta["id"], email)  # expires quickly: fetch per attempt
            with open_url(url) as response, open(partial, "wb") as f:
                while block := response.read(CHUNK_BYTES):
                    f.write(block)
            expected_md5 = meta.get("md5")
            if expected_md5 and md5sum(partial) != expected_md5:
                raise IOError("MD5 checksum mismatch")
            partial.replace(dest)
            secs = time.perf_counter() - start
            print(f"done  {dest.name}  {expected_size / 2**20:.0f} MB in {secs:.0f}s")
            return
        except Exception as err:  # network hiccups are common on long downloads
            print(f"retry {dest.name} (attempt {attempt}/{RETRIES}): {err}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {dest.name} after {RETRIES} attempts")


def verify_files(files: list[dict], out_dir: Path) -> bool:
    """Check every downloaded file against its Dataverse MD5; return True if all pass."""
    ok = True
    for meta in files:
        path = out_dir / meta["filename"]
        if not path.exists():
            print(f"MISSING  {path.name}")
            ok = False
        elif md5sum(path) != meta.get("md5"):
            print(f"CORRUPT  {path.name} (delete it and re-run the download)")
            ok = False
        else:
            print(f"ok       {path.name}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", help="email for the Dataverse guestbook (needed to download)")
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--only", choices=list(DATASETS), help="download one dataset only")
    parser.add_argument("--verify", action="store_true",
                        help="only check MD5 checksums of already-downloaded files")
    args = parser.parse_args()
    if not args.verify and not args.email:
        parser.error("--email is required unless --verify is used")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else list(DATASETS)
    all_ok = True
    for name in names:
        files = sorted(list_files(DATASETS[name]), key=lambda m: m["filename"])
        print(f"== {name}: {len(files)} files")
        if args.verify:
            all_ok &= verify_files(files, out_dir)
        else:
            for meta in files:
                download_file(meta, out_dir, args.email)
    if args.verify:
        print("All files verified." if all_ok else "Some files failed verification.")


if __name__ == "__main__":
    main()
