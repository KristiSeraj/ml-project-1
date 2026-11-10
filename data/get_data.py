"""Download the original files into data/raw/.

Run from a fresh clone before any notebook:

    python data/get_data.py

Idempotent: a file already present is left alone. Nothing in this script
edits or deletes anything under data/raw/.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"

# filename in data/raw/ -> source URL
SOURCES: dict[str, str] = {
    # "rides-june.csv": "https://example.com/rides-june.csv",
}


def fetch(name: str, url: str) -> Path:
    dest = RAW / name
    if dest.exists():
        print(f"have   {name}")
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"get    {name}  <-  {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main() -> int:
    if not SOURCES:
        print("No sources listed yet: fill in SOURCES in data/get_data.py.")
        return 1
    for name, url in SOURCES.items():
        fetch(name, url)
    print(f"\nraw data in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
