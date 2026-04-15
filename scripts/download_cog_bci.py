from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request


ZENODO_RECORD_API = "https://zenodo.org/api/records/7413650"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or download files from the COG-BCI Zenodo record.")
    parser.add_argument("--output-dir", default="data/cog_bci")
    parser.add_argument("--download", action="store_true", help="Download record files. By default only metadata is saved.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of files to download.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _fetch_json(ZENODO_RECORD_API)
    (output_dir / "record_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    files = record.get("files", [])
    print(f"Saved metadata for {len(files)} files to {output_dir / 'record_metadata.json'}")
    for index, file_info in enumerate(files[: args.limit]):
        key = file_info.get("key", f"file_{index}")
        size = file_info.get("size")
        print(f"{index + 1:03d}. {key} size={size}")
        if args.download:
            url = file_info["links"]["self"]
            target = output_dir / key
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"     downloading -> {target}")
            urllib.request.urlretrieve(url, target)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
