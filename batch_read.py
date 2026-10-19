"""Read every meter photo in a folder or ZIP and write the readings to Excel and CSV.

    python batch_read.py path/to/WaterMeters                 # Kaggle folder (data.csv is used for checking)
    python batch_read.py dataset.zip --out results --name run1
    python batch_read.py photos/ --labels labels.csv --decimals 3 --limit 50

Labels are optional. When a CSV with a file-name column and a value column is found (or the file names
carry the value, like Kaggle's id_53_value_595_825.jpg), each reading is marked Match Yes/No.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.dataset import CLI_COLUMNS, Dataset, csv_text, enrich, expected_for, load_split, parse_labels, summarize, xlsx_bytes  # noqa: E402
from app.reader import MeterReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="folder or ZIP of meter photos")
    parser.add_argument("--labels", type=Path, help="CSV with expected readings (file name + value columns)")
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="output folder (default: results/)")
    parser.add_argument("--name", help="output file name without extension (default: dataset name)")
    parser.add_argument("--decimals", type=int, help="decimal places on the meter, e.g. 3 for 00123.456 (default: dashboard calibration)")
    parser.add_argument("--meter-id", default="dataset")
    parser.add_argument("--limit", type=int, help="only read the first N images")
    parser.add_argument("--subset", choices=["train", "val", "test"], help="only read images from this part of the model's training split")
    args = parser.parse_args()

    overrides = {"decimal_places": args.decimals} if args.decimals is not None else {}
    reader = MeterReader(ROOT / "data" / "config.json", overrides)
    split = load_split(ROOT / "models")
    with Dataset(args.dataset) as dataset:
        labels = dict(dataset.labels)
        if args.labels:
            labels.update(parse_labels(args.labels.read_text(encoding="utf-8", errors="replace")))
        items = list(dataset)
        if args.subset:
            if not split:
                parser.error("--subset needs models/meter_split.json from the training script")
            items = [(n, load) for n, load in items if split.get(Path(n).name) == args.subset]
        if args.limit:
            items = items[: args.limit]
        if not items:
            print(f"No images found in {args.dataset}", file=sys.stderr)
            return 1

        print(f"Reading {len(items)} images from {args.dataset}" + (f" ({len(labels)} labels found)" if labels else ""))
        rows, started = [], time.time()
        for index, (name, load) in enumerate(items, start=1):
            row = {"meter_id": args.meter_id, "source_name": name, "expected": expected_for(name, labels),
                   "reading": "", "confidence": 0.0, "roi_source": "", "note": ""}
            try:
                result = reader.read(reader.decode(load()))
                row.update(reading=result["reading"], confidence=result["confidence"], roi_source=result["roi"]["source"])
            except Exception as exc:  # one bad file must not stop a dataset run
                row["note"] = str(exc) or exc.__class__.__name__
            row["captured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            rows.append(enrich(row, split))
            if index % 25 == 0 or index == len(items):
                labelled = [r for r in rows if r["match"]]
                accuracy = f", {100 * sum(r['match'] == 'Yes' for r in labelled) / len(labelled):.1f}% match" if labelled else ""
                rate = (time.time() - started) / index
                print(f"  {index}/{len(items)}{accuracy}, {rate:.2f}s per image, ~{rate * (len(items) - index) / 60:.1f} min left", flush=True)

    name = args.name or (args.dataset.stem if args.dataset.is_file() else args.dataset.resolve().name) or "readings"
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    csv_path, xlsx_path = args.out / f"{name}.csv", args.out / f"{name}.xlsx"
    csv_path.write_text(csv_text(rows, CLI_COLUMNS), encoding="utf-8", newline="")
    xlsx_path.write_bytes(xlsx_bytes(rows, CLI_COLUMNS, summary, title=f"Water meter readings — {name}"))
    print()
    for label, value in summary:
        print(f"  {label}: {value}")
    print(f"\nSaved {xlsx_path}\nSaved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
