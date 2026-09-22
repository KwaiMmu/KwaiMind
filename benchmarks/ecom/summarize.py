from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = []
    for path in sorted(args.scores_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "task_id" in record and "aggregate" in record:
            tasks.append(record)
    values = [task["aggregate"]["overall"] for task in tasks if task["aggregate"]["overall"] is not None]
    summary = {"score": round(sum(values) / len(values), 4) if values else None, "num_tasks": len(tasks), "tasks": tasks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "overall", "scored", "total"])
        for task in tasks:
            writer.writerow([task["task_id"], task["aggregate"]["overall"], task["num_scored"], task["num_cases"]])


if __name__ == "__main__":
    main()
