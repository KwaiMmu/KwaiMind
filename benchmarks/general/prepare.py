from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_imgedit(source: Path, image_root: Path) -> list[dict]:
    data = json.loads(source.read_text(encoding="utf-8"))
    return [
        {"id": str(key), "prompt": item["prompt"], "images": [str(image_root / item["id"])], "metadata": {"edit_type": item.get("edit_type")}}
        for key, item in data.items()
    ]


def convert_rededit(source: Path, image_root: Path, language: str) -> list[dict]:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt_key = "a_to_b_instructions_eng" if language == "en" else "a_to_b_instructions"
    output = []
    for item in records:
        relative = item["source"].removeprefix("redbench/")
        output.append({"id": f"{item['task']}-{item['id']}", "prompt": item[prompt_key], "images": [str(image_root / relative)], "metadata": {"task": item["task"], "language": language}})
    return output


def convert_gedit(source: Path, image_root: Path) -> list[dict]:
    data = json.loads(source.read_text(encoding="utf-8"))
    return [
        {"id": item["key"], "prompt": item["instruction"], "images": [str(image_root / item["source_image"])], "metadata": {"task": item["task_type"], "language": item["instruction_language"]}}
        for item in data
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["imgedit", "rededit", "gedit"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["en", "cn"], default="en")
    args = parser.parse_args()
    if args.benchmark == "imgedit":
        records = convert_imgedit(args.source, args.image_root)
    elif args.benchmark == "rededit":
        records = convert_rededit(args.source, args.image_root, args.language)
    else:
        records = convert_gedit(args.source, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
