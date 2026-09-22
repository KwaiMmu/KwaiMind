from __future__ import annotations

import json
from pathlib import Path


def load_manifest(path: str | Path, data_root: str | Path | None = None) -> list[dict]:
    manifest = Path(path).resolve()
    root = Path(data_root).resolve() if data_root else manifest.parent
    with manifest.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a JSON list")
    normalized = []
    for index, record in enumerate(records):
        item = dict(record)
        item["id"] = str(item.get("id", index))
        images = item.get("images", item.get("edit_image"))
        if isinstance(images, str):
            images = [images]
        if not images or not item.get("prompt"):
            raise ValueError(f"Invalid record at index {index}")
        resolved = []
        for image in images:
            path_obj = Path(image)
            resolved.append(str(path_obj if path_obj.is_absolute() else root / path_obj))
        item["images"] = resolved
        normalized.append(item)
    return normalized
