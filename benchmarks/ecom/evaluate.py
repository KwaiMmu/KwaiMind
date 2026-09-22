from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps


def encode_image(path: Path, max_pixels: int) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def parse_scores(text: str, dimensions: list[str]) -> tuple[dict[str, int], str]:
    scores = {}
    for dimension in dimensions:
        match = re.search(rf"{re.escape(dimension)}\s*[:：]\s*([1-5])", text)
        if match:
            scores[dimension] = int(match.group(1))
    reason_match = re.search(r"简要理由\s*[:：]\s*(.+)", text)
    return scores, reason_match.group(1).strip()[:80] if reason_match else ""


def load_judge(path: Path, task_id: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["task_id"] == task_id:
            return record
    raise ValueError(f"Unknown task: {task_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--judge-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--flush-every", type=int, default=10,
                        help="Write partial scores to disk after this many completed cases")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY before running evaluation")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    judge = load_judge(args.judge_jsonl, args.task_id)
    dimensions = judge["dimension_names_cn"]
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    previous = {}
    if args.output.is_file() and not args.overwrite:
        previous = json.loads(args.output.read_text(encoding="utf-8")).get("cases", {})

    def evaluate(record: dict) -> tuple[str, dict]:
        case_id = str(record["id"])
        result_path = args.result_dir / f"{case_id}.png"
        if not result_path.is_file():
            return case_id, {"missing_result": True}
        parts = []
        for relative in record["images"]:
            parts.append(types.Part.from_bytes(data=encode_image(args.data_root / relative, args.max_pixels), mime_type="image/jpeg"))
        parts.append(types.Part.from_bytes(data=encode_image(result_path, args.max_pixels), mime_type="image/jpeg"))
        prompt = judge["judge_prompt"].replace("<edit_prompt>", str(record["prompt"]))
        parts.append(types.Part.from_text(text=prompt))
        last_error = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(model=args.model, contents=[types.Content(role="user", parts=parts)])
                scores, reason = parse_scores(response.text or "", dimensions)
                if len(scores) != len(dimensions):
                    return case_id, {"parse_error": True, "raw": (response.text or "")[:500]}
                values = [scores[name] for name in dimensions]
                return case_id, {"scores": scores, "overall": round(math.prod(values) ** (1 / len(values)), 4), "reason": reason}
            except Exception as error:
                last_error = error
                time.sleep(2 ** attempt)
        return case_id, {"error": str(last_error)}

    todo = [record for record in records if "scores" not in previous.get(str(record["id"]), {})]
    results = {case_id: result for case_id, result in previous.items() if "scores" in result}
    def provenance() -> dict:
        return {
            "judge_model": args.model,
            "google_genai_version": importlib.metadata.version("google-genai"),
            "judge_prompt_sha256": hashlib.sha256(judge["judge_prompt"].encode("utf-8")).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "max_pixels": args.max_pixels,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def build_output(case_results: dict) -> dict:
        scored = [result for result in case_results.values() if "scores" in result]
        per_dimension = {
            name: round(sum(result["scores"][name] for result in scored) / len(scored), 4) if scored else None
            for name in dimensions
        }
        return {
            "task_id": args.task_id,
            "dimensions": dimensions,
            "num_cases": len(records),
            "num_scored": len(scored),
            "provenance": provenance(),
            "aggregate": {
                "overall": round(sum(result["overall"] for result in scored) / len(scored), 4) if scored else None,
                "per_dim": per_dimension,
            },
            "cases": case_results,
        }

    def flush(case_results: dict) -> None:
        # Write to a sibling temp file and rename so an interrupt cannot corrupt the
        # scores file: a resumed run either sees the previous flush or this one, and
        # never pays the API again for a case that was already scored.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_suffix(".json.partial")
        staging.write_text(json.dumps(build_output(case_results), ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate, record) for record in todo]
        for index, future in enumerate(as_completed(futures), start=1):
            case_id, result = future.result()
            results[case_id] = result
            if index % args.flush_every == 0:
                flush(results)
    flush(results)
    scored = [result for result in results.values() if "scores" in result]
    if len(scored) != len(records):
        raise RuntimeError(f"Scored {len(scored)} of {len(records)} cases")


if __name__ == "__main__":
    main()
