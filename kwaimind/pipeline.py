from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps


DEFAULT_TEXT_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_PROCESSOR_ID = "Qwen/Qwen-Image-Edit"


def resolve_checkpoint(path: str | Path) -> list[str]:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file() and checkpoint.suffix == ".safetensors":
        return [str(checkpoint)]
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    index_path = checkpoint / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        names = sorted(set(index["weight_map"].values()))
        shards = [checkpoint / name for name in names]
    else:
        shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards or any(not shard.is_file() for shard in shards):
        raise FileNotFoundError(f"No complete safetensors checkpoint found in {checkpoint}")
    return [str(shard) for shard in shards]


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def fit_size(width: int, height: int, max_pixels: int | None, rounding: str = "nearest") -> tuple[int, int]:
    """Align an output size to a multiple of 16, honouring an optional pixel budget.

    `rounding="nearest"` (the default and the released behaviour) rounds each side to the
    closest multiple of 16, with ties going up; this reproduces the alignment used for the
    reported E-com results. `rounding="floor"` truncates both sides instead, which shrinks
    non-multiple-of-16 inputs (750 -> 736 rather than 752) and changes benchmark conditions.
    """
    if rounding not in ("nearest", "floor"):
        raise ValueError(f"rounding must be 'nearest' or 'floor', got {rounding!r}")
    if max_pixels and width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width = max(16, int(width * scale))
        height = max(16, int(height * scale))
    if rounding == "floor":
        return max(16, width // 16 * 16), max(16, height // 16 * 16)
    aligned_width = max(16, (width + 8) // 16 * 16)
    aligned_height = max(16, (height + 8) // 16 * 16)
    if max_pixels and aligned_width * aligned_height > max_pixels:
        scale = math.sqrt(max_pixels / (aligned_width * aligned_height))
        aligned_width = max(16, int(aligned_width * scale) // 16 * 16)
        aligned_height = max(16, int(aligned_height * scale) // 16 * 16)
    return aligned_width, aligned_height


def checkpoint_state_keys(shards: list[str]) -> set[str]:
    """Return the parameter names a checkpoint provides, without reading any tensors."""
    from safetensors import safe_open

    keys = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("pipe.dit."):
                    key = key[len("pipe.dit."):]
                elif key.startswith("dit."):
                    key = key[len("dit."):]
                if key in keys:
                    raise ValueError(f"Duplicate checkpoint key: {key}")
                keys.add(key)
    return keys


class KwaiMindPipeline:
    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        device: str = "cuda",
        text_model_id: str = DEFAULT_TEXT_MODEL_ID,
        processor_id: str = DEFAULT_PROCESSOR_ID,
        vram_limit: float | None = None,
    ) -> "KwaiMindPipeline":
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        # A KwaiMind checkpoint is a complete transformer, so it is loaded *as* the
        # transformer. Loading the base transformer first and then overwriting every
        # parameter would read ~40 GiB of weights that are all discarded, and would hold
        # both copies at once — enough to exhaust a 96 GiB card.
        shards = resolve_checkpoint(checkpoint)
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=shards),
                ModelConfig(model_id=text_model_id, origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id=text_model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(model_id=processor_id, origin_file_pattern="processor/"),
            vram_limit=vram_limit,
        )
        if pipe.dit is None:
            raise ValueError(f"DiffSynth did not recognise {checkpoint} as a Qwen-Image transformer")
        expected = set(pipe.dit.state_dict())
        provided = checkpoint_state_keys(shards)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            raise ValueError(
                f"Incompatible checkpoint: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        pipe.dit.eval()
        return cls(pipe)

    def __call__(
        self,
        prompt: str,
        images: Sequence[Image.Image | str | Path],
        seed: int = 42,
        steps: int = 40,
        width: int | None = None,
        height: int | None = None,
        max_pixels: int | None = 1_440_000,
        cfg_scale: float = 4.0,
        negative_prompt: str = "",
        rounding: str = "nearest",
        single_image_mode: str = "list",
        edit_image_auto_resize: bool = True,
        zero_cond_t: bool = True,
    ) -> Image.Image:
        """Edit `images` according to `prompt`.

        `single_image_mode` selects how a lone reference image reaches DiffSynth. "list"
        (the default and the released behaviour) wraps it in a list, which routes through
        `encode_prompt_edit_multi` and matches how the reported results were produced.
        "pil" passes the bare `PIL.Image`, routing through `encode_prompt_edit` and using a
        different prompt template.
        """
        if single_image_mode not in ("list", "pil"):
            raise ValueError(f"single_image_mode must be 'list' or 'pil', got {single_image_mode!r}")
        loaded = [load_image(image) if isinstance(image, (str, Path)) else image.convert("RGB") for image in images]
        if not loaded:
            raise ValueError("At least one input image is required")
        width, height = fit_size(width or loaded[0].width, height or loaded[0].height, max_pixels, rounding)
        edit_image = loaded[0] if len(loaded) == 1 and single_image_mode == "pil" else loaded
        with torch.inference_mode():
            return self.pipe(
                prompt,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                edit_image=edit_image,
                seed=seed,
                num_inference_steps=steps,
                height=height,
                width=width,
                edit_image_auto_resize=edit_image_auto_resize,
                zero_cond_t=zero_cond_t,
            )
