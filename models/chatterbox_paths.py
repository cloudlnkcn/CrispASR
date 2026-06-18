from __future__ import annotations

from pathlib import Path


def select_chatterbox_t3_checkpoint(model_dir: Path) -> Path:
    """Select the best available Chatterbox T3 checkpoint."""

    for name in ("t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    for candidate in sorted(model_dir.glob("t3_mtl*.safetensors")):
        return candidate

    return model_dir / "t3_cfg.safetensors"
