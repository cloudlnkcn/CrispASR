#!/usr/bin/env python
"""Isolated, memory-bounded PatchEncoder reference for the dots-tts diff harness.

Loads ONLY the ~128 M-param VAESemanticEncoder (``patch_encoder.*``) from
``model.safetensors`` — never the 1.5 B LLM / DiT / vocoder — so it stays well
under 1 GB resident and cannot OOM a 16 GB machine. (Loading the full model in
float32 is ~11 GB and crashed the M1; do not do that here.)

It runs ``VAESemanticEncoder.decode_patch`` on a fixed, seeded latent patch with
an empty conv tail and empty KV caches (the "patch 0" case), and writes a small
GGUF the C++ ``crispasr-diff dots-tts`` branch can read:

  penc_in_patch0        (patch_size, latent_dim)   latent fed to decode_patch
  penc_positions_patch0 (out_ds_rate,)             decode positions, as f32
  penc_conv_tail_patch0 (latent_dim, left_pad)     conv tail (zeros for patch 0)
  penc_out_patch0       (1, out_dim=1536)          decode_patch llm_embedding

Usage:
  python tools/dots_penc_reference.py \
      --model-dir /Volumes/backups/ai/crispasr-gguf/dots-tts-soar-src \
      --output    /Volumes/backups/ai/crispasr-gguf/dots-tts-penc-ref.gguf \
      --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


class _Cfg(dict):
    """dict that also supports attribute access + .get (duck-types ConfigBase)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(k) from e


def _install_import_stubs() -> None:
    """Stub heavy text deps so importing the dots_tts modules doesn't pull
    lingua / WeTextProcessing (pynini). We only import the backbone module."""

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, _n):
            return _Dummy()

    def _mod(name, **attrs):
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    try:
        import lingua  # noqa: F401
    except ImportError:
        _mod("lingua", Language=_Dummy, LanguageDetectorBuilder=_Dummy)
    try:
        import tn  # noqa: F401
    except ImportError:
        _mod("tn")
        _mod("tn.chinese")
        _mod("tn.chinese.normalizer", Normalizer=_Dummy)
        _mod("tn.english")
        _mod("tn.english.normalizer", Normalizer=_Dummy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _install_import_stubs()
    import torch
    from safetensors import safe_open
    from dots_tts.modules.backbone.semantic_encoder import VAESemanticEncoder

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((args.model_dir / "config.json").read_text())
    latent_dim = int(cj["latent_dim"])
    pe_cfg = _Cfg(cj["PatchEncoder"])
    top_cfg = _Cfg({
        "patch_size": int(cj["patch_size"]),
        "latent_dim": latent_dim,
        "PatchEncoder": pe_cfg,
    })

    # out_dim = LLM hidden size; read it straight off out_proj to avoid guessing.
    st_path = args.model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        out_dim = list(f.get_slice("patch_encoder.out_proj.weight").get_shape())[0]
        penc_sd = {}
        for k in f.keys():
            if k.startswith("patch_encoder."):
                penc_sd[k[len("patch_encoder."):]] = f.get_tensor(k).float()
    print(f"[penc-ref] out_dim={out_dim} penc tensors={len(penc_sd)}")

    enc = VAESemanticEncoder(in_dim=latent_dim, out_dim=int(out_dim), config=top_cfg)
    missing, unexpected = enc.load_state_dict(penc_sd, strict=False)
    if missing:
        print(f"[penc-ref] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[penc-ref] WARNING unexpected keys: {unexpected}")
    enc.eval().float()

    patch_size = int(cj["patch_size"])
    g = torch.Generator().manual_seed(args.seed)
    latent_patch = torch.randn(1, patch_size, latent_dim, generator=g, dtype=torch.float32)

    state = enc.init_decode_state(
        max_audio_patch_count=4, batch_size=1,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    positions = torch.arange(enc.out_ds_rate, dtype=torch.long)  # patch 0: [0, 1]
    embedding, _conv_tail = enc.decode_patch(
        latent_patch, state.conv_tail, state.layer_caches, positions,
    )

    def npf(t):
        return np.ascontiguousarray(t.detach().cpu().float().numpy())

    tensors = {
        "penc_in_patch0": npf(latent_patch[0]),               # (patch_size, latent_dim)
        "penc_positions_patch0": positions.float().numpy(),   # (out_ds_rate,)
        "penc_conv_tail_patch0": npf(state.conv_tail[0]),     # (latent_dim, left_pad)
        "penc_out_patch0": npf(embedding[0]),                 # (1, out_dim)
    }
    for k, v in tensors.items():
        print(f"[penc-ref] {k}: {v.shape} {v.dtype}")

    import gguf
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.output), arch="crispasr.reference")
    w.add_description("dots-tts isolated PatchEncoder reference")
    w.add_int32("crispasr.ref.seed", args.seed)
    w.add_int32("crispasr.ref.patch_size", patch_size)
    w.add_int32("crispasr.ref.out_ds_rate", int(enc.out_ds_rate))
    for name, arr in sorted(tensors.items()):
        a = np.ascontiguousarray(arr.astype(np.float32))
        if a.ndim == 0:
            a = a.reshape(1)
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[penc-ref] wrote {args.output} ({args.output.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
