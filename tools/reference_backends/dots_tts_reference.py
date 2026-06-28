"""dots.tts reference dump backend.

Captures stage-by-stage activations from the official PyTorch dots.tts
model (rednote-hilab/dots.tts-soar) so the CrispASR C++ dots-tts backend
can be diffed via `crispasr-diff dots-tts`.

Architecture (see src/dots_tts/models/dots_tts/core.py):
  Qwen2.5-1.5B LLM backbone (28L, GQA 12Q/2KV) generates audio latents
  patch-by-patch via an 18-layer DiT (``velocity_field_predictor``)
  flow-matching head. A 24-layer PatchEncoder (``patch_encoder``,
  VAESemanticEncoder) maps each generated latent patch back to one LLM
  embedding via ``decode_patch``. A BigVGAN vocoder decodes the full latent
  stream to 48 kHz audio.

Per-patch loop (model.py ``_decode`` / ``_consume_audio_patch``):
  DiT ODE  ->  denormalize  ->  patch_encoder.decode_patch  ->  step_llm
                                                             ->  eos_proj

The PatchEncoder is the critical stage to validate: its ``decode_patch``
takes a denormalized latent patch ``(B, patch_size, latent_dim)`` plus a
conv tail and KV caches, and returns one LLM embedding ``(B, 1, 1536)``.
For patch 0 the conv tail is zero and the KV caches are empty, so it is
the cleanest isolation point.

Stages dumped (all f32, GGUF column-major == numpy row-major flat):
  penc_in_patch0        (patch_size, latent_dim)  latent fed to decode_patch
  penc_positions_patch0 (out_ds_rate,)            decode positions (as f32)
  penc_out_patch0       (1, 1536)                 decode_patch llm_embedding
  dit_vel_step0         (patch_size, latent_dim)  DiT velocity, first call
  llm_prefill_hidden    (T_text, 1536)            LLM hidden after text prefill
  audio                 (1, T_audio)              final PCM at 48 kHz

Drive with:
  DOTS_MODEL_DIR   local snapshot dir (default: the crispasr-gguf src dir).
  DOTS_TEXT        input text (default "Hi there.").
  DOTS_NUM_STEPS   ODE steps (default 4 for dump speed).
  DOTS_CFG_SCALE   guidance scale (default 1.2, runtime default).
  DOTS_MAX_GEN_LEN cap LLM generate length / patches (default 96).
  DOTS_SEED        deterministic seed (default 42).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_MODEL_DIR = "/Volumes/backups/ai/crispasr-gguf/dots-tts-soar-src"

DEFAULT_STAGES = [
    "penc_in_patch0",
    "penc_positions_patch0",
    "penc_out_patch0",
    "dit_vel_step0",
    "llm_prefill_hidden",
    "audio",
]


def required_packages() -> list[str]:
    return ["dots_tts @ git+https://github.com/rednote-hilab/dots.tts.git"]


def _np(t) -> np.ndarray:
    """Detach a torch tensor to a contiguous float32 numpy array."""
    return t.detach().to("cpu").float().contiguous().numpy()


def _install_import_stubs() -> None:
    """Stub heavy text-frontend deps that dots_tts imports at module load.

    ``dots_tts.utils.text`` hard-imports ``lingua`` (language detection) and
    ``tn`` (WeTextProcessing / pynini, a pain to build on macOS) at module
    top. We only ever run with ``language=None`` and ``normalize_text=False``,
    so those code paths are never executed — only imported. Register inert
    stub modules so the import succeeds without the real packages.
    """
    import sys
    import types

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, _name):
            return _Dummy()

    def _mod(name, **attrs):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    try:
        import lingua  # noqa: F401  (real package present → use it)
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


def dump(
    model_dir: "Path | str | None" = None,
    audio: np.ndarray | None = None,
    stages: Set[str] | None = None,
    max_new_tokens: int = 20,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Run dots.tts reference and return per-stage intermediates.

    The unified harness (tools/dump_reference.py) serializes the returned
    dict to a GGUF archive itself, so this only returns numpy arrays.
    ``audio`` is ignored (dots.tts is text-to-speech); the input text comes
    from ``$DOTS_TEXT``.
    """
    import torch

    if stages is None:
        stages = set(DEFAULT_STAGES)

    model_dir = os.environ.get("DOTS_MODEL_DIR") or (
        str(model_dir) if model_dir else DEFAULT_MODEL_DIR)
    text = os.environ.get("DOTS_TEXT", "Hi there.")
    num_steps = int(os.environ.get("DOTS_NUM_STEPS", "4"))
    cfg_scale = float(os.environ.get("DOTS_CFG_SCALE", "1.2"))
    max_gen_len = int(os.environ.get("DOTS_MAX_GEN_LEN", "96"))
    seed = int(os.environ.get("DOTS_SEED", "42"))

    print(f"[dots-ref] model_dir={model_dir}")
    print(f"[dots-ref] text={text!r} num_steps={num_steps} cfg={cfg_scale} "
          f"max_gen_len={max_gen_len} seed={seed}")

    _install_import_stubs()
    from dots_tts.runtime import DotsTtsRuntime

    runtime = DotsTtsRuntime.from_pretrained(
        model_dir,
        precision="float32",
        max_generate_length=max_gen_len,
    )
    model = runtime.model
    core = model.core
    core.eval()

    results: Dict[str, np.ndarray] = {}
    captured: Dict[str, Any] = {}

    # ---- Hook: PatchEncoder.decode_patch (capture first patch only) ----
    orig_decode_patch = core.patch_encoder.decode_patch

    def decode_patch_hook(latent_patch, conv_tail, layer_caches, positions, *a, **kw):
        out = orig_decode_patch(latent_patch, conv_tail, layer_caches, positions, *a, **kw)
        if "penc_in_patch0" not in captured:
            embedding = out[0] if isinstance(out, tuple) else out
            captured["penc_in_patch0"] = _np(latent_patch[0])      # (patch_size, latent_dim)
            captured["penc_conv_tail_patch0"] = _np(conv_tail[0])  # (latent_dim, left_pad)
            captured["penc_positions_patch0"] = _np(positions).astype(np.float32)
            captured["penc_out_patch0"] = _np(embedding[0])        # (1, 1536) or (1536,)
            print(f"[dots-ref] penc patch0: in={captured['penc_in_patch0'].shape} "
                  f"pos={captured['penc_positions_patch0']} "
                  f"out={captured['penc_out_patch0'].shape}")
        return out

    core.patch_encoder.decode_patch = decode_patch_hook

    # ---- Hook: DiT velocity field (capture first forward output) ----
    def dit_hook(_module, _inp, out):
        if "dit_vel_step0" not in captured:
            vel = out[0] if isinstance(out, (tuple, list)) else out
            captured["dit_vel_step0"] = _np(vel)
            print(f"[dots-ref] dit velocity step0: {captured['dit_vel_step0'].shape}")

    dit_handle = core.velocity_field_predictor.register_forward_hook(dit_hook)

    # ---- Run generation ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("[dots-ref] generating...")
    try:
        gen = runtime.generate(
            text=text,
            num_steps=num_steps,
            guidance_scale=cfg_scale,
        )
        audio_t = gen["audio"] if isinstance(gen, dict) else gen
        captured["audio"] = _np(audio_t)
        print(f"[dots-ref] audio: {captured['audio'].shape}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[dots-ref] generation failed: {e}")
        traceback.print_exc()
    finally:
        dit_handle.remove()
        core.patch_encoder.decode_patch = orig_decode_patch

    # ---- LLM prefill hidden (independent of the loop) ----
    if "llm_prefill_hidden" in stages:
        try:
            with torch.no_grad():
                token_ids = core.tokenizer.encode(text)
                input_ids = torch.tensor([token_ids], dtype=torch.long)
                embeds = core.llm.get_input_embeddings()(input_ids)
                outputs = core.llm(inputs_embeds=embeds, output_hidden_states=True,
                                   use_cache=False)
                captured["llm_prefill_hidden"] = _np(outputs.hidden_states[-1][0])
                print(f"[dots-ref] llm_prefill_hidden: "
                      f"{captured['llm_prefill_hidden'].shape}")
        except Exception as e:  # noqa: BLE001
            print(f"[dots-ref] llm prefill failed: {e}")

    # ---- Collect + return (the harness serializes to GGUF) ----
    for k, v in captured.items():
        a = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
        if a.ndim == 0:
            a = a.reshape(1)
        results[k] = a
    print(f"[dots-ref] returning {len(results)} tensors: {sorted(results)}")
    return results
