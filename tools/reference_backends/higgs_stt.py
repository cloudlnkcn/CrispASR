"""bosonai/higgs-audio-v3-stt reference dump backend.

Captures the architectural boundaries that the C++ runtime (src/higgs_stt.cpp)
reproduces, so crispasr-diff can localize the first divergence:

  raw_audio         the 16 kHz mono PCM fed in
  mel_spectrogram   WhisperFeatureExtractor log-mel (128, 3000)
  encoder_hidden    audio_tower output (post-AvgPool + final LayerNorm) (750, 1280)
  encoder_out       audio_encoder_proj output (projected audio embeds)  (375, 2048)
  generated_text    best-effort greedy transcript (optional)

The encoder/projector stages are the priority — they are the genuinely new
ggml code (Whisper encoder + AvgPool, depthwise-temporal-conv MLP projector).
The model is loaded in bf16 (≈5.4 GB) to stay within a 16 GB box; that is more
than enough precision to localize a structural bug (a layout error collapses
cosine to ~0 regardless of dtype).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_hidden",
    "encoder_out",
    "generated_text",
]


def _pad_or_trim_mel(mel: np.ndarray, target_t: int = 3000) -> np.ndarray:
    """mel: (n_mels, T) -> (n_mels, target_t), zero-padded / truncated."""
    n_mels, T = mel.shape
    if T == target_t:
        return mel
    out = np.zeros((n_mels, target_t), dtype=mel.dtype)
    out[:, : min(T, target_t)] = mel[:, : min(T, target_t)]
    return out


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoModel

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32)

    # ---- mel (Whisper-large-v3 feature extractor), padded to the 30 s window ----
    from transformers import WhisperFeatureExtractor
    try:
        fe = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    except Exception:
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    feats = fe(np.asarray(audio, dtype=np.float32), sampling_rate=16000,
               return_tensors="pt").input_features  # (1, 128, T)
    mel_np = feats[0].detach().cpu().float().numpy()  # (128, T)
    mel_np = _pad_or_trim_mel(mel_np, 3000)
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = mel_np

    need_model = bool({"encoder_hidden", "encoder_out", "generated_text"} & stages)
    if not need_model:
        return out

    print(f"  loading HiggsAudio3 (bf16) from {model_dir}")
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()

    # Locate the audio submodules (top-level on HiggsAudio3Model; tolerate a
    # `.model` wrapper just in case a transformers version nests them).
    root = model
    if not hasattr(root, "audio_tower") and hasattr(model, "model"):
        root = model.model
    audio_tower = getattr(root, "audio_tower", None)
    projector = getattr(root, "audio_encoder_proj", None)

    feats_bf16 = torch.from_numpy(mel_np).to(torch.bfloat16).unsqueeze(0)  # (1,128,3000)

    with torch.no_grad():
        enc_hidden = None
        if ("encoder_hidden" in stages or "encoder_out" in stages) and audio_tower is not None:
            enc = audio_tower(feats_bf16)
            enc_hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc
            if isinstance(enc_hidden, (tuple, list)):
                enc_hidden = enc_hidden[0]
            if "encoder_hidden" in stages:
                out["encoder_hidden"] = enc_hidden[0].detach().cpu().float().numpy()  # (750,1280)

        if "encoder_out" in stages and projector is not None and enc_hidden is not None:
            proj = projector(enc_hidden)
            if isinstance(proj, (tuple, list)):
                proj = proj[0]
            out["encoder_out"] = proj[0].detach().cpu().float().numpy()  # (375,2048)

    # ---- best-effort transcript via the model's own helper ----
    if "generated_text" in stages:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            # Reuse the repo's transcribe helper if importable; otherwise skip.
            import importlib.util
            tp = Path(model_dir) / "transcribe.py"
            text = ""
            if tp.exists():
                spec = importlib.util.spec_from_file_location("higgs_transcribe", str(tp))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                # transcribe.py exposes a transcribe(...)-style entry; signatures
                # vary across releases, so guard the call.
                fn = getattr(mod, "transcribe", None)
                if callable(fn):
                    text = fn(model, tok, audio) if fn.__code__.co_argcount >= 3 else ""
            out["generated_text"] = text or ""
        except Exception as e:  # noqa: BLE001
            print(f"  generated_text skipped: {e}")
            out["generated_text"] = ""

    return out
