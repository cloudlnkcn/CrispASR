"""
CrispASR -- orpheus talker CUDA diff: localize the 0-byte sweep failure.

Orpheus passes on M1 Metal/CPU but emits 0 bytes on CUDA (PLAN §201). The
SNAC-only `orpheus` diff doesn't touch the talker (Llama-3.2-3B AR decode,
§176b Lk-bucket + device KV) where the failure must live. This kernel runs
the talker-level diff against PyTorch ground truth, GPU vs CPU:

  1. Build CrispASR with CUDA (crispasr-diff).
  2. Download unsloth/orpheus-3b-0.1-ft (tara) + convert to F16 GGUF.
  3. Generate the talker reference (greedy codec-token stream) in PyTorch.
  4. Run A: crispasr-diff orpheus-talker on CPU  → expect PASS vs ref.
  5. Run B: crispasr-diff orpheus-talker on GPU  → expect FAIL/0-byte.
  Comparing A vs B vs the PyTorch ground truth localizes the CUDA bug and
  prints the first diverging codec token.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
HF_DIR = WORK / "orpheus-3b-hf"
GGUF = WORK / "orpheus-3b-0.1-ft-f16.gguf"
REF = WORK / "orpheus-talker-ref.gguf"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("ORPHEUS_HF_MODEL", "unsloth/orpheus-3b-0.1-ft")
TEXT = os.environ.get("ORPHEUS_TEXT", "Hey there, my name is Tara.")
SPEAKER = os.environ.get("ORPHEUS_SPEAKER", "tara")
MAXGEN = os.environ.get("ORPHEUS_DIFF_MAXGEN", "48")

PROGRESS = RESULTS / "progress.jsonl"
_T0 = time.time()


def step(name, **kv):
    line = json.dumps({"t": round(time.time() - _T0, 2), "step": name, **kv})
    print(f"[step] {line}", flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + build ──────────────────────────────────────────────────────────
step("start", ref=CRISPASR_REF)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
     CRISPASR_REPO, str(REPO)])
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
step("cloned", sha=sha)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
     "-DBUILD_SHARED_LIBS=ON"] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-diff "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")

DIFF = BUILD / "bin" / "crispasr-diff"
if not DIFF.exists():
    cands = [c for c in BUILD.rglob("crispasr-diff") if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit("crispasr-diff binary not found")
    DIFF = cands[0]
step("diff_found", path=str(DIFF))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ── Download HF model ──────────────────────────────────────────────────────
hf_token = kh.resolve_hf_token()
step("hf_token", have=bool(hf_token))
from huggingface_hub import snapshot_download  # noqa: E402
snapshot_download(HF_MODEL, local_dir=str(HF_DIR), token=hf_token,
                  allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"])
step("hf_downloaded")

# ── Convert to F16 GGUF ────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=False)
run([sys.executable, str(REPO / "models" / "convert-orpheus-to-gguf.py"),
     "--input", str(HF_DIR), "--output", str(GGUF), "--outtype", "f16",
     "--variant", "fixed_speaker"],
    env={"OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
step("converted", gguf_mb=round(GGUF.stat().st_size / 1e6, 1))

# read codec block range from the GGUF so the reference filter matches
import gguf as _g  # noqa: E402
_r = _g.GGUFReader(str(GGUF))
offset = int(_r.fields["orpheus.custom_token_offset"].parts[-1][0])
count = int(_r.fields["orpheus.custom_token_count"].parts[-1][0])
step("codec_range", offset=offset, count=count)

# ── Generate the talker reference (PyTorch greedy) ─────────────────────────
with kh.build_heartbeat("ref.gen"):
    run([sys.executable, str(REPO / "tools" / "dump_reference.py"),
         "--backend", "orpheus-talker", "--model-dir", str(HF_DIR),
         "--audio", str(REPO / "samples" / "jfk.wav"), "--output", str(REF),
         "--max-new-tokens", MAXGEN],
        env={"ORPHEUS_TEXT": TEXT, "ORPHEUS_SPEAKER": SPEAKER,
             "ORPHEUS_CUSTOM_OFFSET": str(offset), "ORPHEUS_CUSTOM_COUNT": str(count),
             "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
step("ref_generated", ref_mb=round(REF.stat().st_size / 1e6, 3))

# ── Run the diff: CPU then GPU ─────────────────────────────────────────────
def run_diff(label, extra_env, timeout=900):
    step(f"{label}_start")
    cmd = [str(DIFF), "orpheus-talker", str(GGUF), str(REF),
           str(REPO / "samples" / "jfk.wav")]
    env = {"ORPHEUS_DIFF_MAXGEN": MAXGEN, **extra_env}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env={**os.environ, **env})
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc, out = -1, (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    (RESULTS / f"{label}.txt").write_text(out or "")
    print((out or "")[-4000:], flush=True)
    verdict = "PASS" if "→ PASS" in (out or "") else ("FAIL" if "FAIL" in (out or "") else "?")
    step(f"{label}_done", rc=rc, elapsed=round(time.time() - t0, 2), verdict=verdict)
    return {"label": label, "rc": rc, "verdict": verdict}

results = []
results.append(run_diff("diff_cpu", {}))                       # parity baseline
results.append(run_diff("diff_gpu", {"ORPHEUS_DIFF_GPU": "1"}))  # CUDA repro

(RESULTS / "summary.json").write_text(json.dumps(
    {"sha": sha, "gpu": gpu_name, "model": HF_MODEL, "codec_offset": offset,
     "codec_count": count, "results": results}, indent=2))
step("done", results=results)
print("\n=== SUMMARY ===", flush=True)
for r in results:
    print(f"  {r['label']:10s} rc={r['rc']} verdict={r['verdict']}", flush=True)
