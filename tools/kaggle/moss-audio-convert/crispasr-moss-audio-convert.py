# %% [markdown]
# # CrispASR — MOSS-Audio-4B-Instruct GGUF conversion
#
# Convert `OpenMOSS-Team/MOSS-Audio-4B-Instruct` to F16 GGUF,
# quantize to Q4_K, upload to `cstr/MOSS-Audio-4B-Instruct-GGUF`.

# %% [code]
import os, subprocess, sys, shutil
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# HF_TOKEN
try:
    from kaggle_secrets import UserSecretsClient
    hf_token_secret = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token_secret
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret
    print("[1] HF_TOKEN OK", flush=True)
except Exception as exc:
    hf_token_secret = None
    print(f"[1] HF_TOKEN fail: {exc}", flush=True)

# %% [code]
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT_F16 = WORK / "moss-audio-4b-instruct-f16.gguf"
OUT_Q4K = WORK / "moss-audio-4b-instruct-q4_k.gguf"
BRANCH = os.environ.get("CRISPASR_REF", "feature/moss-audio")

print(f"[2] cloning {BRANCH}", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
])
print("[2] clone done", flush=True)

# %% [code]
print("[3] installing deps", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "gguf", "huggingface_hub", "hf_transfer",
])
print("[3] deps done", flush=True)

# %% [code]
print("[4] downloading model", flush=True)
from huggingface_hub import snapshot_download

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "moss-audio-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"[4] scratch: {scratch} (free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)", flush=True)

src = snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    cache_dir=str(scratch),
)
print(f"[4] source: {src}", flush=True)

# %% [code]
print("[5] converting F16", flush=True)
subprocess.check_call([
    sys.executable, str(REPO / "models" / "convert-moss-audio-to-gguf.py"),
    "--input", src,
    "--output", str(OUT_F16),
    "--outtype", "f16",
])
print(f"[5] F16: {OUT_F16.stat().st_size / (1024**3):.1f} GiB", flush=True)

# %% [code]
print("[6] building quantizer", flush=True)
BUILD = WORK / "build"
BUILD.mkdir(exist_ok=True)
# install ninja if missing
subprocess.run(["apt-get", "install", "-y", "-qq", "ninja-build"], check=False)
subprocess.check_call([
    "cmake", "-G", "Ninja", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
    "-DCRISPASR_BUILD_TESTS=OFF",
])
subprocess.check_call([
    "cmake", "--build", str(BUILD), "--target", "crispasr-quantize", "-j4",
])
QUANTIZE = BUILD / "bin" / "crispasr-quantize"
print(f"[6] quantizer: {QUANTIZE}", flush=True)

subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q4K), "q4_k"])
print(f"[6] Q4K: {OUT_Q4K.stat().st_size / (1024**3):.1f} GiB", flush=True)

# free F16
OUT_F16.unlink(missing_ok=True)

# %% [code]
HF_REPO = "cstr/MOSS-Audio-4B-Instruct-GGUF"
if hf_token_secret:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_secret)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[7] repo: {e}", flush=True)
    if OUT_Q4K.exists():
        print(f"[7] uploading Q4K ({OUT_Q4K.stat().st_size / (1024**3):.1f} GiB)", flush=True)
        api.upload_file(
            path_or_fileobj=str(OUT_Q4K),
            path_in_repo="moss-audio-4b-instruct-q4_k.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message="Add Q4_K GGUF (PLAN #58)",
        )
        print("[7] uploaded", flush=True)
else:
    print("[7] no token — staged locally", flush=True)
    if OUT_Q4K.exists():
        print(f"  {OUT_Q4K} ({OUT_Q4K.stat().st_size / (1024**3):.1f} GiB)")

print("[DONE]", flush=True)
