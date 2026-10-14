#!/usr/bin/env python3
"""mirror-modelscope.py — keep a ModelScope mirror of every model URL built
into CrispASR's auto-download paths.

The C++ sources hardcode huggingface.co resolve URLs (registry, VAD/LID/punc
loaders, TTS companions). crispasr_cache.cpp rewrites those URLs to a
ModelScope mirror when huggingface.co fails or CRISPASR_MIRROR=modelscope is
set — but the mirror only exists if someone uploaded it. This script is that
someone:

  1. inventory   — extract every `https://huggingface.co/<org>/<repo>/resolve/
                   main/<file>` URL from the source tree
  2. diff        — check which (repo, file) pairs are missing on ModelScope
                   (namespace `cloudlnk/<repo>` by default)
  3. mirror      — per repo: download the missing files from hf-mirror.com,
                   create the ModelScope repo, upload, verify

Modes:
  python3 scripts/mirror-modelscope.py check             # report gaps only
  python3 scripts/mirror-modelscope.py mirror            # fill the gaps
  ... --namespace cloudlnk --max-bytes 2e9 --budget 600  # slicing knobs

Notes:
  * ModelScope repo creation has been observed to trip modelscope.cn's WAF
    ("YUNWAF_CLIENT_UNCLASSIFIED") from some networks/clients while uploads
    to existing repos keep working. When `ms create` fails this script marks
    the repo `needs-create` and keeps going; run it again from a network
    where creation works, or create the repos once via the web UI —
    re-running then fills them.
  * `ms` CLI must be logged in (`ms login`) for mirror mode.
  * State file (.mirror-state.json, resumable) lives in the output dir.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

SOURCE_DIRS = ["src", "examples/cli", "examples/server", "include", "tools",
               "bindings", "python", "ci", "scripts"]
HF_RE = re.compile(r'https://huggingface\.co/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)'
                   r'/resolve/([^"\s<>)]+)')
DATASET_RE = re.compile(r'https://huggingface\.co/datasets/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)'
                        r'/resolve/([^"\s<>)]+)')
DEFAULT_NS = "cloudlnk"
HF_MIRROR = "https://hf-mirror.com"


def extract_inventory(root):
    """-> ({hf_repo: {files}}, {dataset_repo: {files}})"""
    models, datasets = {}, {}
    for d in SOURCE_DIRS:
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        out = subprocess.run(["grep", "-rh", "https://huggingface.co", p],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            for repo, ref_and_path in DATASET_RE.findall(line):
                path = ref_and_path.split("/", 1)[1] if "/" in ref_and_path else ref_and_path
                datasets.setdefault(repo, set()).add(path.rstrip("\\,;"))
            for repo, ref_and_path in HF_RE.findall(line):
                path = ref_and_path.split("/", 1)[1] if "/" in ref_and_path else ref_and_path
                models.setdefault(repo, set()).add(path.rstrip("\\,;"))
    # drop empty paths and runtime-templated filenames ("ggml-#{name}.bin",
    # "…{HF_REPO}…") — those are substituted per-request and cannot be
    # mirrored statically.
    for bag in (models, datasets):
        for repo in list(bag):
            bag[repo] = {f for f in bag[repo] if f and "{" not in f and "}" not in f}
            if not bag[repo]:
                del bag[repo]
    return models, datasets


def ms_repo_files(repo_id, repo_type="model"):
    kind = "models" if repo_type == "model" else "datasets"
    url = f"https://modelscope.cn/api/v1/{kind}/{repo_id}/repo/files?Recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
        return {f["Path"] for f in d["Data"]["Files"] if f.get("Type") != "tree"}
    except Exception:
        return None


def ms_dataset_has(repo_id, path):
    """HEAD the resolve URL — the same path the C++ fallback downloads from.
    The dataset listing API (oss/tree) lags or 404s unpredictably, so
    existence is judged by whether the file actually resolves."""
    url = f"https://modelscope.cn/datasets/{repo_id}/resolve/master/{path}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def hf_tree(repo):
    """file sizes from the hf-mirror tree API (empty on failure)"""
    url = f"{HF_MIRROR}/api/models/{repo}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            tree = json.loads(r.read())
        return {e["path"]: e.get("size", 0) for e in tree if e.get("type") == "file"}
    except Exception:
        return {}


def content_length(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def ms_dataset_repo_exists(repo_id):
    url = f"https://modelscope.cn/api/v1/datasets/{repo_id}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read()).get("Code") == 200
    except Exception:
        return False


def diff_against_modelscope(models, ns):
    """-> ({repo: [missing files]}, [repos absent entirely])"""
    partial, absent = {}, []
    repos = sorted(models)
    def probe(repo):
        return repo, ms_repo_files(f"{ns}/{repo.split('/', 1)[1]}")
    with cf.ThreadPoolExecutor(12) as ex:
        for repo, files in ex.map(probe, repos):
            if files is None:
                absent.append(repo)
            else:
                miss = sorted(set(models[repo]) - files)
                if miss:
                    partial[repo] = miss
    return partial, absent


def mirror(args):
    models, datasets = extract_inventory(args.root)
    ns = args.namespace
    state_path = os.path.join(args.output, ".mirror-state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": {}, "failed": {}}
    t0 = time.time()

    # dataset repos (g2p-dicts) mirror to ModelScope datasets
    work = []
    for repo, files in sorted(datasets.items()):
        repo_id = f"{ns}/{repo.split('/', 1)[1]}"
        miss = sorted(f for f in files if not ms_dataset_has(repo_id, f))
        if miss:
            work.append({"repo": repo, "files": miss, "repo_type": "dataset"})
    for repo, files in sorted(models.items()):
        existing = ms_repo_files(f"{ns}/{repo.split('/', 1)[1]}")
        miss = sorted(set(files) - (existing or set()))
        if miss:
            work.append({"repo": repo, "files": miss, "repo_type": "model"})

    if args.check:
        for w in work:
            print(f"missing {len(w['files']):2d} file(s) on {ns}/{w['repo'].split('/', 1)[1]}: {w['files']}")
        print(f"{len(work)} repo(s) need mirroring; nothing uploaded in check mode")
        return 0

    os.makedirs(args.output, exist_ok=True)
    processed = 0
    for w in work:
        if args.budget and time.time() - t0 > args.budget:
            print(f"time budget reached ({processed} repos this run)")
            break
        repo, files = w["repo"], w["files"]
        key = f"{w['repo_type']}:{repo}"
        if key in state["done"]:
            continue  # already mirrored; failed entries ARE retried on re-run
        repo_id = f"{ns}/{repo.split('/', 1)[1]}"
        print(f"== {repo} -> {repo_id} ({len(files)} file(s))", flush=True)

        # create FIRST so a WAF-blocked/absent repo never burns download
        # bandwidth (uploads to existing repos are unaffected by the WAF).
        create = subprocess.run(["ms", "create", repo_id, "--repo-type", w["repo_type"],
                                 "--exist-ok"], capture_output=True, text=True)
        exists = ms_repo_files(repo_id) is not None if w["repo_type"] == "model" \
            else ms_dataset_repo_exists(repo_id)
        if create.returncode != 0 and not exists:
            print(f"   ms create blocked/failed — repo needs creating once manually", flush=True)
            state["failed"][key] = "needs-create"
            json.dump(state, open(state_path, "w"), indent=1)
            continue

        stage = os.path.join(args.output, "stage", repo.split("/", 1)[1])
        os.makedirs(stage, exist_ok=True)
        ok = True
        for f in files:
            dest = os.path.join(stage, f)
            os.makedirs(os.path.dirname(dest) or stage, exist_ok=True)
            src = f"{HF_MIRROR}/{repo}/resolve/main/{f}" if w["repo_type"] == "model" \
                else f"{HF_MIRROR}/datasets/{repo}/resolve/main/{f}"
            want = content_length(src)
            if want > args.max_bytes:
                print(f"   skip {f}: {want/1e6:.0f} MB > --max-bytes", flush=True)
                continue
            rc = subprocess.run(["curl", "-fSL", "--retry", "3", "--connect-timeout", "20",
                                 "-C", "-", "-o", dest, src],
                                capture_output=True, timeout=3600).returncode
            if rc != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
                print(f"   download FAILED {f}", flush=True)
                ok = False
                break
        if not ok:
            state["failed"][key] = "download"
            json.dump(state, open(state_path, "w"), indent=1)
            subprocess.run(["rm", "-rf", stage])
            continue

        # Per-file uploads, NOT folder upload: ms upload's folder flow
        # pre-validates against ModelScope's global content-addressed store —
        # when the blob already exists platform-wide (it does: the original
        # cstr repos are on modelscope.cn), it concludes "All files already
        # committed" and skips the commit, leaving the target repo empty.
        # Single-file mode always issues the commit.
        staged = []
        for root_, _, names_ in os.walk(stage):
            for n in names_:
                local = os.path.join(root_, n)
                rel = os.path.relpath(local, stage).replace(os.sep, "/")
                staged.append((local, rel))
        upload_failed = None
        for local, rel in staged:
            up = subprocess.run(["ms", "upload", repo_id, local, rel, "--repo-type", w["repo_type"]],
                                capture_output=True, text=True, timeout=14400)
            out = up.stdout + up.stderr
            if up.returncode != 0 or "Upload complete" not in out:
                upload_failed = f"{rel}: rc={up.returncode} {out[-160:]}"
                print(f"   upload FAILED {rel}", flush=True)
                break
        if upload_failed:
            state["failed"][key] = "upload"
        else:
            if w["repo_type"] == "model":
                remote = ms_repo_files(repo_id) or set()
                still = [f for f in files if f not in remote]
            else:
                still = [f for f in files if not ms_dataset_has(repo_id, f)]
            if still:
                print(f"   verify FAILED, missing: {still}", flush=True)
                state["failed"][key] = f"verify: {still}"
            else:
                print(f"   OK", flush=True)
                state["done"][key] = len(files)
                processed += 1
        json.dump(state, open(state_path, "w"), indent=1)
        subprocess.run(["rm", "-rf", stage])

    print(f"done={len(state['done'])} failed={len(state['failed'])}")
    if state["failed"]:
        print("failed entries (re-run to retry):")
        for k, v in state["failed"].items():
            print(f"  {k}: {v}")
    return 0


def print_create_commands(args):
    """Emit one `ms create` line per missing repo — run the output on any
    machine/network where ModelScope's WAF allows repo creation, then re-run
    `mirror` here to fill the files."""
    models, datasets = extract_inventory(args.root)
    ns = args.namespace
    lines = []
    for repo in sorted(models):
        repo_id = f"{ns}/{repo.split('/', 1)[1]}"
        if ms_repo_files(repo_id) is None:
            lines.append(f"ms create {repo_id} --repo-type model --exist-ok")
    for repo in sorted(datasets):
        repo_id = f"{ns}/{repo.split('/', 1)[1]}"
        if not any(ms_dataset_has(repo_id, f) for f in list(datasets[repo])[:1]) \
                and not ms_dataset_repo_exists(repo_id):
            lines.append(f"ms create {repo_id} --repo-type dataset --exist-ok")
    print("\n".join(lines))
    print(f"# {len(lines)} repo(s) to create", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "mirror", "commands"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
        s.add_argument("--namespace", default=DEFAULT_NS)
        s.add_argument("--output", default="/tmp/crispasr-msmirror")
        s.add_argument("--max-bytes", type=float, default=2e9,
                       help="skip single files larger than this (default 2 GB)")
        s.add_argument("--budget", type=float, default=0,
                       help="stop after this many seconds (0 = no limit)")
    args = ap.parse_args()
    args.check = args.cmd == "check"
    if args.cmd == "check":
        models, datasets = extract_inventory(args.root)
        partial, absent = diff_against_modelscope(models, args.namespace)
        print(f"inventory: {len(models)} model repos, {len(datasets)} dataset repos")
        print(f"completely missing on ModelScope ({args.namespace}): {len(absent)}")
        for r in absent:
            print(f"  {r}: {sorted(models[r])}")
        print(f"partially missing: {len(partial)}")
        for r, m in partial.items():
            print(f"  {r}: {m}")
        for repo, files in datasets.items():
            repo_id = f"{args.namespace}/{repo.split('/', 1)[1]}"
            miss = sorted(f for f in files if not ms_dataset_has(repo_id, f))
            if miss:
                print(f"dataset {repo}: missing {miss}")
    elif args.cmd == "commands":
        sys.exit(print_create_commands(args))
    else:
        sys.exit(mirror(args))


if __name__ == "__main__":
    main()
