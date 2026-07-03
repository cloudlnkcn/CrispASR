#!/usr/bin/env python3
"""imatrix_ab.py — A/B harness for importance-matrix (imatrix) quantization.

Measures whether feeding an importance matrix to the quantizer improves the
quantized ASR model's fidelity to the f16 gold, using the model's **prefill
logits** (the first-token distribution over the audio) as a fixed-length,
generation-independent parity signal — the ASR analog of CrispEmbed's
embedding-cosine A/B (tools/imatrix_ab.py there).

Everything runs serially (16 GB Mac constraint: never load two heavy models at
once). Pipeline:
  1. calibration : run crispasr with CRISPASR_IMATRIX_OUT over the calib clips
                   (merges across clips) -> <model>.imatrix.gguf
  2. quant A     : crispasr-quantize <src> <out_a> <qtype>              (baseline)
  3. quant B     : crispasr-quantize <src> <out_b> <qtype> --imatrix    (candidate)
  4. eval        : for each held-out clip, dump prefill logits (via
                   CRISPASR_ACTDUMP_OUT) from src / A / B and report
                   mean cosine(A, src) vs mean cosine(B, src).

Accept criterion: mean cos(B,src) >= mean cos(A,src) (imatrix must not regress;
target: close part of the gap to the f16 gold).

Usage:
  python tools/imatrix_ab.py --cli build/bin/crispasr \\
      --quant build/bin/crispasr-quantize --src model-f16.gguf --qtype q4_k \\
      --calib a.wav b.wav --eval c.wav d.wav [--workdir DIR] [--keep]

Note: the src model's backend must have the imatrix/actdump collector installed
on its decode scheduler (crispasr_imatrix_install) — see docs/quantize.md.
"""
import argparse, math, os, struct, subprocess, sys, time


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def read_dump(path):
    """Read an actdump file: int64 n (LE) then n float32. Returns list[float]."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<q", f.read(8))
        return list(struct.unpack(f"<{n}f", f.read(n * 4)))


def logits(cli, model, wav, out):
    """Run the model on one clip, dumping prefill logits to `out`; return vec."""
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, CRISPASR_ACTDUMP_OUT=out)
    r = run([cli, "-m", model, "-f", wav], env=env)
    if not os.path.exists(out):
        sys.exit(f"no logits dumped for {os.path.basename(model)} on {os.path.basename(wav)}:\n{r.stderr[-1500:]}")
    return read_dump(out)


def cosine(a, b):
    n = min(len(a), len(b))
    if n == 0 or len(a) != len(b):
        return float("nan")
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(y * y for y in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--src", required=True, help="f16/f32 (or q8_0) source GGUF = gold")
    ap.add_argument("--qtype", default="q4_k")
    ap.add_argument("--calib", nargs="+", required=True, help="calibration wav clips")
    ap.add_argument("--eval", nargs="+", required=True, help="held-out eval wav clips")
    ap.add_argument("--workdir", default="/tmp/imatrix_ab")
    ap.add_argument("--keep", action="store_true", help="keep intermediate GGUFs")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.src))[0]
    imat = os.path.join(args.workdir, base + ".imatrix.gguf")
    out_a = os.path.join(args.workdir, f"{base}-{args.qtype}.gguf")
    out_b = os.path.join(args.workdir, f"{base}-{args.qtype}-imatrix.gguf")
    dump = os.path.join(args.workdir, "acts.bin")

    # 1. calibration (one process per clip; the collector merges into `imat`)
    if os.path.exists(imat):
        os.remove(imat)
    print(f"[1/4] calibration over {len(args.calib)} clip(s) -> {imat}")
    t0 = time.time()
    for w in args.calib:
        env = dict(os.environ, CRISPASR_IMATRIX_OUT=imat)
        r = run([args.cli, "-m", args.src, "-f", w], env=env)
        if not os.path.exists(imat):
            sys.exit(f"calibration failed on {w}:\n{r.stderr[-1500:]}")
    print(f"      done in {time.time()-t0:.1f}s  ({os.path.getsize(imat)//1024} KB)")

    # 2 + 3. quantize baseline and imatrix
    print(f"[2/4] quantize baseline  -> {out_a}")
    r = run([args.quant, args.src, out_a, args.qtype])
    if r.returncode != 0:
        sys.exit(f"baseline quant failed:\n{r.stderr[-1500:]}\n{r.stdout[-1500:]}")
    print(f"[3/4] quantize +imatrix  -> {out_b}")
    r = run([args.quant, args.src, out_b, args.qtype, "--imatrix", imat])
    if r.returncode != 0:
        sys.exit(f"imatrix quant failed:\n{r.stderr[-1500:]}\n{r.stdout[-1500:]}")
    n_im = sum(1 for l in r.stdout.splitlines() if "(imatrix)" in l)
    print(f"      {n_im} tensor(s) quantized with imatrix weighting")

    # 4. eval (serial: one model in memory at a time)
    print(f"[4/4] eval prefill-logit cosine over {len(args.eval)} held-out clip(s)")
    rows, cos_a_all, cos_b_all = [], [], []
    for w in args.eval:
        name = os.path.basename(w)
        g = logits(args.cli, args.src, w, dump)
        a = logits(args.cli, out_a, w, dump)
        b = logits(args.cli, out_b, w, dump)
        ca, cb = cosine(a, g), cosine(b, g)
        cos_a_all.append(ca)
        cos_b_all.append(cb)
        rows.append((name, ca, cb))

    sz = lambda p: os.path.getsize(p) / 1e6
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("\n===== A/B RESULT (" + args.qtype + ", vs f16 gold) =====")
    print(f"  source     {os.path.basename(args.src):40s} {sz(args.src):7.1f} MB")
    print(f"  A baseline {os.path.basename(out_a):40s} {sz(out_a):7.1f} MB")
    print(f"  B +imatrix {os.path.basename(out_b):40s} {sz(out_b):7.1f} MB")
    print(f"  {'clip':28s}  cos(A,gold)   cos(B,gold)   delta(B-A)")
    for name, ca, cb in rows:
        print(f"  {name:28s}  {ca:.6f}     {cb:.6f}    {cb-ca:+.6f}")
    ma, mb = mean(cos_a_all), mean(cos_b_all)
    print(f"  {'MEAN':28s}  {ma:.6f}     {mb:.6f}    {mb-ma:+.6f}")
    verdict = "PASS (imatrix >= baseline)" if mb >= ma - 1e-6 else "REGRESSION"
    print(f"  VERDICT: {verdict}")

    if not args.keep:
        for p in (out_a, out_b, dump):
            if os.path.exists(p):
                os.remove(p)
        print("  (removed intermediate GGUFs; --keep to retain)")


if __name__ == "__main__":
    main()
