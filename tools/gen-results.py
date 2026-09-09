#!/usr/bin/env python3
"""Regenerates optimize/results.md from the six optimize.py commands its header lists. Run from the guide folder after any
change to optimize.py or data/benchmarks.json:  python3 tools/gen-results.py"""
import subprocess, datetime, os, sys
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = [
    ("Production build (ML-gfx906 fork tile table + fast-path MMVQ kernel), default", []),
    ("Stock b10288 (the reference every report number is measured on)", ["--build", "stock"]),
    ("Model file as a decision variable, quality-bounded (--allow-quant --max-kl 0.04: Q8_0, Q6_K, Q4_K_M)", ["--allow-quant", "--max-kl", "0.04"]),
    ("Production cap for the hyperconverged fleet (--cap 125)", ["--cap", "125"]),
    ("With the q8_0-keys / q4_0-values cache allowed (--kv-q4v; needs a GGML_CUDA_FA_ALL_QUANTS build)", ["--kv-q4v"]),
    ("Measured cells only (--measured-only)", ["--measured-only"]),
]
old = open(f"{G}/optimize/results.md").read().split("\n## ", 1)[0]          # keep the header block as written
out = [old.rstrip("\n"), ""]
for title, args in SECTIONS:
    r = subprocess.run([sys.executable, "optimize.py", *args], cwd=f"{G}/optimize", capture_output=True, text=True)
    body = (r.stdout + (("\n[stderr]\n" + r.stderr) if r.returncode else "")).rstrip("\n")
    out += [f"## {title}", "", "```", body, "```", ""]
open(f"{G}/optimize/results.md", "w").write("\n".join(out))
print("results.md regenerated", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
