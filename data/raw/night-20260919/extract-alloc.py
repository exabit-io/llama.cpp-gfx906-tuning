#!/usr/bin/env python3
"""Extract §5.2's per-die memory itemisation from probe-alloc logs.

probe-alloc.sh grepped for `ROCm0 ... buffer size`, but under `-sm tensor` llama.cpp reports
buffers against a composite device named `Meta(ROCm0,ROCm1,ROCm2,ROCm3)`, and those figures are
PER DIE. Hence the empty columns in probe-alloc.md; this recovers them from the saved -v logs.
"""
import re, glob, os
rows = []
for f in sorted(glob.glob('/root/night-20260919/probe-alloc-logs/*.log')):
    slots, depth = map(int, os.path.basename(f)[:-4].split('-'))
    t = open(f, errors='replace').read()
    g = lambda p: (float(m.group(1)) if (m := re.search(p, t)) else None)
    mdl_rep = g(r'Meta\([^)]*Repacked[^)]*\) model buffer size = *([0-9.]+) MiB') or 0
    mdl_met = g(r'load_tensors: Meta\(ROCm0,ROCm1,ROCm2,ROCm3\) model buffer size = *([0-9.]+) MiB') or 0
    kv      = g(r'llama_kv_cache: Meta\([^)]*\) KV buffer size = *([0-9.]+) MiB')
    comp    = g(r'sched_reserve: Meta\([^)]*\) compute buffer size = *([0-9.]+) MiB')
    host_c  = g(r'sched_reserve: +ROCm_Host compute buffer size = *([0-9.]+) MiB')
    cpu_m   = g(r'CPU_Mapped model buffer size = *([0-9.]+) MiB')
    off     = (m.group(0) if (m := re.search(r'offloaded \d+/\d+ layers to GPU', t)) else '?')
    tot = re.search(r'llama_kv_cache: size = *([0-9.]+) MiB \( *(\d+) cells, *(\d+) layers', t)
    rows.append(dict(slots=slots, depth=depth, model=mdl_rep+mdl_met, kv=kv, comp=comp,
                     host_c=host_c, cpu_m=cpu_m, off=off,
                     kvtot=float(tot.group(1)) if tot else None,
                     cells=int(tot.group(2)) if tot else None,
                     kvlayers=int(tot.group(3)) if tot else None))
rows.sort(key=lambda r: r['slots']*r['depth'])
print("| cell | total KV tok | offload | model MiB/die | KV MiB/die | compute MiB/die | **sum GiB/die** | headroom to 31 |")
print("|---|---:|---|---:|---:|---:|---:|---:|")
for r in rows:
    if r['kv'] is None: print(f"| {r['slots']}x{r['depth']//1024}K | — | {r['off']} | did not allocate |"); continue
    s = (r['model']+r['kv']+r['comp'])/1024
    print(f"| {r['slots']} x {r['depth']//1024}K | {r['slots']*r['depth']/1e6:.3f}M | {r['off'].replace(' layers to GPU','')} "
          f"| {r['model']:.0f} | {r['kv']:.1f} | {r['comp']:.1f} | **{s:.2f}** | {31-s:.2f} |")
print("\nPer-token costs (per die):")
for r in rows:
    if r['kv'] is None: continue
    n = r['slots']*r['depth']
    print(f"  {r['slots']}x{r['depth']//1024}K: KV {r['kv']*1024/n:.2f} KiB/tok/die, compute {r['comp']*1024/n:.2f} KiB/tok/die")
r0 = rows[0]
print(f"\nKV cache structure: {r0['kvlayers']} KV-bearing layers of 65 blocks; host-side compute "
      f"buffer {r0['host_c']:.0f} MiB, CPU-mapped model {r0['cpu_m']:.0f} MiB.")
