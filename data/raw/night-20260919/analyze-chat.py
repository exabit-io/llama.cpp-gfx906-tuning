#!/usr/bin/env python3
"""Turn chat-client.py's per-request JSONL into the views the requirements ask for.

  per-turn      decode / TTFT by turn index, so a cost that only appears deep in a session
                (context shift, cache eviction) is visible instead of averaged away.
  crossing      for a run with --seed-ctx against a known slot limit: the turns BEFORE the
                session outgrew its slot vs the turns AFTER. That difference is R2.4's
                context-compression cost.
  R3.1          median / p10 / min decode, the gate being p10 (decision N1).

usage: analyze-chat.py FILE.jsonl [--slot-ctx N] [--label L]
"""
import json, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+")
ap.add_argument("--slot-ctx", type=int, default=0)
ap.add_argument("--label", default="")
a = ap.parse_args()

def q(v, p):
    v = sorted(v)
    return v[min(int(len(v)*p), len(v)-1)] if v else 0.0

for fn in a.files:
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    ok = [r for r in rows if not r.get("failed") and not r.get("seeded")]
    fails = [r for r in rows if r.get("failed")]
    seeds = [r for r in rows if r.get("seeded")]
    label = a.label or fn.split("/")[-1].replace(".jsonl", "")
    print(f"\n### {label}   ({len(rows)} records: {len(ok)} measured, {len(seeds)} seed, {len(fails)} refused)")
    if seeds:
        print(f"seed prefill: {q([r['prefill_tok'] for r in seeds],0.5):.0f} tok median, "
              f"{q([r['ttft_s'] for r in seeds],0.5):.1f} s median  (SEEDED — not grown)")
    if ok:
        d = [r["decode_tps"] for r in ok]
        print(f"decode tok/s per request: median {q(d,.5):.2f}  p10 {q(d,.1):.2f}  min {min(d):.2f}"
              f"   -> R3.1 (p10>=12): {'PASS' if q(d,.1)>=12 else 'FAIL'}"
              f"   [median>=12: {'pass' if q(d,.5)>=12 else 'fail'}]")
        t = [r["ttft_s"] for r in ok]
        print(f"TTFT s: median {q(t,.5):.2f}  p90 {q(t,.9):.2f}  max {max(t):.2f}")
        print(f"cache misses (whole conversation re-prefilled): {sum(1 for r in ok if r.get('cache_miss'))}/{len(ok)}")
        print("\n| turn | n | ctx_est med | prefill med | decode med | TTFT med | cache miss |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for tn in sorted({r["turn"] for r in ok}):
            g = [r for r in ok if r["turn"] == tn]
            print(f"| {tn} | {len(g)} | {q([r['ctx_est'] for r in g],.5):.0f} | "
                  f"{q([r['prefill_tok'] for r in g],.5):.0f} | {q([r['decode_tps'] for r in g],.5):.2f} | "
                  f"{q([r['ttft_s'] for r in g],.5):.2f} | {sum(1 for r in g if r.get('cache_miss'))} |")
    if a.slot_ctx and ok:
        # ctx_est is the context BEFORE this turn's generation was added, near enough for the split
        before = [r for r in ok if r["ctx_est"] - r["gen_tok"] < a.slot_ctx]
        after  = [r for r in ok if r["ctx_est"] - r["gen_tok"] >= a.slot_ctx]
        print(f"\ncrossing the {a.slot_ctx}-token slot limit:")
        for nm, g in (("before", before), ("after", after)):
            if g:
                print(f"  {nm:6s} n={len(g):3d}  decode med {q([r['decode_tps'] for r in g],.5):6.2f}  "
                      f"TTFT med {q([r['ttft_s'] for r in g],.5):6.2f}  "
                      f"prefill med {q([r['prefill_tok'] for r in g],.5):8.0f}")
            else:
                print(f"  {nm:6s} n=0 — no turns on this side of the limit")
        if before and after:
            db = q([r["decode_tps"] for r in before], .5); da = q([r["decode_tps"] for r in after], .5)
            print(f"  => context-compression cost: decode {db:.2f} -> {da:.2f} tok/s "
                  f"({(da/db-1)*100:+.1f}%)")
    if fails:
        print(f"\nrefusals: {len(fails)}; first at ctx ~{min(r['ctx_est'] for r in fails)}")
        print(f"  {fails[0]['failed'][:220]}")
