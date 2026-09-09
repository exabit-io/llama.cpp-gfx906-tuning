#!/usr/bin/env python3
"""Concurrency sweep against llama-server /completion using its exact per-request timings.
usage: server-bench.py URL TAG [--conc 1,2,4,8,16,32] [--prompt-tokens 1300] [--gen 256]"""
import argparse, json, random, statistics as st, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("tag")
ap.add_argument("--conc", default="1,2,4,8,16,32"); ap.add_argument("--prompt-tokens", type=int, default=1300)
ap.add_argument("--gen", type=int, default=256); ap.add_argument("--min-req", type=int, default=8)
a = ap.parse_args()

def post(path, body, timeout=1800):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())

words = "the quick brown fox jumps over lazy dog alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
def make_prompt(seed):
    rnd = random.Random(seed); txt = " ".join(rnd.choice(words) for _ in range(a.prompt_tokens * 2))
    toks = post("/tokenize", {"content": txt})["tokens"][: a.prompt_tokens]
    return post("/detokenize", {"tokens": toks})["content"]

def one(i):
    t0 = time.perf_counter()
    r = post("/completion", {"prompt": prompts[i % len(prompts)], "n_predict": a.gen, "ignore_eos": True,
                             "cache_prompt": False, "temperature": 0.0, "seed": i})
    t = r["timings"]; return dict(wall=time.perf_counter() - t0, pn=t["prompt_n"], pms=t["prompt_ms"],
                                  gn=t["predicted_n"], gms=t["predicted_ms"])

prompts = [make_prompt(s) for s in range(8)]
one(0)  # warm-up
print(f"# {a.tag}  prompt≈{a.prompt_tokens} tok  gen={a.gen} tok (ignore_eos)  {time.strftime('%FT%T')}")
print("| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
out = []
for c in [int(x) for x in a.conc.split(",")]:
    n = max(a.min_req, 2 * c)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex: res = list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    gn = sum(r["gn"] for r in res); pn = sum(r["pn"] for r in res)
    agg_pp = (pn + gn) / wall  # total tokens (prompt+gen) per second of wall time
    row = dict(conc=c, reqs=n, wall=wall, agg_gen=gn / wall, agg_pp=agg_pp, rpm=n / wall * 60,
               req_gen=st.mean(r["gn"] / (r["gms"] / 1000) for r in res), ttft=st.mean(r["pms"] / 1000 for r in res),
               req_wall=st.mean(r["wall"] for r in res))
    out.append(row)
    print(f"| {c} | {n} | {wall:.1f} | {row['agg_gen']:.1f} | {row['agg_pp']:.0f} | {row['rpm']:.1f} | {row['req_gen']:.1f} | {row['ttft']:.2f} | {row['req_wall']:.1f} |", flush=True)
json.dump(out, open(f"/root/rocm-tests/bench/{a.tag}.json", "w"), indent=1)
