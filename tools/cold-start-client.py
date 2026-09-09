#!/usr/bin/env python3
"""cold-start-client.py URL TAG READY_TS: against a FRESH llama-server, sequential 2K-prompt requests (first = prefill+64 tokens,
then five decode-only 64-token continuations with cache_prompt) and two 8-way batches of distinct 1300-token prompts (first use of the
batch-8 shape, then its second use). Prints one row per step with the server's own timings and the seconds since the server was ready."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, tag, ready = sys.argv[1], sys.argv[2], float(sys.argv[3])
def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r: return json.loads(r.read())
txt = open('/root/models/wikitext-2-raw/wiki.train.raw').read()
def prompt(i, n):
    toks = post("/tokenize", {"content": txt[i * n * 6:(i + 1) * n * 6]})["tokens"][:n]
    return post("/detokenize", {"tokens": toks})["content"]
def one(p, gen, cache):
    t0 = time.perf_counter(); r = post("/completion", {"prompt": p, "n_predict": gen, "temperature": 0.0, "cache_prompt": cache, "ignore_eos": True})
    t = r["timings"]; return dict(ts=time.time() - ready, wall=time.perf_counter() - t0, pn=t["prompt_n"], pms=t["prompt_ms"], gn=t["predicted_n"], gms=t["predicted_ms"])
def row(step, r): print(f"| {tag} | {step} | {r['ts']:.1f} | {r['pn']} | {r['pms']:.0f} | {r['gn']} | {r['gn'] / (r['gms'] / 1000):.1f} |", flush=True)
p2k = prompt(0, 2048)
cum = 0
for k in range(6):
    r = one(p2k, 64, k > 0); cum += r["gn"]; row(f"seq{k + 1} (cum {cum} tok)", r)
p8 = [prompt(10 + i, 1300) for i in range(8)]
for k in range(2):
    with ThreadPoolExecutor(max_workers=8) as ex: rs = list(ex.map(lambda p: one(p, 64, False), p8))
    agg = sum(x["gn"] / (x["gms"] / 1000) for x in rs); mean = agg / len(rs)
    print(f"| {tag} | batch8 #{k + 1} | {max(x['ts'] for x in rs):.1f} | {rs[0]['pn']}x8 | {max(x['pms'] for x in rs):.0f} | {sum(x['gn'] for x in rs)} | {mean:.1f} per req, {agg:.1f} agg |", flush=True)
