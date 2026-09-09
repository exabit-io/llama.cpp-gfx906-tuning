#!/usr/bin/env python3
"""BENCHMARKS-TODO item 5: N concurrent long-context requests (distinct wikitext slices + a summary instruction), greedy,
reporting the server's timings and the draft-acceptance lines it logs.
usage: mtp-depth-client.py URL TAG SERVER_LOG --context-tokens 32768 --conc 4 [--gen 300]"""
import argparse, json, re, statistics as st, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("tag"); ap.add_argument("logf")
ap.add_argument("--context-tokens", type=int, required=True); ap.add_argument("--conc", type=int, default=4); ap.add_argument("--gen", type=int, default=300)
ap.add_argument("--warmup", type=int, default=0, help="generate this many tokens on a short prompt before building the prompts (the branch builds run their first 100-300 tokens after load slow)")
ap.add_argument("--waves", type=int, default=1, help="2 = send the same prompts twice with cache_prompt on; the second wave is decode-only (all slots decoding together)")
a = ap.parse_args()
def post(path, body):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=4 * 3600) as r: return json.loads(r.read())
txt = open('/root/models/wikitext-2-raw/wiki.train.raw').read()
N = a.context_tokens; span = N * 6
def prompt(i):
    toks = post("/tokenize", {"content": txt[i * span: (i + 1) * span]})["tokens"][:N]
    assert len(toks) == N, f"slice {i}: {len(toks)} tokens"
    return post("/detokenize", {"tokens": toks})["content"] + "\n\nWrite a detailed summary of the text above, section by section, in plain prose.\n\n"
prompts = [prompt(i) for i in range(a.conc)]
post("/completion", {"prompt": "Say OK.", "n_predict": 4, "temperature": 0.0})
if a.warmup: post("/completion", {"prompt": "Write a long essay about the history of computing.", "n_predict": a.warmup, "temperature": 0.0, "ignore_eos": True})
cache = a.waves > 1
def one(i):
    t0 = time.perf_counter()
    r = post("/completion", {"prompt": prompts[i], "n_predict": a.gen, "temperature": 0.0, "cache_prompt": cache, "ignore_eos": True})
    t = r["timings"]; return dict(wall=time.perf_counter() - t0, pn=t["prompt_n"], pms=t["prompt_ms"], gn=t["predicted_n"], gms=t["predicted_ms"])
out = []
for wave in range(1, a.waves + 1):
    n0 = sum(1 for _ in open(a.logf, errors='replace'))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.conc) as ex: res = list(ex.map(one, range(a.conc)))
    wall = time.perf_counter() - t0; time.sleep(1)
    lines = open(a.logf, errors='replace').read().splitlines()[n0:]
    acc = [m for m in (re.search(r'acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+)', l) for l in lines) if m]
    accs = f"{sum(int(m[2]) for m in acc)} / {sum(int(m[3]) for m in acc)} = {sum(int(m[2]) for m in acc) / max(1, sum(int(m[3]) for m in acc)):.2f}" if acc else "-"
    gn = sum(r["gn"] for r in res)
    label = a.tag if a.waves == 1 else f"{a.tag} wave {wave}" + (" (decode only)" if wave > 1 else " (prefill + decode)")
    print(f"| {label} | {N} | {a.conc} | {wall:.0f} | {st.mean(r['pn'] for r in res):.0f} | {st.mean(r['pms'] / 1000 for r in res):.0f} / {max(r['pms'] / 1000 for r in res):.0f} | {st.mean(r['gn'] / (r['gms'] / 1000) for r in res):.1f} | {sum(r['gn'] / (r['gms'] / 1000) for r in res):.1f} | {gn / wall:.1f} | {accs} |", flush=True)
    out.append(dict(tag=a.tag, wave=wave, ctx=N, conc=a.conc, wall=wall, res=res, acc=accs))
json.dump(out, open(f"/root/rocm-tests/bench/{a.tag}.json", "w"), indent=1)
