#!/usr/bin/env python3
"""m1-analyze.py DIR PREFIX [--ref-ms MS] [--gib-per-die G] [--api]
Splits a rocprofv3 kernel trace of llama-bench tg into decode tokens per die and reports, per token:
span, busy (union of kernel intervals), kernel time by category, counts, gaps; optionally HIP/RCCL API calls per token."""
import csv, sys, collections, statistics, argparse
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('prefix')
ap.add_argument('--ref-ms', type=float, default=None, help='unprofiled ms per token (for the idle-by-subtraction line)')
ap.add_argument('--gib-per-die', type=float, default=None, help='weights streamed per die per token')
ap.add_argument('--api', action='store_true'); ap.add_argument('--first', type=int, default=8, help='steady-state window start token')
ap.add_argument('--last-skip', type=int, default=2); ap.add_argument('--show-token', type=int, default=None)
a = ap.parse_args()
rows = list(csv.DictReader(open(f'{a.dir}/{a.prefix}_kernel_trace.csv')))
by_agent = collections.defaultdict(list)
for r in rows:
    by_agent[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'], int(r['Grid_Size_X']), r['Queue_Id']))
def cat(name):
    n = name.lower()
    if 'nccl' in n or 'rccl' in n or 'allreduce' in n or 'ggml_cuda_tp' in n or 'cross_device' in n: return 'rccl'
    if 'mul_mat_vec' in n: return 'mmvq'
    if 'mul_mat_q' in n or 'mmq' in n: return 'mmq'
    if 'flash_attn' in n: return 'fa'
    if 'norm' in n: return 'norm'
    if 'cpy' in n or 'copy' in n or 'fillbuffer' in n: return 'cpy'
    if 'rocclr' in n: return 'rt'
    return 'other'
def union(iv):
    iv = sorted(iv); tot = 0; cs, ce = iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    return tot + ce - cs
summary = {}
delims_by_agent = {}
for ag, ks in sorted(by_agent.items()):
    ks.sort()
    # delimiter: the largest-grid mul_mat_vec kernel (the output projection), once per token
    mm = [k for k in ks if 'mul_mat_vec' in k[2]]
    if not mm: print(ag, 'no mmvq kernels'); continue
    key = max(((k[2], k[3]) for k in mm), key=lambda t: t[1])
    delim_idx = [i for i, k in enumerate(ks) if (k[2], k[3]) == key]
    delims_by_agent[ag] = [ks[i][1] for i in delim_idx]
    toks = []; prev = 0
    for i in delim_idx:
        toks.append(ks[prev:i+1]); prev = i+1
    n = len(toks); sel = toks[a.first:n-a.last_skip]
    per = []
    for t in sel:
        span = t[-1][1] - t[0][0]
        busy = union([(s, e) for s, e, *_ in t])
        c = collections.Counter(); d = collections.Counter()
        for s, e, name, g, q in t: c[cat(name)] += 1; d[cat(name)] += e - s
        gaps = []; ce = t[0][1]
        for s, e, *_ in sorted(t):
            if s > ce: gaps.append(s - ce)
            ce = max(ce, e)
        per.append(dict(span=span, busy=busy, n=len(t), c=c, d=d, gaps=gaps))
    med = lambda f: statistics.median(f(p) for p in per)
    print(f'\n== {ag}: {n} tokens (delimiter {key[0][:60]}... grid {key[1]}), steady window {len(sel)} tokens, median per token:')
    periods = [b-a for a,b in zip(delims_by_agent[ag][a.first-1:], delims_by_agent[ag][a.first:n-a.last_skip])]
    print(f'  token period (delimiter to delimiter) {statistics.median(periods)/1e6:.2f} ms; inter-token gap {statistics.median(periods)/1e6 - med(lambda p: p["span"])/1e6:.2f} ms')
    print(f'  kernels {med(lambda p: p["n"]):.0f}   span {med(lambda p: p["span"])/1e6:.2f} ms   busy(union) {med(lambda p: p["busy"])/1e6:.2f} ms   idle-in-span {med(lambda p: p["span"]-p["busy"])/1e6:.2f} ms')
    cats = ['mmvq', 'mmq', 'fa', 'rccl', 'norm', 'cpy', 'other', 'rt']
    print('  ' + '  '.join(f'{k}: {med(lambda p, k=k: p["d"][k])/1e6:.2f} ms/{med(lambda p, k=k: p["c"][k]):.0f}' for k in cats))
    g = [x for p in per for x in p['gaps']]; ng = len(per)
    bins = [(0, 10e3), (10e3, 30e3), (30e3, 100e3), (100e3, 300e3), (300e3, 1e12)]
    print('  gaps per token: ' + ', '.join(f'{lo/1e3:.0f}-{hi/1e3:.0f}us: {sum(1 for x in g if lo <= x < hi)/ng:.0f} ({sum(x for x in g if lo <= x < hi)/ng/1e6:.2f} ms)' for lo, hi in bins))
    if a.gib_per_die:
        mm_ms = med(lambda p: p['d']['mmvq'])/1e6
        print(f'  weight streaming: {a.gib_per_die} GiB in {mm_ms:.2f} ms of mmvq = {a.gib_per_die*1.0737/mm_ms*1e3:.0f} GB/s achieved')
    if a.ref_ms:
        print(f'  unprofiled token {a.ref_ms:.2f} ms - busy {med(lambda p: p["busy"])/1e6:.2f} ms = {a.ref_ms - med(lambda p: p["busy"])/1e6:.2f} ms true idle per token ({(a.ref_ms - med(lambda p: p["busy"])/1e6)/a.ref_ms*100:.0f}%)')
    summary[ag] = per
    if a.show_token is not None and a.show_token < len(sel):
        t = sel[a.show_token]; c = collections.Counter(); d = collections.Counter()
        for s, e, name, g_, q in t: c[name[:70]] += 1; d[name[:70]] += e - s
        print('  kernel inventory of one token (count, total us, name):')
        for name, cnt in sorted(c.items(), key=lambda kv: -d[kv[0]]): print(f'    {cnt:4d} {d[name]/1e3:9.1f} {name}')
if a.api:
    import os
    for kind in ('hip_api_trace', 'rccl_api_trace'):
        p = f'{a.dir}/{a.prefix}_{kind}.csv'
        if not os.path.exists(p): print('no', p); continue
        api = list(csv.DictReader(open(p)))
        ag0 = sorted(delims_by_agent)[0]; dl = delims_by_agent[ag0]
        n = len(dl); lo, hi = dl[a.first-1], dl[n-a.last_skip-1]; ntok = n - a.last_skip - a.first
        c = collections.Counter(); d = collections.Counter()
        for r in api:
            s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            if lo <= s < hi: c[r['Function']] += 1; d[r['Function']] += e - s
        print(f'\n== {kind}: calls per token (host time per token) over {ntok} tokens:')
        for f, cnt in sorted(c.items(), key=lambda kv: -d[kv[0]])[:14]: print(f'    {cnt/ntok:8.1f}  {d[f]/ntok/1e6:7.2f} ms  {f}')
