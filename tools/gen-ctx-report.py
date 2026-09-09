#!/usr/bin/env python3
"""Builds /root/qwen38-27b-ctx-gfx906.html from the 2026-09-06 context / optimisation / extras results (ctx_data.load)."""
import json, math, os, sys
sys.path.insert(0, '/root/rocm-tests/bench')
from ctx_data import load, buffers
D = load()
import ctx_predict as CP
OUT = '/root/qwen38-27b-ctx-gfx906.html'
PROBE = '/tmp/claude-0/-root/9db04509-6845-4917-9582-a8859b8799cb/scratchpad/probe'

# ---------------- derived numbers ----------------
def fit(rows, key):
    xs = [r['pp'] / 1024 for r in rows]; ys = [key(r) for r in rows]; n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); return k, my - k * mx
L = D['ladder']; MODEL = {}
for name, rows in L.items():
    k2, c0 = fit(rows, lambda r: 1000 / r['s_pp']); ks, base = fit(rows, lambda r: 1000 * r['b'] / r['s_tg'])
    MODEL[name] = dict(b=rows[0]['b'], c0=c0, k=2 * k2, base=base, ks=ks, kseq=ks / rows[0]['b'])
PP = MODEL['f16-b8']; pp_rate = lambda d: 1000 / (PP['c0'] + PP['k'] * d / 1024)           # instantaneous prefill t/s at depth d tokens
def tg_rate(name, d): m = MODEL[name]; return 1000 * m['b'] / (m['base'] + m['ks'] * d / 1024)
ATT_TFLOPS = 16 * 2 * 24 * 256 * 2 * 1024 / (PP['k'] / 1000) / 1e12
KV_GBS = 64 * 1024 * 1024 / (MODEL['f16-b8']['kseq'] / 1000) / 1e9
Q8_RATIO = MODEL['q8-b8']['kseq'] / MODEL['f16-b8']['kseq']
def row(name, pp): return next(r for r in L[name] if r['pp'] == pp)

# memory probes (per die, MiB), verbose runs of 2026-09-06 10:4x
def probe(name):
    b = buffers(f'{PROBE}/{name}.log'); return b
MEM = [  # label, log name, npl, ctx per seq, kv type, fits
    ('8 × 64K',  'f16-8x64K',      8,  65536, 'f16', True),
    ('8 × 128K', 'f16-8x128K-ref', 8, 131072, 'f16', True),
    ('8 × 160K', 'f16-8x160K-ub2048', 8, 163840, 'f16', True),
    ('4 × 256K', 'f16-4x256K',     4, 262144, 'f16', True),
    ('8 × 192K', 'q8q8-8x192K-ub2048', 8, 196608, 'q8_0', True),
    ('8 × 192K', 'f16-8x192K-ub512', 8, 196608, 'f16 (ub 512)', True),
    ('8 × 256K', 'q8-2M-v',        8, 262144, 'q8_0', False),
    ('8 × 256K', 'f16-2M',         8, 262144, 'f16', False)]
MEMROWS = []
for label, log, npl, ctx, kv, fits in MEM:
    b = probe(log); model = 6495.03
    kv_mib = b.get('kv') or (npl * ctx * 16 / 1024 * (1.0625 if kv.startswith('q8') else 1))
    rs = b.get('rs') or npl * 299.25 / 8
    comp = b.get('compute')
    if comp is None:  # allocation failed at the compute buffer: use the size the failing allocation asked for
        comp = {'q8-2M-v': 9536.34, 'f16-2M': 2468.0}.get(log, 0)
    MEMROWS.append(dict(label=label, kv=kv, npl=npl, ctx=ctx, model=model, kvm=kv_mib, rs=rs, comp=comp, fits=fits, total=model + kv_mib + rs + comp))
DIE_MIB = 32768

def f0(v): return f'{v:,.0f}'
def f1(v): return f'{v:.1f}'
def f2(v): return f'{v:.2f}'
def ctxk(t): return f'{t // 1024}K' if t >= 1024 else str(t)

# ---------------- svg helpers ----------------
def grid_axis(L_, R, T, Bt, ymax, ticks, fmt=lambda v: f'{v:g}', ymin=0):
    g = ['<g class="grid">']
    for tv in ticks:
        y = Bt - (tv - ymin) / (ymax - ymin) * (Bt - T); g.append(f'<line x1="{L_}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}"/>')
    g.append('</g>'); g.append(f'<line class="axis" x1="{L_}" y1="{Bt}" x2="{R}" y2="{Bt}"/>')
    for tv in ticks:
        y = Bt - (tv - ymin) / (ymax - ymin) * (Bt - T); g.append(f'<text class="tick" x="{L_ - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(tv)}</text>')
    return '\n'.join(g)

DEPTHS = [2048, 8192, 32768, 65536, 131072, 262016]
def xlog(d, L_, R, lo=2048, hi=262144): return L_ + (math.log2(d) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * (R - L_)

def depth_lines(pid, title, series, ymax, ticks, ylabel, aria, model_line=None, points=None, W=560, H=300, extra_x=None):
    """series: list of dict(name, var, values{depth: y}); model_line: (label, fn depth->y); points: list of dict(name, var, values{depth:y}, note)"""
    L_, R, T, Bt = 58, W - 16, 30, H - 40
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" data-kind="lines" data-panel="{pid}" tabindex="0">',
         f'<text class="val" x="{L_}" y="16">{title}</text>', grid_axis(L_, R, T, Bt, ymax, ticks)]
    xt = [2048, 8192, 32768, 131072, 262144]
    for d in xt: s.append(f'<text class="tick" x="{xlog(d, L_, R):.1f}" y="{Bt + 18}" text-anchor="middle">{ctxk(d)}</text>')
    s.append(f'<text class="tick" x="{R}" y="{Bt + 34}" text-anchor="end">context per sequence, tokens (log scale)</text>')
    s.append(f'<text class="tick" x="{L_ - 8}" y="{T - 10}" text-anchor="end">{ylabel}</text>')
    yv = lambda v: Bt - v / ymax * (Bt - T)
    if model_line:
        lab, fn = model_line; pts = []
        for i in range(0, 101):
            d = 2048 * 2 ** (i / 100 * 7); pts.append(f'{xlog(d, L_, R):.1f},{yv(fn(d)):.1f}')
        s.append(f'<polyline class="model" points="{" ".join(pts)}"/>')
    xs_all = sorted({d for se in series for d in se['values']})
    data = dict(x=[round(xlog(d, L_, R), 1) for d in xs_all], labels=[ctxk(d) + ' context' for d in xs_all], series=[])
    for se in series:
        pts = [(xlog(d, L_, R), yv(v)) for d, v in sorted(se['values'].items())]
        if se.get('draw', True):
            s.append(f'<polyline class="line" stroke="var(--{se["var"]})" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
            for x, y in pts: s.append(f'<circle class="pt" fill="var(--{se["var"]})" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        if se.get('end'):
            lx, ly = pts[-1]; s.append(f'<text class="val" x="{lx + 8:.1f}" y="{ly + 4:.1f}">{se.get("end", "")}</text>')
        data['series'].append(dict(name=se['name'], var=se['var'], values=[se['values'].get(d) for d in xs_all], unit=se.get('unit', '')))
    if points:
        for p in points:
            for d, v in p['values'].items():
                x, y = xlog(d, L_, R), yv(v); s.append(f'<rect class="pt" fill="var(--{p["var"]})" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" transform="rotate(45 {x:.1f} {y:.1f})"/>')
            data['series'].append(dict(name=p['name'], var=p['var'], values=[p['values'].get(d) for d in xs_all], unit=p.get('unit', '')))
    s.append(f'<line class="xh" x1="0" y1="{T}" x2="0" y2="{Bt}" hidden/>')
    s.append('</svg>'); s.append(f'<script type="application/json" data-for="{pid}">{json.dumps(data)}</script>')
    return '\n'.join(s)

def mem_chart():
    W, rowh, top = 760, 34, 30; H = top + rowh * len(MEMROWS) + 46; L_, R = 210, W - 70
    sc = lambda mib: L_ + mib / 40000 * (R - L_)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Stacked horizontal bars of per-die memory for eight context configurations: weights, KV cache, recurrent state and compute buffer, against the 32 GiB a die holds. Eight sequences at 256K exceed the die at both f16 and q8_0; four at 256K and eight at 128K or 160K fit.">']
    s.append(f'<text class="val" x="{L_}" y="16">memory per die, GiB (weights · KV cache · recurrent state · compute buffer)</text>')
    for g in (8, 16, 24, 32, 40): s.append(f'<line class="{"limit" if g == 32 else "grid"}" x1="{sc(g * 1024):.1f}" y1="{top}" x2="{sc(g * 1024):.1f}" y2="{H - 40}"/><text class="tick" x="{sc(g * 1024):.1f}" y="{H - 24}" text-anchor="middle">{g}</text>')
    s.append(f'<text class="region-l" x="{sc(32768) + 4:.1f}" y="{top - 4}">32 GiB die</text>')
    parts = [('model', 'series-dim', 'weights'), ('kvm', 'series', 'KV cache'), ('rs', 'series-3', 'recurrent state'), ('comp', 'series-2', 'compute buffer')]
    for i, m in enumerate(MEMROWS):
        y = top + i * rowh + 6; x = L_
        s.append(f'<text class="val" x="{L_ - 10}" y="{y + 15}" text-anchor="end">{m["label"]} · {m["kv"]}</text>')
        tip = f'{m["label"]} at {m["kv"]}: weights {m["model"]/1024:.1f} GiB + KV {m["kvm"]/1024:.1f} + state {m["rs"]/1024:.2f} + compute {m["comp"]/1024:.1f} = {m["total"]/1024:.1f} GiB per die; {"fits" if m["fits"] else "does not fit"}'
        s.append(f'<g class="bar" tabindex="0" data-tip="{tip}">')
        for key, var, _ in parts:
            w = m[key] / 40000 * (R - L_)
            s.append(f'<rect class="mark" fill="var(--{var})" x="{x + 1:.1f}" y="{y}" width="{max(0, w - 2):.1f}" height="22"{"" if m["fits"] else " opacity=\"0.45\""}/>'); x += w
        s.append(f'<text class="{"val" if m["fits"] else "ratio"}" x="{x + 8:.1f}" y="{y + 15}">{m["total"]/1024:.1f} GiB{"" if m["fits"] else " · out of memory"}</text>')
        s.append('</g>')
    s.append('</svg>'); return '\n'.join(s)

def cols_panel(pid, title, groups, ymax, ticks, ylabel, aria, W=560, H=280):
    """groups: list of (label, [(name, var, value, tip)])"""
    L_, R, T, Bt = 58, W - 16, 30, H - 44; n = len(groups); gw = (R - L_) / n
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">', f'<text class="val" x="{L_}" y="16">{title}</text>', grid_axis(L_, R, T, Bt, ymax, ticks), f'<text class="tick" x="{L_ - 8}" y="{T - 10}" text-anchor="end">{ylabel}</text>']
    for gi, (label, bars) in enumerate(groups):
        k = len(bars); bw = min(24, (gw - 16) / k - 2); x0 = L_ + gi * gw + (gw - (bw + 2) * k) / 2
        for bi, (name, var, val, tip) in enumerate(bars):
            x = x0 + bi * (bw + 2); y = Bt - val / ymax * (Bt - T)
            s.append(f'<g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{x - 2:.1f}" y="{T}" width="{bw + 4:.1f}" height="{Bt - T}"/>')
            s.append(f'<path class="mark" fill="var(--{var})" d="M{x:.1f} {Bt} V{y + 4:.1f} a4 4 0 0 1 4 -4 H{x + bw - 4:.1f} a4 4 0 0 1 4 4 V{Bt} Z"/>')
            s.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{val:.0f}</text></g>')
        s.append(f'<text class="tick" x="{L_ + gi * gw + gw / 2:.1f}" y="{Bt + 18}" text-anchor="middle">{label}</text>')
    s.append('</svg>'); return '\n'.join(s)

def hbars(pid, title, items, xmax, aria, fmt=lambda v: f'{v:.1f}', W=560, ref=None, unit=''):
    """items: (label, var, value, tip); horizontal bars with a reference line"""
    rowh = 26; top = 30; H = top + rowh * len(items) + 40; L_, R = 200, W - 70
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}">', f'<text class="val" x="{L_}" y="16">{title}</text>']
    for i, (label, var, val, tip) in enumerate(items):
        y = top + i * rowh; w = val / xmax * (R - L_)
        s.append(f'<text class="val" x="{L_ - 10}" y="{y + 14}" text-anchor="end">{label}</text><g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{L_}" y="{y}" width="{R - L_ + 70}" height="{rowh}"/>')
        s.append(f'<path class="mark" fill="var(--{var})" d="M{L_} {y + 3} H{L_ + w - 4:.1f} a4 4 0 0 1 4 4 V{y + 15} a4 4 0 0 1 -4 4 H{L_} Z"/><text class="val" x="{L_ + w + 8:.1f}" y="{y + 14}">{fmt(val)}{unit}</text></g>')
    if ref: s.append(f'<line class="limit" x1="{L_ + ref / xmax * (R - L_):.1f}" y1="{top - 6}" x2="{L_ + ref / xmax * (R - L_):.1f}" y2="{top + rowh * len(items)}"/>')
    s.append(f'<line class="axis" x1="{L_}" y1="{top + rowh * len(items) + 2}" x2="{R}" y2="{top + rowh * len(items) + 2}"/>')
    s.append('</svg>'); return '\n'.join(s)

# ---------------- tables ----------------
def ladder_table():
    depths = [r['pp'] for r in L['f16-b4']]
    h = ['<div class="tablewrap"><table><thead><tr><th>context per sequence</th>', ''.join(f'<th class="num">{ctxk(d)}</th>' for d in depths), '</tr></thead><tbody>']
    def line(label, name, key, fmt):
        cells = ''.join(f'<td class="num">{fmt(next((r for r in L[name] if r["pp"] == d), None))}</td>' for d in depths); return f'<tr><td>{label}</td>{cells}</tr>'
    v = lambda r, k: (f'{r[k]:.0f}' if k == 's_pp' else f'{r[k]:.1f}') if r else '—'
    h += [line('prefill t/s, 8 sequences, f16', 'f16-b8', 's_pp', lambda r: v(r, 's_pp')), line('prefill t/s, 4 sequences, f16', 'f16-b4', 's_pp', lambda r: v(r, 's_pp')),
          line('prefill t/s, 8 sequences, q8_0 KV', 'q8-b8', 's_pp', lambda r: v(r, 's_pp')), line('prefill t/s, 4 sequences, q8_0 KV', 'q8-b4', 's_pp', lambda r: v(r, 's_pp')),
          line('decode t/s, 8 sequences, f16', 'f16-b8', 's_tg', lambda r: v(r, 's_tg')), line('decode t/s, 4 sequences, f16', 'f16-b4', 's_tg', lambda r: v(r, 's_tg')),
          line('decode t/s, 8 sequences, q8_0 KV', 'q8-b8', 's_tg', lambda r: v(r, 's_tg')), line('decode t/s, 4 sequences, q8_0 KV', 'q8-b4', 's_tg', lambda r: v(r, 's_tg')),
          line('decode t/s per sequence, 8 × f16', 'f16-b8', 's_tg', lambda r: f'{r["s_tg"] / 8:.1f}' if r else '—'), line('decode t/s per sequence, 4 × f16', 'f16-b4', 's_tg', lambda r: f'{r["s_tg"] / 4:.1f}' if r else '—'),
          line('prefill wall, 8 × f16 (s)', 'f16-b8', 't_pp', lambda r: f'{r["t_pp"]:.0f}' if r else '—'), line('prefill wall, 4 × f16 (s)', 'f16-b4', 't_pp', lambda r: f'{r["t_pp"]:.0f}' if r else '—')]
    h.append('</tbody></table></div>'); return ''.join(h)

def mem_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>slots × context</th><th>KV type</th><th class="num">KV cache</th><th class="num">recurrent state</th><th class="num">compute buffer</th><th class="num">total with 6.3 GiB of weights</th><th>result</th></tr></thead><tbody>']
    for m in MEMROWS:
        h.append(f'<tr><td>{m["label"]}</td><td class="mono">{m["kv"]}</td><td class="num">{m["kvm"]/1024:.1f} GiB</td><td class="num">{m["rs"]/1024:.2f} GiB</td><td class="num">{m["comp"]/1024:.1f} GiB</td><td class="num{" hi" if m["fits"] else ""}">{m["total"]/1024:.1f} GiB</td><td class="note">{"fits" if m["fits"] else "out of memory"}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def server_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>slots</th><th class="num">prompt tokens</th><th class="num">wall s</th><th class="num">first token, mean s</th><th class="num">first token, last slot s</th><th class="num">wave prefill t/s</th><th class="num">generation per request t/s</th><th class="num">tokens/s over the wave</th></tr></thead><tbody>']
    for k, rows in D['server'].items():
        for r in rows:
            h.append(f'<tr><td>{r["conc"]}</td><td class="num">{r["prompt_tokens"]:,}</td><td class="num">{r["wall"]:.0f}</td><td class="num">{r["ttft_mean"]:.0f}</td><td class="num">{r["ttft_max"]:.0f}</td><td class="num">{r["reqs"] * r["prompt_tokens"] / r["ttft_max"]:.0f}</td><td class="num">{r["req_gen"]:.1f}</td><td class="num">{r["agg_total"]:.0f}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def dp4_rows():
    out = []
    for pp in (2048, 4096):
        for b in (1, 4, 8):
            rs = [x for i in range(4) for x in D['dp4'][i] if x['pp'] == pp and x['b'] == b]
            if len(rs) == 4: out.append(dict(pp=pp, b=b, pp_sum=sum(r['s_pp'] for r in rs), tg_sum=sum(r['s_tg'] for r in rs), pp_die=(min(r['s_pp'] for r in rs), max(r['s_pp'] for r in rs)), tg_die=(min(r['s_tg'] for r in rs), max(r['s_tg'] for r in rs))))
    return out
DP = dp4_rows()
TP_REF = {1: (838, 44.2), 4: (832.24, 118.72), 8: (837.73, 158.85), 32: (838.47, 181.48)}   # tensor split 4 dies at pp 512..2048 (batched-fine2 / this run)

def ppl_table():
    P = D['ppl']; rows = [('f16', 'tensor split', 'f16-tensor', 'baseline'), ('q8_0 keys and values', 'tensor split', 'q8q8-tensor', 'runs on the dies'), ('f16', 'layer split', 'f16-layer', 'control for the rows below'),
                          ('q8_0 keys, q4_0 values', 'layer split', 'q8q4-layer', 'attention ran on the CPU: 55 min for the 98k tokens against 5.5, dies at 39 W'), ('q8_0 keys, iq4_nl values', 'layer split', 'q8iq4-layer', 'same fallback; stopped after 1 min to give the dies back'),
                          ('q8_0 keys, q4_0 or iq4_nl values', 'tensor split', None, 'aborts in the split backend: GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)')]
    h = ['<div class="tablewrap"><table><thead><tr><th>KV cache</th><th>placement</th><th class="num">perplexity, wikitext-2, 6 × 16K</th><th>note</th></tr></thead><tbody>']
    for kv, pl, key, note in rows:
        p = P.get(key) if key else None; val = f'{p["ppl"]:.3f} ± {p["err"]:.3f}' if p and p.get('ppl') else ('—' if key else 'not runnable')
        h.append(f'<tr><td>{kv}</td><td>{pl}</td><td class="num">{val}</td><td class="note">{note}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def therm_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>stage</th><th>window (UTC)</th><th class="num">samples</th><th class="num">samples with a die at 1000 MHz</th><th>max junction °C (0b / 0e / 1b / 1e)</th><th class="num">mean W of dies at 1730 MHz</th></tr></thead><tbody>']
    for name, t0, t1 in D['ctx_stages']:
        c = D['ctx_clocks'].get(name)
        if not c or not c['samples']: continue
        h.append(f'<tr><td>{name}</td><td class="mono">{t0[:5]}–{t1[:5]}</td><td class="num">{c["samples"]}</td><td class="num">{c["low"]} ({100 * c["low"] / c["samples"]:.0f}%)</td><td class="mono">{" / ".join(map(str, c["max_t"]))}</td><td class="num">{c["mean_w_hi"]:.0f}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)


def scaling_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>model</th><th class="num">blocks</th><th class="num">GiB on disk</th><th class="num">one die, ms / token</th><th class="num">fit</th><th class="num">four dies, ms / token</th><th class="num">fit</th><th class="num">four dies, prefill t/s</th></tr></thead><tbody>']
    for k, v in CP.DENSE.items():
        if not v['pts'][1]: continue
        one = 1000 / v['one'][1] if v['one'] and v['one'][1] else None
        h.append(f'<tr><td>{k}</td><td class="num">{v["layers"]}</td><td class="num">{v["gib"]:.2f}</td><td class="num">{f"{one:.1f}" if one else "—"}</td><td class="num">{CP.t1(v["layers"], v["gib"]):.1f}</td><td class="num">{1000 / v["pts"][1]:.1f}</td><td class="num">{CP.t4(v["layers"], v["gib"]):.1f}</td><td class="num">{v["pts"][0]:.0f}</td></tr>')
    m = CP.MOE
    h.append(f'<tr><td>35B-A3B Q8_0 (mixture, {m["active_gib"]:.1f} GiB active)</td><td class="num">40</td><td class="num">{m["gib_file"]:.2f}</td><td class="num">two dies: {1000 / m["tg2"]:.1f}</td><td class="num">—</td><td class="num">{m["t_meas"]:.1f}</td><td class="num">{m["t_dense"]:.1f} + {m["penalty_per_layer"] * 40:.1f}</td><td class="num">{m["pp4"]:.0f}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def predict_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>Flash-Next file and placement</th><th class="num">weights per die</th><th class="num">cache room (f16)</th><th class="num">slots at 256K</th><th class="num">one stream t/s</th><th class="num">at 256K depth</th><th class="num">eight streams t/s</th><th class="num">prefill t/s, short / 256K avg</th><th class="num">256K prompt, min</th></tr></thead><tbody>']
    for k, p in CP.PRED.items():
        h.append(f'<tr><td>{k}</td><td class="num">{p["per_die_gb"]:.1f} GB</td><td class="num">{p["kv_tokens"] / 1024:.0f}K tokens</td><td class="num">{p["slots_256k"]}</td><td class="num">{p["tg1"]:.0f}</td><td class="num">{p["tg1_256k"]:.0f}</td><td class="num">{p["tg8"]:.0f}</td><td class="num">{p["pp_rate"]:.0f} / {p["pp_avg_256k"]:.0f}</td><td class="num">{p["pp_256k_min"]:.0f}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)


def E_row(v, pr): return next((r for r in D['opt_E'] if r['variant'] == v and r['prompt'] == pr), None)
def E_tps(v, pr): r = E_row(v, pr); return float(r['tps']) if r else 0.0
def E_rate(v, pr):
    r = E_row(v, pr)
    if not r: return 0.0
    import re as _re; m = _re.search(r'acceptance = ([\d.]+)', r['acc']); return 100 * float(m[1]) if m else 0.0

# ---------------- optional sections (opt-sweep / extras) ----------------
A = D['opt_A']
def opt_A_section():
    if not A: return '<div class="prose"><p class="note">The allreduce and split-mode comparison is still running; this section fills in when it finishes.</p></div>'
    base = A.get('base', {}); order = [k for k in ['base', 'topo-fixed', 'topo-fixed-16ch', 'internal', 'internal-p2p', 'none', 'nccl-p2p', 'no-graphs', 'row-split'] if k in A]
    labels = {'base': 'default (RCCL)', 'topo-fixed': 'RCCL, fixed topology', 'topo-fixed-16ch': 'fixed topology, 16 ch', 'internal': 'internal allreduce', 'internal-p2p': 'internal + peer access',
              'none': 'generic reduce', 'nccl-p2p': 'RCCL + peer access', 'no-graphs': 'HIP graphs off', 'row-split': 'row split (-sm row)'}
    items_tg = []; items_pp = []; items_b8 = []
    for k in order:
        a = A[k]
        if 'fail' in a: continue
        tip = f'{labels[k]}: pp2048 {a["pp"]:.0f} t/s, tg256 {a["tg"]:.2f} t/s, batch 8: prefill {a["b8pp"]:.0f}, decode {a["b8tg"]:.1f}; {a["env"]}'
        items_tg.append((labels[k], 'series' if k != 'base' else 'series-dim', a['tg'], tip)); items_pp.append((labels[k], 'series-2' if k != 'base' else 'series-dim', a['pp'], tip)); items_b8.append((labels[k], 'series-3' if k != 'base' else 'series-dim', a['b8tg'], tip))
    best = max((k for k in order if 'fail' not in A[k] and k not in ('base', 'row-split')), key=lambda k: A[k]['tg'], default=None)
    h = ['<div class="two">',
         f'<figure><div class="chart">{hbars("optA-tg", "single-stream decode, tg256, tokens/s", items_tg, 60, "Horizontal bars of single-stream generation throughput per allreduce variant.", ref=base.get("tg"))}<div class="tip" role="status" aria-live="polite"></div></div></figure>',
         f'<figure><div class="chart">{hbars("optA-pp", "prefill, pp2048, tokens/s", items_pp, 1000, "Horizontal bars of prompt-processing throughput per variant.", fmt=lambda v: f"{v:.0f}", ref=base.get("pp"))}<div class="tip" role="status" aria-live="polite"></div></div></figure>', '</div>']
    tbl = ['<div class="tablewrap"><table><thead><tr><th>variant</th><th>setting</th><th class="num">pp2048</th><th class="num">tg256</th><th class="num">batch 8 prefill</th><th class="num">batch 8 decode</th><th class="num">tg256 vs default</th></tr></thead><tbody>']
    for k in order:
        a = A[k]
        if 'fail' in a: tbl.append(f'<tr><td>{labels[k]}</td><td class="mono">{a["env"]}</td><td colspan="5" class="note">failed</td></tr>'); continue
        d = (a['tg'] / base['tg'] - 1) * 100 if base else 0
        tbl.append(f'<tr><td>{labels[k]}</td><td class="mono">{a["env"]}</td><td class="num">{a["pp"]:.0f}</td><td class="num{" hi" if k == best else ""}">{a["tg"]:.2f}</td><td class="num">{a["b8pp"]:.0f}</td><td class="num">{a["b8tg"]:.1f}</td><td class="num">{d:+.1f}%</td></tr>')
    tbl.append('</tbody></table></div>')
    return '\n'.join(h) + ''.join(tbl)

def opt_B_section():
    Bt = D['opt_B']
    if not Bt: return '<div class="prose"><p class="note">The micro-batch comparison is still running.</p></div>'
    h = ['<div class="tablewrap"><table><thead><tr><th class="num">prompt tokens</th><th class="num">micro-batch</th><th class="num">compute buffer per die</th><th class="num">prefill t/s, 8 sequences</th><th class="num">decode t/s</th></tr></thead><tbody>']
    for r in Bt: h.append(f'<tr><td class="num">{r["npp"]:,}</td><td class="num">{r["ub"]}</td><td class="num">{r["compute"]} MiB</td><td class="num">{r["pp"]:.0f}</td><td class="num">{r["tg"]:.1f}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def spec_section():
    E = D['opt_E']
    if not E: return '<div class="prose"><p class="note">The speculative-decoding comparison is still running.</p></div>'
    h = ['<div class="tablewrap"><table><thead><tr><th>variant</th><th>prompt</th><th class="num">prompt tokens</th><th class="num">generated</th><th class="num">tokens/s</th><th class="num">first token s</th><th>acceptance (server log)</th></tr></thead><tbody>']
    names = {'none': 'none', 'mtp': 'MTP head, 3 drafted', 'mtp-n2': 'MTP head, 2 drafted', 'ngram-mod': 'n-gram lookup (ngram-mod)', 'draft-0.8b': 'draft Qwen3.5-0.8B, 6 drafted', 'draft-2b': 'draft Qwen3.5-2B, 6 drafted'}
    for r in E:
        import re as _re; m = _re.search(r'acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+)', r['acc']); acc = f'{100 * float(m[1]):.0f}% ({m[2]} of {m[3]})' if m else '—'
        h.append(f'<tr><td>{names.get(r["variant"], r["variant"])}</td><td>{"free generation" if r["prompt"] == "free" else "code rewrite"}</td><td class="num">{r["pn"]}</td><td class="num">{r["gn"]}</td><td class="num">{float(r["tps"]):.1f}</td><td class="num">{r["ttft"]}</td><td class="note">{acc}</td></tr>')
    h.append('</tbody></table></div>'); return ''.join(h)


def extras_section():
    b1 = D['x_b1']; c160 = D['x_160k']; c192 = D['x_192k']; lad2 = D['x_ladder2']
    if not (b1 or c160 or c192 or lad2 or D['x2_cAB'] or D['x2_topo'] or D['x2_160k'] or D['x2_192k']):
        return '<div class="prose"><p class="note">The follow-up runs (single stream to 256K, the 8-slot ceiling cells at 160K and 192K, a data-parallel run with a q8_0 cache, and the batch-8 ladder under the corrected topology) are still in progress; this section fills in when they finish.</p></div>'
    h = []
    if b1:
        h.append('<h3>One stream to 256K</h3><div class="tablewrap"><table><thead><tr><th class="num">context</th><th class="num">prefill t/s</th><th class="num">prefill wall s</th><th class="num">decode t/s</th></tr></thead><tbody>')
        for r in b1: h.append(f'<tr><td class="num">{ctxk(r["pp"])}</td><td class="num">{r["s_pp"]:.0f}</td><td class="num">{r["t_pp"]:.0f}</td><td class="num">{r["s_tg"]:.1f}</td></tr>')
        h.append('</tbody></table></div>')
        r = b1[-1]; h.append(f'<div class="prose"><p>A single 256K prompt is read in {r["t_pp"] / 60:.1f} minutes ({r["s_pp"]:.0f} tokens/s averaged) and the stream then generates at {r["s_tg"]:.1f} tokens/s, against {b1[0]["s_tg"]:.1f} at 2K: the same 46 µs per 1K of depth as sections 3 found for four and eight streams, on top of the 21.9 ms single-stream step.</p></div>')

    ab = D['x2_cAB']
    if ab:
        h.append('<h3>The 8K cell, four fresh runs, two allocations</h3><div class="tablewrap"><table><thead><tr><th>run</th><th class="num">cache allocation, cells</th><th class="num">prefill t/s</th><th class="num">decode t/s</th></tr></thead><tbody>')
        for i, (k, rows) in enumerate(ab.items()):
            c = k.split('-c')[1].split('-')[0]
            for r in rows: h.append(f'<tr><td>{i + 1}</td><td class="num">{int(c):,}</td><td class="num">{r["s_pp"]:.0f}</td><td class="num">{r["s_tg"]:.1f}</td></tr>')
        h.append('</tbody></table></div><div class="prose"><p>The ladder measured 152 tokens/s for eight sequences at 8K, the micro-batch runs 130–134. Four fresh processes at the two allocations give 133–142 with no pattern by allocation, so this cell simply varies by 5–7% between runs; prefill does not vary.</p></div>')
    c160 = c160 or D['x2_160k']; c192 = c192 or D['x2_192k']
    topo = D['x2_topo']
    if topo:
        h.append('<h3>The batch-8 ladder under the corrected topology</h3><div class="tablewrap"><table><thead><tr><th>context</th>' + ''.join(f'<th class="num">{ctxk(r["pp"])}</th>' for r in topo) + '</tr></thead><tbody>')
        h.append('<tr><td>prefill t/s, default topology</td>' + ''.join(f'<td class="num">{row("f16-b8", r["pp"])["s_pp"]:.0f}</td>' for r in topo) + '</tr>')
        h.append('<tr><td>prefill t/s, corrected topology</td>' + ''.join(f'<td class="num">{r["s_pp"]:.0f}</td>' for r in topo) + '</tr>')
        h.append('<tr><td>decode t/s, default topology</td>' + ''.join(f'<td class="num">{row("f16-b8", r["pp"])["s_tg"]:.1f}</td>' for r in topo) + '</tr>')
        h.append('<tr><td>decode t/s, corrected topology</td>' + ''.join(f'<td class="num">{r["s_tg"]:.1f}</td>' for r in topo) + '</tr></tbody></table></div>')
    if c160 or c192:
        h.append('<h3>Eight slots at their ceiling</h3><div class="tablewrap"><table><thead><tr><th>configuration</th><th class="num">prefill t/s</th><th class="num">prefill wall</th><th class="num">decode t/s</th><th class="num">per sequence</th></tr></thead><tbody>')
        for lab, rows in (('8 × 160K, f16', c160), ('8 × 192K, q8_0', c192)):
            for r in rows: h.append(f'<tr><td>{lab}</td><td class="num">{r["s_pp"]:.0f}</td><td class="num">{r["t_pp"] / 60:.0f} min</td><td class="num">{r["s_tg"]:.1f}</td><td class="num">{r["s_tg"] / 8:.1f}</td></tr>')
        h.append('</tbody></table></div>')
    dq = [x for i in range(4) for x in D['x_dp4q8'][i]]
    if dq:
        h.append('<h3>Four instances with a q8_0 cache</h3><div class="tablewrap"><table><thead><tr><th class="num">prompt</th><th class="num">batch per die</th><th class="num">prefill t/s, four dies</th><th class="num">decode t/s, four dies</th><th class="num">per stream</th></tr></thead><tbody>')
        for pp in (2048, 8192):
            rs = [x for x in dq if x['pp'] == pp]
            if len(rs) == 4: h.append(f'<tr><td class="num">{pp}</td><td class="num">8</td><td class="num">{sum(r["s_pp"] for r in rs):.0f}</td><td class="num">{sum(r["s_tg"] for r in rs):.1f}</td><td class="num">{sum(r["s_tg"] for r in rs) / 32:.1f}</td></tr>')
        h.append('</tbody></table></div>')
    if lad2:
        name = list(lad2)[0]; rows = lad2[name]
        h.append(f'<h3>The batch-8 ladder again{" under the corrected topology" if "variant" in name else ""}</h3><div class="tablewrap"><table><thead><tr><th>context</th>' + ''.join(f'<th class="num">{ctxk(r["pp"])}</th>' for r in rows) + '</tr></thead><tbody>')
        h.append('<tr><td>prefill t/s, first run</td>' + ''.join(f'<td class="num">{row("f16-b8", r["pp"])["s_pp"]:.0f}</td>' for r in rows) + '</tr>')
        h.append('<tr><td>prefill t/s, this run</td>' + ''.join(f'<td class="num">{r["s_pp"]:.0f}</td>' for r in rows) + '</tr>')
        h.append('<tr><td>decode t/s, first run</td>' + ''.join(f'<td class="num">{row("f16-b8", r["pp"])["s_tg"]:.1f}</td>' for r in rows) + '</tr>')
        h.append('<tr><td>decode t/s, this run</td>' + ''.join(f'<td class="num">{r["s_tg"]:.1f}</td>' for r in rows) + '</tr>')
        h.append('</tbody></table></div>')
    return ''.join(h)


def md_table(rows, num_from=1, hi=None):
    """rows: list of cell lists incl. header + separator; renders a table; cells from num_from right-aligned"""
    rows = [r for r in rows if r and not all(set(c) <= set('-: ') for c in r)]
    if not rows: return ''
    h = ['<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th{" class=num" if i >= num_from else ""}>{c}</th>' for i, c in enumerate(rows[0])) + '</tr></thead><tbody>']
    for r in rows[1:]:
        h.append('<tr>' + ''.join(f'<td class="{"num" if i >= num_from else ""}{" hi" if hi and hi(r) and i == hi(r) else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>')
    h.append('</tbody></table></div>'); return ''.join(h)

def unified_section():
    kb = D['k_bb']; ks = D['k_srv']
    if not (kb['kvu-b8'] or kb['kvu-b4'] or ks): return '<div class="prose"><p class="note">The shared-pool runs are in progress; this part fills in when they finish.</p></div>'
    h = []
    if kb['kvu-b8'] or kb['kvu-b4']:
        h.append('<div class="tablewrap"><table><thead><tr><th>batched bench</th><th class="num">context per sequence</th><th class="num">prefill t/s, per-slot caches</th><th class="num">prefill t/s, one shared pool</th><th class="num">decode t/s, per-slot caches</th><th class="num">decode t/s, one shared pool</th></tr></thead><tbody>')
        for name, lad in (('kvu-b8', 'f16-b8'), ('kvu-b4', 'f16-b4')):
            for r in kb[name]:
                ref = min(L[lad], key=lambda x: abs(x['pp'] - r['pp']))
                h.append(f'<tr><td>{r["b"]} sequences in a 256K pool</td><td class="num">{ctxk(r["pp"] + 256 if r["pp"] % 1024 else r["pp"])}</td><td class="num">{ref["s_pp"]:.0f}</td><td class="num">{r["s_pp"]:.0f}</td><td class="num">{ref["s_tg"]:.1f}</td><td class="num">{r["s_tg"]:.1f}</td></tr>')
        h.append('</tbody></table></div>')
    top = ks.get('(top)', [])
    if top:
        waves = [r for r in top if len(r) == 10]; mix = [r for r in top if len(r) == 9]
        if waves: h.append('<h3>Waves on the shared pool</h3>' + md_table([['server', 'wave', 'wall s', 'agg gen t/s', 'per-request gen t/s', 'per-request prefill t/s', 'first token mean / max s', 'request wall mean / max s'], ['-'] * 8] + [[r[0], f'{r[2]} × {int(r[1]):,} tokens', r[4], r[5], r[6], r[7], r[8], r[9]] for r in waves if r[0] != 'server' and not r[0].startswith('-')], num_from=2))
        if mix: h.append('<h3>One long request beside seven short ones</h3>' + md_table([r for r in mix], num_from=2))
    x3 = D.get('x3')
    if x3:
        if x3['single']:
            rows = [r for r in x3['single'] if len(r) == 10 and r[0] != 'server' and not r[0].startswith('-')]; pool = K_wave(261888)
            if rows or pool:
                h.append('<h3>One 256K request alone</h3><div class="tablewrap"><table><thead><tr><th>server</th><th class="num">first token s</th><th class="num">prefill t/s</th><th class="num">generation t/s</th><th class="num">wall s</th></tr></thead><tbody>')
                if pool: h.append(f'<tr><td>shared 256K pool, 8 slots</td><td class="num">{pool["ttft"].split(" / ")[0]}</td><td class="num">{pool["pp"]:.0f}</td><td class="num">{pool["gen"]:.1f}</td><td class="num">{pool["wall"]:.0f}</td></tr>')
                for r in rows: h.append(f'<tr><td>private caches, 4 slots × 256K</td><td class="num">{r[8].split(" / ")[0]}</td><td class="num">{r[7]}</td><td class="num">{r[6]}</td><td class="num">{r[4]}</td></tr>')
                b1 = next((r for r in D['x_b1'] if r['pp'] == 262016), None)
                if b1: h.append(f'<tr><td>batched bench, one sequence, private cache</td><td class="num">{b1["t_pp"]:.0f}</td><td class="num">{b1["s_pp"]:.0f}</td><td class="num">{b1["s_tg"]:.1f}</td><td class="num">{b1["t"]:.0f}</td></tr>')
                k1 = next((r for r in x3['kvu_b1'] if r['pp'] == 262016), None)
                if k1: h.append(f'<tr><td>batched bench, one sequence, shared pool</td><td class="num">{k1["t_pp"]:.0f}</td><td class="num">{k1["s_pp"]:.0f}</td><td class="num">{k1["s_tg"]:.1f}</td><td class="num">{k1["t"]:.0f}</td></tr>')
                from ctx_data import rows_bb as _rb; k8 = next((r for r in _rb('/root/rocm-tests/bench/qwen38-27b-q8_0-x4-kvu-b1-seq8.md') if r['pp'] == 262016), None)
                if k8: h.append(f'<tr><td>batched bench, one sequence, shared pool sized for eight</td><td class="num">{k8["t_pp"]:.0f}</td><td class="num">{k8["s_pp"]:.0f}</td><td class="num">{k8["s_tg"]:.1f}</td><td class="num">{k8["t"]:.0f}</td></tr>')
                h.append('</tbody></table></div>')
                h.append(f'<div class="prose"><p>Four of the five read the prompt at the same rate; only the shared-pool <em>server</em> takes twice as long. A private-cache server, the bench with a private cache, the bench with a pool, and the bench with a pool sized for eight sequences all agree, so the penalty is not in the cache layout or the attention kernels but in how the server feeds a prompt through a pooled cache in this build. Generation afterwards is the same 30.6 tokens/s everywhere.</p></div>')
        if x3['mix']:
            mixrows = [r for r in x3['mix'] if len(r) == 9 and r[0] != 'server' and not r[0].startswith('-')]; base = K_mix('per-slot')
            h.append('<h3>The mixed wave with smaller prefill chunks</h3><div class="tablewrap"><table><thead><tr><th>server</th><th class="num">wall s</th><th class="num">long: first token s</th><th class="num">long: gen t/s</th><th class="num">short: first token mean / max s</th><th class="num">short: gen t/s</th><th class="num">short: wall mean / max s</th></tr></thead><tbody>')
            if base: h.append(f'<tr><td>private caches, batch 2048</td><td class="num">{base["wall"]:.0f}</td><td class="num">{base["lttft"]:.0f}</td><td class="num">{base["lgen"]:.1f}</td><td class="num">{base["sttft"]}</td><td class="num">{base["sgen"]:.1f}</td><td class="num">{base["swall"]}</td></tr>')
            for r in mixrows: h.append(f'<tr><td>{r[0].replace("per-slot 8 x 128K, batch", "private caches, batch")}</td><td class="num">{r[2]}</td><td class="num">{r[3]}</td><td class="num">{r[4]}</td><td class="num">{r[6]}</td><td class="num">{r[7]}</td><td class="num">{r[8]}</td></tr>')
            h.append('</tbody></table></div>')
    return ''.join(h)

def f_serve_section():
    F = D['f_serve']; base = F.get('base (OPT=base)', []); best = next((v for k, v in F.items() if k.startswith('best-rerun')), [])
    if not (base and best): return ''
    rows = [['clients', 'agg gen t/s, default', 'agg gen t/s, corrected topology', 'per-request gen t/s, default', 'per-request, corrected', 'first token s, default', 'first token s, corrected'], ['-'] * 7]
    for b, t in zip([r for r in base if r[0].isdigit()], [r for r in best if r[0].isdigit()]): rows.append([b[0], b[3], t[3], b[6], t[6], b[7], t[7]])
    return '<h3>The corrected topology in the server</h3>' + md_table(rows) + '<div class="prose"><p>1300-token prompts, 256 generated, the first report\'s client: the corrected RCCL topology with sixteen channels gives the server +3% at one client, +2% at eight and nothing at four, the same small, real gain the bench showed. It costs nothing to set (<code>NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16</code> in the server\'s environment).</p></div>'

def mtp_more_section():
    Mx = D['m']
    if not Mx: return '<div class="prose"><p class="note">The draft-length, sampling, depth and concurrency runs are in progress; this part fills in when they finish.</p></div>'
    h = []
    for head, rows in Mx.items():
        if head == '(top)' or not rows: continue
        h.append(f'<h3>{head}</h3>' + md_table(rows, num_from=2))
    return ''.join(h)


def M_rows(head_prefix):
    for k, v in D['m'].items():
        if k.startswith(head_prefix): return [r for r in v if r and r[0] not in ('variant',) and not r[0].startswith('-')]
    return []
def M1(v, pr):
    r = next((r for r in M_rows('M1') if r[0] == v and r[1] == pr), None); return float(r[4]) if r else 0.0
def M2(v): r = next((r for r in M_rows('M2') if r[0] == v and r[1] == 'edit'), None); return float(r[4]) if r else 0.0
def M3(v): r = next((r for r in M_rows('M3') if r[0] == v), None); return float(r[4]) if r else 0.0
def M4(v, c):
    r = next((r for r in M_rows('M4') if r[0] == v and r[1] == str(c)), None); return (float(r[4]), float(r[5])) if r else (0.0, 0.0)
def K_wave(prompt):
    for r in D['k_srv'].get('(top)', []):
        if len(r) == 10 and r[1] == str(prompt): return dict(wall=float(r[4]), agg=float(r[5]), gen=float(r[6]), pp=float(r[7]), ttft=r[8])
    return None
def K_mix(server):
    for r in D['k_srv'].get('(top)', []):
        if len(r) == 9 and r[0].startswith(server): return dict(wall=float(r[2]), lttft=float(r[3]), lgen=float(r[4]), sttft=r[6], sgen=float(r[7]), swall=r[8])
    return None
def kvu_cell(name, pp):
    r = next((r for r in D['k_bb'][name] if r['pp'] == pp), None); return r


def topo_gain():
    t = D['x2_topo']
    if not t: return 0.0
    return 100 * (sum(r['s_pp'] / row('f16-b8', r['pp'])['s_pp'] for r in t) / len(t) - 1)


def x3_single_ttft():
    x3 = D.get('x3') or {}
    for r in x3.get('single', []):
        if len(r) == 10 and r[0] != 'server' and not r[0].startswith('-'): return float(r[8].split(' / ')[0])
    return 0.0
def x3_mix_row(bsz):
    x3 = D.get('x3') or {}
    for r in x3.get('mix', []):
        if len(r) == 9 and f'batch {bsz}' in r[0] and not r[0].startswith('-'): return r
    return None
def x3_short_gen(bsz): r = x3_mix_row(bsz); return float(r[7]) if r else 0.0
def x3_long_ttft(bsz): r = x3_mix_row(bsz); return float(r[3]) if r else 0.0

# ---------------- page ----------------
def wave_prefill(r): return r['reqs'] * r['prompt_tokens'] / r['ttft_max']
srv_points = {r['prompt_tokens']: wave_prefill(r) for k in D['server'] for r in D['server'][k]}
S8 = row('f16-b8', 131072); S4 = row('f16-b4', 262016); Q4 = row('q8-b4', 262016); Q8 = row('q8-b8', 131072)
srv = {(r['conc'], r['prompt_tokens']): r for k in D['server'] for r in D['server'][k]}
w4 = srv.get((4, 261888)); w8 = srv.get((8, 130816)); w8s = srv.get((8, 4096)); w8m = srv.get((8, 32768))
dp32 = next((r for r in DP if r['pp'] == 2048 and r['b'] == 8), None); dp4_ = next((r for r in DP if r['pp'] == 2048 and r['b'] == 1), None)

pre_line = depth_lines('pre', 'prefill throughput by context, tokens/s', [
    dict(name='8 sequences, f16 (line)', var='series', values={r['pp']: r['s_pp'] for r in L['f16-b8']}, end='', unit=' t/s'),
    dict(name='4 sequences, f16', var='series', values={r['pp']: r['s_pp'] for r in L['f16-b4']}, end=f'{S4["s_pp"]:.0f} at 256K', unit=' t/s', draw=False),
    dict(name='8 sequences, q8_0 KV', var='series', values={r['pp']: r['s_pp'] for r in L['q8-b8']}, end='', unit=' t/s', draw=False),
    dict(name='4 sequences, q8_0 KV', var='series', values={r['pp']: r['s_pp'] for r in L['q8-b4']}, end='', unit=' t/s', draw=False)],
    1000, [0, 250, 500, 750, 1000], 'tokens/s', 'Line chart of prompt-processing throughput against context length from 2K to 256K on a log axis: about 835 tokens per second at 2K, 727 at 32K, 510 at 128K and 363 at 256K, identical for four and eight sequences and for f16 and q8_0 caches; diamonds mark the server waves slightly above the line.',
    model_line=('model', lambda d: 1000 / (PP['c0'] + PP['k'] * d / 2048)),
    points=[dict(name='server wave (prompt tokens / last first-token)', var='series-3', values={(4096 if t == 4096 else 32768 if t == 32768 else 131072 if t == 130816 else 262016): v for t, v in srv_points.items()}, unit=' t/s')])
dec8 = depth_lines('dec8', 'decode, 8 sequences, tokens/s (all sequences)', [
    dict(name='f16 KV', var='series', values={r['pp']: r['s_tg'] for r in L['f16-b8']}, end=f'{S8["s_tg"]:.0f}', unit=' t/s'),
    dict(name='q8_0 KV', var='series-2', values={r['pp']: r['s_tg'] for r in L['q8-b8']}, end=f'{Q8["s_tg"]:.0f}', unit=' t/s')],
    200, [0, 50, 100, 150, 200], 'tokens/s', 'Line chart of batch-8 generation throughput against context: f16 falls from 159 tokens per second at 2K to 83 at 128K; q8_0 from 153 to 58.')
dec4 = depth_lines('dec4', 'decode, 4 sequences, tokens/s (all sequences)', [
    dict(name='f16 KV', var='series', values={r['pp']: r['s_tg'] for r in L['f16-b4']}, end=f'{S4["s_tg"]:.0f}', unit=' t/s'),
    dict(name='q8_0 KV', var='series-2', values={r['pp']: r['s_tg'] for r in L['q8-b4']}, end=f'{Q4["s_tg"]:.0f}', unit=' t/s')],
    200, [0, 50, 100, 150, 200], 'tokens/s', 'Line chart of batch-4 generation throughput against context: f16 falls from 119 tokens per second at 2K to 50 at 256K; q8_0 from 114 to 32.')

dp_groups_pp = [(f'{4 * b} streams', [('tensor split', 'series', TP_REF[b if b > 1 else 4][0] if b > 1 else TP_REF[4][0], f'tensor split, {4 * b} sequences: {TP_REF[b if b > 1 else 4][0]:.0f} t/s prefill'), ('4 instances', 'series-2', r['pp_sum'], f'four single-die instances at batch {b}: {r["pp_sum"]:.0f} t/s prefill in total ({r["pp_die"][0]:.0f}–{r["pp_die"][1]:.0f} per die)')]) for b, r in [(1, dp4_), (8, dp32)] if r]
dp_groups_tg = [(f'{4 * b} streams', [('tensor split', 'series', TP_REF[4 if b == 1 else 32][1], f'tensor split, {4 * b} sequences: {TP_REF[4 if b == 1 else 32][1]:.1f} t/s decode, {TP_REF[4 if b == 1 else 32][1] / (4 * b):.1f} per stream'), ('4 instances', 'series-2', r['tg_sum'], f'four single-die instances at batch {b}: {r["tg_sum"]:.1f} t/s decode in total, {r["tg_sum"] / (4 * b):.1f} per stream')]) for b, r in [(1, dp4_), (8, dp32)] if r]


html = f'''<title>Qwen3.8-27B at 256K on gfx906</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --bg: #EEF1F4; --surface: #F8FAFB; --surface-2: #E3E8ED;
  --ink: #17212C; --ink-2: #485664; --muted: #76828E; --rule: #D2D9E0; --rule-2: #BFC8D1;
  --accent: #B04F1C; --accent-ink: #8E3E13; --accent-soft: rgba(176,79,28,0.10);
  --series: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-dim: #a9b4be; --grid: #E0E5EA; --axis: #BFC8D1;
  --code-bg: #E4E9EE; --focus: #2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #0E141A; --surface: #151D25; --surface-2: #1C2630;
    --ink: #E7ECF1; --ink-2: #B2BCC6; --muted: #7E8A96; --rule: #29343F; --rule-2: #35414E;
    --accent: #E28B55; --accent-ink: #F0A97C; --accent-soft: rgba(226,139,85,0.14);
    --series: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-dim: #4c5966; --grid: #253039; --axis: #35414E;
    --code-bg: #19232D; --focus: #3987e5;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #0E141A; --surface: #151D25; --surface-2: #1C2630;
  --ink: #E7ECF1; --ink-2: #B2BCC6; --muted: #7E8A96; --rule: #29343F; --rule-2: #35414E;
  --accent: #E28B55; --accent-ink: #F0A97C; --accent-soft: rgba(226,139,85,0.14);
  --series: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-dim: #4c5966; --grid: #253039; --axis: #35414E;
  --code-bg: #19232D; --focus: #3987e5;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif; font-size: 17.5px; line-height: 1.55; -webkit-font-smoothing: antialiased; }}
a {{ color: var(--accent-ink); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
a:focus-visible, [tabindex]:focus-visible {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.page {{ max-width: 74rem; margin: 0 auto; padding: 2.75rem 1.5rem 4rem; display: grid; gap: 3.25rem; }}
h1, h2, h3 {{ font-family: "Schibsted Grotesk", "Helvetica Neue", Arial, sans-serif; text-wrap: balance; margin: 0; line-height: 1.1; letter-spacing: -0.012em; }}
h1 {{ font-size: clamp(2.1rem, 4.5vw, 3.1rem); font-weight: 800; max-width: 22ch; }}
h2 {{ font-size: 1.5rem; font-weight: 700; }}
h3 {{ font-size: 1.05rem; font-weight: 700; }}
p {{ margin: 0; }}
.prose {{ max-width: 66ch; display: grid; gap: 0.9rem; }}
.prose ul {{ margin: 0; padding-left: 1.2rem; }}
.prose li {{ margin: 0.2rem 0; }}
.eyebrow {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent-ink); font-weight: 500; }}
.mono, code, pre, td.num, th.num, td.mono {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; }}
code {{ font-size: 0.86em; background: var(--code-bg); padding: 0.08em 0.35em; border-radius: 3px; }}
pre {{ margin: 0; background: var(--code-bg); border-radius: 6px; padding: 0.95rem 1.1rem; overflow-x: auto; font-size: 0.82rem; line-height: 1.5; color: var(--ink); }}
pre code {{ background: none; padding: 0; font-size: inherit; }}
.mast {{ display: grid; gap: 1.25rem; padding-bottom: 2rem; border-bottom: 1px solid var(--rule); }}
.mast .lede {{ font-size: 1.2rem; color: var(--ink-2); max-width: 60ch; }}
.claim {{ border-left: 3px solid var(--accent); padding: 0.35rem 0 0.35rem 1rem; max-width: 60ch; font-size: 1.05rem; }}
.claim b {{ font-weight: 600; }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 0.6rem 1.75rem; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.8rem; color: var(--ink-2); line-height: 1.45; }}
.facts dt {{ color: var(--muted); font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }}
.facts dd {{ margin: 0.1rem 0 0; }}
.facts div {{ display: grid; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); }}
.tile {{ background: var(--surface); padding: 1.1rem 1.2rem 1.15rem; display: grid; gap: 0.3rem; align-content: start; }}
.tile .v {{ font-family: "Schibsted Grotesk", sans-serif; font-weight: 800; font-size: 2.35rem; line-height: 1; letter-spacing: -0.02em; }}
.tile .v small {{ font-size: 0.95rem; font-weight: 600; color: var(--ink-2); margin-left: 0.25rem; letter-spacing: 0; }}
.tile .l {{ font-size: 0.92rem; color: var(--ink-2); }}
section {{ display: grid; gap: 1.25rem; }}
.sec-head {{ display: grid; gap: 0.35rem; }}
figure {{ margin: 0; display: grid; gap: 0.6rem; }}
figcaption {{ font-size: 0.9rem; color: var(--ink-2); max-width: 70ch; }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--rule); background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-weight: 600; color: var(--ink-2); font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; background: var(--surface-2); }}
tr:last-child td {{ border-bottom: none; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.hi {{ font-weight: 600; }}
td.note, .note {{ color: var(--ink-2); }}
td.mono {{ white-space: nowrap; font-size: 0.85rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; font-size: 0.88rem; color: var(--ink-2); align-items: center; }}
.legend span {{ display: inline-flex; align-items: center; gap: 0.45rem; }}
.sw {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.lk {{ width: 18px; height: 3px; border-radius: 2px; display: inline-block; }}
.dm {{ width: 9px; height: 9px; transform: rotate(45deg); display: inline-block; }}
.lk.dash {{ height: 0; border-top: 2px dashed var(--rule-2); background: none; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1.5rem; align-items: start; }}
.three {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 1.5rem; align-items: start; }}
svg {{ max-width: 100%; height: auto; display: block; }}
.chart {{ background: var(--surface); border: 1px solid var(--rule); padding: 1rem 1rem 0.6rem; position: relative; }}
.chart text {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; fill: var(--ink-2); }}
.chart text.val {{ fill: var(--ink); font-weight: 500; }}
.chart text.tick {{ fill: var(--muted); font-size: 11px; }}
.chart text.ratio {{ fill: var(--ink-2); font-size: 11px; }}
.chart .grid line, .chart line.grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .axis {{ stroke: var(--axis); stroke-width: 1; }}
.chart .limit {{ stroke: var(--accent); stroke-width: 1.5; }}
.chart .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.chart .model {{ fill: none; stroke: var(--rule-2); stroke-width: 1.5; stroke-dasharray: 5 4; }}
.chart .pt {{ stroke: var(--surface); stroke-width: 2; }}
.chart .xh {{ stroke: var(--rule-2); stroke-width: 1; }}
.chart text.region-l {{ fill: var(--accent-ink); font-size: 10.5px; letter-spacing: 0.04em; }}
.chart .bar {{ cursor: default; }}
.chart .bar rect.hit {{ fill: transparent; }}
.chart .bar:hover .mark, .chart .bar:focus .mark {{ filter: brightness(0.92); }}
.chart svg[data-kind="lines"] {{ cursor: crosshair; }}
.tip {{ position: absolute; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 0.8rem; line-height: 1.4; padding: 0.45rem 0.6rem; border-radius: 4px; max-width: 22rem; opacity: 0; transition: opacity 120ms; }}
.tip[data-show="1"] {{ opacity: 1; }}
.tip .row {{ display: flex; gap: 0.5rem; align-items: baseline; }}
.tip .row b {{ font-weight: 600; min-width: 3.6em; text-align: right; font-variant-numeric: tabular-nums; }}
.tip .row i {{ display: inline-block; width: 12px; height: 3px; border-radius: 2px; }}
.tip .h {{ font-weight: 600; margin-bottom: 0.15rem; }}
.files {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.8rem; color: var(--ink-2); display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: 0.35rem 1.5rem; }}
footer {{ border-top: 1px solid var(--rule); padding-top: 1.5rem; font-size: 0.9rem; color: var(--ink-2); display: grid; gap: 0.6rem; }}
@media (prefers-reduced-motion: reduce) {{ .tip {{ transition: none; }} }}
@media (max-width: 40rem) {{ body {{ font-size: 16.5px; }} .tile .v {{ font-size: 2rem; }} }}
</style>

<main class="page">

<header class="mast">
  <div class="eyebrow">Measurement report · Mac Pro (2019) · 2026-09-06</div>
  <h1>Qwen3.8-27B at 256K context on four Vega 20 dies</h1>
  <p class="lede">The same 27 GiB Q8_0 model as the <a href="https://claude.ai/code/artifact/e248ff49-e4e3-4a21-967e-927f21f5af71">first report</a>, now driven to its full 262 144-token context on the four gfx906 dies of a 2019 Mac Pro: what fits, how prompt reading and generation slow down as the context grows, what a quantised KV cache buys and costs, and what the real server does when four or eight long requests land at once.</p>
  <p class="claim"><b>The short version:</b> a die holds 6.3 GiB of weights and 16 KiB of KV cache per token, so eight slots reach 128K to 160K at f16 and four slots reach the full 256K; eight at 256K fit nothing. Reading a prompt costs 1.18 ms per token plus 0.012 ms per thousand tokens already in context, so a 256K prompt takes 12 minutes and its last tokens go by at 233 tokens/s. Generation for four streams falls from 119 tokens/s at 2K to 50 at 256K. q8_0 for keys and values is free in quality and buys 192K at eight slots, but it makes generation up to 36% slower at depth, and 4-bit values do not run on these dies at all. Four independent single-die instances beat the tensor split by 12–14% on short-prompt throughput and lose everywhere else. The model's own next-token head nearly doubles a single stream and still helps at four; eight busy slots run better without it, for a reason the first report's batch staircase predicts. Of the platform knobs, only the corrected XGMI topology for RCCL moves anything, by 1–4%. A calibrated cost model puts a 105 GB Qwen3.8-Flash-Next at about 58 tokens/s on these dies with six 256K slots, once llama.cpp is updated.</p>
  <dl class="facts">
    <div><dt>Model</dt><dd>Qwen3.8-27B, Unsloth GGUF Q8_0, 27.04 GiB; 64 blocks, 16 full attention (4 KV heads × 256) and 48 linear attention; trained context 262 144</dd></div>
    <div><dt>Machine</dt><dd>macpro2019-01: Xeon W-3275M, 377 GiB RAM, 2 × Radeon Pro Vega II Duo = 4 × Vega 20 (gfx906), 32 GB HBM2 each, one XGMI hive</dd></div>
    <div><dt>Software</dt><dd>Ubuntu 24.04, Linux 7.0.0-30, ROCm 7.14 (TheRock, gfx906), llama.cpp b10288 (360e134), tensor split over 4 dies, flash attention, batch and micro-batch 2048</dd></div>
    <div><dt>Conditions</dt><dd>perf level high (1730 MHz), chassis fans at maximum through the T2 fan daemon, junction ≤ 87 °C, dies at their 200 W cap in prefill; one continuous 7.5 h run plus the follow-up runs of the same evening</dd></div>
  </dl>
</header>

<section aria-label="Key figures">
  <div class="tiles">
    <div class="tile"><div class="v">16<small>KiB / token</small></div><div class="l">KiB of f16 KV cache per token per die; 64 KiB across the four. Only the 16 full-attention blocks hold a cache; the 48 linear blocks keep a fixed 150 MiB state per sequence.</div></div>
    <div class="tile"><div class="v">4 × 256K</div><div class="l">The largest full-context configuration that fits: four slots at 256K with 16 GiB of cache per die. Eight slots stop at 160K (f16) or 192K (q8_0).</div></div>
    <div class="tile"><div class="v">{S4['s_pp']:.0f}<small>tok/s</small></div><div class="l">Average prompt-reading rate over a 256K prompt (four at once); {pp_rate(262016):.0f} tokens/s by its last tokens. A single 256K prompt takes {262016 / S4['s_pp'] / 4 / 60:.0f} minutes on its own.</div></div>
    <div class="tile"><div class="v">{S4['s_tg']:.0f}<small>tok/s</small></div><div class="l">Generation for four streams each at 256K depth, {S4['s_tg'] / 4:.1f} per stream, against {row('f16-b4', 2048)['s_tg']:.0f} at 2K. Eight streams at 128K: {S8['s_tg']:.0f} in total, {S8['s_tg'] / 8:.1f} each.</div></div>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">1 · What fits</div><h2>Six and a third GiB of weights per die, then 16 KiB of cache for every token in every slot</h2></div>
  <div class="prose">
    <p>In tensor split every die carries a quarter of the weights (6.3 GiB; the 1.3 GiB token embedding stays in host memory) and a quarter of the KV cache: the cache tensors are split across the dies like the weights, 16 KiB per token per die. The hybrid architecture is what makes 256K thinkable at all: 48 of the 64 blocks are linear attention with a fixed recurrent state of 150 MiB per sequence, and only the 16 full-attention blocks (4 key/value heads of 256, f16) grow with the context. A dense 64-block model of this width would need four times the cache.</p>
    <p>The compute buffer is the third occupant. At f16 it is 2 to 2.5 GiB per die with a 2048-token micro-batch; with a q8_0 cache the flash-attention path needs a scratch copy that grows with the context, 4.9 GiB at a million cells and 9.5 GiB at two million, which is why q8_0 buys less headroom than halving the cache suggests. The chart puts the eight configurations that matter against the 32 GiB of a die; each bar is what llama.cpp reported when it loaded (or what it asked for when it failed).</p>
  </div>
  <figure>
    <div class="chart">{mem_chart()}<div class="tip" role="status" aria-live="polite"></div></div>
    <div class="legend"><span><i class="sw" style="background:var(--series-dim)"></i>weights</span><span><i class="sw" style="background:var(--series)"></i>KV cache</span><span><i class="sw" style="background:var(--series-3)"></i>recurrent state</span><span><i class="sw" style="background:var(--series-2)"></i>compute buffer</span><span>faded: the allocation that failed</span></div>
    <figcaption>Per-die memory by configuration (slots × context per slot). Hover or focus a bar for the figures. Eight slots at 256K would need 32 GiB of f16 cache per die before anything else; at q8_0 the cache fits but its 9.5 GiB compute buffer does not, at any micro-batch size. Eight at 192K f16 fits only with the micro-batch cut to 512, which costs 16% of prefill speed.</figcaption>
  </figure>
  {mem_table()}
  <div class="prose"><p>So the choice for full-context serving is four slots at f16, or eight slots at 128K (160K if nothing else runs on the dies). A server can also be given one shared cache pool (<code>--kv-unified</code>) so that any one slot may reach 256K while the eight together stay within the pool, which suits traffic where long requests are the exception; that mode was not benchmarked here.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">2 · Reading the prompt</div><h2>1.18 ms per token, plus 0.012 ms for every thousand tokens already in context</h2></div>
  <div class="prose"><p><code>llama-batched-bench</code> on the four dies, four or eight sequences each with its own prompt of the stated length, 128 generated tokens per sequence, f16 or q8_0 cache. The prefill rate is the average over the whole prompt; the dashed line is the two-parameter model fitted to it, an instantaneous cost per token of {PP['c0']:.2f} ms plus {PP['k']:.4f} ms per 1K tokens of depth. Four and eight sequences, f16 and q8_0 all lie on the same curve within 1.5%: reading a prompt is a property of depth alone.</p></div>
  <figure>
    <div class="chart">{pre_line}<div class="tip" role="status" aria-live="polite"></div></div>
    <div class="legend"><span><i class="lk" style="background:var(--series)"></i>batched bench, 8 sequences f16 (4 sequences and q8_0 coincide; in the tooltip)</span><span><i class="dm" style="background:var(--series-3)"></i>server waves, section 5</span><span><i class="lk dash"></i>fitted model</span></div>
    <figcaption>Prompt-processing throughput against context length. Move across the chart, or focus it and use the arrow keys, for the measured values at each depth. The attention term is quadratic in the prompt, so the average rate over a prompt falls to {S4['s_pp']:.0f} tokens/s at 256K while the rate at its last tokens is {pp_rate(262016):.0f}.</figcaption>
  </figure>
  <div class="prose">
    <p>The attention share of a token's cost rises from a few percent at 2K to {100 * PP['k'] * 256 / (PP['c0'] + PP['k'] * 256):.0f}% at 256K. That slope is a throughput figure for the flash-attention kernel: 16 blocks × 24 heads × 256 wide × two matrix products is {16 * 2 * 24 * 256 * 2 * 1024 / 1e6:.0f} MFLOP per token per 1K of depth, and {PP['k']:.4f} ms for it means {ATT_TFLOPS:.0f} TFLOP/s over the four dies, {ATT_TFLOPS / 4:.1f} per die, about 60% of the {13.8:.1f} TFLOP/s the dies reach on a pure FMA loop. The 1.18 ms base is the dense part, unchanged from the first report's 830 tokens/s.</p>
    <p>What it means in minutes: a 32K prompt is read in {32768 / row('f16-b4', 32768)['s_pp']:.0f} s, 128K in {131072 / row('f16-b4', 131072)['s_pp'] / 60:.1f} minutes, 256K in {262016 / S4['s_pp'] / 60:.0f} minutes, when the dies do nothing else. Eight 128K prompts arriving together take {8 * 131072 / S8['s_pp'] / 60:.0f} minutes before the last of them sees a token (section 5 measures exactly that on the server).</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">3 · Generating at depth</div><h2>Every thousand tokens of context adds 46 µs to a sequence's step; q8_0 adds 86</h2></div>
  <div class="prose"><p>The same runs, decode phase: all sequences advance one token per step, so the aggregate rate is the batch divided by the step time. Two panels, one per batch size, f16 against q8_0 cache.</p></div>
  <div class="two">
    <figure><div class="chart">{dec8}<div class="tip" role="status" aria-live="polite"></div></div></figure>
    <figure><div class="chart">{dec4}<div class="tip" role="status" aria-live="polite"></div></div></figure>
  </div>
  <div class="legend"><span><i class="lk" style="background:var(--series)"></i>f16 KV cache</span><span><i class="lk" style="background:var(--series-2)"></i>q8_0 keys and values</span></div>
  <figcaption>Aggregate generation throughput against context per sequence. The step time is linear in depth: {MODEL['f16-b8']['base']:.0f} ms + {MODEL['f16-b8']['ks']:.3f} ms per 1K for eight sequences, {MODEL['f16-b4']['base']:.0f} ms + {MODEL['f16-b4']['ks']:.3f} ms per 1K for four, i.e. {MODEL['f16-b8']['kseq'] * 1000:.0f} µs per sequence per 1K of context in both cases. With q8_0 the per-sequence slope is {MODEL['q8-b8']['kseq'] * 1000:.0f} µs.</figcaption>
  {ladder_table()}
  <div class="prose">
    <p>The slope is the cache being read: 64 MiB of f16 keys and values per 1K of depth per sequence, in {MODEL['f16-b8']['kseq'] * 1000:.0f} µs, is {KV_GBS:.0f} GB/s across the four dies, {KV_GBS / 4:.0f} GB/s per die, 44% of the 840 GB/s a plain read of the HBM2 achieves. At 256K depth a four-stream step reads 16 GiB of cache and takes {MODEL['f16-b4']['base'] + MODEL['f16-b4']['ks'] * 256:.0f} ms, so each stream gets {S4['s_tg'] / 4:.1f} tokens/s; at 2K it was {row('f16-b4', 2048)['s_tg'] / 4:.1f}.</p>
    <p>q8_0 halves the bytes and still takes {Q8_RATIO:.1f}× longer per 1K: the flash-attention vector kernel dequantises every key and value block as it goes, and on Vega 20 that arithmetic costs more than the bandwidth it saves. The cache type therefore trades generation speed for memory, not for speed: at 256K with four streams, q8_0 gives {Q4['s_tg']:.0f} tokens/s against {S4['s_tg']:.0f} for f16 ({100 * (1 - Q4['s_tg'] / S4['s_tg']):.0f}% less), and at 128K with eight, {Q8['s_tg']:.0f} against {S8['s_tg']:.0f}. Prefill is untouched.</p>
    <p>The base step (the part that does not depend on depth) is {MODEL['f16-b4']['base']:.0f} ms for four sequences and {MODEL['f16-b8']['base']:.0f} ms for eight, against 21.9 ms for one: streaming the weights once per step and synchronising the four dies once per layer, as the first report found. Eight streams at 128K therefore give {S8['s_tg']:.0f} tokens/s, {S8['s_tg'] / 8:.1f} each; four at 128K give {row('f16-b4', 131072)['s_tg']:.0f}, {row('f16-b4', 131072)['s_tg'] / 4:.1f} each.</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">4 · Cache types</div><h2>q8_0 keys and values cost nothing in quality; 4-bit values do not run here</h2></div>
  <div class="prose"><p><code>llama-perplexity</code> on wikitext-2, six chunks of 16 384 tokens (98k tokens), with the cache types llama.cpp offers. The idea that keys tolerate 8 bits and values 4 is testable only outside the tensor split, whose backend cannot place the converted value tensors and aborts; on layer split the operation runs, but on the CPU.</p></div>
  {ppl_table()}
  <div class="prose"><p>The two perplexities that ran on the dies agree to three decimals, so q8_0 for both keys and values is safe at this context. The 4-bit-value figure is also within the error bar but came from a run in which the dies drew 39 W and 25 CPU cores did the attention, ten times slower than f16; there is no supported GPU kernel for a 256-wide head with a 4-bit value cache in this build. The only quantised cache worth configuring on this machine is q8_0 for both, and section 3 says when it is worth it: when the extra context matters more than the generation rate.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">5 · The server with long requests</div><h2>Eight 128K requests at once: the last one waits {(w8['ttft_max'] / 60) if w8 else 0:.0f} minutes for its first token</h2></div>
  <div class="prose"><p><code>llama-server</code> in tensor split with eight slots of 128K (f16), then four slots of 256K, continuous batching, no prompt cache in RAM. One wave per prompt size: as many simultaneous requests as slots, each a distinct random-word prompt of the stated length, 256 tokens generated, timings from the server's own per-request figures. The 4K and 32K waves are the same server as the 128K one.</p></div>
  {server_table()}
  <div class="prose">
    <p>The server processes the prompts one slot at a time in 2048-token chunks, so the first request sees a token after its own prompt is read and the last after all of them are; the mean first-token time is the middle of that ramp. The wave prefill column, all prompt tokens over the last first-token time, is the server's effective prompt rate: {wave_prefill(w8s) if w8s else 0:.0f} tokens/s at 4K, {wave_prefill(w8m) if w8m else 0:.0f} at 32K, {wave_prefill(w8) if w8 else 0:.0f} at 128K and {wave_prefill(w4) if w4 else 0:.0f} at 256K, a little above the batched bench because a chunk holds one sequence rather than eight slices. Per-request generation is what a client sees once its answer starts: {w8['req_gen'] if w8 else 0:.1f} tokens/s at 128K × 8 and {w4['req_gen'] if w4 else 0:.1f} at 256K × 4, slowed below section 3's figures by the prompts still being read for the slots behind it.</p>
    <p>The practical reading is that long context is a prefill problem before it is a generation problem. Eight simultaneous 128K requests occupy the dies for {w8['wall'] / 60 if w8 else 0:.0f} minutes, four 256K requests for {w4['wall'] / 60 if w4 else 0:.0f}; a queue of such requests is served at about {wave_prefill(w8) if w8 else 0:.0f} to {wave_prefill(w4) if w4 else 0:.0f} prompt tokens per second whatever the slot count. Prompt caching (<code>--cache-ram</code>, on by default and disabled for these measurements) is what turns a long conversation into a short one: the second turn on the same context is read from the cache, not recomputed.</p>
  </div>
  <h3>One shared pool instead of eight private caches</h3>
  <div class="prose"><p>llama.cpp can give the slots one cache pool (<code>--kv-unified</code>): every slot may reach the pool\'s size on its own, and the slots together may not exceed it. With a 256K pool that is eight slots that can each take a 256K request, provided the others are idle, in 4 GiB of cache per die instead of 16. The cost is in the attention: in a shared pool every sequence\'s attention scans the whole used pool behind a mask, not just its own tokens. The batched bench measures that directly, then the server runs the traffic pattern the mode is meant for.</p></div>
  {unified_section()}
  <div class="prose">
    <p>The batched bench puts a number on the pool: with eight sequences at 32K in a 256K pool, prefill runs at the rate of a single 256K prompt ({(kvu_cell("kvu-b8", 32512) or {"s_pp": 0})["s_pp"]:.0f} tokens/s against {row("f16-b8", 32768)["s_pp"]:.0f} with private caches) and generation loses a third ({(kvu_cell("kvu-b8", 32512) or {"s_tg": 0})["s_tg"]:.0f} against {row("f16-b8", 32768)["s_tg"]:.0f}); at 2K the pool is nearly free (7%). What a sequence pays is the attention over everything in the pool, not over its own context. The server tells the same story in wall-clock: eight 32K requests at once take {(K_wave(32512) or {"wall": 0})["wall"]:.0f} s on the pool against {w8m["wall"] if w8m else 0:.0f} on private caches, and a 256K request alone waited {(K_wave(261888) or {"ttft": "?"})["ttft"].split(" / ")[0]} s for its first token on the pool against {x3_single_ttft():.0f} s on private caches (the table below); eight 4K requests are within 12%. The mode is worth having only when long requests are rare and the pool is otherwise nearly empty: then one slot can take 256K in 4 GiB of cache per die that eight private 256K caches could never fit.</p>
    <p>The mixed wave shows something the pool cannot fix, on either server. One 128K prompt beside seven 4K ones: on private caches the short requests got their first token in {(K_mix("per-slot") or {"sttft": "?"})["sttft"]} s but then generated at {(K_mix("per-slot") or {"sgen": 0})["sgen"]:.1f} tokens/s until the long prompt was read, because every one of their decode steps rides in a batch with a 2048-token chunk of the long prompt and takes as long as that chunk. On the pool the short slots decoded at {(K_mix("unified") or {"sgen": 0})["sgen"]:.1f} tokens/s but two of them waited over 300 s, and the long prompt itself slowed from {(K_mix("per-slot") or {"lttft": 0})["lttft"]:.0f} to {(K_mix("unified") or {"lttft": 0})["lttft"]:.0f} s. A long prefill is a head-of-line block for everyone sharing the dies. The batch size, which sets how large each prompt chunk is, was the obvious lever and it is not one: at 1024 and 512 the short requests still generate at {x3_short_gen(1024):.1f} and {x3_short_gen(512):.1f} tokens/s until the long prompt is read, and the long prompt only gets slower ({x3_long_ttft(1024):.0f} and {x3_long_ttft(512):.0f} s to its first token against {(K_mix('per-slot') or {'lttft': 0})['lttft']:.0f}). The server adds the generating slots' tokens to every batch before the prompt chunk, so the pacing of the short slots is set by something other than chunk size in this build; whatever it is, a server that must answer short requests promptly while reading long prompts needs either a second instance for the short traffic (section 6) or an admission policy that keeps very long prompts off the busy slots.</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">6 · Four instances instead of one split</div><h2>Independent dies win 12–14% on short-prompt throughput and lose the rest</h2></div>
  <div class="prose"><p>One llama.cpp instance per die, no split, all four running at once with 2048- and 4096-token prompts. Each die holds the whole model (25.4 GiB) beside 2 GiB of cache, 1.2 GiB of recurrent state and 2 GiB of compute buffer, about 1 GiB from the edge; that leaves room for roughly 48K tokens of f16 cache per die in total, so this mode is for short requests only. The four dies did not slow each other at all: each matched a lone die within 2%.</p></div>
  <div class="two">
    <figure><div class="chart">{cols_panel('dp-pp', 'prefill, tokens/s in total', dp_groups_pp, 1000, [0, 250, 500, 750, 1000], 'tokens/s', 'Column chart comparing prompt throughput: tensor split 832 versus four instances 929 at four streams, 838 versus 929 at thirty-two.')}<div class="tip" role="status" aria-live="polite"></div></div></figure>
    <figure><div class="chart">{cols_panel('dp-tg', 'decode, tokens/s in total', dp_groups_tg, 250, [0, 50, 100, 150, 200, 250], 'tokens/s', 'Column chart comparing generation throughput: tensor split 119 versus four instances 77 at four streams, 181 versus 206 at thirty-two.')}<div class="tip" role="status" aria-live="polite"></div></div></figure>
  </div>
  <div class="legend"><span><i class="sw" style="background:var(--series)"></i>tensor split, one instance on four dies</span><span><i class="sw" style="background:var(--series-2)"></i>four instances, one per die</span></div>
  <figcaption>Totals over the four dies for the same number of streams (2048-token prompts). Four instances read prompts at {dp32['pp_sum'] if dp32 else 0:.0f} tokens/s whatever the batch, four single-die rates added; the split reaches {TP_REF[8][0]:.0f}. Decode: four instances at batch 8 give {dp32['tg_sum'] if dp32 else 0:.0f} tokens/s for 32 streams ({dp32['tg_sum'] / 32 if dp32 else 0:.1f} each) against the split's {TP_REF[32][1]:.0f} ({TP_REF[32][1] / 32:.1f} each), but four single streams get {dp4_['tg_sum'] / 4 if dp4_ else 0:.1f} tokens/s each against {TP_REF[4][1] / 4:.1f} on the split and one stream {19.3:.1f} against 45.7.</figcaption>
  <div class="prose"><p>The split's cross-die synchronisation costs it about a third of the ideal in prefill (four dies read at {TP_REF[8][0] / 234:.1f}× one die's rate) and more in decode at small batches, and that is what the four instances recover; the price is single-die latency, four separate queues, and no long context. For a workload of many short requests where nobody waits on one answer, four instances are the higher-throughput configuration; for everything in this report, the split is.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">7 · Platform knobs</div><h2>The cross-die allreduce, micro-batch size and split mode</h2></div>
  <div class="prose"><p>The first report attributed the split's generation cost to the synchronisation between dies; the XGMI report found that RCCL builds its rings from a firmware topology that mislabels the bridge links and runs 40% faster with a corrected topology file. This build of llama.cpp uses RCCL for its tensor-split allreduce on Linux, so the two meet here: the same corrected topology, RCCL's channel count, llama.cpp's own internal allreduce, explicit peer access, HIP graphs and the legacy row split, each measured with <code>llama-bench</code> (2048-token prompt, 256 generated, two repetitions) and with the eight-sequence batched bench.</p></div>
  {opt_A_section()}
  <div class="prose">
    <p>RCCL is the right allreduce, and the corrected topology helps it a little. With the firmware's hop table RCCL builds its rings as 0-1-2-3 and 0-3-2-1, each crossing two bridge pairs that are not directly linked; the corrected file gives it the physical ring 0-1-3-2 and 0-2-3-1. For llama.cpp that is worth {(A["topo-fixed"]["pp"] / A["base"]["pp"] - 1) * 100 if A else 0:+.1f}% in prefill and {(A["topo-fixed-16ch"]["tg"] / A["base"]["tg"] - 1) * 100 if A else 0:+.1f}% in single-stream generation with sixteen channels, against the 40% the XGMI report measured on gigabyte transfers: the split's messages are a few hundred kilobytes per layer in prefill and a few kilobytes in decode, so latency, not the ring's bandwidth, is what they pay. llama.cpp's own allreduce and the generic reduce are {(1 - A["internal"]["tg"] / A["base"]["tg"]) * 100 if A else 0:.0f}% slower in generation and {(1 - A["internal"]["pp"] / A["base"]["pp"]) * 100 if A else 0:.0f}% in prefill, whether or not peer access is switched on; HIP graphs are worth {(1 - A["no-graphs"]["tg"] / A["base"]["tg"]) * 100 if A else 0:.0f}% of single-stream generation; the legacy row split does not load this model. At eight sequences every RCCL variant lands within 1% of the others, so for a server at batch 8 the topology file is a tidy-up rather than a gain. The batch-8 ladder run again under the corrected topology (section 10) says the same at depth: prefill {topo_gain():+.1f}% on average from 2K to 128K, decode within the run-to-run spread, and every cell of the original ladder reproduced within 2.5%.</p>
    <p>The dies leave no other knob: their power cap is already the board maximum (200 W, no headroom in sysfs), the memory clock has one level under load, the CPU governor was already at performance, and the fans were at maximum throughout.</p>
  </div>
  <h3>Micro-batch size at depth</h3>
  <div class="prose"><p>The micro-batch is the number of prompt tokens a die processes per kernel launch; it sets the compute buffer (section 1) and the prefill efficiency. Eight sequences, f16 cache, at 8K and 32K of context.</p></div>
  {opt_B_section()}
  <div class="prose"><p>2048 is the setting: 512 costs 15% of prefill and 1024 costs 4–5%, while 4096 adds 1% at 8K and 0.5% at 32K for twice the compute buffer (section 1 shows what that buffer is worth at the memory edge). Generation does not depend on the micro-batch at all. The eight-sequence decode figures in this table sit below the ladder's 152 at 8K; four fresh runs of that cell (section 10) gave 133 to 142 whatever the cache allocation, so the 8K decode figure carries a 5–7% run-to-run spread and the ladder's value, taken in the same process right after the 2K cell, is at the top of it.</p></div>
  {f_serve_section()}
</section>

<section>
  <div class="sec-head"><div class="eyebrow">8 · Speculative decoding</div><h2>The model's own next-token head, two small drafts and an n-gram lookup</h2></div>
  <div class="prose"><p>Single stream on the tensor split, greedy, 400 tokens, two prompts: free generation of a Python module, and a rewrite of a 1300-token module with type hints (the copy-heavy case). The GGUF carries an unused next-token-prediction block that this build can use as a draft head (<code>--spec-type draft-mtp</code>); the drafts are Qwen3.5-0.8B and 2B at Q8_0 on one die; n-gram lookup needs no model.</p></div>
  {spec_section()}
  <div class="prose">
    <p>The next-token head is the one that pays. It is a single extra block evaluated inside the target's own forward pass, so a draft costs almost nothing to make; with three tokens drafted a step, {E_rate("mtp", "edit"):.0f}% of them are accepted on the code rewrite and {E_rate("mtp", "free"):.0f}% on free generation, and the stream runs at {E_tps("mtp", "edit"):.1f} and {E_tps("mtp", "free"):.1f} tokens/s against {E_tps("none", "edit"):.1f} without ({E_tps("mtp", "edit") / E_tps("none", "edit"):.2f}× and {E_tps("mtp", "free") / E_tps("none", "free"):.2f}×). Two drafted tokens accept more often and do a little better on free text ({E_tps("mtp-n2", "free"):.1f}), a little worse on the rewrite. Quality is unchanged by construction: every drafted token is verified against the target's own distribution and rejected if it differs.</p>
    <p>A separate draft model loses on this machine, at 0.8B and at 2B alike ({E_tps("draft-0.8b", "edit"):.1f} and {E_tps("draft-2b", "edit"):.1f} tokens/s on the rewrite, {E_rate("draft-0.8b", "edit"):.0f}% and {E_rate("draft-2b", "edit"):.0f}% accepted): section 9's cost model says a token costs a lone die about 5 ms in fixed per-layer overhead whatever the model's size, so six drafted tokens cost as much as the 27B's own step before the verification is paid for. The n-gram lookup finds nothing to draft in these prompts and changes nothing. For a single user the flag is <code>--spec-type draft-mtp</code>; what it does under load, with sampling, and at depth is measured below.</p>
  </div>
  <h3>Draft length, sampling, depth and concurrency</h3>
  {mtp_more_section()}
  <div class="prose">
    <p><b>Draft length.</b> One stream, greedy: on the code rewrite the head keeps paying up to six drafted tokens ({M1("mtp-n6", "edit"):.0f} tokens/s, {M1("mtp-n6", "edit") / M1("none", "edit"):.1f}× the {M1("none", "edit"):.0f} without it); on free prose it peaks at three or four ({M1("mtp-n3", "free"):.0f}, {M1("mtp-n3", "free") / M1("none", "free"):.1f}×). Eight drafted tokens collapse both ({M1("mtp-n8", "edit"):.0f} and {M1("mtp-n8", "free"):.0f}): the verify step is a batch of one plus the draft, and nine tokens leave the 8-wide matrix-vector kernel of the first report's staircase for a tile that costs twice as much. On this platform the draft must keep slots × (draft + 1) at or below eight.</p>
    <p><b>Sampling.</b> The gain does not depend on greedy decoding. With the model's own defaults (temperature 1.0, top-k 20, top-p 0.95) the rewrite runs at {M2("mtp-t1.0"):.0f} tokens/s with the head against {M2("none-t1.0"):.0f} without; at temperature 0.7 with top-p 0.95, {M2("mtp-t0.7"):.0f} against {M2("none-t0.7"):.0f}, with 97% of drafts accepted.</p>
    <p><b>Depth.</b> Summarising real text, the head is worth more the deeper the context: {M3("mtp-32k"):.0f} against {M3("none-32k"):.0f} tokens/s at 32K ({M3("mtp-32k") / max(M3("none-32k"), 1):.2f}×) and {M3("mtp-128k"):.0f} against {M3("none-128k"):.0f} at 128K ({M3("mtp-128k") / max(M3("none-128k"), 1):.2f}×). At depth a step is mostly the cache being read (section 3), and the verify batch reads it once for four tokens instead of once per token, so the head recovers exactly the cost that long context adds.</p>
    <p><b>Concurrency.</b> Eight slots, eight realistic prompts, 300 tokens each, one wave per level. With one client the head gives {M4("mtp-n3", 1)[0]:.0f} aggregate tokens/s against {M4("none", 1)[0]:.0f}; with two, {M4("mtp-n3", 2)[0]:.0f} against {M4("none", 2)[0]:.0f}; with four, drafting one token still helps ({M4("mtp-n1", 4)[0]:.0f} against {M4("none", 4)[0]:.0f}) while three drafted tokens make a 16-token verify batch and the gain is gone ({M4("mtp-n3", 4)[0]:.0f}). With eight clients every draft length loses ({M4("mtp-n1", 8)[0]:.0f} to {M4("mtp-n3", 8)[0]:.0f} against {M4("none", 8)[0]:.0f}): eight slots already fill the fast kernel with their own eight tokens, and any draft pushes the step onto the slower tiles while acceptance stays at 70–88%. So the head is the setting for one to four concurrent streams, drafting three tokens up to two clients and one token at four; a busy eight-slot server should run without it, or with slots × (draft + 1) held at eight by a scheduler that this build does not have. Sixteen slots with sixteen clients confirm the tile arithmetic from the other side: drafting one token there makes 32-token batches, the efficient full tile of the first report, and lands within 4% of no head ({M4("mtp-n1", 16)[0]:.0f} against {M4("none", 16)[0]:.0f}), while three drafted tokens make 64-token batches and lose 13%. Implemented with that constraint in mind, a draft that adapts its length to the number of active slots would keep the whole curve above the baseline.</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">9 · Scaling and a prediction</div><h2>A token on four dies costs {CP.A4:.1f} ms, plus {CP.C4 * 1000:.0f} µs per layer, plus {CP.G4 * 1000:.0f} µs per GiB of weights</h2></div>
  <div class="prose">
    <p>To say anything about a model this machine has not run, the decode cost has to be separated into its parts. <code>llama-bench</code> measured five Qwen3.5 / 3.8 sizes at Q8_0 (0.8B, 2B, 9B and 27B, 24 to 64 layers, 0.8 to 27 GiB) plus three other 27B quantisations, on one die and on four, and one mixture-of-experts model, Qwen3.5-35B-A3B at Q8_0 (40 layers, 256 experts, 8 routed and one shared per token, 34 GiB on disk, {CP.MOE["active_gib"]:.1f} GiB read per token). A three-term model, a fixed cost per token, a cost per layer and a cost per GiB of weights read, fits the four Q8_0 sizes on four dies within 1%:</p>
  </div>
  {scaling_table()}
  <div class="prose">
    <p>The per-layer term is the split's synchronisation (the RCCL allreduces every block needs) and the launch overhead of the linear-attention blocks; the per-GiB term is the weights streamed from HBM2, {0.25 / CP.G4 * 1024 / 1000 * 1000:.0f} GB/s effective per die since each reads a quarter. On one die the same data give {CP.G1:.2f} ms/GiB ({1024 / CP.G1 / 1000 * 1000:.0f} GB/s) and {CP.C1 * 1000:.0f} µs per layer with no fixed cost to speak of, which is the whole story of the split: four dies cut the streaming term by four and add a synchronisation cost per layer that the small models cannot hide. Q4_0 and the K-quants are the exceptions in the table: they read fewer bytes than the model credits them for, because their dequantisation is arithmetic Vega 20 does not do for free; Q4_K_M at 27B is {CP.DENSE["27B Q4_K_M"]["pts"][1] / CP.DENSE["27B Q8_0"]["pts"][1]:.2f}× Q8_0 on four dies for 0.57× the bytes.</p>
    <p>The mixture model is the second calibration. Its {CP.MOE["active_gib"]:.1f} GiB per token would cost {CP.MOE["t_dense"]:.1f} ms by the dense model; it measured {CP.MOE["t_meas"]:.1f} ms on four dies ({CP.MOE["tg4"]:.0f} tokens/s) and {1000 / CP.MOE["tg2"]:.1f} ms on two ({CP.MOE["tg2"]:.0f} tokens/s), so four dies are slower than two for it. The difference, {CP.MOE["penalty_per_layer"] * 1000:.0f} µs per layer, is the expert routing and nine small matrix-vector products per block, and it is what a sparse model pays on this platform. In prefill the mixture reached {CP.MOE["pp4"]:.0f} tokens/s on four dies, {CP.MOE["pp_tflops"]:.0f} TFLOP/s on its active weights against {CP.DENSE_PP_TFLOPS:.0f} for the dense 27B: each expert sees only a slice of the 2048-token micro-batch, so its matrix products run at a fraction of the dense kernel's efficiency. Batching suits it better than the dense model: {CP.MOE["batch"][8]["s_tg"]:.0f} tokens/s at eight sequences and {CP.MOE["batch"][32]["s_tg"]:.0f} at thirty-two, {CP.MOE["batch"][32]["s_tg"] / CP.MOE["batch"][1]["s_tg"]:.1f}× one stream against the 27B's 4.1×.</p>
  </div>
  <h3>Qwen3.8-Flash-Next on this machine</h3>
  <div class="prose">
    <p>Qwen3.8-Flash-Next is 177 billion parameters of which 51 billion are an n-gram embedding table, 48 blocks (12 full attention with 24 query heads over 2 key/value heads of 256, 36 linear attention), 512 experts of which 10 plus a shared one fire per token, a 248k vocabulary and a one-block next-token head. Per token that is {CP.FN["active_params"] / 1e9:.1f} billion active parameters and a KV cache of {CP.FN["kv_per_token"] / 1024:.0f} KiB, a third of the 27B's. Its GGUF architecture, <code>qwen4exp</code>, went into llama.cpp on 2026-08-27; the build on this machine (b10288, 2026-08-05) predates it, so the figures below are the model above applied to those shapes, with the assumptions stated. The user's 105 GB file is a 4.7-bit quantisation ({105e9 / 177e9:.2f} bytes per parameter); Unsloth's UD-Q4_K_XL is 111 GB and UD-IQ4_XS 93.7 GB.</p>
  </div>
  {predict_table()}
  <div class="prose">
    <p>Where the n-gram table lives decides what fits. llama.cpp keeps a model's token embedding in host memory and reads rows on demand; if the 51 billion n-gram parameters are handled the same way (30 GB of the 105 at this quantisation), the dies hold {CP.PRED[list(CP.PRED)[0]]["per_die_gb"]:.0f} GB of weights each and have room for {CP.PRED[list(CP.PRED)[0]]["kv_tokens"] / 1024 / 1024:.1f} million tokens of f16 cache, six slots at the full 256K. If the table must sit on the dies, they hold {CP.PRED[list(CP.PRED)[1]]["per_die_gb"]:.0f} GB each and the cache room shrinks to two slots at 256K, or eight at 64K. The dies' 377 GiB of host RAM makes the first case the one to aim for.</p>
    <p>Speed follows from the calibrations: {CP.PRED[list(CP.PRED)[0]]["tg1"]:.0f} tokens/s for one stream ({1000 / CP.PRED[list(CP.PRED)[0]]["t_dense"]:.0f} before the expert routing cost, which is why it lands near the 27B's 46 despite a fifth of the bytes), {CP.PRED[list(CP.PRED)[0]]["tg1_256k"]:.0f} at 256K of context, about {CP.PRED[list(CP.PRED)[0]]["tg8"]:.0f} tokens/s for eight streams; prompts read at about {CP.PRED[list(CP.PRED)[0]]["pp_rate"]:.0f} tokens/s when short and {CP.PRED[list(CP.PRED)[0]]["pp_avg_256k"]:.0f} averaged over a 256K prompt, {CP.PRED[list(CP.PRED)[0]]["pp_256k_min"]:.0f} minutes for the whole of one. The uncertain parts are named: the expert term is scaled from nine experts of 512 to eleven of 640 in proportion; the K-quant penalty is the 27B's ({CP.PRED[list(CP.PRED)[0]]["kq"]:.2f}×) applied to the byte term; the sparse attention is costed as dense attention over the 12 blocks, which is an upper bound if llama.cpp implements the sparsity. Two things would move the answer: the MTP head of section 8, which this model also carries and which nearly doubled the 27B's single stream; and a two-die placement, which the mixture calibrator preferred, which the 105 GB file rules out but a 75 GB residue (table on the host) does not on three dies.</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">10 · Follow-up runs</div><h2>The ceiling cells, a single stream to 256K, and the ladder once more</h2></div>
  {extras_section()}
</section>

<section>
  <div class="sec-head"><div class="eyebrow">11 · Conditions</div><h2>Seven and a half hours at the power cap, 87 °C at the hottest</h2></div>
  <div class="prose"><p>A sampler read every die's clock, junction temperature, power and the four chassis fans every five seconds. The fans held their maximum (1220 and 2520 rpm) throughout; the dies sat at 1730 MHz and their 200 W cap in prefill, and the samples with a die at 1000 MHz are the idle moments between stages and the perplexity runs that fell to the CPU. Nothing throttled and the 1000 MHz clamp of earlier sessions did not appear. The evening's follow-up runs (sections 7–10, another eight hours) kept the same conditions; their hottest moment was the eight-slot 192K cell at 89 °C junction, still below the point where these dies throttle.</p></div>
  {therm_table()}
</section>

<section>
  <div class="sec-head"><div class="eyebrow">12 · Reproduce</div><h2>One driver script, its client, and the follow-up runs</h2></div>
<pre><code># context ladders: 8 sequences to 128K, 4 sequences to 256K (add -ctk q8_0 -ctv q8_0 for the q8_0 rows)
llama-batched-bench -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -b 2048 -ub 2048 -ntg 128 -npl 8 -npp 2048,8192,32768,65536,131072 -c 1050624
llama-batched-bench -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -b 2048 -ub 2048 -ntg 128 -npl 4 -npp 2048,8192,32768,65536,131072,262016 -c 1048576
# server waves: 8 slots x 128K, then 4 slots x 256K; one wave of NP simultaneous long prompts per size
llama-server -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -np 8 -cb -c 1048576 -b 2048 -ub 2048 --cache-ram 0 -to 7200 --host 127.0.0.1 --port 8089
python3 /root/rocm-tests/bench/server-ctx-bench.py http://127.0.0.1:8089 TAG --prompt-tokens 130816 --conc 8 --reqs 8 --gen 256
# the next-token head for a single user (section 8): add to any llama-server line; draft 3 for 1-2 clients, 1 at 4, off at 8
llama-server ... --spec-type draft-mtp --spec-draft-n-max 3
# corrected RCCL topology (section 7): +1-4%
NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 llama-server ...
# one shared 256K pool for 8 slots (section 5): --kv-unified -np 8 -c 262144
# cache-type perplexity (tensor split; -sm layer for the 4-bit-value rows)
llama-perplexity -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -ctk q8_0 -ctv q8_0 -f wiki.test.raw -c 16384 --chunks 6 -b 2048 -ub 2048</code></pre>
  <div class="files">
    <span>rocm-tests/bench/ctx-sweep.sh · the 7.5 h run (ladders, dp4, perplexity, server waves)</span><span>qwen38-27b-q8_0-ctx-{{f16,q8}}-b{{4,8}}.md · the ladders</span>
    <span>qwen38-27b-q8_0-ctx-server-{{b8,b4}}-f16.md/.jsonl · server waves</span><span>qwen38-27b-q8_0-ctx-dp4-die*.md · four instances</span>
    <span>qwen38-27b-q8_0-ctx-ppl-*.md · perplexity by cache type</span><span>qwen38-27b-q8_0-ctx-clocks.txt · clocks, temperatures, power, fans, 5 s</span>
    <span>opt-sweep.sh · knobs, scaling, mixture calibrator, speculative (qwen38-27b-q8_0-opt-*)</span><span>extras-sweep.sh, extras2.sh, extras3.sh, extras4.sh · single stream, ceilings, pool, MTP matrix, mixed waves (-ctx2-*, -x2-*, -x3-*, -x4-*)</span>
    <span>server-ctx-bench.py, mixed-client.py, mtp-conc-client.py, spec-client.py · the clients</span><span>ctx_data.py, ctx_predict.py, gen-ctx-report.py · loader, cost model, this page</span>
  </div>
  <div class="prose"><p>Related: <a href="https://claude.ai/code/artifact/e248ff49-e4e3-4a21-967e-927f21f5af71">Qwen3.8-27B at 8 bits on four Vega 20 dies</a> (scaling, batching and concurrency at short context) and <a href="https://claude.ai/code/artifact/f811df1d-b2bb-4d95-9157-4726d0519194">Four Vega 20 dies, one XGMI ring</a> (the fabric the split synchronises over).</p></div>
</section>

<footer>
  <p>Measured 2026-09-06 on macpro2019-01 with llama.cpp b10288 on ROCm 7.14. Tokens per second as llama.cpp reports them; server figures from the server's own per-request timings; memory as llama.cpp reported it at load; bandwidths in 10⁹ bytes per second.</p>
</footer>

</main>

<script>
(function () {{
  var charts = document.querySelectorAll('.chart');
  charts.forEach(function (chart) {{
    var tip = chart.querySelector('.tip'); if (!tip) return;
    function place(x, y) {{
      var r = chart.getBoundingClientRect();
      var px = x - r.left + 12, py = y - r.top - 8;
      if (px + tip.offsetWidth > r.width - 8) px = Math.max(4, x - r.left - tip.offsetWidth - 12);
      tip.style.left = px + 'px'; tip.style.top = Math.max(4, py - tip.offsetHeight) + 'px';
      tip.setAttribute('data-show', '1');
    }}
    function hide() {{ tip.removeAttribute('data-show'); }}
    chart.querySelectorAll('.bar').forEach(function (bar) {{
      function show(evt) {{
        tip.textContent = bar.getAttribute('data-tip');
        var m = bar.querySelector('.mark'); var b = m ? m.getBoundingClientRect() : bar.getBoundingClientRect();
        place(evt && evt.clientX ? evt.clientX : b.right, b.top);
      }}
      bar.addEventListener('mouseenter', show); bar.addEventListener('mousemove', show); bar.addEventListener('mouseleave', hide);
      bar.addEventListener('focus', function () {{ show(null); }}); bar.addEventListener('blur', hide);
    }});
    var svg = chart.querySelector('svg[data-kind="lines"]'); if (!svg) return;
    var data = JSON.parse(chart.querySelector('script[data-for="' + svg.getAttribute('data-panel') + '"]').textContent);
    var xh = svg.querySelector('.xh'), vb = svg.viewBox.baseVal, idx = -1;
    function showIdx(i, clientX) {{
      idx = i; var x = data.x[i];
      xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.removeAttribute('hidden');
      while (tip.firstChild) tip.removeChild(tip.firstChild);
      var h = document.createElement('div'); h.className = 'h'; h.textContent = data.labels[i]; tip.appendChild(h);
      data.series.forEach(function (s) {{
        if (s.values[i] == null) return;
        var row = document.createElement('div'); row.className = 'row';
        var k = document.createElement('i'); k.style.background = 'var(--' + s.var + ')';
        var v = document.createElement('b'); v.textContent = Number(s.values[i]).toFixed(s.values[i] < 100 ? 1 : 0) + (s.unit || '');
        var n = document.createElement('span'); n.textContent = s.name;
        row.appendChild(k); row.appendChild(v); row.appendChild(n); tip.appendChild(row);
      }});
      var r = svg.getBoundingClientRect(); var sx = r.left + (x / vb.width) * r.width;
      place(clientX || sx, r.top + 24);
    }}
    function hideAll() {{ hide(); xh.setAttribute('hidden', ''); idx = -1; }}
    svg.addEventListener('pointermove', function (e) {{
      var r = svg.getBoundingClientRect(); var vx = (e.clientX - r.left) / r.width * vb.width;
      var best = 0, bd = Infinity; data.x.forEach(function (x, i) {{ var d = Math.abs(x - vx); if (d < bd) {{ bd = d; best = i; }} }});
      showIdx(best, e.clientX);
    }});
    svg.addEventListener('pointerleave', hideAll);
    svg.addEventListener('focus', function () {{ showIdx(idx < 0 ? data.x.length - 1 : idx); }});
    svg.addEventListener('blur', hideAll);
    svg.addEventListener('keydown', function (e) {{
      if (e.key === 'ArrowRight') {{ showIdx(Math.min(data.x.length - 1, (idx < 0 ? 0 : idx) + 1)); e.preventDefault(); }}
      else if (e.key === 'ArrowLeft') {{ showIdx(Math.max(0, (idx < 0 ? data.x.length - 1 : idx) - 1)); e.preventDefault(); }}
      else if (e.key === 'Escape') hideAll();
    }});
  }});
}})();
</script>
'''
open(OUT, 'w', encoding='utf-8').write(html)
print(f'wrote {OUT}: {len(html)} bytes; model pp c0 {PP["c0"]:.3f} k {PP["k"]:.5f}; tg slopes {[(k, round(v["kseq"]*1000, 1)) for k, v in MODEL.items()]}; att {ATT_TFLOPS:.1f} TFLOP/s; kv {KV_GBS:.0f} GB/s; mem rows {[(m["label"], m["kv"], round(m["total"]/1024, 1), m["fits"]) for m in MEMROWS]}')
