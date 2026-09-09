#!/usr/bin/env python3
import json
OUT = '/root/qwen38-27b-q8_0-gfx906.html'

# ---------------- data (2026-09-04 rerun unless noted) ----------------
LB = {1: dict(pp=234.65, tg=20.25, pp16=200.47, tg16=18.83, pp32=174.71, tg32=17.69),
      2: dict(pp=439.51, tg=31.23, pp16=377.46, tg16=29.71, pp32=331.81, tg32=28.51),
      4: dict(pp=829.69, tg=45.69, pp16=714.32, tg16=43.99, pp32=627.28, tg32=42.88)}
Q4 = {1: dict(pp=201.74, tg=24.71, pp32=155.49, tg32=20.95), 2: dict(pp=383.84, tg=35.52, pp32=299.58, tg32=32.11), 4: dict(pp=732.92, tg=47.33, pp32=569.79, tg32=44.32)}
BATCH = {512: [(1, 664.13, 44.19, 174.50), (2, 782.79, 71.38, 261.52), (4, 827.20, 121.32, 382.32), (8, 832.50, 160.87, 453.68), (16, 834.62, 152.13, 439.91), (32, 835.75, 181.50, 485.64)],
         2048: [(1, 819.10, 43.06, 397.58), (2, 825.07, 72.37, 511.89), (4, 828.68, 117.44, 611.01), (8, 830.03, 159.57, 665.54), (16, 830.16, 150.36, 655.76), (32, 830.31, 179.22, 684.12)]}
CONC = [1, 2, 4, 8, 16, 32]
# conc: (reqs, wall, agg_gen, total, rpm, per_req_gen, ttft, per_req_wall)
SRV = {'tp4':    [(8, 62.7, 32.7, 199, 7.7, 44.6, 1.84, 7.8), (8, 45.0, 45.5, 277, 10.7, 34.4, 3.66, 11.2), (8, 34.3, 59.8, 363, 14.0, 26.5, 5.81, 17.1), (16, 63.6, 64.4, 391, 15.1, 12.9, 7.00, 31.6), (32, 128.1, 64.0, 389, 15.0, 5.7, 7.45, 63.7), (64, 247.7, 66.1, 402, 15.5, 2.8, 8.31, 123.2)],
       'layer4': [(8, 151.0, 13.6, 82, 3.2, 20.0, 5.88, 18.9), (8, 98.6, 20.8, 126, 4.9, 18.0, 10.20, 24.6), (8, 77.6, 26.4, 160, 6.2, 11.9, 13.74, 38.8), (16, 158.2, 25.9, 157, 6.1, 4.9, 16.80, 79.0), (32, 299.4, 27.4, 166, 6.4, 2.5, 19.05, 149.4), (64, 582.2, 28.1, 171, 6.6, 1.2, 19.69, 289.9)],
       'tp2':    [(8, 93.9, 21.8, 133, 5.1, 30.9, 3.27, 11.7), (8, 65.5, 31.3, 190, 7.3, 26.8, 6.59, 16.4), (8, 53.2, 38.5, 234, 9.0, 18.4, 9.98, 26.6), (16, 101.2, 40.5, 246, 9.5, 8.2, 11.20, 50.5), (32, 198.0, 41.4, 252, 9.7, 3.8, 13.93, 98.7), (64, 385.6, 42.5, 258, 10.0, 1.8, 13.44, 192.0)]}
MODES = [('tp4', 'tensor split, 4 dies', 'series'), ('layer4', 'layer split, 4 dies', 'series-2'), ('tp2', 'tensor split, 2 dies (one module)', 'series-3')]
# fine serving sweeps, every concurrency 1..32 (server-fine.sh); fall back to the six-level run per mode if absent
def load_srv(mode):
    d = {}; done = False
    try:
        for l in open(f'/root/rocm-tests/bench/qwen38-27b-q8_0-serverfine-server-{mode}.md'):
            c = [x.strip() for x in l.strip().strip('|').split('|')]
            if len(c) >= 9 and c[0].isdigit(): d[int(c[0])] = tuple(float(x) for x in c[1:9])
            if l.startswith('# client exit=0'): done = True
    except OSError: pass
    return d if done and 32 in d else {}   # only a finished sweep replaces the six-level run
SRVF = {}
for mode, _, _ in MODES:
    f = load_srv(mode)
    SRVF[mode] = f if len(f) >= 6 else {c: tuple(float(v) for v in row) for c, row in zip(CONC, SRV[mode])}
FINE_SRV = all(len(SRVF[m]) > 6 for m, _, _ in MODES)

THERM = [('llama-bench sweep', '05:26–05:39', 165, 6, '81 / 81 / 69 / 67', '226 / 244 / 202 / 210', 90),
         ('batched bench, 4 dies', '05:39–05:45', 64, 5, '78 / 80 / 75 / 73', '235 / 232 / 213 / 206', 169),
         ('server, tensor split 4', '05:45–05:55', 121, 26, '76 / 78 / 71 / 70', '198 / 226 / 233 / 214', 150),
         ('server, layer split 4', '05:55–06:18', 277, 71, '68 / 67 / 64 / 62', '237 / 250 / 284 / 249', 112),
         ('server, tensor split 2', '06:18–06:34', 184, 7, '81 / 82 / 40 / 40', '211 / 219 / 40 / 42', 96)]

def f1(v): return f'{v:.1f}'
def f2(v): return f'{v:.2f}'

# ---------------- svg helpers ----------------
def rounded_col(x, y, w, ybase, fill, cls='mark'):
    if ybase - y < 4: return f'<rect class="{cls}" fill="{fill}" x="{x:.1f}" y="{y:.1f}" width="{w}" height="{ybase - y:.1f}"/>'
    return f'<path class="{cls}" fill="{fill}" d="M{x:.1f} {ybase:.1f} V{y + 4:.1f} a4 4 0 0 1 4 -4 H{x + w - 4:.1f} a4 4 0 0 1 4 4 V{ybase:.1f} Z"/>'

def grid_and_axis(L, R, T, Bt, ymax, ticks, fmt=lambda v: f'{v:g}'):
    g = ['<g class="grid">']
    for tv in ticks:
        y = Bt - (tv / ymax) * (Bt - T)
        g.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}"/>')
    g.append('</g>')
    g.append(f'<line class="axis" x1="{L}" y1="{Bt}" x2="{R}" y2="{Bt}"/>')
    for tv in ticks:
        y = Bt - (tv / ymax) * (Bt - T)
        g.append(f'<text class="tick" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(tv)}</text>')
    return '\n'.join(g)

def scaling_panel(key, title, ymax, ticks, unit_label, aria):
    W, H, L, R, T, Bt = 360, 262, 52, 346, 30, 214
    cats = [1, 2, 4]; base = LB[1][key]
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" data-kind="cols">', grid_and_axis(L, R, T, Bt, ymax, ticks)]
    bw, gap = 22, 2
    for i, d in enumerate(cats):
        cx = L + (i + 0.5) * (R - L) / 3
        meas = LB[d][key]; ideal = base * d
        x1 = cx - bw - gap / 2; x2 = cx + gap / 2
        y1 = Bt - (meas / ymax) * (Bt - T); y2 = Bt - (ideal / ymax) * (Bt - T)
        ratio = meas / base
        tip_m = f'{d} die{"s" if d > 1 else ""}: {meas:.1f} {unit_label}, {ratio:.2f}× one die'
        tip_i = f'Linear scaling from one die would be {ideal:.0f} {unit_label}'
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip_m}"><rect class="hit" x="{x1 - 6:.1f}" y="{T}" width="{bw + 12}" height="{Bt - T + 30}"/>{rounded_col(x1, y1, bw, Bt, "var(--series)")}<text class="val" x="{x1 + bw / 2:.1f}" y="{y1 - 7:.1f}" text-anchor="middle">{meas:.0f}</text></g>')
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip_i}"><rect class="hit" x="{x2 - 6:.1f}" y="{T}" width="{bw + 12}" height="{Bt - T + 30}"/>{rounded_col(x2, y2, bw, Bt, "var(--series-dim)")}</g>')
        svg.append(f'<text class="tick" x="{cx:.1f}" y="{Bt + 18}" text-anchor="middle">{d} die{"s" if d > 1 else ""}</text>')
        svg.append(f'<text class="ratio" x="{cx:.1f}" y="{Bt + 36}" text-anchor="middle">×{ratio:.2f}</text>')
    svg.append(f'<text class="tick" x="{L}" y="13">{title}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)

def batch_panel():
    W, H, L, R, T, Bt = 760, 296, 52, 740, 30, 250
    ymax, ticks = 200, [0, 50, 100, 150, 200]
    rows = BATCH[512]
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Column chart of aggregate generation throughput versus batch size on four dies: 44 tokens per second at batch 1 rising to 181 at batch 32, with per-sequence rate falling from 44 to 5.7." data-kind="cols">', grid_and_axis(L, R, T, Bt, ymax, ticks)]
    bw = 24
    for i, (b, spp, stg, s) in enumerate(rows):
        cx = L + (i + 0.5) * (R - L) / 6; x = cx - bw / 2
        y = Bt - (stg / ymax) * (Bt - T)
        per = stg / b
        tip = f'Batch {b}: {stg:.1f} tokens/s in total, {per:.1f} per sequence; prompt phase {spp:.0f} tokens/s; with 2048-token prompts {BATCH[2048][i][2]:.1f} total'
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{cx - 40:.1f}" y="{T}" width="80" height="{Bt - T + 30}"/>{rounded_col(x, y, bw, Bt, "var(--series)")}<text class="val" x="{cx:.1f}" y="{y - 7:.1f}" text-anchor="middle">{stg:.0f}</text></g>')
        svg.append(f'<text class="tick" x="{cx:.1f}" y="{Bt + 18}" text-anchor="middle">batch {b}</text>')
        svg.append(f'<text class="ratio" x="{cx:.1f}" y="{Bt + 36}" text-anchor="middle">{per:.1f} each</text>')
    svg.append(f'<text class="tick" x="{L}" y="13">generated tokens per second, all sequences together (512-token prompts)</text>')
    svg.append('</svg>')
    return '\n'.join(svg)

def line_panel(pid, metric_idx, ymax, ticks, fmt, ylabel, aria, W=560, R=456):
    H, L, T, Bt = 296, 52, 26, 250
    xpos = lambda c: L + (c - 1) / 31 * (R - L)
    union = sorted(set().union(*[set(SRVF[m]) for m, _, _ in MODES]))
    dense = len(union) > 8
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" data-kind="lines" data-panel="{pid}" tabindex="0">', grid_and_axis(L, R, T, Bt, ymax, ticks, fmt)]
    for c in ([1, 4, 8, 12, 16, 20, 24, 28, 32] if dense else union):
        svg.append(f'<text class="tick" x="{xpos(c):.1f}" y="{Bt + 18}" text-anchor="middle">{c}</text>')
    svg.append(f'<text class="tick" x="{(L + R) / 2:.1f}" y="{Bt + 36}" text-anchor="middle">concurrent requests</text>')
    svg.append(f'<line class="xh" x1="{xpos(1):.1f}" x2="{xpos(1):.1f}" y1="{T}" y2="{Bt}" hidden/>')
    series_json = {'x': [xpos(c) for c in union], 'conc': union, 'series': []}
    for mode, label, var in MODES:
        pts = [(xpos(c), Bt - (SRVF[mode][c][metric_idx] / ymax) * (Bt - T)) for c in sorted(SRVF[mode])]
        d = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in pts)
        svg.append(f'<path class="line" stroke="var(--{var})" d="{d}"/>')
        marks = pts if len(pts) <= 8 else [pts[0], pts[-1]]
        for x, y in marks: svg.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--{var})"/>')
        ex, ey = pts[-1]; last_c = max(SRVF[mode])
        svg.append(f'<text class="val" x="{ex + 10:.1f}" y="{ey + 4:.1f}">{mode} {fmt(SRVF[mode][last_c][metric_idx])}</text>')
        series_json['series'].append({'name': label, 'short': mode, 'var': var, 'values': [SRVF[mode][c][metric_idx] if c in SRVF[mode] else None for c in union]})
    svg.append(f'<text class="tick" x="{L}" y="13">{ylabel}</text>')
    svg.append('</svg>')
    svg.append(f'<script type="application/json" data-for="{pid}">{json.dumps(series_json)}</script>')
    return '\n'.join(svg)

# ---------------- fine batch sweep (2026-09-04 08:4x) ----------------
import re as _re
def load_fine(*names):
    d = {}
    for name in names:
        try:
            for l in open(f'/root/rocm-tests/bench/{name}'):
                c = [x.strip() for x in l.strip().strip('|').split('|')]
                if len(c) >= 10 and c[0].isdigit(): d[int(c[2])] = (float(c[5]), float(c[7]), float(c[9]))
        except OSError: pass
    return d
# PP 512: one run over every size 1..32; PP 2048: the even sizes from the first fine pass plus the odd sizes from the second
FINE = {512: load_fine('qwen38-27b-q8_0-batched-fine2-pp512-all.md'), 2048: load_fine('qwen38-27b-q8_0-batched-fine-pp2048.md', 'qwen38-27b-q8_0-batched-fine2-pp2048-odd.md')}
FB = sorted(FINE[512]) if FINE[512] else [b for b, *_ in BATCH[512]]
def step_ms(b, stg): return b / stg * 1000

def fine_cols_panel():
    W, H, L, R, T, Bt = 760, 296, 52, 740, 30, 250
    ymax, ticks = 200, [0, 50, 100, 150, 200]
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Column chart of aggregate generation throughput for every batch size from 1 to 32: it rises to about 162 tokens per second at batch 8, falls by a third at batch 9, rises to about 153 at 16, falls at 17, rises to about 163 at 24, falls at 25 and rises to about 181 at 32." data-kind="cols">', grid_and_axis(L, R, T, Bt, ymax, ticks)]
    n = len(FB); step = (R - L) / n; bw = 22 if n <= 20 else 14
    LABELLED = {1, 8, 9, 16, 17, 24, 25, 32} if n > 20 else set(FB)
    TICKS = {1, 4, 8, 12, 16, 20, 24, 28, 32} if n > 20 else set(FB)
    for i, b in enumerate(FB):
        spp, stg, stot = FINE[512][b]; cx = L + (i + 0.5) * step; y = Bt - (stg / ymax) * (Bt - T)
        tip = f'Batch {b}: {stg:.1f} tokens/s in total, {stg / b:.1f} per sequence, {step_ms(b, stg):.0f} ms per decode step'
        lab = f'<text class="val" x="{cx:.1f}" y="{y - 7:.1f}" text-anchor="middle">{stg:.0f}</text>' if b in LABELLED else ''
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{cx - step / 2:.1f}" y="{T}" width="{step:.1f}" height="{Bt - T + 30}"/>{rounded_col(cx - bw / 2, y, bw, Bt, "var(--series)")}{lab}</g>')
        if b in TICKS: svg.append(f'<text class="tick" x="{cx:.1f}" y="{Bt + 18}" text-anchor="middle">{b}</text>')
    svg.append(f'<text class="tick" x="{(L + R) / 2:.1f}" y="{Bt + 36}" text-anchor="middle">batch size (sequences decoding together)</text>')
    svg.append(f'<text class="tick" x="{L}" y="13">generated tokens per second, all sequences together (512-token prompts)</text>')
    svg.append('</svg>')
    return '\n'.join(svg)

def staircase_panel():
    W, H, L, R, T, Bt = 760, 296, 52, 740, 30, 250
    ymax, ticks = 200, [0, 50, 100, 150, 200]
    n = len(FB); step = (R - L) / n
    xs = {b: L + (i + 0.5) * step for i, b in enumerate(FB)}
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Step chart of decode step time versus batch size for every size from 1 to 32: about 22 to 49 milliseconds for batches 1 to 8 on the mat-vec kernel, roughly 95 to 105 for 9 to 16, 137 to 147 for 17 to 24, 166 to 176 for 25 to 32; each region is one MMQ tile width." data-kind="cols">', grid_and_axis(L, R, T, Bt, ymax, ticks)]
    # kernel regions: boundaries between 8|10, 16|18, 24|26
    nxt = lambda b: b + 1 if (b + 1) in xs else (b + 2 if (b + 2) in xs else b + 1)
    regions = [(1, 8, 'mat-vec kernel, up to 8 tokens'), (nxt(8), 16, 'MMQ, 16-wide tile'), (nxt(16), 24, 'MMQ, 24-wide tile'), (nxt(24), 32, 'MMQ, 32-wide tile')]
    TICKS = {1, 4, 8, 12, 16, 20, 24, 28, 32} if n > 20 else set(FB)
    for lo, hi, label in regions:
        if lo not in xs or hi not in xs: continue
        x0 = xs[lo] - step / 2; x1 = xs[hi] + step / 2
        if lo != 1: svg.append(f'<line class="region" x1="{x0:.1f}" x2="{x0:.1f}" y1="{T - 4}" y2="{Bt}"/>')
        svg.append(f'<text class="region-l" x="{(x0 + x1) / 2:.1f}" y="{T - 8}" text-anchor="middle">{label}</text>')
    pts = [(xs[b], Bt - (step_ms(b, FINE[512][b][1]) / ymax) * (Bt - T)) for b in FB]
    svg.append('<path class="line" stroke="var(--series)" d="M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in pts) + '"/>')
    for b, (x, y) in zip(FB, pts):
        ms = step_ms(b, FINE[512][b][1])
        tip = f'Batch {b}: {ms:.0f} ms per decode step, {FINE[512][b][1]:.1f} tokens/s in total'
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{x - step / 2:.1f}" y="{T}" width="{step:.1f}" height="{Bt - T + 30}"/><circle class="pt mark" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--series)"/></g>')
        if b in (1, 8, nxt(8), 16, nxt(16), 24, nxt(24), 32): svg.append(f'<text class="val" x="{x:.1f}" y="{y + (16 if b in (nxt(8), nxt(16), nxt(24)) else -9):.1f}" text-anchor="middle">{ms:.0f}</text>')
        if b in TICKS: svg.append(f'<text class="tick" x="{x:.1f}" y="{Bt + 18}" text-anchor="middle">{b}</text>')
    svg.append(f'<text class="tick" x="{(L + R) / 2:.1f}" y="{Bt + 36}" text-anchor="middle">batch size (sequences decoding together)</text>')
    svg.append(f'<text class="tick" x="{L}" y="13">milliseconds per decode step (all sequences advance one token)</text>')
    svg.append('</svg>')
    return '\n'.join(svg)

def mmvq16_addendum():
    """2026-09-07 item 9: the 16-column MMVQ patch (run-through report s.9)"""
    # (batch, stock, 16-column patch, Q8_0 rewrite 'fh') on four dies at 2048-token prompts; one die at 512-token prompts
    tp4 = [(1, 45.6, 45.8, 45.5), (4, 116.8, 117.4, 118.1), (8, 160.9, 160.6, 174.1), (9, 96.5, 162.6, None), (12, 122.3, 176.6, 197.7), (16, 152.3, 171.3, 207.2)]
    die = [(1, 19.5, 19.4, 19.5), (4, 48.3, 48.2, 55.0), (8, 52.7, 52.7, 72.2), (12, 54.7, 51.7, None), (16, 63.0, 46.6, None)]
    r = ['<div class="tablewrap"><table><thead><tr><th class="num">batch</th><th class="num">four dies, stock</th><th class="num">16-column patch</th><th class="num">Q8_0 rewrite</th><th class="num">change</th><th class="num">one die, stock</th><th class="num">16-column patch</th><th class="num">Q8_0 rewrite</th><th class="num">change</th></tr></thead><tbody>']
    dd = {b: (a, c, e) for b, a, c, e in die}
    f = lambda v: f'{v:.1f}' if v is not None else '—'
    for b, a, c, e in tp4:
        best = e if e is not None else c
        ch = f'{100 * (best / a - 1):+.0f}%' if a and best else ''
        d1, d2, d3 = dd.get(b, (None, None, None))
        dbest = d3 if d3 is not None else d2
        dch = f'{100 * (dbest / d1 - 1):+.0f}%' if d1 and dbest else ''
        r.append(f'<tr><td class="num">{b}</td><td class="num">{f(a)}</td><td class="num">{f(c)}</td><td class="num hi">{f(e)}</td><td class="num">{ch}</td><td class="num">{f(d1)}</td><td class="num">{f(d2)}</td><td class="num hi">{f(d3)}</td><td class="num">{dch}</td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

def fine_table():
    r = ['<div class="tablewrap"><table><thead><tr><th class="num">batch</th><th class="num">prompt phase t/s</th><th class="num">generation t/s, all sequences</th><th class="num">per sequence</th><th class="num">ms per decode step</th><th class="num">generation t/s with 2048-token prompts</th><th class="num">per sequence</th></tr></thead><tbody>']
    for b in FB:
        spp, stg, _ = FINE[512][b]; f2k = FINE[2048].get(b)
        r.append(f'<tr><td class="num">{b}</td><td class="num">{spp:.0f}</td><td class="num hi">{stg:.1f}</td><td class="num">{stg / b:.1f}</td><td class="num">{step_ms(b, stg):.0f}</td><td class="num">{f2k[1]:.1f}</td><td class="num">{f2k[1] / b:.1f}</td></tr>' if f2k else f'<tr><td class="num">{b}</td><td class="num">{spp:.0f}</td><td class="num hi">{stg:.1f}</td><td class="num">{stg / b:.1f}</td><td class="num">{step_ms(b, stg):.0f}</td><td class="num note">pending</td><td class="num note"></td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

# ---------------- tables ----------------
def lb_table():
    r = ['<div class="tablewrap"><table><thead><tr><th>dies (tensor split)</th><th class="num">pp2048</th><th class="num">tg256</th><th class="num">pp2048 at 16k depth</th><th class="num">tg256 at 16k</th><th class="num">pp2048 at 32k depth</th><th class="num">tg256 at 32k</th><th class="num">pp / tg kept at 32k</th></tr></thead><tbody>']
    for d in (1, 2, 4):
        v = LB[d]
        r.append(f'<tr><td>{d}</td><td class="num">{v["pp"]:.1f}</td><td class="num hi">{v["tg"]:.2f}</td><td class="num">{v["pp16"]:.1f}</td><td class="num">{v["tg16"]:.2f}</td><td class="num">{v["pp32"]:.1f}</td><td class="num">{v["tg32"]:.2f}</td><td class="num note">{100 * v["pp32"] / v["pp"]:.0f}% / {100 * v["tg32"] / v["tg"]:.0f}%</td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

def q4_table():
    r = ['<div class="tablewrap"><table><thead><tr><th>dies</th><th class="num">Q8_0 pp2048</th><th class="num">Q4_K_M pp2048</th><th class="num">Q8_0 tg256</th><th class="num">Q4_K_M tg256</th><th class="num">Q8_0 tg256 at 32k</th><th class="num">Q4_K_M tg256 at 32k</th></tr></thead><tbody>']
    for d in (1, 2, 4):
        a, b = LB[d], Q4[d]
        r.append(f'<tr><td>{d}</td><td class="num">{a["pp"]:.1f}</td><td class="num">{b["pp"]:.1f} <span class="note">({100 * (b["pp"] / a["pp"] - 1):+.0f}%)</span></td><td class="num">{a["tg"]:.2f}</td><td class="num">{b["tg"]:.2f} <span class="note">({100 * (b["tg"] / a["tg"] - 1):+.0f}%)</span></td><td class="num">{a["tg32"]:.2f}</td><td class="num">{b["tg32"]:.2f} <span class="note">({100 * (b["tg32"] / a["tg32"] - 1):+.0f}%)</span></td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

def batch_table():
    r = ['<div class="tablewrap"><table><thead><tr><th class="num">batch</th><th class="num">prompt phase t/s (512-token prompts)</th><th class="num">generation t/s, all sequences</th><th class="num">per sequence</th><th class="num">prompt phase t/s (2048-token prompts)</th><th class="num">generation t/s, all sequences</th><th class="num">per sequence</th></tr></thead><tbody>']
    for i in range(6):
        b, p5, t5, _ = BATCH[512][i]; _, p20, t20, _ = BATCH[2048][i]
        r.append(f'<tr><td class="num">{b}</td><td class="num">{p5:.0f}</td><td class="num hi">{t5:.1f}</td><td class="num">{t5 / b:.1f}</td><td class="num">{p20:.0f}</td><td class="num">{t20:.1f}</td><td class="num">{t20 / b:.1f}</td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

def srv_table():
    if not FINE_SRV:
        r = ['<div class="tablewrap"><table><thead><tr><th>mode</th><th class="num">concurrency</th><th class="num">requests</th><th class="num">generation t/s, aggregate</th><th class="num">tokens/s incl. prompts</th><th class="num">requests / min</th><th class="num">generation t/s per request</th><th class="num">time to first token, s</th><th class="num">per-request wall, s</th></tr></thead><tbody>']
        for mode, label, var in MODES:
            for i, c in enumerate(sorted(SRVF[mode])):
                reqs, wall, agg, tot, rpm, per, ttft, pw = SRVF[mode][c]
                first = f'<span class="key" style="background:var(--{var})"></span>{label}' if i == 0 else ''
                r.append(f'<tr><td>{first}</td><td class="num">{c}</td><td class="num">{reqs:.0f}</td><td class="num hi">{agg:.1f}</td><td class="num">{tot:.0f}</td><td class="num">{rpm:.1f}</td><td class="num">{per:.1f}</td><td class="num">{ttft:.2f}</td><td class="num">{pw:.1f}</td></tr>')
        r.append('</tbody></table></div>')
        return '\n'.join(r)
    heads = ''.join(f'<th class="num"><span class="key" style="background:var(--{var})"></span>{mode}</th>' for mode, _, var in MODES)
    r = ['<div class="tablewrap"><table><thead><tr><th class="num" rowspan="2">concurrency</th><th class="num" rowspan="2">requests</th><th colspan="3">generation t/s, aggregate</th><th colspan="3">time to first token, s</th><th colspan="3">generation t/s per request</th><th colspan="3">requests / min</th></tr><tr>' + heads * 4 + '</tr></thead><tbody>']
    for c in range(1, 33):
        cells = [f'<td class="num">{c}</td><td class="num">{max(8, 2 * c)}</td>']
        for idx, fmt in ((2, '{:.1f}'), (6, '{:.2f}'), (5, '{:.1f}'), (4, '{:.1f}')):
            for mode, _, _ in MODES:
                v = SRVF[mode].get(c)
                cells.append(f'<td class="num{" hi" if idx == 2 and mode == "tp4" else ""}">{fmt.format(v[idx]) if v else "–"}</td>')
        r.append('<tr>' + ''.join(cells) + '</tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

def therm_table():
    r = ['<div class="tablewrap"><table><thead><tr><th>stage</th><th>window (UTC)</th><th class="num">5-s samples</th><th class="num">samples with an idle die at 1000 MHz</th><th>max junction °C, dies 0b / 0e / 1b / 1e</th><th>max W, instantaneous</th><th class="num">mean W of dies at 1730 MHz</th></tr></thead><tbody>']
    for n, w, s, pct, t, p, m in THERM:
        r.append(f'<tr><td>{n}</td><td class="mono">{w}</td><td class="num">{s}</td><td class="num">{pct}%</td><td class="mono">{t}</td><td class="mono">{p}</td><td class="num">{m}</td></tr>')
    r.append('</tbody></table></div>')
    return '\n'.join(r)

# ---------------- derived numbers for the prose ----------------


def srv_cols_panel(mode, label, var):
    W, H, L, R, T, Bt = 760, 296, 52, 740, 30, 250
    ymax, ticks = 80, [0, 20, 40, 60, 80]
    data = SRVF[mode]; concs = sorted(data); pitch = (R - L) / 32; bw = 14
    xpos = lambda c: L + (c - 0.5) * pitch
    vals = {c: data[c][2] for c in concs}
    hi_c = max(concs, key=lambda c: vals[c]); lo_c = min((c for c in concs if c > 1), key=lambda c: vals[c])
    labelled = {1, 4, 8, 16, 32, hi_c, lo_c} & set(concs)
    aria = f'Column chart of aggregate generation throughput for {label} at each measured concurrency from 1 to 32: highest {vals[hi_c]:.0f} tokens per second at {hi_c} clients, lowest above one client {vals[lo_c]:.0f} at {lo_c}.'
    svg = [f'<figure><div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" data-kind="cols">', grid_and_axis(L, R, T, Bt, ymax, ticks)]
    for c in concs:
        reqs, wall, agg, tot, rpm, per, ttft, pw = data[c]; x = xpos(c); y = Bt - (agg / ymax) * (Bt - T)
        tip = f'{c} concurrent: {agg:.1f} tokens/s aggregate, {per:.1f} per request, first token after {ttft:.2f} s, {rpm:.1f} requests/min ({reqs:.0f} requests, {wall:.0f} s)'
        lab = f'<text class="val" x="{x:.1f}" y="{y - 7:.1f}" text-anchor="middle">{agg:.0f}</text>' if c in labelled else ''
        svg.append(f'<g class="bar" tabindex="0" data-tip="{tip}"><rect class="hit" x="{x - pitch / 2:.1f}" y="{T}" width="{pitch:.1f}" height="{Bt - T + 30}"/>{rounded_col(x - bw / 2, y, bw, Bt, f"var(--{var})")}{lab}</g>')
    for c in (1, 4, 8, 12, 16, 20, 24, 28, 32):
        svg.append(f'<text class="tick" x="{xpos(c):.1f}" y="{Bt + 18}" text-anchor="middle">{c}</text>')
    svg.append(f'<text class="tick" x="{(L + R) / 2:.1f}" y="{Bt + 36}" text-anchor="middle">concurrent requests</text>')
    svg.append(f'<text class="tick" x="{L}" y="13">{mode}: {label} · generated tokens/s, all requests together</text>')
    svg.append('</svg><div class="tip" role="status" aria-live="polite"></div></div></figure>')
    return '\n'.join(svg)
sparse = [mode for mode, _, _ in MODES if len(SRVF[mode]) <= 6]
SPARSE_NOTE = (' The ' + ' and '.join(sparse) + (' panels show' if len(sparse) > 1 else ' panel shows') + ' the six levels measured in the morning run; the sweep at every level is still running for ' + ('them' if len(sparse) > 1 else 'it') + '.') if sparse else ''


def rec_numbers():
    t = SRVF['tp4']; six = dict(zip(CONC, SRV['tp4']))
    a4, a8, a32 = t[4][2], t[8][2], t[32][2]
    gain = (a8 - a4) * 86400
    lo = min(a8, six[8][2]) - max(a4, six[4][2]); hi = max(a8, six[8][2]) - min(a4, six[4][2])
    return dict(a4=a4, a8=a8, a32=a32, p4=t[4][5], p8=t[8][5], p32=t[32][5], r8=t[8][4], r32=t[32][4], t4=t[4][6], t8=t[8][6], t8b=six[8][6],
                gain=gain, lo=lo * 86400, hi=hi * 86400, day8=a8 * 86400 / 1e6, day4=a4 * 86400 / 1e6, req8=t[8][4] * 1440)
REC = rec_numbers()

def jagged_para():
    t = SRVF['tp4']
    if len(t) <= 6: return ''
    peaks = {c: t[c][2] for c in (4, 8, 32) if c in t}
    trough_c = min((c for c in t if 5 <= c <= 31), key=lambda c: t[c][2]); trough = t[trough_c][2]
    six = dict(zip(CONC, SRV['tp4']))
    return (f"<p><b>Why the curve is jagged.</b> The client sends identical requests, 1300 tokens in and 256 out, so at each level they run in synchronised waves. Whenever one wave's prompts arrive while another is still generating, the 1300-token prefill is batched with the decode tokens, and for the two seconds it takes every running sequence advances by one token. How often that happens depends on how the waves drift apart, which differs from level to level and from run to run, and the batch-size stairs of section 3 add their own steps. The peaks, {' and '.join(f'{v:.0f} tokens/s at {c}' for c, v in peaks.items())} clients, are what the server delivers when the waves line up; the trough, {trough:.0f} at {trough_c}, is what it delivers when they do not. The same level can land on either side: sixteen clients gave {six[16][2]:.0f} tokens/s in the six-level run earlier in the day and {t[16][2]:.0f} in this one. Traffic with varied prompt and output lengths sits between the two.</p>")
JAGGED = jagged_para()

gb = 27.04 * 1.073741824
bw1 = gb * LB[1]['tg']; ms1 = 1000 / LB[1]['tg']; ms4 = 1000 / LB[4]['tg']; stream1 = gb / 840 * 1000; stream4 = gb / 4 / 840 * 1000
sc_pp4 = LB[4]['pp'] / LB[1]['pp']; sc_tg4 = LB[4]['tg'] / LB[1]['tg']; sc_pp2 = LB[2]['pp'] / LB[1]['pp']; sc_tg2 = LB[2]['tg'] / LB[1]['tg']

html = f'''<title>Qwen3.8-27B Q8_0 on gfx906</title>
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
.mast .claim {{ border-left: 3px solid var(--accent); padding: 0.35rem 0 0.35rem 1rem; max-width: 60ch; font-size: 1.05rem; }}
.mast .claim b {{ font-weight: 600; }}
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
.key {{ display: inline-block; width: 14px; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 0.45rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; font-size: 0.88rem; color: var(--ink-2); align-items: center; }}
.legend span {{ display: inline-flex; align-items: center; gap: 0.45rem; }}
.sw {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.lk {{ width: 18px; height: 3px; border-radius: 2px; display: inline-block; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1.5rem; align-items: start; }}
svg {{ max-width: 100%; height: auto; display: block; }}
.chart {{ background: var(--surface); border: 1px solid var(--rule); padding: 1rem 1rem 0.6rem; position: relative; }}
.chart text {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; fill: var(--ink-2); }}
.chart text.val {{ fill: var(--ink); font-weight: 500; }}
.chart text.tick {{ fill: var(--muted); font-size: 11px; }}
.chart text.ratio {{ fill: var(--ink-2); font-size: 11px; }}
.chart .grid line {{ stroke: var(--grid); stroke-width: 1; }}
.chart .axis {{ stroke: var(--axis); stroke-width: 1; }}
.chart .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.chart .pt {{ stroke: var(--surface); stroke-width: 2; }}
.chart .xh {{ stroke: var(--rule-2); stroke-width: 1; }}
.chart .region {{ stroke: var(--rule-2); stroke-width: 1; }}
.chart text.region-l {{ fill: var(--accent-ink); font-size: 10.5px; letter-spacing: 0.04em; }}
.chart .bar {{ cursor: default; }}
.chart .bar rect.hit {{ fill: transparent; }}
.chart .bar:hover .mark, .chart .bar:focus .mark {{ filter: brightness(0.92); }}
.chart svg[data-kind="lines"] {{ cursor: crosshair; }}
.tip {{ position: absolute; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 0.8rem; line-height: 1.4; padding: 0.45rem 0.6rem; border-radius: 4px; max-width: 20rem; opacity: 0; transition: opacity 120ms; }}
.tip[data-show="1"] {{ opacity: 1; }}
.tip .row {{ display: flex; gap: 0.5rem; align-items: baseline; }}
.tip .row b {{ font-weight: 600; min-width: 3.2em; text-align: right; font-variant-numeric: tabular-nums; }}
.tip .row i {{ display: inline-block; width: 12px; height: 3px; border-radius: 2px; }}
.tip .h {{ font-weight: 600; margin-bottom: 0.15rem; }}
.files {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.8rem; color: var(--ink-2); display: grid; grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr)); gap: 0.35rem 1.5rem; }}
footer {{ border-top: 1px solid var(--rule); padding-top: 1.5rem; font-size: 0.9rem; color: var(--ink-2); display: grid; gap: 0.6rem; }}
@media (prefers-reduced-motion: reduce) {{ .tip {{ transition: none; }} }}
@media (max-width: 40rem) {{ body {{ font-size: 16.5px; }} .tile .v {{ font-size: 2rem; }} }}
</style>

<main class="page">

<header class="mast">
  <div class="eyebrow">Measurement report · Mac Pro (2019) · 2026-09-04</div>
  <h1>Qwen3.8-27B at 8 bits on four Vega 20 dies</h1>
  <p class="lede">A 27-billion-parameter hybrid-attention model, 27 GiB of Q8_0 weights, served by llama.cpp on the four gfx906 dies of a 2019 Mac Pro: how fast it reads a prompt, how fast it writes, how that scales from one die to four, and what happens under thirty-two concurrent requests.</p>
  <p class="claim"><b>The short version:</b> four dies in tensor split give 45.7 tokens/s to one stream and {REC['a8']:.0f} tokens/s to eight, reading 1300-token prompts in under two seconds; thirty-two clients get no more in total and a quarter each. Layer split is 2.4× slower; the dies are limited by their 200 W power cap in prefill and by memory bandwidth plus cross-die synchronisation in generation. Batching climbs a staircase with a step at every multiple of 8 sequences. Run it with eight slots. Every number was measured twice, a day apart, and agreed within 2%.</p>
  <dl class="facts">
    <div><dt>Model</dt><dd>Qwen3.8-27B, Unsloth GGUF Q8_0, 27.32 B params, 27.04 GiB, 866 tensors (506 Q8_0 + 360 F32)</dd></div>
    <div><dt>Machine</dt><dd>macpro2019-01: Xeon W-3245, 377 GiB RAM, 2 × Radeon Pro Vega II Duo = 4 × Vega 20 (gfx906), 32 GB HBM2 each, one XGMI hive</dd></div>
    <div><dt>Software</dt><dd>Ubuntu 24.04, Linux 7.0.0-30, ROCm 7.14 (TheRock, gfx906), llama.cpp b10288 (360e134), HIP graphs, flash attention on</dd></div>
    <div><dt>Conditions</dt><dd>fresh boot, sclk 1730 MHz, perf level high, chassis fans at max (T2 fan daemon and SMC sensor module installed that morning); the same suite on 2026-09-03, default DPM, no fan daemon, agreed within 2%</dd></div>
  </dl>
</header>

<section aria-label="Key figures">
  <div class="tiles">
    <div class="tile"><div class="v">45.7<small>tok/s</small></div><div class="l">Generation for a single stream on all four dies (tensor split), 256 tokens after a 2048-token prompt. One die alone: 20.3.</div></div>
    <div class="tile"><div class="v">830<small>tok/s</small></div><div class="l">Prompt processing on four dies; 235 on one. A 1300-token prompt is read in 1.8 s.</div></div>
    <div class="tile"><div class="v">{REC['a8']:.0f}<small>tok/s</small></div><div class="l">Aggregate generation serving eight concurrent 1300-token requests: {REC['p8']:.1f} tokens/s each, {REC['r8']:.1f} requests per minute. Thirty-two clients: the same total, {REC['p32']:.1f} each.</div></div>
    <div class="tile"><div class="v">200<small>W</small></div><div class="l">Per-die power cap, reached in prompt processing. Clocks held at 1730 MHz; junction peaked at 82 °C with the fans at maximum.</div></div>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">1 · The model, as llama.cpp sees it</div><h2>A hybrid of linear attention and full attention, quantised to 8 bits</h2></div>
  <div class="prose"><p>The GGUF declares the architecture <code>qwen35</code>. Of its 65 blocks, 64 are used; the last is a next-token-prediction head llama.cpp ignores. Every fourth block is full attention (16 in all) with 24 query heads over 4 key/value heads, each 256 wide; the other 48 are linear-attention blocks with a 128-wide recurrent state and a 4-tap convolution. That mix is why the KV cache stays small: only the 16 attention blocks keep one, so 32 slots of 4096 tokens fit beside the weights on four dies.</p></div>
  <div class="tablewrap"><table>
    <thead><tr><th>field</th><th>value</th><th>field</th><th>value</th></tr></thead>
    <tbody>
      <tr><td>architecture</td><td class="mono">qwen35 (hybrid)</td><td>context length</td><td class="mono">262 144</td></tr>
      <tr><td>blocks</td><td class="mono">65 (64 used + 1 next-token-prediction, unused)</td><td>embedding / feed-forward width</td><td class="mono">5 120 / 17 408</td></tr>
      <tr><td>full-attention interval</td><td class="mono">every 4th block: 16 attention, 48 linear</td><td>attention heads</td><td class="mono">24 query, 4 key/value, 256 wide, RoPE 64 dims at base 10<sup>7</sup></td></tr>
      <tr><td>linear-attention state</td><td class="mono">128 × 16 groups, conv kernel 4, inner size 6 144</td><td>tokenizer</td><td class="mono">gpt2-style, pre-tokenizer qwen35</td></tr>
      <tr><td>weights</td><td class="mono">506 Q8_0 tensors + 360 F32 norms = 27.04 GiB</td><td>per die, tensor split of 4</td><td class="mono">6.76 GiB of weights + its share of the KV cache</td></tr>
      <tr><td>quantised by</td><td class="mono">Unsloth, from Qwen/Qwen3.8-27B (Apache-2.0)</td><td>file</td><td class="mono">/root/models/Qwen3.8-27B-Q8_0.gguf</td></tr>
    </tbody></table></div>
  <div class="prose"><p>Two fits matter for what follows. One die holds the whole model with room for a single 32k-token context, so the llama-bench single-die rows are real single-die numbers. The 32-slot server's cache, recurrent state and compute buffers would not fit beside 27 GiB of weights on one 32 GB die, so the serving sweeps use two or four dies.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">2 · Scaling across dies</div><h2>Prompts scale 3.5× on four dies; generation 2.3×</h2></div>
  <div class="prose"><p><code>llama-bench</code> with a 2048-token prompt and 256 generated tokens, tensor split, flash attention, batch and micro-batch 2048, two repetitions. The grey column is what linear scaling from one die would give.</p></div>
  <div class="two">
    <figure>
      <div class="chart">{scaling_panel('pp', 'prompt processing, pp2048, tokens/s', 1000, [0, 250, 500, 750, 1000], 'tokens/s', 'Column chart of prompt processing throughput: 235 tokens per second on one die, 440 on two, 830 on four, against linear scaling of 469 and 939.')}<div class="tip" role="status" aria-live="polite"></div></div>
    </figure>
    <figure>
      <div class="chart">{scaling_panel('tg', 'generation, tg256, tokens/s', 100, [0, 25, 50, 75, 100], 'tokens/s', 'Column chart of generation throughput: 20.3 tokens per second on one die, 31.2 on two, 45.7 on four, against linear scaling of 40.5 and 81.')}<div class="tip" role="status" aria-live="polite"></div></div>
    </figure>
  </div>
  <div class="legend"><span><i class="sw" style="background:var(--series)"></i>measured</span><span><i class="sw" style="background:var(--series-dim)"></i>linear scaling from one die</span></div>
  <figcaption>Hover or focus a column for the figure. Prompt processing is compute-bound and splits almost cleanly; generation is bound by memory bandwidth and by the synchronisation every layer needs between dies.</figcaption>
  {lb_table()}
  <div class="prose">
    <p>The generation numbers are a bandwidth story. One die streams its {gb:.0f} GB of weights once per token: at {LB[1]['tg']:.2f} tokens/s that is {bw1:.0f} GB/s, {100 * bw1 / 840:.0f}% of the 840 GB/s a plain read of its HBM2 achieves. Four dies each stream a quarter of the weights, which would take {stream4:.1f} ms of the {ms4:.1f} ms a token costs; the other {ms4 - stream4:.0f} ms is everything that does not divide by four: the reduction across dies after every layer, kernel launches, and the sequential parts of the graph.</p>
    <p>Context depth costs prompt processing more than generation: at 32k tokens of context, four dies keep {100 * LB[4]['pp32'] / LB[4]['pp']:.0f}% of their prompt speed and {100 * LB[4]['tg32'] / LB[4]['tg']:.0f}% of their generation speed. With only 16 attention blocks, the attention over a long context is a small share of each token.</p>
  </div>
  <h3>The same model at Q4_K_M</h3>
  <div class="prose"><p>Unsloth's Q4_K_M build (15.3 GiB) was measured with the same command on 2026-09-03. Generation gains from the smaller weights on one die and almost nothing on four, where synchronisation dominates; prompt processing is faster at Q8_0 on this hardware, whose 8-bit matrix kernels are the better fit for Vega 20.</p></div>
  {q4_table()}
</section>

<section>
  <div class="sec-head"><div class="eyebrow">3 · Batching</div><h2>Thirty-two sequences generate 4.1× the tokens of one, up a staircase with steps at every eighth</h2></div>
  <div class="prose"><p><code>llama-batched-bench</code> on four dies in tensor split, a 73 728-token KV budget, 512-token prompts (2048 in the table) and 128 generated tokens per sequence. The first pass used batch sizes 1, 2, 4, 8, 16 and 32 and showed batch 16 below batch 8; the 512-token charts below come from one run over every size from 1 to 32, and the 2048-token column of the table from two passes covering the same sizes. Prompt processing sits at 830 tokens/s from batch 4 up whatever the batch, so both charts show the generation phase.</p></div>
  <figure>
    <div class="chart">{staircase_panel()}<div class="tip" role="status" aria-live="polite"></div></div>
    <figcaption>Time for one decode step (every sequence in the batch advances one token) by batch size. The step cost is nearly flat within each region and jumps between 8 and 9, 16 and 17, 24 and 25: batches of up to 8 tokens use llama.cpp's quantised matrix-vector kernel; larger batches use its MMQ matrix kernel, which picks the smallest tile width in multiples of 8 that covers the batch and computes the whole tile, so batch 9 pays for 16 and batch 17 for 24.</figcaption>
  </figure>
  <figure>
    <div class="chart">{fine_cols_panel()}<div class="tip" role="status" aria-live="polite"></div></div>
    <figcaption>The same runs as aggregate throughput. Within a tile the fixed step cost is shared by more sequences, so throughput climbs; crossing into the next tile drops it by a third. Batch 16 delivers no more than batch 8, and batch 24 barely more; only the full 32-wide tile clearly beats eight. Hover or focus a column for any size; the table has them all.</figcaption>
  </figure>
  {fine_table()}
  <div class="prose">
    <p><b>Update, 2026-09-07.</b> The 8-to-9 cliff is a compile-time constant, and raising <code>MMVQ_MAX_BATCH_SIZE</code> from 8 to 16 removes it on the tensor split: batch 9 goes from 96 to 163 tokens/s and 12 becomes a peak at 175. A rewrite of the kernel's Q8_0 path the same afternoon (scales converted once per row and column per block, four rows per thread block up to 8 columns and two above; five measured iterations, every build correct on 1186 of 1186 matrix-multiply tests) goes further on both placements from one binary: on the four-die split 174 / 198 / 207 tokens/s at 8 / 12 / 16 sequences against 161 / 122 / 152 stock, and on a single die 55 / 72 at 4 / 8 sequences against 48 / 53. The table below has both; the staircase rule stands for the stock build. The same kernel inside the ML-gfx906 fork, whose own gfx906 tile table reads prompts a third faster (1130 against 848 tokens/s on the split, 314 against 230 on one die), is one binary with both gains at perplexity identical to stock (5.5969): the box's production build since the evening of 2026-09-07, with the decode cells re-measured within 3% of the table below. 2048-token prompts on the split, 512 on one die, same session as the stock control; the measurements and the patch are in the <a href="https://claude.ai/code/artifact/5c2a1ae8-dd59-476e-9739-9dc1f179be93">run-through page</a>.</p>
  </div>
  {mmvq16_addendum()}
  <div class="prose">
    <p>The kernel boundaries are in the source of this build: the matrix-vector path is limited to <code>MMVQ_MAX_BATCH_SIZE</code> of 8 columns, and the MMQ launcher loops over tile widths from 8 to 128 in steps of 8 and keeps the first one that covers the batch in a single tile. On Vega 20 the matrix-vector kernel is also the more efficient one: eight tokens cost 49 ms a step through it, and sixteen cost about 100 ms through a 16-wide MMQ tile.</p>
    <p>For serving that gives a rule: the number of sequences decoding together should sit at the top of a stair. Eight slots give the best latency for the throughput; thirty-two give 12% more in this pure-decode test but nothing more in serving (section 4); anything from 9 to 15 or 17 to 23 active sequences pays a full tile for a partial one, and 9, 17 and 25 are the worst sizes of all. Batch {max(FB)} reaches {FINE[512][max(FB)][1]:.0f} tokens/s at {FINE[512][max(FB)][1] / max(FB):.1f} per sequence. Prompt length made no difference to the generation phase: the 2048-token rows in the table track the 512-token rows within a few percent.</p>
  </div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">4 · Serving under concurrency</div><h2>Tensor split over four dies: {SRVF['tp4'][1][2]:.0f} tokens/s for one client, {REC['a8']:.0f} for eight, no more for thirty-two</h2></div>
  <div class="prose"><p><code>llama-server</code> with 32 slots of 4096 tokens, continuous batching, flash attention, batch 2048. The client sends 1300-token prompts, asks for 256 tokens with end-of-sequence ignored, and runs 2× the concurrency in requests (at least 8) with no prompt caching, reading the server's own per-request timings. Three placements of the model: tensor split across all four dies, layer split across all four, and tensor split across the two dies of one MPX module. {'Every concurrency from 1 to 32 was run, as one ascending sweep per placement on a fresh server.' if FINE_SRV else ''}</p></div>
  <div class="legend">{''.join(f'<span><i class="lk" style="background:var(--{var})"></i>{mode}: {label}</span>' for mode, label, var in MODES)}</div>
  {''.join(srv_cols_panel(mode, label, var) for mode, label, var in MODES)}
  <figcaption>Aggregate generation throughput at each concurrency, one panel per placement on the same scale; a column stands only where a level was measured. Hover or focus a column for the per-request rate, time to first token and requests per minute at that level.{SPARSE_NOTE}</figcaption>
  <figure>
    <div class="chart">{line_panel('ttft', 6, 20, [0, 5, 10, 15, 20], lambda v: f'{v:.2f}' if isinstance(v, float) and v != int(v) else f'{v:g}', 'time to first token, seconds (1300-token prompt)', 'Line chart of time to first token versus concurrency for the three placements: tensor split on four dies rises from 1.8 seconds to about 8 by eight clients and stays there; tensor split on two dies and layer split on four dies rise higher.', W=760, R=656)}<div class="tip" role="status" aria-live="polite"></div></div>
    <figcaption>Time to first token for a 1300-token prompt. Move across the chart, or focus it and use the arrow keys, for all three placements at one concurrency; markers sit on measured levels where a placement has only six. On four dies it rises to about 8 s by eight clients and stays there, because every new prompt shares the dies with the streams already generating.</figcaption>
  </figure>
  {srv_table()}
  <div class="prose">
    <p>Tensor split across four dies is the configuration. It leads at every concurrency, saturates at about {max(v[2] for v in SRVF['tp4'].values()):.0f} tokens/s, and reads a prompt in {SRVF['tp4'][1][6]:.2f} s when alone ({1300 / SRVF['tp4'][1][6]:.0f} tokens/s of prefill). Layer split runs the four dies one after another, so each token waits on a chain of transfers: {SRVF['tp4'][1][2] / SRVF['layer4'][1][2]:.1f}× slower for one client, {SRVF['tp4'][32][2] / SRVF['layer4'][32][2]:.1f}× slower for thirty-two, with three times the wait for the first token. Two dies in tensor split reach {SRVF['tp2'][32][2]:.1f} tokens/s, {100 * SRVF['tp2'][32][2] / SRVF['tp4'][32][2]:.0f}% of four.</p>
    <p>Why 66 and not the 181 of the batched bench? The server's clients each hold a 1300-token context, and their prompts arrive while other slots are generating; prefill and generation then share the dies. During the generation phase alone the thirty-two slots still produce about {32 * SRVF['tp4'][32][5]:.0f} tokens/s ({SRVF['tp4'][32][5]:.1f} each); the aggregate over the whole request, prompt included, is {SRVF['tp4'][32][2]:.0f}.</p>
    {JAGGED}
  </div>
  <p class="claim"><b>What to run:</b> eight slots (<code>-np 8</code>), and let the HTTP layer queue the rest. Eight clients already reach the ceiling, {REC['a8']:.1f} tokens/s against {REC['a32']:.1f} at thirty-two and the same {REC['r8']:.1f} requests per minute, but each request streams at {REC['p8']:.1f} tokens/s instead of {REC['p32']:.1f}. Four clients keep {REC['p4']:.1f} per request at {REC['a4']:.1f} in total; the {REC['a8'] - REC['a4']:.1f} tokens/s between four and eight is about {REC['gain']:,.0f} tokens a day ({REC['lo']:,.0f} to {REC['hi']:,.0f} across the two runs), while the {REC['t8'] - REC['t4']:.1f} s of extra first-token wait is smaller than the spread between two measurements of eight ({REC['t8']:.2f} and {REC['t8b']:.2f} s). Beyond eight, each added client only slows the others: at thirty-two everyone reads at {REC['p32']:.1f} tokens/s, below a reader's five to six.</p>
  <div class="prose"><p>Eight is also the size that keeps every decode step on the matrix-vector kernel of section 3, so the figure holds instead of drifting through the padded tile sizes as a 32-slot server does. Held there, the box delivers about {REC['day8']:.1f} million generated tokens and {REC['req8']:,.0f} requests of this shape per day, against {REC['day4']:.1f} million at four slots.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">5 · What limits it</div><h2>Power in prefill, bandwidth and synchronisation in generation</h2></div>
  <div class="prose">
    <p>A sampler read every die's clock, junction temperature and power every five seconds for the whole run. Under prompt processing a die sits at its 200 W cap (instantaneous readings up to 284 W) with its clock at 1730 MHz, so prefill is compute-limited at the power cap and forcing the top DPM level adds nothing. Generation draws far less: the mean over dies at full clock was 90 to 170 W by stage, the signature of memory-bound work.</p>
    <p>The 1000 MHz samples are idle moments, not throttling: in the tensor-split sweep the dies reading 1000 MHz averaged 57 W against 150 W for those at 1730, and in layer split three dies idle while the fourth works, which is what its 71% is. With the chassis fans at maximum (set by hand; the fan speed itself is not readable from Linux on this box), junction temperatures peaked at 82 °C on the Slot-1 dies during single- and two-die work and stayed under 80 °C with all four loaded.</p>
  </div>
  {therm_table()}
</section>

<section>
  <div class="sec-head"><div class="eyebrow">6 · Repeatability</div><h2>Measured twice, a day apart, under different clock conditions</h2></div>
  <div class="prose"><p>The whole set was first run on 2026-09-03 with default clock management and default fan control, then again on 2026-09-04 after a reboot with the performance level forced high and the fans at maximum. Across the 126 comparable cells the median change is +0.6% and everything lies within ±2%, except one: layer split at concurrency 32, where the earlier figure came from a separate retry on a fresh server after a GPU page fault rather than from the end of a full sweep. That fault did not recur.</p></div>
  <div class="tablewrap"><table>
    <thead><tr><th>measurement</th><th class="num">2026-09-03</th><th class="num">2026-09-04</th><th class="num">change</th></tr></thead>
    <tbody>
      <tr><td>llama-bench, one die, pp2048 / tg256</td><td class="num">234.4 / 20.06</td><td class="num">234.7 / 20.25</td><td class="num">+0.1% / +0.9%</td></tr>
      <tr><td>llama-bench, four dies, pp2048 / tg256</td><td class="num">823.5 / 45.40</td><td class="num">829.7 / 45.69</td><td class="num">+0.8% / +0.6%</td></tr>
      <tr><td>batched bench, batch 32, generation (2048-token prompts)</td><td class="num">178.7</td><td class="num">179.2</td><td class="num">+0.3%</td></tr>
      <tr><td>server, tensor split 4, concurrency 8 / 32</td><td class="num">64.1 / 66.3</td><td class="num">64.4 / 66.1</td><td class="num">+0.5% / −0.3%</td></tr>
      <tr><td>server, tensor split 2, concurrency 8 / 32</td><td class="num">40.2 / 42.1</td><td class="num">40.5 / 42.5</td><td class="num">+0.7% / +1.0%</td></tr>
      <tr><td>server, layer split 4, concurrency 8 / 32</td><td class="num">25.8 / 30.3 (retry)</td><td class="num">25.9 / 28.1</td><td class="num">+0.4% / −7.3%</td></tr>
    </tbody></table></div>
  <div class="prose"><p>This box has shown a failure mode in which the dies silently drop to 1000 MHz after hours of load and stay there until a reboot; it invalidated other benchmark runs on 2026-09-03. Those observations were made with no fan daemon and no SMC sensor module installed (the T2 fan daemon <code>t2fanrd</code> and <code>applesmc-t2</code> went in on 2026-09-04 at 03:48 and 03:58 UTC, minutes before the boot these measurements ran on) and with rocm-smi at its defaults. Under this day's conditions, fans set to maximum by hand and the performance level forced high, 7.4 hours of GPU load including the 6.3-hour serving sweep showed no clamp: no sample had three or more dies at 1000 MHz under load, and no die read 1000 MHz at load for longer than 20 s. Whether the fans, the forced level, or neither made the difference is untested. Both runs of this set fell outside the clamp either way.</p></div>
</section>

<section>
  <div class="sec-head"><div class="eyebrow">7 · Reproduce</div><h2>Three commands and one client script</h2></div>
<pre><code># scaling: 1, 2 and 4 dies, tensor split, depths 0 / 16k / 32k
llama-bench -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm0/rocm1,rocm0/rocm1/rocm2/rocm3 \\
  -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 256 -d 0,16384,32768 -r 2 -o md
# batching: four dies, 512 / 2048-token prompts, 128 generated tokens, batch 1..32
llama-batched-bench -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -c 71680 -b 2048 -ub 2048 -npp 512,2048 -ntg 128 -npl 1,2,4,8,16,32
# serving: 32 slots x 4096 tokens; then the sweep client (tp4 shown; layer4 = -sm layer; tp2 = --device rocm0,rocm1)
llama-server -m /root/models/Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on \\
  -np 32 -cb -c 131072 -b 2048 -ub 2048 --host 127.0.0.1 --port 8089
python3 /root/rocm-tests/bench/server-bench.py http://127.0.0.1:8089 TAG --conc 1,2,4,8,16,32 --gen 256 --prompt-tokens 1300</code></pre>
  <div class="files">
    <span>rocm-tests/bench/rerun-27b-q8_0-unclamped.sh · runs all of the above in order</span><span>qwen38-27b-q8_0-unclamped.md · llama-bench sweep</span>
    <span>qwen38-27b-q8_0-unclamped-batched.md · batched bench</span><span>qwen38-27b-q8_0-unclamped-server-{{tp4,layer4,tp2}}.md · server sweeps (+ .json, .log)</span>
    <span>qwen38-27b-q8_0-unclamped-clocks.txt · clock / temperature / power trace, 5 s</span><span>qwen38-27b-q8_0-unclamped-COMPARE.md · 2026-09-03 vs 2026-09-04, all 126 cells</span>
    <span>qwen38-27b-q8_0*.md · the 2026-09-03 set</span><span>qwen38-27b-ud-q4_k_m.md · the Q4_K_M llama-bench sweep</span>
  </div>
  <div class="prose"><p>Related: <a href="https://claude.ai/code/artifact/f811df1d-b2bb-4d95-9157-4726d0519194">gfx906 Four-Way XGMI Ring</a>, the measurement of the fabric these four dies synchronise over, including why RCCL needs a corrected topology file on this machine.</p></div>
</section>

<footer>
  <p>Measured 2026-09-04 on macpro2019-01 with llama.cpp b10288 on ROCm 7.14. Tokens per second as llama.cpp reports them; server figures from the server's own per-request timings; bandwidths in 10⁹ bytes per second.</p>
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
    // column charts: each bar carries its own tooltip
    chart.querySelectorAll('.bar').forEach(function (bar) {{
      function show(evt) {{
        tip.textContent = bar.getAttribute('data-tip');
        var b = bar.querySelector('.mark').getBoundingClientRect();
        place(evt && evt.clientX ? evt.clientX : b.right, b.top);
      }}
      bar.addEventListener('mouseenter', show); bar.addEventListener('mousemove', show); bar.addEventListener('mouseleave', hide);
      bar.addEventListener('focus', function () {{ show(null); }}); bar.addEventListener('blur', hide);
    }});
    // line charts: crosshair snapped to the nearest concurrency, one tooltip for every series
    var svg = chart.querySelector('svg[data-kind="lines"]'); if (!svg) return;
    var data = JSON.parse(chart.querySelector('script[data-for="' + svg.getAttribute('data-panel') + '"]').textContent);
    var xh = svg.querySelector('.xh'), vb = svg.viewBox.baseVal, idx = -1;
    function showIdx(i, clientX) {{
      idx = i; var x = data.x[i];
      xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.removeAttribute('hidden');
      while (tip.firstChild) tip.removeChild(tip.firstChild);
      var h = document.createElement('div'); h.className = 'h'; h.textContent = data.conc[i] + ' concurrent request' + (data.conc[i] > 1 ? 's' : ''); tip.appendChild(h);
      data.series.forEach(function (s) {{
        var row = document.createElement('div'); row.className = 'row';
        var k = document.createElement('i'); k.style.background = 'var(--' + s.var + ')';
        var v = document.createElement('b'); v.textContent = s.values[i] == null ? '–' : Number(s.values[i]).toFixed(s.values[i] < 10 ? 2 : 1);
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
print(f'wrote {OUT}: {len(html)} bytes, {html.count(chr(10))} lines')
print(f'derived: GB {gb:.1f}, 1-die {bw1:.0f} GB/s ({100*bw1/840:.0f}%), ms/token 1 die {ms1:.1f} / 4 dies {ms4:.1f}, stream 4 dies {stream4:.1f} ms; scaling pp4 {sc_pp4:.2f} tg4 {sc_tg4:.2f} pp2 {sc_pp2:.2f} tg2 {sc_tg2:.2f}')
