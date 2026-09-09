#!/usr/bin/env python3
"""gen-0908-report.py: builds reports/2026-09-08-next-steps-measurements.html from data/benchmarks.json (sibling of the run-through page)."""
import json, html, datetime, os
G = '/root/llama.cpp-benchmarking'
D = json.load(open(f'{G}/data/benchmarks.json'))
def esc(x): return html.escape(str(x))
def fmt(v, nd=1):
    if v is None: return '—'
    if isinstance(v, bool): return 'yes' if v else 'no'
    if isinstance(v, float): return f'{v:.{nd}f}'
    return esc(v)
def table(cols, rows, num_from=1, hi=None, nd=1, caption=None):
    out = ['<div class="tw"><table><thead><tr>']
    for i, c in enumerate(cols): out.append(f'<th class="{"num" if i >= num_from else ""}">{esc(c)}</th>')
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>')
        for i, v in enumerate(r):
            cls = ('num' if i >= num_from else '') + (' hi' if hi and hi(r, i) else '')
            out.append(f'<td class="{cls.strip()}">{fmt(v, nd)}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    if caption: out.append(f'<p class="cap">{caption}</p>')
    return '\n'.join(out)
def bars(title, groups, series_names, unit='tok/s', width=860, colors=('var(--series)','var(--series-2)','var(--series-3)','var(--series-dim)')):
    """grouped horizontal bars: groups = [(label, [v1, v2, ...])]; one scale across all"""
    vals = [v for _, vs in groups for v in vs if v is not None]
    vmax = max(vals) * 1.08
    rowh = 16 * len(series_names) + 10; labw = 150; top = 8
    h = top + rowh * len(groups) + 26
    s = [f'<figure><div class="legend">' + ''.join(f'<span style="--sw:{colors[i]}">{esc(n)}</span>' for i, n in enumerate(series_names)) + '</div>',
         f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}">']
    scale = (width - labw - 70) / vmax
    for t in range(0, int(vmax) + 1, 50 if vmax > 150 else 10):
        x = labw + t * scale
        s.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h-22}" stroke="var(--grid)" stroke-width="1"/><text x="{x:.1f}" y="{h-8}" font-size="11" fill="var(--muted)" text-anchor="middle">{t}</text>')
    for gi, (label, vs) in enumerate(groups):
        y0 = top + gi * rowh
        s.append(f'<text x="{labw-8}" y="{y0 + rowh/2 + 4}" font-size="12" fill="var(--ink-2)" text-anchor="end">{esc(label)}</text>')
        for si, v in enumerate(vs):
            if v is None: continue
            y = y0 + 4 + si * 16; w = v * scale
            s.append(f'<rect class="mark" tabindex="0" x="{labw}" y="{y}" width="{w:.1f}" height="12" fill="{colors[si]}" data-tip="{esc(label)} · {esc(series_names[si])}: {v:g} {unit}"/>')
            s.append(f'<text x="{labw + w + 5:.1f}" y="{y + 10}" font-size="11" fill="var(--ink)" font-family="IBM Plex Mono, monospace">{v:g}</text>')
    s.append(f'</svg><figcaption>{esc(title)}</figcaption></figure>')
    return '\n'.join(s)

m2 = D['m2_slots_prod']; m4 = D['kv_unified_b10837']['batched_8x32k_m4']; fk = D['fork_knobs_sweep']; ag = D['ar_gate_sweep']
fu = D['fusion_nq_test']; sub = fu['gdn_subfold']; go = D['graph_opt_split']; m3 = D['mtp_prod_build']; pr = D['tp2x2_prod_server']; b1 = D['mmvq_batch1_variants']
fin = D.get('final_config'); fa = D.get('fa_counters_head256'); ci = D.get('paired_ci_prod_table'); sf = D.get('server_final_prod'); gv = D.get('governor_threshold_m7'); cp = D.get('power_cap_phases'); ab = D.get('tile_table_ablation')
now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

# M2 chart: decode by slots per depth
m2rows = {(r[0], r[1]): r for r in m2['rows']}
m2chart = bars('M2 — production build, tp4 decode tok/s by slots at 2K / 8K / 32K prompts',
               [(f'{n} slots', [m2rows.get((2048, n), [None]*4)[3], m2rows.get((8192, n), [None]*4)[3], m2rows.get((32768, n), [None]*4)[3]]) for n in (8, 12, 16, 24, 32)],
               ['2K', '8K', '32K'])
agchart = bars('Allreduce size gate — decode tok/s at 1 / 2 / 4 / 8 / 16 slots (2K), fusion build',
               [(r[0], r[2:7]) for r in ag['rows'] if r[0] in ('base (RCCL)', 'peer-write, fork default gate', 'peer-write, 4 rows', 'peer-write, 8 rows')],
               ['1', '2', '4', '8', '16'], colors=('var(--series)','var(--series-2)','var(--series-3)','var(--accent)','var(--series-dim)'))
m3rows = [(r[0], r[1], r[2], r[3]) for r in m3['rows']]
m3chart = bars('M3 — MTP on the production build, decode-only wave, tok/s',
               [(f'{p//1024}K · {n} slots', [next((r[3] for r in m3rows if r[0]==p and r[1]==n and r[2]==d), None) for d in (0,1,2,3)]) for p in (2048, 32768) for n in (4, 8)],
               ['no draft', 'draft 1', 'draft 2', 'draft 3'], colors=('var(--series-dim)','var(--series)','var(--series-2)','var(--series-3)'))

final_section = ''
if fin:
    final_section = f'''
<h2 id="final">The combined configuration against the 2026-09-07 production build</h2>
<p>{esc(fin.get("summary",""))}</p>
{table(fin["lb"]["columns"], fin["lb"]["rows"], num_from=2, nd=1)}
{table(fin["batched"]["columns"], fin["batched"]["rows"], num_from=1, nd=1, caption="llama-batched-bench tp4, 2K prompts, decode tok/s.")}
{table(fin["mtp"]["columns"], fin["mtp"]["rows"], num_from=1, nd=1, caption="Single user with MTP draft 3, llama-server -np 1, decode-only wave.")}
<p class="small">{esc(fin.get("numerics",""))}</p>
'''

sf_section = ''
if sf:
    rows = {(r[0], r[3]): r for r in sf['rows']}
    def g(p, c, i=6):
        r = rows.get((p, c)); return r[i] if r else None
    sfchart = bars('Server level, 1300/256 requests — aggregate generation tok/s by clients',
                   [('final, 16 slots', [g('team16', 4), g('team16', 8), g('team16', 12), g('team16', 16), None]),
                    ('final, 32 slots', [None, None, None, g('busy32', 16), g('busy32', 32)]),
                    ('2026-09-07 build, 32 slots', [None, None, None, g('prod0907-np32', 16), g('prod0907-np32', 32)]),
                    ('final, two tp2 pairs', [None, g('pairs8', 8), None, g('pairs8', 16), None])],
                   ['4 clients', '8', '12', '16', '32'], colors=('var(--series-dim)', 'var(--series)', 'var(--series-2)', 'var(--series-3)', 'var(--accent)'))
    sf_section = f'''
<h2 id="server">The final build and 32 slots at the server level <span class="verdict v-change">busy profile → 16 slots</span></h2>
<p><code>llama-server</code> tp4 on the final build with <code>gfx906.env</code>, 1300-token prompts / 256 generated. Key <code>server_final_prod</code>.</p>
{sfchart}
{table(['profile','build','slots','clients','agg gen tok/s','total tok/s','req/min','per-user tok/s','TTFT s','per-request s'], [[r[0], r[1], r[2], r[3], r[6], r[7], r[8], r[9], r[10], r[11]] for r in sf['rows']], num_from=2, nd=1)}
<p>{esc(sf['reading'])}</p>
'''
gv_section = ''
if gv:
    gv_section = f'''
<h2 id="governor">Host load beside the server and the governor threshold (M7) <span class="verdict v-trade">175 W host cap at 200 W dies</span></h2>
<p>Production build tp4 at 16 slots, 16 clients on 1300/256 requests, beside one <code>yes</code> per hardware thread under stepped host RAPL caps; dies at 200 W caps; the job stops at DC 1180 W. Key <code>governor_threshold_m7</code>.</p>
{table(['host','agg gen tok/s','total tok/s','per user','TTFT s','per-request s','DC max W'], gv['serving_cost']['rows'], num_from=1, nd=1, caption="A. The serving cost of an all-core host load at the 150 W cap.")}
{table(['host cap W','DC max W (40 s)','die W mean'], gv['dc_vs_host_cap']['rows'], num_from=0, nd=0, caption="B. DC total against the host cap with the 16 clients running; stopped at 200 W (1206 W > the 1180 W guard). No clamp: 1730 MHz throughout.")}
<p>{esc(gv['reading'])}</p>
'''
cp_section = ''
if cp:
    cprows = cp['rows'][:5]
    cpchart = bars('Cap sweep, phases separated — percent of the 200 W figure',
                   [(f'{r[0]} W', [round(100*r[2],1), round(100*r[4],1)]) for r in cprows[1:]], ['decode-only, 16 slots', 'prefill-only, pp2048'], unit='%')
    cp_section = f'''
<h2 id="caps">Cap sweep with the phases separated <span class="verdict v-change">optimiser curves replaced</span></h2>
<p>Production build tp4; per cap, decode-only at 16 slots (2K) and prefill-only pp2048, caps set live and 200 W repeated at the end. Key <code>power_cap_phases</code>.</p>
{cpchart}
{table(['cap W','decode 16 slots tok/s','vs 200 W','prefill pp2048 tok/s','vs 200 W','die W (decode)','sclk','die W (prefill)','sclk'], [[r[0], r[1], f"{100*(r[2]-1):+.1f}%", r[3], f"{100*(r[4]-1):+.1f}%", r[5], r[6], r[7], r[8]] for r in cp['rows']], num_from=99)}
<p>{esc(cp['reading'])}</p>
'''
ab_section = ''
if ab:
    q8 = {r[0]: r for r in ab['rows'] if r[1] == 'Q8_0'}; q6 = {r[0]: r for r in ab['rows'] if r[1] == 'Q6_K'}; q4 = {r[0]: r for r in ab['rows'] if r[1] == 'Q4_K_M'}
    abchart = bars('Tile-table ablation — Q8_0 pp2048 tok/s, four dies and one die', [(b, [q8[b][4], q8[b][2]]) for b in ('stock', 'a-tileload', 'b-config', 'c-kunroll', 'fork-b10254')], ['four dies (tp4)', 'one die'])
    bb = {r[0]: r for r in ab['batched']['rows']}
    ab_section = f'''
<h2 id="ablation">Tile-table ablation (TODO 12) <span class="verdict v-change">one header is the gain</span></h2>
<p>The fork's three MMQ commits applied cumulatively on stock b10288 (a = tile-load threads, b = + the gfx906 MMQ config table with its dispatch guards, c = + the Q8_0 partial-k unroll), against the whole fork b10254. Key <code>tile_table_ablation</code>.</p>
{abchart}
{table(['build','Q8_0 tp4 pp2048','Q8_0 one die','Q6_K tp4','Q4_K_M tp4','decode 16 / 24 / 32 slots'], [[b, q8[b][4], q8[b][2], q6[b][4], q4[b][4], (f"{bb[b][1]:.0f} / {bb[b][2]:.0f} / {bb[b][3]:.0f}" if b in bb else '')] for b in ('stock', 'a-tileload', 'b-config', 'c-kunroll', 'fork-b10254')], num_from=1, nd=1)}
<p>{esc(ab['reading'])}</p>
'''
fa_section = ''
if fa:
    pk = fa['prefill_kernel']; dk = fa['decode_kernel']
    fa_section = f'''
<h2 id="fa">Flash-attention counters at head size 256 (S4) <span class="verdict v-trade">two targets</span></h2>
<p>Production build tp4, <code>rocprofv3</code> kernel trace with <code>SQ_INSTS_VALU</code> (the only one of the four requested counters that exists for gfx906 on ROCm 7.14; the gfx906-set pass is queued), plus the instruction counts of the gfx906 code object. Primary source <code>reports/2026-09-08-fa-counters.md</code>; key <code>fa_counters_head256</code>.</p>
{table(['phase','kernel','share %','dispatches','mean µs','max µs'], [[r[0], r[1], r[2], r[3], r[4], r[5] if r[5] is not None else ''] for r in fa['time_share']['rows']], num_from=2, nd=1, caption="Kernel time share under the profiler; attention is 17% of a 32K prompt on the split and 16 × 304 µs + 0.5 ms combine = 19% of a 128K decode token.")}
{table(['kernel','occupancy','VALU issue','KV loop','reading'], [
 [pk['name'] + ' (prefill)', f"{pk['waves_per_simd']} waves/SIMD: {pk['vgpr']} VGPRs, {pk['lds_bytes']//1024} KB LDS", f"{pk['valu_issue_pct_of_peak']}% of peak", f"{pk['loop_instr']} instr: {pk['loop_v_dot2']} v_dot2, {pk['loop_v_pk']} v_pk, {pk['loop_ds_read']} ds_read, {pk['loop_s_waitcnt']} s_waitcnt", 'latency-bound at low occupancy; a tile at 3 waves/SIMD needs ≤ 84 VGPRs and ≤ 21.8 KB'],
 [dk['name'] + ' (decode 128K)', f"{dk['waves_per_simd']} waves/SIMD: {dk['vgpr']} VGPRs, {dk['lds_bytes']//1024} KB", f"{dk['valu_issue_pct_of_peak']}%", f"{dk['loop_instr']} instr: {dk['loop_v_dot2']} v_dot2, {dk['loop_v_pk']} v_pk, {dk['loop_ds_read']} ds_read, {dk['loop_s_waitcnt']} s_waitcnt", f"KV-streaming at ≥ {dk['hbm_gbps_min']} GB/s and {dk['kv_passes']} passes over the same KV (GQA 6 packed as ncols2 = 2); a six-head packing reads it once, floor 150 µs vs 304"]], num_from=99)}
<p><b>Second pass with the gfx906 counter set</b> (queue 23): the prefill tile kernel measures 46–47% VALU busy, 24 LDS instructions per global load, LDS wait 7.5 shader-engine cycles per active cycle, no bank conflicts and a 95% L2 hit rate on the K/V tiles — an LDS-bound loop at two waves per SIMD, as the disassembly said. The MMQ Q8_0 × 128 kernel beside it runs 61% VALU with 6 M LDS bank conflicts per dispatch (S7 material). The one-column decode kernel at depth needs a separate decode-heavy pass.</p>
<p>The packed-fp16 dot is already in use in both kernels, so S4's order becomes: the corrected counters, then the six-head decode packing (a dispatch and template change worth up to 9% of the 128K token), then a prefill tile shape at three waves per SIMD.</p>
'''
ci_section = ''
if ci:
    s = ci['summary']
    def c(b, k, nd=1):
        m, sd = s[b][k]; return f"{m:.{nd}f} ± {sd:.{nd}f}"
    ci_section = f'''
<h2 id="ci">Paired confidence intervals for the production table <span class="verdict v-none">gains stand</span></h2>
<p>Three builds interleaved per round; <code>llama-bench -r 5</code> × 3 rounds and <code>llama-batched-bench</code> × 2 rounds. Mean ± sd across rounds. Key <code>paired_ci_prod_table</code>.</p>
{table(['cell','stock','2026-09-07 production','production'], [
 ['tp4 pp2048', c('stock','tp4_pp'), c('prod0907','tp4_pp'), c('prod','tp4_pp')],
 ['tp4 tg128', c('stock','tp4_tg'), c('prod0907','tp4_tg'), c('prod','tp4_tg')],
 ['rocm0 pp2048', c('stock','rocm0_pp'), c('prod0907','rocm0_pp'), c('prod','rocm0_pp')],
 ['rocm0 tg128', c('stock','rocm0_tg',2), c('prod0907','rocm0_tg',2), c('prod','rocm0_tg',2)],
 ['tp4 decode 8 slots', c('stock','tp4_8'), c('prod0907','tp4_8'), c('prod','tp4_8')],
 ['tp4 decode 12 slots', c('stock','tp4_12'), c('prod0907','tp4_12'), c('prod','tp4_12')],
 ['tp4 decode 16 slots', c('stock','tp4_16'), c('prod0907','tp4_16'), c('prod','tp4_16')],
 ['rocm0 decode 4 slots', c('stock','rocm0_4'), c('prod0907','rocm0_4'), c('prod','rocm0_4')],
 ['rocm0 decode 8 slots', c('stock','rocm0_8'), c('prod0907','rocm0_8'), c('prod','rocm0_8')]], num_from=99)}
<p>{esc(ci['reading'])} Within one <code>llama-bench</code> run the split's single-stream spread is 1–2.7 tok/s (the RCCL path), so rounds, not repeats, are the unit for the split.</p>
'''

page = f'''<title>gfx906 Next-Steps Measurements</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{open(f"{G}/reports/2026-09-07-todo-runthrough.html").read().split("<style>")[1].split("</style>")[0]}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);margin:18px 0 10px}}
.kpi div{{background:var(--surface);padding:12px 14px}}
.kpi .n{{font-family:"Schibsted Grotesk",sans-serif;font-size:1.7rem;font-weight:800;line-height:1.1;font-variant-numeric:tabular-nums}}
.kpi .l{{font-size:.84rem;color:var(--ink-2);margin-top:2px}}
</style>
<div class="wrap">
<div class="eyebrow">gfx906 · Qwen3.8-27B Q8_0 · NEXT-STEPS measurements · 2026-09-08, updated {now}</div>
<h1>Next-Steps Measurements</h1>
<p class="lede">The roadmap's diagnostics and its first code, measured on the four Vega 20 dies in one day: where a decode token's time goes, which of the planned changes paid, and what the fork tree already had waiting.</p>
<div class="claim"><b>Single stream:</b> the fork's own peer-write allreduce, gated to four rows, is +14.5% on its own; the one-column whole-block load and two exact kernel fusions add ~5%; MTP draft 3 multiplies the rest.</div>
<div class="claim"><b>Kernel fusion measured ~3%, not the 15–20% the trace implied:</b> the small kernels are latency-bound on each block's critical path, and neither fusing them nor running them on more streams recovers that time.</div>
<div class="claim"><b>Serving:</b> 32 slots beat 16 by 4–5% on the decode-only bench but not at the server level (80–85 vs 83 tok/s, half the per-user rate); the two pairs beat the four-die server by 20% at the request level; MTP's verify batch may reach 16 rows on the production build.</div>
<div class="cond">
<div><div class="k">Builds</div><div class="v">production = fork tile table + MMVQ patches (2026-09-07); fusion build = production + the nine-patch series</div></div>
<div><div class="k">Conditions</div><div class="v">perf level high, fans max, host RAPL 150 W, SMC log and clock sampler beside every job, watchdog v2</div></div>
<div><div class="k">Model</div><div class="v">Qwen3.8-27B Q8_0, 27.04 GiB, 64 blocks (48 linear-attention), n_embd 5120</div></div>
<div><div class="k">Placement</div><div class="v">tp4 = tensor split over four dies; one die = rocm0; pairs = two tp2 servers</div></div>
</div>
<p class="cap">Every number here is in <code>data/benchmarks.json</code> (keys named per section) and reproduced by the scripts in <code>tools/</code>; the Markdown source is <code>reports/2026-09-08-next-steps-measurements.md</code>.</p>

<h2>What the day settled</h2>
<div class="kpi">
<div><div class="n">56.6</div><div class="l">tok/s single stream with the fork's allreduce gated to four rows (49.4 before)</div></div>
<div><div class="n">+4–5%</div><div class="l">32 slots over 16 at 2K, 8K and 32K on the production build</div></div>
<div><div class="n">101.9</div><div class="l">tok/s at 16 clients from two tp2 pairs on the final build, against 83.4 from the four-die server</div></div>
<div><div class="n">80.0</div><div class="l">tok/s at 32 clients on 32 slots at the server level: slots beyond 16 add nothing there (busy profile → 16)</div></div>
<div><div class="n">+23%</div><div class="l">4 slots × draft 3 at 32K on the production build (draft 2 lost 12% on stock)</div></div>
<div><div class="n">~3%</div><div class="l">what three kernel fusions returned against a +15–20% estimate</div></div>
<div><div class="n">0</div><div class="l">effect of multi-stream graph optimisation on either placement</div></div>
</div>

<h2>M1 recap — where a token goes</h2>
<p>The morning's kernel trace (<code>reports/2026-09-08-m1-kernel-trace.md</code>) split the 21.5 ms single-stream token on the split into 11.2 ms of matrix-vector kernels at 604 GB/s, 3.4 ms of RCCL kernels (128 allreduces at 27 µs), 5.8 ms of 1,300 small kernels and 1.1 ms idle: the die is busy 95% of the time, so launch gaps were never the lever. That ordering — allreduce, small kernels, batch-1 matvec — set the day's sequence; the results below are what each turned out to be worth.</p>

<h2>M2 — 24 and 32 slots, and 12–32 at depth <span class="verdict v-change">busy profile → 32 slots</span></h2>
<p>Production build, tp4 <code>llama-batched-bench -ntg 128</code>, f16 KV. Key <code>m2_slots_prod</code>.</p>
{m2chart}
{table(['prompt','slots','prefill t/s','decode t/s','per stream','e2e t/s'], m2['rows'], num_from=0, nd=1)}
<ul>
<li>32 slots beat 16 by 4.4% at 2K, 5.2% at 8K and 5.2% at 32K; 24 slots sit below 16 at every depth. The fork's tile table lifts the 24/32-slot decode 19% over stock against +33% on prefill, so a decode-shaped tile still has ~10% (S7).</li>
<li>16 slots hold at depth (−5% at 8K, −22% at 32K). The optimiser's busy-server scenario moved to <code>-np 32</code> (<code>launch.sh busy</code>); the team scenario stays at 16.</li>
<li>The SMC DC total peaked at 1199 W during the 32 × 32K prefill at the 200 W caps with the host at 150 W: 29 W under the envelope.</li>
</ul>

<h2>M4 — the pooled cache on b10837 <span class="verdict v-none">capacity mode</span></h2>
{table(m4['columns'][:4], [r[:4] for r in m4['rows']], num_from=1, nd=1, caption="b10837 tp4, 8 × 32K, llama-batched-bench. b10288 reference: 130 / 86 decode.")}
<p>The pool still costs 34% of decode and 46% of prefill at 8 × 32K; the upstream change fixed the server's lone-prompt path only. Nothing in the guide changes.</p>

<h2>The fork's own allreduce and whole-token graph <span class="verdict v-change">adopted with a gate</span></h2>
<p>The fork tree (mx-llama.cpp b10254, commit 751b611) carries a peer-write custom allreduce (<code>tp-allreduce.cu</code>, vLLM-style, one kernel per rank) and a whole-token HIP graph per lane — the two things the roadmap's S2 was going to build — behind environment variables that no measurement had set: <code>GGML_ENABLE_CUSTOM_AR=1</code> and, on gfx906, <code>HSA_FORCE_FINE_GRAIN_PCIE=1</code>. Key <code>fork_knobs_sweep</code>.</p>
{table(fk['columns'], fk['rows'], num_from=2, nd=1)}
<ul>
<li><b>Single stream +14%</b> (47.7 → 54.3): the kernel +10%, the whole-token graph another +3.6%. Perplexity through the custom path 5.5969, identical. The trace shows 4 <code>hipGraphLaunch</code> per token instead of 516, and <code>k_broadcast_reduce&lt;4&gt;</code> at 46 µs per allreduce including its spin-wait.</li>
<li>8 and 16 slots lose 8% because the fork's size gate (<code>ne &lt; 262144</code>) also sends their 8–16-row messages through the peer-write kernel. Disabling the gate puts prefill through the F32 two-shot kernel: −36%.</li>
<li>Fine-grained memory by itself changes nothing; serial lane dispatch costs 1.7%.</li>
</ul>
<h3>The size gate</h3>
<p><code>GGML_TP_AR_MAX_NE</code> (fusion-tree commit 25e1d46) replaces the fixed gate with an element count; one decode row is 5,120 elements. Key <code>ar_gate_sweep</code>.</p>
{agchart}
{table(ag['columns'], ag['rows'], num_from=1, nd=1, caption="The batched bench's one-slot cell is unreliable with the custom path (the whole-token graph warms up during the first batch); llama-bench tg128 is the single-stream figure.")}
<p><b>Adopted</b> for builds with the knob: <code>GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481</code>. One stream +14.5% (49.4 → 56.6), two streams +19%, four +1–3%, eight and more untouched. <code>settings/launch.sh</code> drops the settings on a binary without the knob, where the default gate would cost 8% at 8–16 slots.</p>

<h2>Kernel fusions (S3) <span class="verdict v-trade">~3%, and a lesson</span></h2>
<p>Three fusions on the production source, each with an off switch: the fused norm emits the Q8_1 copy of its output (128 quantize launches), the residual add runs inside the norm kernel (129), and the delta-net block's q/k norms and beta sigmoid are recomputed inside the GDN kernel (144). Kernels per token per die 1866 → 1466, busy 20.4 → 19.3 ms. Key <code>fusion_nq_test</code>.</p>
{table(fu['columns'], fu['rows'], num_from=2, nd=2)}
<ul>
<li><b>Correctness:</b> the q8 and add+norm fusions are bit-exact. The GDN fold's perplexity shift (5.6083) came from running inside the 2048-token prefill steps, now excluded; isolated at decode, the sigmoid part is exact and the L2-norm part diverged late in greedy decoding through its summation order — commit 797124c reproduces the standalone kernel's order, checked in the final run below.</li>
<li><b>Speed:</b> add+norm +1.6%, GDN fold +1.8%, q8 fusion 0 (−0.6%). The trace says why: the q8-emitting norm takes 10.5 µs where norm + quantize took 6.4 + 3.9. <b>The small kernels are latency-bound, not launch-bound</b> — their time is their own dependent chain, and fusing two dependent phases keeps both latencies.</li>
<li>The fold also cost 7% of prefill on tp4 (strided raw views walked token by token inside the prefill GDN kernel); the two folds are now restricted to ≤ 64 rows.</li>
</ul>
<h3>Overlap instead of fusion <span class="verdict v-none">0</span></h3>
<p>If fusion cannot remove the latency, concurrency might hide it: the fork's multi-stream graph optimisation (<code>GGML_CUDA_GRAPH_OPT</code>) was gated to single-device processes, and <code>=2</code> (commit a1462e0) allows it per split lane. Key <code>graph_opt_split</code>.</p>
{table(go['columns'], go['rows'], num_from=2, nd=2)}
<p>Nothing moves on either placement (perplexity intact). The small kernels sit on each block's single dependency chain; their 5.8 ms is intrinsic latency, recoverable neither by fusing consecutive kernels nor by stream-level concurrency. Only a different execution model — a persistent per-block kernel — would change it, and that is beyond this roadmap.</p>

<h2>Batch-1 matrix-vector variants (S6) <span class="verdict v-change">whole-block load adopted</span></h2>
<p>New one-column knobs (rows and warps per block, whole-block load at one column). Key <code>mmvq_batch1_variants</code>.</p>
{table(b1['columns'], b1['rows'], num_from=1, nd=2)}
<p>The whole-block load (nine aligned dwords and a funnel shift per 32-weight block) is +2.8% on the split and +7.1% on one die and is now the default (commit d49570b); it lost at batch 8 in the run-through, which is why it is restricted to one column. More rows per block help only one die; four warps lose.</p>

<h2>M3 — MTP on the production build <span class="verdict v-change">verify batch ≤ 16</span></h2>
<p>tp4 <code>llama-server</code>, greedy, 300 generated, N concurrent distinct prompts, two waves; the decode-only wave is shown. Key <code>mtp_prod_build</code>.</p>
{m3chart}
{table(['prompt','slots','draft','decode t/s','vs none','acceptance'], [[r[0], r[1], r[2], r[3], (f"{r[4]*100:+.0f}%" if r[2] else "—"), (f"{r[5]:.2f}" if r[5] else "")] for r in m3['rows']], num_from=0, nd=1)}
<p>On stock, 4 slots × draft 2 lost 12% at 32K (the 12-row verify batch fell into the MMQ tile); on the production build the 16-column kernel takes it, and 4 slots × draft 3 is +23% at 32K, +14% at 2K. 8 slots × draft 1 (16 rows) is neutral at 2K and +5% at 32K. Schedule for adaptive drafting (S5): slots × (draft + 1) ≤ 16. The optimiser carries these factors; the 128K and 160K profiles gained draft 1.</p>

<h2>Two tensor-split pairs on the production build <span class="verdict v-change">request traffic</span></h2>
{table(pr['columns'], pr['rows'], num_from=0, nd=1, caption="Two llama-server instances (rocm0+1, rocm2+3), dual-server-bench, 1300-token prompts / 256 generated, aggregate over the wave's wall clock. Key tp2x2_prod_server.")}
<p>Against the production four-die server at 16 slots (75.5 / 82.1 tok/s at 8 / 16 clients) the pairs lead by 20% at 8 and 22% at 16 clients, at about 3 s more first-token latency; 16 slots per pair add nothing. The decode-only bench still favours tp4 (204 at 16 slots, 213 at 32): one server reads prompts in micro-batches that stall its decoders, two servers overlap one's prefill with the other's decode. The guide keeps both: tp4 for decode-heavy work, the pairs for request traffic.</p>
<p><b>Fairness caveat (2026-09-08 evening, key <code>pair_link_sensitivity</code>):</b> on the ring as cabled each pair runs on one XGMI link (~33 GB/s per direction) while tp4 has the whole ring; a two-isolated-pairs bridge would give each pair two links. Measured bound: a pair with <i>no</i> direct link at all (the diagonal, two hops or PCIe) loses only 2.5–4.6% of prefill and nothing at decode against the one-link pairs, so a second link could add at most a few percent of prefill and first-token latency — the pairs' 20% server-level lead is the overlap of two servers' prefill and decode, not the link. Not tested physically by choice: models above 64 GB need all four dies on one fabric.</p>
{sf_section}
{gv_section}
{cp_section}
{ab_section}
{fa_section}
{ci_section}
{final_section}
<h2>What changed in the guide</h2>
<ul>
<li><code>settings/gfx906.env</code>: the custom allreduce with the four-row gate (builds with the knob); <code>settings/launch.sh</code>: <code>busy</code> profile at 32 slots, draft 1 on the 128K and 160K profiles, a guard for binaries without the gate knob.</li>
<li><code>optimize/optimize.py</code>: measured 24/32-slot factors for the production build; production-build draft factors and the 16-row verify rule.</li>
<li><code>NEXT-STEPS.md</code>: S2 closed by the fork's allreduce; S3 revised — fusion ~3%, overlap 0, the small-kernel time is intrinsic; S6 batch-1 load adopted; M2–M4 closed. <code>patches/</code>: the nine-patch series with switches and measured effects.</li>
<li>Open: M6 (mixed-wave instrumentation), M7's tenant-latency half (needs a fleet node), the two S4 kernels, the decode-shaped MMQ tile (S7); the tile-table ablation settled S1's scope (one header).</li>
</ul>
<p class="small">Generated from <code>data/benchmarks.json</code> by <code>tools/gen-0908-report.py</code>.</p>
</div>
<div class="tip" id="tip"></div>
<script>
(function(){{var tip=document.getElementById('tip');function show(e){{var t=e.target.getAttribute('data-tip');if(!t)return;tip.textContent=t;tip.style.opacity=1;var x=(e.clientX||0)+14,y=(e.clientY||0)+14;tip.style.left=x+'px';tip.style.top=y+'px';}}
document.querySelectorAll('.mark').forEach(function(m){{m.addEventListener('mousemove',show);m.addEventListener('focus',function(e){{var r=m.getBoundingClientRect();tip.textContent=m.getAttribute('data-tip');tip.style.opacity=1;tip.style.left=(r.left+r.width/2)+'px';tip.style.top=(r.top-28)+'px';}});m.addEventListener('mouseleave',function(){{tip.style.opacity=0;}});m.addEventListener('blur',function(){{tip.style.opacity=0;}});}});}})();
</script>
'''
open(f'{G}/reports/2026-09-08-next-steps-measurements.html', 'w').write(page)
print('written', len(page), 'bytes; final section:', 'yes' if fin else 'placeholder (no final_config key yet)')
