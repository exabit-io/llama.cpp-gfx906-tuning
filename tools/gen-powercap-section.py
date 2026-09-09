#!/usr/bin/env python3
"""Regenerate the power-cap section of the run-through page from the bench data files, splice it into gfx906-runthrough.html."""
import json, math, re, statistics as st, sys, time, collections
B='/root/rocm-tests/bench'; S='/tmp/claude-0/-/6424abb0-1538-4890-ab93-6cb2da934ac8/scratchpad'
def gm(v): return math.exp(st.mean(math.log(x) for x in v))
def load(p): return [json.loads(l) for l in open(p)]
study=load(f'{B}/qwen38-27b-mxxmfh-powercap-adaptive-points.jsonl'); floor=load(f'{B}/qwen38-27b-mxxmfh-powercap-floor-points.jsonl')
ab=[p for p in study if p['stage'] in ('A','B') and p['conc']==16]
caps=sorted({p['cap'] for p in ab}, reverse=True)
agg={c:dict(n=sum(1 for p in ab if p['cap']==c), th=gm([p['agg_gen'] for p in ab if p['cap']==c]), kj=gm([p['gen_per_kJ_bays'] for p in ab if p['cap']==c]),
            wb=st.mean(p['w_bays'] for p in ab if p['cap']==c), wd=st.mean(p['w_die'] for p in ab if p['cap']==c), t=st.mean(p['temp'] for p in ab if p['cap']==c), ttft=st.mean(p['ttft'] for p in ab if p['cap']==c)) for c in caps}
ref=agg[200]
fcaps=sorted({p['cap'] for p in floor}, reverse=True)
fl={c:dict(n=sum(1 for p in floor if p['cap']==c), th=gm([p['agg_gen'] for p in floor if p['cap']==c]), kj=gm([p['gen_per_kJ_bays'] for p in floor if p['cap']==c]), wb=st.mean(p['w_bays'] for p in floor if p['cap']==c), wd=st.mean(p['w_die'] for p in floor if p['cap']==c)) for c in fcaps}
val=[p for p in study if p['stage']=='V']
vt={(c,k):dict(th=gm([p['agg_gen'] for p in val if p['cap']==c and p['conc']==k]), kj=gm([p['gen_per_kJ_bays'] for p in val if p['cap']==c and p['conc']==k]), wb=st.mean(p['w_bays'] for p in val if p['cap']==c and p['conc']==k), ttft=st.mean(p['ttft'] for p in val if p['cap']==c and p['conc']==k)) for c in (70,200) for k in (8,16)}
def best_within(loss):
    ok=[c for c in caps if agg[c]['th']>=ref['th']*(1-loss)]; return max(ok, key=lambda c: agg[c]['kj'])
b2,b5,b10=best_within(.02),best_within(.05),best_within(.10)
# ---------- chart 1: two panels, shared x = cap W
X0,X1=60,600; xs=lambda w: X0+(X1-X0)*w/200
A0,A1=30,180; ya=lambda v: A1-(A1-A0)*v/100
B0,B1=250,400; yb=lambda v: B1-(B1-B0)*v/160
def pts(f, key): return ' '.join(f'{xs(c):.1f},{f(agg[c][key]):.1f}' for c in sorted(caps))
svg=[f'<svg viewBox="0 0 640 470" role="img" aria-label="Aggregate decode tokens per second and generated tokens per kilojoule against the per-die power cap, 16 clients">']
svg.append('<g font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--muted)">')
for y0,y1,ticks,f,lab in ((A0,A1,(0,25,50,75,100),ya,'agg gen tok/s'),(B0,B1,(0,40,80,120,160),yb,'gen tok/kJ (bays)')):
    svg.append(f'<rect x="{X0}" y="{y0}" width="{xs(85)-X0:.1f}" height="{y1-y0}" fill="var(--accent-soft)"/>')
    svg.append(f'<line x1="{X0}" y1="{y0}" x2="{X0}" y2="{y1}" stroke="var(--axis)"/><line x1="{X0}" y1="{y1}" x2="{X1}" y2="{y1}" stroke="var(--axis)"/>')
    for t in ticks[1:]: svg.append(f'<line x1="{X0}" y1="{f(t):.1f}" x2="{X1}" y2="{f(t):.1f}" stroke="var(--grid)"/>')
    for t in ticks: svg.append(f'<text x="{X0-8}" y="{f(t)+4:.1f}" text-anchor="end">{t}</text>')
    svg.append(f'<text x="14" y="{(y0+y1)/2:.0f}" transform="rotate(-90 14 {(y0+y1)/2:.0f})" text-anchor="middle">{lab}</text>')
    svg.append(f'<line x1="{xs(85):.1f}" y1="{y0}" x2="{xs(85):.1f}" y2="{y1}" stroke="var(--rule-2)" stroke-dasharray="3 3"/>')
for w in (0,50,100,150,200): svg.append(f'<text x="{xs(w):.1f}" y="{B1+18}" text-anchor="middle">{w}</text>')
svg.append(f'<text x="{xs(85):.1f}" y="{B1+18}" text-anchor="middle" fill="var(--ink-2)">85</text>')
svg.append(f'<text x="{(X0+X1)/2:.0f}" y="{B1+40}" text-anchor="middle" fill="var(--ink-2)">per-die power cap, W</text>')
svg.append(f'<text x="{X0+8}" y="{A0+14}" fill="var(--ink-2)">below 85 W the cap is not honoured</text>')
svg.append(f'<text x="{X0+8}" y="{B1-8}" fill="var(--ink-2)">one operating point from 85 W down</text>')
svg.append('</g>')
svg.append(f'<polyline points="{pts(ya,"th")}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>')
svg.append(f'<polyline points="{pts(yb,"kj")}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>')
svg.append('<g stroke="var(--surface)" stroke-width="2">')
for c in caps:
    a=agg[c]; svg.append(f'<circle class="mark" tabindex="0" data-t="{c} W · {a["th"]:.1f} tok/s · {a["kj"]:.0f} gen tok/kJ · {a["n"]} samples" cx="{xs(c):.1f}" cy="{ya(a["th"]):.1f}" r="4" fill="var(--series)"/>')
    svg.append(f'<circle class="mark" tabindex="0" data-t="{c} W · {a["kj"]:.1f} gen tok/kJ · {a["th"]:.1f} tok/s" cx="{xs(c):.1f}" cy="{yb(a["kj"]):.1f}" r="4" fill="var(--series)"/>')
svg.append('</g><g stroke="var(--series-2)" stroke-width="2" fill="var(--surface)">')
for c in fcaps:
    a=fl[c]; svg.append(f'<circle class="mark" tabindex="0" data-t="floor test · {c} W · {a["th"]:.1f} tok/s · {a["kj"]:.0f} gen tok/kJ" cx="{xs(c):.1f}" cy="{ya(a["th"]):.1f}" r="4.5"/>')
    svg.append(f'<circle class="mark" tabindex="0" data-t="floor test · {c} W · {a["kj"]:.1f} gen tok/kJ" cx="{xs(c):.1f}" cy="{yb(a["kj"]):.1f}" r="4.5"/>')
svg.append('</g>')
svg.append('<g font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--ink-2)">')
svg.append(f'<text x="{xs(200)-8:.1f}" y="{ya(ref["th"])-9:.1f}" text-anchor="end">{ref["th"]:.1f}</text><text x="{xs(75):.1f}" y="{ya(agg[65]["th"])-10:.1f}" text-anchor="middle">{agg[65]["th"]:.1f}</text>')
svg.append(f'<text x="{xs(200)-8:.1f}" y="{yb(ref["kj"])+16:.1f}" text-anchor="end">{ref["kj"]:.0f}</text><text x="{xs(75):.1f}" y="{yb(agg[65]["kj"])-10:.1f}" text-anchor="middle">{agg[65]["kj"]:.0f}</text>')
svg.append('</g></svg>')
chart1='\n'.join(svg)
# ---------- HBM chart + table (if the sweep has run)
hbm=''; rows=[]
try:
    for l in open(f'{B}/hbm-bw-cap-sweep.md'):
        m=re.match(r'\|\s*(\d+)\s*\|\s*([0-9a-f]{2})\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|',l)
        if m: rows.append((int(m[1]),m[2],float(m[3]),float(m[4]),m[5],m[6]))
except FileNotFoundError: pass
if len(rows)>=8:
    order=[]; per=collections.OrderedDict()
    for r in rows:
        k=(r[0],len([o for o in order if o==r[0]])//4); 
        if r[0] not in order or (order and order[-1]!=r[0]): order.append(r[0])
    # group in sweep order: cap value repeats (200 at start and end); key by position
    groups=[]; cur=None
    for r in rows:
        if cur is None or cur['cap']!=r[0] or len(cur['rows'])==4: cur=dict(cap=r[0],rows=[]); groups.append(cur)
        cur['rows'].append(r)
    for g in groups:
        g['copy']=st.mean(r[2] for r in g['rows']); g['read']=st.mean(r[3] for r in g['rows']); g['cmin']=min(r[2] for r in g['rows']); g['cmax']=max(r[2] for r in g['rows']); g['rmin']=min(r[3] for r in g['rows']); g['rmax']=max(r[3] for r in g['rows'])
        sc=[r[4].replace('Mhz','') for r in g['rows']]; g['sclk']=' / '.join(sc) if len(set(sc))>1 else sc[0]
        pw=[int(r[5].rstrip('W')) for r in g['rows'] if r[5].rstrip('W').isdigit()]; g['w']=f'{min(pw)}–{max(pw)}' if pw and min(pw)!=max(pw) else (str(pw[0]) if pw else '?')
    main=[g for g in groups][:-1] if groups[-1]['cap']==200 and len(groups)>1 else groups   # plot the descending pass; the closing 200 W is a drift check
    main=sorted(main, key=lambda g:g['cap'])
    H0,H1=30,210; yh=lambda v: H1-(H1-H0)*v/1100
    sv=[f'<svg viewBox="0 0 640 262" role="img" aria-label="HBM bandwidth in gigabytes per second against the per-die power cap: read-only and copy kernels">']
    sv.append('<g font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--muted)">')
    sv.append(f'<rect x="{X0}" y="{H0}" width="{xs(85)-X0:.1f}" height="{H1-H0}" fill="var(--accent-soft)"/>')
    sv.append(f'<line x1="{X0}" y1="{H0}" x2="{X0}" y2="{H1}" stroke="var(--axis)"/><line x1="{X0}" y1="{H1}" x2="{X1}" y2="{H1}" stroke="var(--axis)"/>')
    for t in (200,400,600,800,1000): sv.append(f'<line x1="{X0}" y1="{yh(t):.1f}" x2="{X1}" y2="{yh(t):.1f}" stroke="var(--grid)"/>')
    for t in (0,200,400,600,800,1000): sv.append(f'<text x="{X0-8}" y="{yh(t)+4:.1f}" text-anchor="end">{t}</text>')
    sv.append(f'<text x="14" y="{(H0+H1)/2:.0f}" transform="rotate(-90 14 {(H0+H1)/2:.0f})" text-anchor="middle">GB/s per die</text>')
    sv.append(f'<line x1="{X0}" y1="{yh(1024):.1f}" x2="{X1}" y2="{yh(1024):.1f}" stroke="var(--rule-2)" stroke-dasharray="4 3"/><text x="{X1}" y="{yh(1024)-5:.1f}" text-anchor="end" fill="var(--ink-2)">HBM2 peak 1024 GB/s</text>')
    sv.append(f'<line x1="{xs(85):.1f}" y1="{H0}" x2="{xs(85):.1f}" y2="{H1}" stroke="var(--rule-2)" stroke-dasharray="3 3"/>')
    for w in (0,50,100,150,200): sv.append(f'<text x="{xs(w):.1f}" y="{H1+18}" text-anchor="middle">{w}</text>')
    sv.append(f'<text x="{xs(85):.1f}" y="{H1+18}" text-anchor="middle" fill="var(--ink-2)">85</text>')
    sv.append(f'<text x="{(X0+X1)/2:.0f}" y="{H1+40}" text-anchor="middle" fill="var(--ink-2)">per-die power cap, W</text></g>')
    for key,col in (('read','var(--series)'),('copy','var(--series-2)')):
        sv.append(f'<polyline points="{" ".join(f"{xs(g["cap"]):.1f},{yh(g[key]):.1f}" for g in main)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
    sv.append('<g stroke="var(--surface)" stroke-width="2">')
    for g in main:
        sv.append(f'<circle class="mark" tabindex="0" data-t="{g["cap"]} W · read {g["read"]:.0f} GB/s · sclk {g["sclk"]}" cx="{xs(g["cap"]):.1f}" cy="{yh(g["read"]):.1f}" r="4" fill="var(--series)"/>')
        sv.append(f'<circle class="mark" tabindex="0" data-t="{g["cap"]} W · copy {g["copy"]:.0f} GB/s · sclk {g["sclk"]}" cx="{xs(g["cap"]):.1f}" cy="{yh(g["copy"]):.1f}" r="4" fill="var(--series-2)"/>')
    sv.append('</g><g font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--ink-2)">')
    top=main[-1]; low=main[0]
    sv.append(f'<text x="{xs(200)-8:.1f}" y="{yh(top["read"])-9:.1f}" text-anchor="end">read {top["read"]:.0f}</text><text x="{xs(200)-8:.1f}" y="{yh(top["copy"])+16:.1f}" text-anchor="end">copy {top["copy"]:.0f}</text>')
    sv.append(f'<text x="{xs(low["cap"])+8:.1f}" y="{yh(low["read"])-9:.1f}">{low["read"]:.0f}</text><text x="{xs(low["cap"])+8:.1f}" y="{yh(low["copy"])+16:.1f}">{low["copy"]:.0f}</text>')
    sv.append('</g></svg>')
    trows=''.join(f'<tr><td class="num">{g["cap"]}{" again" if i==len(groups)-1 and g["cap"]==200 and len(groups)>1 else ""}</td><td class="num">{g["read"]:.0f}</td><td class="num">{g["copy"]:.0f}</td><td class="num">{g["sclk"]}</td><td class="num">{g["w"]}</td></tr>' for i,g in enumerate(groups))
    hbm=f'''<h3>HBM2 bandwidth under the cap</h3>
<p>The memory clock table on this firmware has one entry, 1000 MHz, marked DPM disabled; the SoC clock is fixed at 971 MHz and the fabric clock at 1166 MHz. The cap can only move sclk. Whether that limits what the memory delivers was measured directly: a HIP probe runs a 1 GiB float4 copy and a read-only reduction on all four dies at once, eight seconds per cap, median of about 1,900 kernel timings; sclk and die power read mid-run.</p>
<figure>
<div class="legend"><span style="--sw:var(--series)">read-only kernel</span><span style="--sw:var(--series-2)">copy kernel (read + write)</span></div>
{chr(10).join(sv)}
<figcaption>Mean of the four dies, which agree within 0.5% at every cap; the descending pass is drawn, the closing 200 W point is the drift check in the table.</figcaption>
</figure>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">read GB/s</th><th class="num">copy GB/s</th><th class="num">sclk mid-run</th><th class="num">die W mid-run</th></tr>
{trows}
</table></div>
<p><b>Bandwidth does not move.</b> Read stays between {min(g['read'] for g in groups):.0f} and {max(g['read'] for g in groups):.0f} GB/s and copy between {min(g['copy'] for g in groups):.0f} and {max(g['copy'] for g in groups):.0f} GB/s at every cap, a {(max(g['read'] for g in groups)/min(g['read'] for g in groups)-1)*100:.1f}% spread on read, with the dies at 1730 MHz at 200 W and pinned at 999 MHz from 110 W down. The L2, which runs at the GFX clock, passes 1024 bytes per clock and so still clears the HBM's 1024 GB/s peak at the floor; the memory is the limit at any clock. Two numbers describe the streaming case: the cap begins to bind at about 140 W (1445 MHz) rather than the 90 W of the decode load, because a streaming kernel keeps the HBM busy and the die draws more at a given clock; and the floor under a streaming load is about 115 W per die, drawn at any cap of 110 W or below. So the minimum cap for full HBM2 bandwidth is none: any cap keeps it, and a die cannot be made to stream for less than about 115 W.</p>
'''
# ---------- tables
tro=''.join(f'<tr><td class="num{" hi" if c in (b2,b5,b10) else ""}">{c}</td><td class="num">{agg[c]["n"]}</td><td class="num">{agg[c]["th"]:.1f}</td><td class="num">{(agg[c]["th"]/ref["th"]-1)*100:+.1f}%</td><td class="num">{agg[c]["kj"]:.1f}</td><td class="num">{(agg[c]["kj"]/ref["kj"]-1)*100:+.1f}%</td><td class="num">{agg[c]["ttft"]:.2f}</td><td class="num">{agg[c]["wb"]:.0f}</td><td class="num">{agg[c]["wd"]:.0f}</td><td class="num">{agg[c]["t"]:.0f}</td></tr>' for c in caps)
rec=[('max throughput',200,'reference'),('within 2%',b2,f'+{(agg[b2]["kj"]/ref["kj"]-1)*100:.0f}% per kJ'),('within 5%',b5,f'+{(agg[b5]["kj"]/ref["kj"]-1)*100:.0f}% per kJ'),('within 10%',b10,f'+{(agg[b10]["kj"]/ref["kj"]-1)*100:.0f}% per kJ'),('max per-watt',85,f'the DPM floor: {(agg[85]["th"]/ref["th"]-1)*100:.0f}% throughput, +{(agg[85]["kj"]/ref["kj"]-1)*100:.0f}% per kJ; any lower cap gives the same point')]
trec=''.join(f'<tr><td>{g}</td><td class="num hi">{c}{" or below" if c==85 else ""}</td><td class="num">{agg[c]["th"]:.1f}</td><td class="num">{agg[c]["kj"]:.0f}</td><td>{n}</td></tr>' for g,c,n in rec)
tval=''.join(f'<tr><td class="num">{c}</td><td class="num">{k}</td><td class="num">{vt[(c,k)]["th"]:.1f}</td><td class="num">{vt[(c,k)]["kj"]:.1f}</td><td class="num">{vt[(c,k)]["ttft"]:.2f}</td><td class="num">{vt[(c,k)]["wb"]:.0f}</td></tr>' for c in (200,70) for k in (8,16))
tfl=''.join(f'<tr><td class="num">{c}</td><td class="num">{fl[c]["n"]}</td><td class="num">{fl[c]["th"]:.1f}</td><td class="num">{fl[c]["kj"]:.1f}</td><td class="num">{fl[c]["wb"]:.0f}</td><td class="num">{fl[c]["wd"]:.0f}</td><td>{"accepted, ignored" if c<85 else "reference"}</td></tr>' for c in fcaps)
# ---------- fleet: when a second node pays (DC watts; idle 244 W DC measured 2026-09-08 04:5x at perf auto; host flat out = CPU zone 303 W from the power-draw test)
IDLE=244.0; HOST_FULL=303.0; HOST_STUDY=86.0
mrows=[(c,agg[c]) for c in sorted(caps) if c>=85]
marg=[]
for (c0,a0),(c1,a1) in zip(mrows,mrows[1:]):
    if a1['th']-a0['th']>0.3: marg.append((c0,c1,a0['th'],a1['th'],(a1['wd']*0+ (a1.get('wdc',0)))))
# DC per cap from the points (w_dc), geomean throughput
dcs={c:st.mean(p['w_dc'] for p in ab if p['cap']==c) for c in caps}
marg=[(c0,c1,a0['th'],a1['th'],(dcs[c1]-dcs[c0])/(a1['th']-a0['th'])) for (c0,a0),(c1,a1) in zip(mrows,mrows[1:]) if a1['th']-a0['th']>0.3]
TF,PF=agg[85]['th'],dcs[85]; T8,P8=vt[(70,8)]['th'],st.mean(p['w_dc'] for p in val if p['cap']==70 and p['conc']==8)
tm=''.join(f'<tr><td class="num">{c0} to {c1}</td><td class="num">{t0:.1f} to {t1:.1f}</td><td class="num{" hi" if m>=15 else ""}">{m:.1f}</td></tr>' for c0,c1,t0,t1,m in marg)
def bays_pinned(c): return 1.093*max(4*c,332)+56
tenv=''.join(f'<tr><td class="num">{c}</td><td class="num">{agg[c]["wb"]:.0f}</td><td class="num">{bays_pinned(c):.0f}</td>'+''.join(f'<td class="num{" hi" if v>1178 else ""}">{v:.0f}</td>' for v in (HOST_FULL+55+18+92+agg[c]['wb'], HOST_FULL+55+18+92+bays_pinned(c), 430+55+18+92+bays_pinned(c)))+'</tr>' for c in (85,110,125,140,155,170,185,200))
fleet=f'''<h3>When a second node pays</h3>
<p>The per-node figures above do not say when to raise a cap and when to bring another node's GPUs in. That is a marginal question: the DC watts each extra token per second costs on the node already serving, against what the same tokens cost on the next node. The first column is the study's 16-client curve, differenced.</p>
<div class="tw"><table>
<tr><th class="num">cap step, W</th><th class="num">tok/s</th><th class="num">marginal DC W per tok/s</th></tr>
{tm}
</table></div>
<p>A node at the floor with 16 clients makes {TF:.1f} tok/s for {PF:.0f} W DC. The box idles at {IDLE:.0f} W DC (302 W at the wall) with the GPUs on auto, so the serving increment at the floor is {(PF-IDLE)/TF:.1f} W per tok/s, {(PF-IDLE)/TF/3.6:.1f} kWh per million generated tokens; at 8 clients it is {(P8-IDLE)/T8:.1f}. Every cap step above 95 W costs more than that.</p>
<ul>
  <li><b>Hyperconverged nodes in a three-node quorum, the case here.</b> The boxes run KVM, databases and Ceph whether or not they serve tokens, and high availability keeps at least three of them up, so capital, idle draw and bring-up are sunk on every node and only the increment counts. The serving pool is therefore three sets of four dies that are on anyway. Spread across all of them at the floor first: {3*TF:.0f} tok/s for {3*(PF-IDLE):.0f} W above idle. No node's cap should rise while any node has floor capacity; past {3*TF:.0f} tok/s raise the three caps in lockstep, and stop at 125 W ({3*agg[125]['th']:.0f} tok/s), where a cap step costs three times what a floor slot does and where the production envelope below also stops. For HA sizing, plan the demand ceiling at what two nodes deliver inside the envelope, {2*agg[155]['th']:.0f} tok/s at 155 W, so losing a node costs latency and not capacity.</li>
  <li><b>A node powered only to serve.</b> Its whole draw counts, {PF/TF:.1f} W per tok/s at the floor. One node wins up to 86 tok/s, at a cap of about 184 W; two nodes at the floor win above that, by 3% at 87 tok/s.</li>
  <li><b>A node that has to be bought.</b> One node until its ceiling, no exceptions. A node costing $3,000 a year at $0.15 per kWh is worth 2,300 W of continuous draw, more than twice this box's peak; per million generated tokens that is about $1.58 of capital against $0.40 to $0.48 of electricity.</li>
</ul>
<h3>The production envelope, with the planned hardware</h3>
<p>The study ran with the host RAPL-capped at 150 W, which was a test-bench precaution and is not available in production: a node running Ceph, databases and VMs keeps its CPU uncapped. The nodes will also carry four M.2 NVMe drives on a PLX x16-to-four-x4 switch card, two Mellanox ConnectX-4 Lx cards and two 3.5-inch SATA drives. The SMC clamps every die to 1000 MHz until a cold power cycle once the DC total passes 1228 W, so the cap has to be chosen so that no simultaneous worst case can cross it. The budget below uses the SMC's own zones as measured on this box and datasheet figures for the additions.</p>
<div class="tw"><table>
<tr><th>component</th><th class="num">idle W</th><th class="num">serving W</th><th class="num">worst W</th><th>source</th></tr>
<tr><td>CPU zone (W-3275M, six DIMMs, VRM)</td><td class="num">29</td><td class="num">86</td><td class="num">303</td><td>SMC PZ1G; 303 W is all 56 threads on openssl; the RAPL limit allows 413 W package, zone envelope 450 W</td></tr>
<tr><td>unzoned: fans, board, T2, boot SSD</td><td class="num">42</td><td class="num">53</td><td class="num">55</td><td>PZ0G minus the zones, median over the study log</td></tr>
<tr><td>PCIe slot zone as fitted today</td><td class="num">17</td><td class="num">17</td><td class="num">18</td><td>PZ5G; 300 W envelope</td></tr>
<tr><td>two ConnectX-4 Lx, with optics</td><td class="num">20</td><td class="num">20</td><td class="num">28</td><td>datasheet, about 10 W typical and 12 W max per card plus SFP28 modules</td></tr>
<tr><td>PLX PEX8747 switch card</td><td class="num">10</td><td class="num">10</td><td class="num">12</td><td>datasheet, switch about 8 W plus card regulators</td></tr>
<tr><td>four M.2 NVMe</td><td class="num">6</td><td class="num">24</td><td class="num">34</td><td>1.5 W idle, 6 W active, 8.5 W M.2 ceiling each</td></tr>
<tr><td>two 3.5-inch SATA, 7200 rpm</td><td class="num">10</td><td class="num">18</td><td class="num">18</td><td>5 W idle, 9 W active; spin-up adds about 25 W per drive for 10 s</td></tr>
<tr><td>the two MPX bays</td><td class="num">172</td><td class="num">by cap</td><td class="num">by cap</td><td>PZ3G+PZ4G; serving values measured, pinned values fitted through 419 W at the floor and 913 W with all dies at 200 W</td></tr>
</table></div>
<p>Everything except the bays comes to {HOST_STUDY+55+17+72:.0f} W under serving with the host at its study level, {HOST_FULL+55+18+92:.0f} W with the host all-core and every addition at its maximum, and {430+55+18+92:.0f} W if the host reaches its RAPL ceiling. The table puts those against the bays at each cap; the pinned column is a prefill-heavy load holding every die at the cap, the serving column the study's measured average.</p>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">bays serving</th><th class="num">bays pinned</th><th class="num">host 303 W, serving</th><th class="num">host 303 W, pinned</th><th class="num">host at RAPL ceiling, pinned</th></tr>
{tenv}
</table></div>
<p class="cap">Bold cells are within 50 W of the 1228 W clamp or over it. Drive spin-up, about 50 W for ten seconds with both drives at once, comes out of the same margin.</p>
<p><b>Production cap: 125 W per die.</b> It is where the economics stop, and it is also the highest cap that stays inside the envelope with the host at its RAPL ceiling and every addition at its maximum: {1228-(430+55+18+92)-(1.093*500+56):.0f} W of margin pinned, {1228-(HOST_FULL+55+18+92)-agg[125]['wb']:.0f} W under serving with the host all-core. At 125 W a node gives {agg[125]['th']:.1f} tok/s, the three-node pool {3*agg[125]['th']:.0f}. Anything above 140 W depends on the host never running all-core while the dies are pinned, which a hyperconverged node cannot promise. The additions change the economics not at all: their 46 to 92 W is paid for Ceph and the VMs whether or not the dies serve, and the per-token increment stays 5.5 W per tok/s at the floor. What would restore headroom is a small governor that reads the DC total each second and lowers the four caps when it passes about 1150 W; the SMC clamps about 20 s after the envelope is crossed, so there is time to act. Not built yet.</p>
'''
section=f'''<h2>Power cap: throughput, energy, and where the cap stops working <span class="verdict v-trade">a trade-off, not one number</span></h2>
<p>The morning sweep, stock build at 8 clients with sampler-window power, put serving at 175 W. It could not resolve anything finer: its per-watt column moved 10% between two runs at the same cap. The evening study measures energy instead of power, on the production build, with the cap switched live under one four-die server at 16 slots. A point is one wave of 32 requests (1300 in, 256 out) at 16 clients; the sampler integrates die power (hwmon) and bay power (the SMC's two MPX zones) at 1 s over the wave, and the figure is generated tokens per kilojoule at the bays. The cap walked 200 to 80 W in 15 W steps down and back up, then 65 to 95 W in 5 W steps four times; down and up samples at the same cap agree within 0.3%, so the noise the morning saw is gone.</p>
<figure>
<div class="legend"><span style="--sw:var(--series)">cap study, 16 clients, 2 to 10 samples per cap</span><span style="--sw:var(--series-2)">floor test, a second run down to 10 W</span></div>
{chart1}
<figcaption>Every sample from both ladders, geometric mean per cap. The shaded band is where the cap has no effect. Per-watt keeps rising as the cap falls until the dies hit their clock floor; there is no interior optimum.</figcaption>
</figure>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">samples</th><th class="num">agg gen tok/s</th><th class="num">vs 200 W</th><th class="num">gen tok/kJ bays</th><th class="num">vs 200 W</th><th class="num">first token s</th><th class="num">bays W</th><th class="num">dies W</th><th class="num">die °C</th></tr>
{tro}
</table></div>
<p><b>Where the cap stops working.</b> From 85 W down every sample is the same point: 60.0 to 60.6 tok/s, 411 to 419 W at the bays, 331 to 343 W at the dies, 146 to 148 gen tok/kJ. The clock trace shows the dies at sclk level 0, 999 to 1000 MHz, the firmware's minimum GFX clock, for most samples; with the memory, SoC and fabric clocks fixed on this firmware, the SMU has no lower state to reach. A die at that clock draws about 83 W under this load and 34 W idle, so a cap below 85 W is accepted by the driver and then simply not met. The controller's "70 W winner" is the plateau picking a label by noise, four pairs, lead 0.1% against a 0.3% spread; the cap starts to bind at about 85 to 90 W, where the dies begin to mix in 1204 MHz.</p>
<h3>Recommendation</h3>
<div class="tw"><table>
<tr><th>goal</th><th class="num">cap per die</th><th class="num">agg gen tok/s</th><th class="num">gen tok/kJ</th><th>note</th></tr>
{trec}
</table></div>
<p>The morning's 175 W sits between the 2% and 5% rows and remains a fine serving default; the machine's 1228 W envelope is met at any of them. The 4-bit values in the table are geometric means; the recommendation rows are the best per-kJ cap whose throughput clears the floor.</p>
<h3>At 8 clients, concurrency beats the cap</h3>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">clients</th><th class="num">agg gen tok/s</th><th class="num">gen tok/kJ bays</th><th class="num">first token s</th><th class="num">bays W</th></tr>
{tval}
</table></div>
<p>Two samples each, alternating. Dropping to the floor at 8 clients buys +32% per kJ for −26% throughput, but 200 W at 16 clients (105 gen tok/kJ) is about as efficient as the floor at 8 (110). Filling slots moves energy per token more than the cap does; lower the cap after the server is full, not instead.</p>
<h3>Caps below idle</h3>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">samples</th><th class="num">agg gen tok/s</th><th class="num">gen tok/kJ bays</th><th class="num">bays W</th><th class="num">dies W</th><th>driver</th></tr>
{tfl}
</table></div>
<p>A second run with the ladder 200, 65, 40, 20, 10 W, down and up. The hwmon minimum for the cap reads 0, and every value is accepted and read back; below the floor none of them changes anything, 10 W included. The closing 200 W point matched the opening one, so nothing drifted.</p>
{hbm}{fleet}<h3>The morning sweep, for the record</h3>
<div class="tw"><table>
<tr><th class="num">cap W</th><th class="num">one die pp2048</th><th class="num">tp4 tg256</th><th class="num">server, 8 clients</th><th class="num">four dies W</th><th class="num">bays W (SMC)</th></tr>
<tr><td class="num">200</td><td class="num">232.9</td><td class="num">47.4</td><td class="num">65.7</td><td class="num">614</td><td class="num">761</td></tr>
<tr><td class="num">185</td><td class="num">225.9</td><td class="num">47.1</td><td class="num">65.5</td><td class="num">655</td><td class="num">778</td></tr>
<tr><td class="num">175</td><td class="num">222.3</td><td class="num">47.2</td><td class="num">64.5</td><td class="num">551</td><td class="num">736</td></tr>
<tr><td class="num">160</td><td class="num">215.1</td><td class="num">46.5</td><td class="num">63.0</td><td class="num">532</td><td class="num">704</td></tr>
<tr><td class="num">150</td><td class="num">209.6</td><td class="num">45.8</td><td class="num">61.6</td><td class="num">531</td><td class="num">687</td></tr>
<tr><td class="num">200 again</td><td class="num">226.7</td><td class="num">47.1</td><td class="num">65.7</td><td class="num">684</td><td class="num">747</td></tr>
</table></div>
<p class="cap">Stock build, 8-slot server, power averaged over the server phase only. Single-stream decode never reaches the cap, so on the stock build at 8 clients only prefill paid for a lower cap; the energy study above supersedes the per-watt reading.</p>

'''
html=open('/root/llama.cpp-benchmarking/reports/2026-09-07-todo-runthrough-1900.html').read()   # always start from the kept 19:00 original so reruns are idempotent
a=html.index('<h2>Power cap versus throughput'); b=html.index('<h2>Still running, and what comes after')
html=html[:a]+section+html[b:]
now=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
html=html.replace('2026-09-07, updated 19:00 UTC', f'2026-09-07, updated {now}')
old_claim='And the "clock clamp" is the Mac Pro\'s 1228 W DC envelope, cleared only by a cold power cycle; a 175 W per-die cap keeps serving within 2% and the machine inside it.'
new_claim=f'And the "clock clamp" is the Mac Pro\'s 1228 W DC envelope, cleared only by a cold power cycle. An energy-integrated cap study on the production build turns the cap into a trade-off table: {b2} W keeps 16-client serving within 2%, {b5} W within 5%, {b10} W within 10%, and below 85 W the cap does nothing at all, because the dies are already at their 1000 MHz floor drawing 83 W each whatever number is written. On the hyperconverged three-node fleet, where the host runs uncapped beside NVMe, NICs and SATA drives, the marginal economics and the 1228 W envelope both stop at 125 W per die: spread serving across all three nodes at the floor first, raise caps in lockstep to 125 W, never higher.'
assert html.count(old_claim)==1; html=html.replace(old_claim,new_claim)
a=html.index('<h2>Still running, and what comes after'); b=html.index('<p class="small">Sources:')
standing=f'''<h2>Where things stand</h2>
<ul>
  <li><b>Eleven queues finished by 04:35 UTC on 2026-09-08</b>: the six tuning queues, the server-level check, two cap sweeps, the floor test and the bandwidth sweep. Peak DC draw stayed under 1130 W against the 1228 W envelope with the host capped at 150 W throughout.</li>
  <li><b>Serving</b>: production build, four-die split at 16 slots. On a dedicated node the cap from the trade-off table: {b2} W within 2%, {b5} W within 5%, {b10} W within 10%. On the hyperconverged three-node fleet: all nodes at the floor first, then caps in lockstep to 125 W, the limit both the marginal economics and the production envelope set. Left to measure: the 8 × 32K pooled-cache decode on b10837, and what serving costs the host's other tenants.</li>
  <li><b>Code</b>: the kernel work is closed for the day. Both patches are in the guide's patches folder and apply to upstream b10288 and to the fork; the aligned-load, whole-block-load and LDS-staging variants are measured losers and documented. Larger ideas remain: packed-fp16 attention at head size 256, and the Q8_1 activation-scale handling inside the fast path.</li>
  <li><b>Tools added</b>: an energy-integrated cap search with down/up pairs and a stop rule (<code>powercap-adaptive.py</code>), and a HIP bandwidth probe (<code>hbm-bw</code>), both in the bench folder.</li>
</ul>
'''
html=html[:a]+standing+html[b:]
old_src='<p class="small">Sources: <code>reports/2026-09-07-todo-runthrough.md</code> and <code>data/benchmarks.json</code> in the tuning-guide folder; raw outputs and scripts in <code>/root/rocm-tests/bench</code>.'
new_src='<p class="small">Sources: <code>reports/2026-09-07-todo-runthrough.md</code> and <code>data/benchmarks.json</code> in the tuning-guide folder; raw outputs and scripts in <code>/root/rocm-tests/bench</code>, where the cap study is <code>qwen38-27b-mxxmfh-powercap-adaptive.md</code> with its points file, the floor test <code>qwen38-27b-mxxmfh-powercap-floor.md</code> and the bandwidth sweep <code>hbm-bw-cap-sweep.md</code>.'
assert html.count(old_src)==1; html=html.replace(old_src,new_src)
open(f'{S}/gfx906-runthrough.html','w').write(html)
print('caps',caps,'b2/b5/b10',b2,b5,b10,'hbm rows',len(rows),'groups',len(rows)//4, 'bytes',len(html))
