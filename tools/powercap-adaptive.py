#!/usr/bin/env python3
"""powercap-adaptive.py URL TAG -- adaptive per-die power-cap search for the best serving performance per watt (2026-09-07, user request).

Method. One llama-server stays up; the cap is switched live (power1_cap on all four dies, 10 s settle). A point is one client wave
of fixed work (2*conc requests of ~1300 prompt tokens + 256 generated, deterministic prompts) at one cap; its energy is the 1 s
sampler's integral of die power (hwmon power1_input) and of the SMC's two MPX-bay readings (PZ3G+PZ4G) over the wave, so the metric
is tokens per kilojoule of a fixed workload, not a mean of power snapshots. Every cap is visited in down-then-up passes (thermal drift
cancels across the pair), and a cap's value is the geometric mean of its samples.
Stage A: the coarse ladder (default 200,185,170,155,140) for --pairs-coarse pass pairs. Stage B: a --fine-step ladder over
+/- --fine-span around the coarse winner, pass pairs until the leader's margin over the runner-up exceeds the pooled between-sample
spread, or --pairs-fine-max pairs. Then a validation block at the winner and at 200 W at 8 and 16 clients.
Primary metric: generated tokens per kJ of bay energy at --conc clients. Output: TAG.md, TAG-points.jsonl, TAG-power1s.csv, TAG.progress."""
import argparse, json, math, os, random, statistics as st, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
B='/root/rocm-tests/bench'
ap=argparse.ArgumentParser(); ap.add_argument('url'); ap.add_argument('tag')
ap.add_argument('--coarse', default='200,185,170,155,140'); ap.add_argument('--pairs-coarse', type=int, default=2)
ap.add_argument('--fine-step', type=int, default=5); ap.add_argument('--fine-span', type=int, default=15); ap.add_argument('--pairs-fine-max', type=int, default=4); ap.add_argument('--pairs-fine-min', type=int, default=2)
ap.add_argument('--conc', type=int, default=16); ap.add_argument('--prompt-tokens', type=int, default=1300); ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--no-fine', action='store_true', help='stop after the coarse ladder (Stage A table only)'); ap.add_argument('--settle', type=float, default=10.0); ap.add_argument('--hard-floor', type=int, default=60, help='lowest cap the fine stage may step to (W)'); ap.add_argument('--selftest', action='store_true')
a=ap.parse_args()
DIES=['0b','0e','1b','1e']
def hw(d, name):
    import glob; return glob.glob(f'/sys/bus/pci/devices/0000:{d}:00.0/hwmon/hwmon*/{name}')[0]
PROG=f'{B}/{a.tag}.progress'; PTS=f'{B}/{a.tag}-points.jsonl'; PW=f'{B}/{a.tag}-power1s.csv'; OUT=f'{B}/{a.tag}.md'
def log(*x):
    with open(PROG,'a') as f: f.write(time.strftime('%FT%T%z')+' '+' '.join(str(v) for v in x)+'\n')
# ---- SMC keys (same interface as smc-read.py)
S='/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00'; MAP=f'{B}/smc-keymap.json'
import struct
_smcmap=json.load(open(MAP)) if os.path.exists(MAP) else {}
def smc(key):
    try:
        i=_smcmap[key]; open(f'{S}/key_at_index','w').write(str(i))
        if open(f'{S}/key_at_index_name').read().strip()!=key: return None
        t=open(f'{S}/key_at_index_type').read().strip(); d=open(f'{S}/key_at_index_data','rb').read()
        return struct.unpack('<f',d)[0] if t=='flt' and len(d)==4 else None
    except Exception: return None
# ---- 1 s sampler
samples=[]  # (t, [w0..w3] W, [temp0..3] C, dc W, bays W)
stop=threading.Event()
def sampler():
    with open(PW,'w') as f:
        f.write('t,w0b,w0e,w1b,w1e,t0b,t0e,t1b,t1e,dc,bays\n')
        while not stop.is_set():
            t=time.time()
            try:
                w=[int(open(hw(d,'power1_input')).read())/1e6 for d in DIES]; tc=[int(open(hw(d,'temp2_input')).read())/1000 for d in DIES]
            except Exception: w=[float('nan')]*4; tc=[float('nan')]*4
            dc=smc('PZ0G'); b3=smc('PZ3G'); b4=smc('PZ4G'); bays=(b3+b4) if (b3 is not None and b4 is not None) else float('nan')
            samples.append((t,w,tc,dc if dc is not None else float('nan'),bays)); f.write(f'{t:.1f},'+','.join(f'{x:.1f}' for x in w+tc)+f',{samples[-1][3]:.1f},{bays:.1f}\n'); f.flush()
            time.sleep(max(0.0, 1.0-(time.time()-t)))
def energy(t0,t1):
    """integrate over samples in [t0,t1]: returns dict of joules (die total, bays, dc), mean W, mean temp, n"""
    s=[x for x in samples if t0<=x[0]<=t1]
    if len(s)<2: return None
    Ed=Eb=Ec=0.0
    for (ta,wa,_,da,ba),(tb,wb,_,db,bb) in zip(s,s[1:]):
        dt=tb-ta; Ed+=dt*(sum(wa)+sum(wb))/2; Eb+=dt*(ba+bb)/2; Ec+=dt*(da+db)/2
    T=s[-1][0]-s[0][0]
    return dict(n=len(s), span=T, E_die=Ed, E_bays=Eb, E_dc=Ec, w_die=Ed/T, w_bays=Eb/T, w_dc=Ec/T,
                temp=st.mean(st.mean(x[2]) for x in s), temp_max=max(max(x[2]) for x in s), w_die_max=max(max(x[1]) for x in s))
# ---- caps
def capfile(d): return hw(d,'power1_cap')
def set_cap(w):
    for d in DIES: open(capfile(d),'w').write(str(int(w)*1000000))
    time.sleep(0.5); got=[int(open(capfile(d)).read())//1000000 for d in DIES]
    if any(g!=w for g in got): log('CAP MISMATCH', w, got)
    return got
def idle_check():
    """6 s with no requests at perf level high: every die must read >= 1700 MHz (the five-second clamp test); True = OK"""
    time.sleep(6); bad=[]
    for d in DIES:
        m=[l for l in open(f'/sys/bus/pci/devices/0000:{d}:00.0/pp_dpm_sclk') if '*' in l][0].split()[1]
        if int(''.join(c for c in m if c.isdigit()))<1700: bad.append(f'{d}:{m}')
    log('idle check', 'OK' if not bad else 'CLAMPED '+' '.join(bad)); return not bad
# ---- client (same request shape as server-bench.py)
def post(path, body, timeout=1800):
    req=urllib.request.Request(a.url+path, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
words="the quick brown fox jumps over lazy dog alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
def make_prompt(seed):
    rnd=random.Random(seed); txt=' '.join(rnd.choice(words) for _ in range(a.prompt_tokens*2))
    toks=post('/tokenize',{'content':txt})['tokens'][:a.prompt_tokens]; return post('/detokenize',{'tokens':toks})['content']
prompts=[]
def one(i):
    t0=time.perf_counter(); r=post('/completion',{'prompt':prompts[i%len(prompts)],'n_predict':a.gen,'ignore_eos':True,'cache_prompt':False,'temperature':0.0,'seed':i})
    t=r['timings']; return dict(wall=time.perf_counter()-t0, pn=t['prompt_n'], pms=t['prompt_ms'], gn=t['predicted_n'], gms=t['predicted_ms'])
def wave(conc):
    n=2*conc; t0=time.time(); tp=time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex: res=list(ex.map(one, range(n)))
    wall=time.perf_counter()-tp; t1=time.time(); gn=sum(r['gn'] for r in res); pn=sum(r['pn'] for r in res)
    e=energy(t0,t1) or {}
    row=dict(conc=conc, reqs=n, wall=wall, gn=gn, pn=pn, agg_gen=gn/wall, agg_total=(pn+gn)/wall, req_gen=st.mean(r['gn']/(r['gms']/1000) for r in res),
             ttft=st.mean(r['pms']/1000 for r in res), t0=t0, t1=t1, **e)
    if e:
        row['gen_per_kJ_bays']=gn/e['E_bays']*1000; row['gen_per_kJ_die']=gn/e['E_die']*1000; row['gen_per_kJ_dc']=gn/e['E_dc']*1000
        row['tok_per_kJ_bays']=(pn+gn)/e['E_bays']*1000
    return row
points=[]
def point(cap, conc, stage, pair, direction):
    set_cap(cap); time.sleep(a.settle)
    r=wave(conc); r.update(cap=cap, stage=stage, pair=pair, dir=direction, idx=len(points)); points.append(r)
    with open(PTS,'a') as f: f.write(json.dumps(r)+'\n')
    log(f'POINT {stage} pair{pair} {direction} cap{cap} conc{conc}: agg_gen {r["agg_gen"]:.1f} t/s, ttft {r["ttft"]:.2f} s, bays {r.get("w_bays",float("nan")):.0f} W, dies {r.get("w_die",float("nan")):.0f} W, temp {r.get("temp",float("nan")):.0f} C, gen/kJ bays {r.get("gen_per_kJ_bays",float("nan")):.1f}, die {r.get("gen_per_kJ_die",float("nan")):.1f}')
    return r
def gmean(v): return math.exp(st.mean(math.log(x) for x in v))
def summarize(stage, key='gen_per_kJ_bays'):
    """per cap: n, geomean of key, log-sd, geomean agg_gen, mean temp; plus pooled between-sample log-sd"""
    caps=sorted({p['cap'] for p in points if p['stage']==stage}, reverse=True); tab={}; resid=[]
    for c in caps:
        v=[p[key] for p in points if p['stage']==stage and p['cap']==c and key in p]
        g=[p['agg_gen'] for p in points if p['stage']==stage and p['cap']==c]
        tt=[p.get('temp',float('nan')) for p in points if p['stage']==stage and p['cap']==c]
        wb=[p.get('w_bays',float('nan')) for p in points if p['stage']==stage and p['cap']==c]
        mu=st.mean(math.log(x) for x in v); resid+=[math.log(x)-mu for x in v]
        tab[c]=dict(n=len(v), gm=gmean(v), sd=(st.pstdev([math.log(x) for x in v]) if len(v)>1 else float('nan')), agg_gen=gmean(g), temp=st.mean(tt), w_bays=st.mean(wb), ttft=st.mean(p['ttft'] for p in points if p['stage']==stage and p['cap']==c))
    pooled=math.sqrt(sum(r*r for r in resid)/max(1,len(resid)-len(caps))) if len(resid)>len(caps) else float('nan')
    return tab, pooled
def table(stage, key='gen_per_kJ_bays'):
    tab,pooled=summarize(stage,key); ref=tab.get(200)
    lines=[f'| cap W | samples | gen tok/kJ bays (geomean) | +/- (log-sd) | vs best | agg gen tok/s | vs 200 W | TTFT s | bays W | die temp C |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    best=max(tab.values(), key=lambda r:r['gm'])['gm']
    for c,r in tab.items():
        lines.append(f"| {c} | {r['n']} | {r['gm']:.1f} | {r['sd']*100 if r['sd']==r['sd'] else float('nan'):.1f}% | {(r['gm']/best-1)*100:+.1f}% | {r['agg_gen']:.1f} | {((r['agg_gen']/ref['agg_gen']-1)*100 if ref else float('nan')):+.1f}% | {r['ttft']:.2f} | {r['w_bays']:.0f} | {r['temp']:.0f} |")
    lines.append(f'\nPooled between-sample spread (log-sd): {pooled*100:.1f}%' if pooled==pooled else '')
    return '\n'.join(lines), tab, pooled
def best_cap(tab): return max(tab.items(), key=lambda kv: kv[1]['gm'])[0]
if a.selftest:
    # synthetic check of energy(), gmean, summarize and the stop rule
    t=1000.0
    for i in range(11): samples.append((t+i,[100,100,100,100],[60,60,60,60],500.0,450.0))
    e=energy(t,t+10); assert abs(e['E_die']-4000)<1e-6 and abs(e['E_bays']-4500)<1e-6 and e['n']==11, e
    points.extend([dict(stage='A',cap=200,gen_per_kJ_bays=100,agg_gen=80,temp=60,w_bays=700,ttft=5),dict(stage='A',cap=200,gen_per_kJ_bays=110,agg_gen=81,temp=65,w_bays=710,ttft=5),
                   dict(stage='A',cap=170,gen_per_kJ_bays=120,agg_gen=78,temp=60,w_bays=600,ttft=5),dict(stage='A',cap=170,gen_per_kJ_bays=126,agg_gen=79,temp=64,w_bays=610,ttft=5)])
    txt,tab,pooled=table('A'); print(txt); assert best_cap(tab)==170 and abs(tab[200]['gm']-math.sqrt(100*110))<1e-9; print('selftest OK'); sys.exit(0)
# ---- run
open(PROG,'w').close(); open(PTS,'w').close()
th=threading.Thread(target=sampler, daemon=True); th.start(); time.sleep(2)
prompts=[make_prompt(s) for s in range(8)]; one(0)
set_cap(200); time.sleep(a.settle); w=wave(a.conc); log(f'WARMUP wave at 200 W: agg_gen {w["agg_gen"]:.1f} t/s (not recorded)')
coarse=[int(x) for x in a.coarse.split(',')]; lo,hi=min(coarse),max(coarse)
log(f'START coarse {coarse} pairs {a.pairs_coarse}; fine step {a.fine_step} span {a.fine_span} pairs {a.pairs_fine_min}-{a.pairs_fine_max}; conc {a.conc}')
with open(OUT,'w') as f: f.write(f'# {a.tag}  {time.strftime("%FT%T")}  adaptive power-cap search, production build, llama-server tp4 -np 16; waves of {2*a.conc} x ({a.prompt_tokens} in / {a.gen} out) at {a.conc} clients; energy from the 1 s sampler (dies: hwmon power1_input; bays: SMC PZ3G+PZ4G)\n')
def check_clamp(r, ref):
    if ref and r['cap']>=170 and r['agg_gen']<0.8*ref:
        log('SUSPECT CLAMP: throughput fell below 80% of the 200 W reference at a cap >= 170 W; idle check')
        if not idle_check(): return False
    return True
ok=True; ref=None
for pair in range(1, a.pairs_coarse+1):
    for direction,caps in (('down',sorted(coarse,reverse=True)),('up',sorted(coarse))):
        for c in caps:
            r=point(c,a.conc,'A',pair,direction)
            if c==200 and ref is None: ref=r['agg_gen']
            if not check_clamp(r,ref): ok=False; break
        if not ok: break
    if not ok: break
if ok:
    txt,tab,pooled=table('A'); cA=best_cap(tab)
    with open(OUT,'a') as f: f.write(f'\n## Stage A: coarse ladder {coarse}, {a.pairs_coarse} down/up pass pairs, {a.conc} clients\n\n{txt}\n\nCoarse winner: {cA} W\n')
    log(f'STAGE A done: winner {cA} W; pooled spread {pooled*100:.1f}%')
    if a.no_fine: log('ALLDONE (no fine stage)'); set_cap(200); stop.set(); sys.exit(0)
    fine=[c for c in range(cA-a.fine_span, cA+a.fine_span+1, a.fine_step) if a.hard_floor<=c<=hi]   # may step below the coarse floor
    log(f'STAGE B ladder {fine}')
    verdict=''
    for pair in range(1, a.pairs_fine_max+1):
        for direction,caps in (('down',sorted(fine,reverse=True)),('up',sorted(fine))):
            for c in caps:
                r=point(c,a.conc,'B',pair,direction)
                if not check_clamp(r,ref): ok=False; break
            if not ok: break
        if not ok: break
        txt,tab,pooled=table('B'); ranked=sorted(tab.items(), key=lambda kv: kv[1]['gm'], reverse=True)
        lead=math.log(ranked[0][1]['gm'])-math.log(ranked[1][1]['gm']) if len(ranked)>1 else float('inf')
        log(f'STAGE B pair {pair}: leader {ranked[0][0]} W ({ranked[0][1]["gm"]:.1f}), runner-up {ranked[1][0]} W ({ranked[1][1]["gm"]:.1f}), lead {lead*100:.1f}% vs pooled spread {pooled*100:.1f}%')
        if pair>=a.pairs_fine_min and pooled==pooled and lead>pooled:
            verdict=f'stopped after pair {pair}: lead {lead*100:.1f}% > pooled spread {pooled*100:.1f}%'; break
    if ok:
        if not verdict: verdict=f'stopped at the pair cap ({a.pairs_fine_max}): lead {lead*100:.1f}% vs pooled spread {pooled*100:.1f}%'
        txt,tab,pooled=table('B'); cB=best_cap(tab)
        with open(OUT,'a') as f: f.write(f'\n## Stage B: fine ladder {fine} ({a.fine_step} W steps), down/up pass pairs, {a.conc} clients\n\n{txt}\n\n{verdict}. Fine winner: {cB} W\n')
        log(f'STAGE B done: winner {cB} W; {verdict}')
        # validation block (2026-09-08: throughput dict renamed thr; it shadowed the sampler thread th and broke the final th.join): winner and 200 W, 8 and 16 clients, twice each, alternating
        for rep in (1,2):
            for c in (cB,200):
                for conc in (8,16): point(c,conc,'V',rep,f'c{conc}')
        with open(OUT,'a') as f:
            f.write(f'\n## Validation: {cB} W vs 200 W at 8 and 16 clients, two samples each\n\n| cap W | clients | samples | gen tok/kJ bays | gen tok/kJ dies | agg gen tok/s | total tok/s | TTFT s | bays W | dies W | die temp C |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
            for c in (cB,200):
                for conc in (8,16):
                    v=[p for p in points if p['stage']=='V' and p['cap']==c and p['conc']==conc]
                    f.write(f"| {c} | {conc} | {len(v)} | {gmean([p['gen_per_kJ_bays'] for p in v]):.1f} | {gmean([p['gen_per_kJ_die'] for p in v]):.1f} | {gmean([p['agg_gen'] for p in v]):.1f} | {gmean([p['agg_total'] for p in v]):.0f} | {st.mean(p['ttft'] for p in v):.2f} | {st.mean(p['w_bays'] for p in v):.0f} | {st.mean(p['w_die'] for p in v):.0f} | {st.mean(p['temp'] for p in v):.0f} |\n")
            allpts=[p for p in points if p['stage'] in ('A','B') and p['conc']==a.conc]; caps=sorted({p['cap'] for p in allpts}, reverse=True)
            g={c:gmean([p['gen_per_kJ_bays'] for p in allpts if p['cap']==c]) for c in caps}; thr={c:gmean([p['agg_gen'] for p in allpts if p['cap']==c]) for c in caps}
            ref200=thr.get(200, max(thr.values())); f.write('\n## Trade-off over every stage A/B sample: cap, gen tok/kJ (bays), throughput vs 200 W\n\n| cap W | samples | gen tok/kJ bays | vs 200 W per-kJ | agg gen tok/s | vs 200 W |\n|---:|---:|---:|---:|---:|---:|\n')
            for c in caps: f.write(f"| {c} | {sum(1 for p in allpts if p['cap']==c)} | {g[c]:.1f} | {(g[c]/g.get(200,g[c])-1)*100:+.1f}% | {thr[c]:.1f} | {(thr[c]/ref200-1)*100:+.1f}% |\n")
            def best_within(loss):
                ok=[c for c in caps if thr[c]>=ref200*(1-loss)]; return max(ok, key=lambda c:g[c]) if ok else None
            f.write(f'\nVerdict: best serving performance per watt at {cB} W per die ({a.conc} clients, production build), throughput {(thr.get(cB,float("nan"))/ref200-1)*100:+.1f}% vs 200 W. Best cap within a throughput floor: -2%: {best_within(0.02)} W; -5%: {best_within(0.05)} W; -10%: {best_within(0.10)} W.\n')
        log(f'ALLDONE winner {cB} W')
set_cap(200); stop.set(); th.join(timeout=3)
with open(OUT,'a') as f: f.write(f'\n# {"done" if ok else "ABORTED (clamp)"} {time.strftime("%FT%T")}\n')
sys.exit(0 if ok else 2)
