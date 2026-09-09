#!/usr/bin/env python3
"""powercap-analyze.py TAG SMCLOG [--json]: per-cap-point table for a powercap-sweep run.
Slices <TAG>-clocks.txt (per-die W, 5 s) and the smc log (PZ0G DC, PZ3G+PZ4G bays) by the PHASE timestamps in <TAG>.progress,
and reads llama-bench / server-bench rows from <TAG>.md."""
import sys, re, json, statistics as st
B='/root/rocm-tests/bench'; tag=sys.argv[1]; smclog=sys.argv[2]; asjson='--json' in sys.argv
def hms(s): h,m,sec=s.split(':'); return int(h)*3600+int(m)*60+int(sec)
prog=[l.rstrip('\n') for l in open(f'{B}/{tag}.progress') if l.strip()]
win={}   # P -> {'bench':(t0,t1), 'srv':(t0,t1)}
for l in prog:
    m=re.match(r'\S+T(\d\d:\d\d:\d\d)\S* PHASE (\S+) (bench start|bench end|server bench start|server end)', l)
    if not m: continue
    t,P,ev=hms(m.group(1)),m.group(2),m.group(3); w=win.setdefault(P,{})
    if ev=='bench start': w['b0']=t
    elif ev=='bench end': w['b1']=t
    elif ev=='server bench start': w['s0']=t
    elif ev=='server end': w['s1']=t
clk=[]  # (t, [w per die])
for l in open(f'{B}/{tag}-clocks.txt'):
    p=l.split()
    if len(p)<5 or not re.match(r'\d\d:\d\d:\d\d',p[0]): continue
    ws=[]
    for f in p[1:5]:
        m=re.search(r'/(\d+)W',f); ws.append(int(m.group(1)) if m else None)
    if None not in ws: clk.append((hms(p[0]),ws))
smc=[]  # (t, dc, bays)
for l in open(smclog):
    if l.startswith('#'): continue
    p=l.split(); d={k:float(v) for k,v in (x.split('=') for x in p[1:] if '=' in x)}
    if 'PZ0G' in d and 'PZ3G' in d: smc.append((hms(p[0]),d['PZ0G'],d['PZ3G']+d['PZ4G']))
def slice_(seq,t0,t1,key): return [key(x) for x in seq if t0<=x[0]<=t1]
md=open(f'{B}/{tag}.md').read()
rows=[]
for P,w in win.items():
    cap=int(re.match(r'cap(\d+)-(\d+)',P).group(1)); idx=int(re.match(r'cap(\d+)-(\d+)',P).group(2))
    r={'cap':cap,'idx':idx}
    sec=md[md.index(f'## {P}: llama-bench'):]; nxt=re.search(r'\n## cap\d+-\d+: llama-bench', sec[10:]); sec=sec if not nxt else sec[:nxt.start()+10]
    lb=re.findall(r'\|\s*(ROCm0(?:/ROCm1/ROCm2/ROCm3)?)\s*\|\s*(pp2048|tg256)\s*\|\s*([\d.]+)', sec)
    for dev,test,v in lb: r[('rocm0_' if dev=='ROCm0' else 'tp4_')+test]=float(v)
    for m in re.finditer(r'^\| (\d+) \| \d+ \| [\d.]+ \| ([\d.]+) \| (\d+) \| [\d.]+ \| ([\d.]+) \| ([\d.]+) \|', sec, re.M):
        c=m.group(1); r[f'srv_c{c}_agg']=float(m.group(2)); r[f'srv_c{c}_total']=int(m.group(3)); r[f'srv_c{c}_perreq']=float(m.group(4)); r[f'srv_c{c}_ttft']=float(m.group(5))
    if 'b0' in w and 'b1' in w:
        s=slice_(clk,w['b0'],w['b1'],lambda x:x[1]); r['n_bench']=len(s)
        if s: r['w4_bench']=round(st.mean(sum(x) for x in s),1); r['w0_bench']=round(st.mean(x[0] for x in s),1)
    if 's0' in w and 's1' in w:
        s=slice_(clk,w['s0'],w['s1'],lambda x:x[1]); r['n_srv']=len(s)
        if s: r['w4_srv']=round(st.mean(sum(x) for x in s),1); r['wmax_die_srv']=max(max(x) for x in s)
        q=slice_(smc,w['s0'],w['s1'],lambda x:(x[1],x[2]))
        if q: r['bays_srv']=round(st.mean(b for _,b in q),1); r['dc_srv']=round(st.mean(d for d,_ in q),1); r['dc_max_srv']=round(max(d for d,_ in q),1)
        qb=slice_(smc,w['b0'],w['b1'],lambda x:(x[1],x[2])) if 'b0' in w else []
        if qb: r['dc_max_bench']=round(max(d for d,_ in qb),1)
    rows.append(r)
rows.sort(key=lambda r:r['idx'])
if asjson: print(json.dumps(rows)); sys.exit()
cols=['cap','rocm0_pp2048','rocm0_tg256','tp4_pp2048','tp4_tg256']+sorted({k for r in rows for k in r if k.startswith('srv_') and (k.endswith('_agg') or k.endswith('_ttft'))})+['w4_bench','w4_srv','bays_srv','dc_srv','dc_max_srv','dc_max_bench']
print('| '+' | '.join(cols)+' |'); print('|'+'---:|'*len(cols))
for r in rows: print('| '+' | '.join(str(r.get(c,'')) for c in cols)+' |')
