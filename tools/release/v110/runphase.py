#!/usr/bin/env python3
# runphase.py CELL...  (read-only) per run of a screen-acc-<CELL> session: the samples in which all four dies are
# >= 95 % busy (the prefill phase), their mean clock, die power, die temperature, bay power, CPU-zone power, fans.
import sys,re,datetime as dt,os
W='/root/rebase-v110'
def ts(s): return dt.datetime.fromisoformat(s)
for cell in sys.argv[1:]:
    pre=f'{W}/screen-{cell}'
    runs=[];prev=None
    for l in open(pre+'.progress'):
        m=re.match(r'(\S+) \[screen\] (.*)',l)
        if not m: continue
        t=ts(m.group(1)); body=m.group(2)
        if body.startswith('block') and 'order' in body:
            if prev is None or True: prev=t if prev is None else prev
        m2=re.match(r'(.+?) (\S+) b(\d+): decode ([\d.]+), prefill ([\d.]+)',body)
        if m2:
            runs.append((prev,t,m2.group(2),int(m2.group(3)),float(m2.group(4)),float(m2.group(5)))); prev=t
    wd=[]
    for l in open(pre+'-watchdog.out'):
        p=l.split()
        if len(p)<5 or not p[1].startswith('card0'): continue
        cards=[re.match(r'card\d:(\d+)MHz/(\d+)%',x) for x in p[1:5]]
        if not all(cards): continue
        wd.append((ts(p[0]),[int(c.group(1)) for c in cards],[int(c.group(2)) for c in cards]))
    day0=wd[0][0].date()
    def daytime(hms,last=[None]):
        t=dt.datetime.combine(day0,dt.time.fromisoformat(hms),tzinfo=dt.timezone.utc)
        while last[0] is not None and t<last[0]-dt.timedelta(hours=12): t+=dt.timedelta(days=1)
        last[0]=t; return t
    ck=[]
    for l in open(pre+'-clocks.txt'):
        p=l.split()
        if len(p)<6: continue
        d=[re.match(r'\w+:(\d+)Mhz/(\d+)C/(\d+)W',x) for x in p[1:5]]
        f=re.match(r'fans:(\d+)/(\d+)/(\d+)/(\d+)',p[5])
        if not all(d) or not f: continue
        ck.append((daytime(p[0]),[int(x.group(1)) for x in d],[int(x.group(2)) for x in d],[int(x.group(3)) for x in d],[int(f.group(i)) for i in range(1,5)]))
    sm=[];last=[None]
    for l in open(pre+'-smc.log'):
        if l.startswith('#'): continue
        p=l.split(); kv=dict(x.split('=') for x in p[1:] if '=' in x)
        try: sm.append((daytime(p[0],last),{k:float(v) for k,v in kv.items()}))
        except Exception: pass
    print(f'== {cell}: arm blk | prefill tok/s decode | full-busy samples | MHz mean (per die) | die W | die C | bays W (PZ3G+PZ4G) | PZ1G W | PZ0G W | fans')
    for (a,b,arm,blk,dec,pf) in runs:
        full=[(t,mh) for (t,mh,bu) in wd if a<=t<=b and min(bu)>=95]
        if not full: print(f'  {arm:5s} b{blk} {pf:8.1f} {dec:8.4f}  no full-busy samples'); continue
        t0,t1=full[0][0],full[-1][0]
        mh=[sum(x[1][i] for x in full)/len(full) for i in range(4)]
        c=[x for x in ck if t0-dt.timedelta(seconds=3)<=x[0]<=t1+dt.timedelta(seconds=3)]
        s=[x for x in sm if t0-dt.timedelta(seconds=3)<=x[0]<=t1+dt.timedelta(seconds=3)]
        mean=lambda v: sum(v)/len(v) if v else float('nan')
        dw=mean([sum(x[3])/4 for x in c]); dc=mean([sum(x[2])/4 for x in c]); fans=[mean([x[4][i] for x in c]) for i in range(4)]
        bays=mean([x[1].get('PZ3G',0)+x[1].get('PZ4G',0) for x in s]); cpu=mean([x[1].get('PZ1G',0) for x in s]); tot=mean([x[1].get('PZ0G',0) for x in s])
        low=sum(1 for x in full for v in x[1] if v<=1000)
        print(f'  {arm:5s} b{blk} {pf:8.1f} {dec:8.4f}  n={len(full):3d} at1000={low:3d}  {sum(mh)/4:6.0f} ({mh[0]:.0f} {mh[1]:.0f} {mh[2]:.0f} {mh[3]:.0f})  {dw:6.1f}  {dc:5.1f}  {bays:6.1f}  {cpu:6.1f}  {tot:6.1f}  {fans[0]:.0f}/{fans[1]:.0f}/{fans[2]:.0f}/{fans[3]:.0f}')
