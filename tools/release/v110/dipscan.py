#!/usr/bin/env python3
# dipscan.py [-v] (read-only): for every screen-*.progress session with an -smc.log in /root/rebase-v110, per run the
# number of 5-s samples with bay power (PZ3G+PZ4G) under 300 W (idle is 175-205 W) INSIDE the loaded span of the run
# (between its first and its last sample of >= 550 W). A stall of the GPU pipeline shows as such a dip.
import sys,re,glob,os,datetime as dt
W='/root/rebase-v110'; verbose='-v' in sys.argv
sess=sorted(glob.glob(W+'/screen-*.progress'),key=os.path.getmtime)
for pf in sess:
    pre=pf[:-9]
    if not os.path.exists(pre+'-smc.log'): continue
    runs=[];prev=None
    for l in open(pf):
        m=re.match(r'(\S+) \[screen\] (.*)',l)
        if not m: continue
        try: t=dt.datetime.fromisoformat(m.group(1))
        except ValueError: continue
        body=m.group(2)
        if prev is None and 'order' in body: prev=t
        m2=re.match(r'(.+?) (\S+) b(\d+): decode ([\d.]+), prefill ([\d.]+)',body)
        if m2 and prev is not None:
            runs.append((prev,t,m2.group(2),int(m2.group(3)),float(m2.group(5)))); prev=t
    if not runs: continue
    day=runs[0][0].date(); last=None; sm=[]
    for l in open(pre+'-smc.log'):
        if l.startswith('#'):
            m=re.match(r'# (\S+) smc-log start',l)
            if m: day=dt.datetime.fromisoformat(m.group(1)).date(); last=None
            continue
        p=l.split()
        try:
            t=dt.datetime.combine(day,dt.time.fromisoformat(p[0]),tzinfo=dt.timezone.utc)
            kv=dict(x.split('=') for x in p[1:] if '=' in x); v=float(kv['PZ3G'])+float(kv['PZ4G'])
        except Exception: continue
        if last is not None and t<last-dt.timedelta(hours=12): day+=dt.timedelta(days=1); t+=dt.timedelta(days=1)
        last=t; sm.append((t,v))
    tot=0; withdip=0; rows=[]
    for (a,b,arm,blk,pfr) in runs:
        s=[v for (t,v) in sm if a<=t<=b]
        hi=[i for i,v in enumerate(s) if v>=550]
        d=sum(1 for v in s[hi[0]:hi[-1]+1] if v<300) if hi else -1
        rows.append((arm,blk,pfr,d)); tot+=max(d,0); withdip+= d>0
    print('%s  %-22s runs %2d, runs with a dip %2d, dip samples %3d'%(runs[0][0].strftime('%m-%d %H:%M'),os.path.basename(pre)[7:],len(runs),withdip,tot))
    if verbose:
        print('      '+'  '.join('%s b%d %.0f:%d'%r for r in rows))
