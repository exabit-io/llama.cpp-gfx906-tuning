#!/usr/bin/env python3
"""binrecords.py — write the survey/<patchset>.md and <patchset>-single.md records for a binrun round.

Every number comes from binstats.py's own output (effects, p, q, per-axis verdicts, bins) and from
binrun.tsv (run values for median/p10/min); nothing is typed by hand. Refuses to write anything if a
patchset is NOT BINNED (incomplete or mixed: mixed goes to the lead) or if the compat gate has no row.

usage: binrecords.py BINRUN_TSV FNCOMPAT_TSV OUTDIR ROUND_LABEL WOULDCHANGE_JSON [ROUND_JSON ARMS_FILE]
  ROUND_JSON / ARMS_FILE: a later round's binstats family file and arms file (round 1 when omitted).
  WOULDCHANGE_JSON: {arm: text} — the observation that would overturn each verdict, written after reading the
  results; every arm must have one (a placeholder would pass survey-lint, so this script refuses instead).
"""
import sys, re, subprocess, statistics as st, collections, os, json

tsv, fnc, outdir, rnd, wcj = sys.argv[1:6]
rj = sys.argv[6] if len(sys.argv) > 6 else None
WOULD = json.load(open(wcj))
W = os.path.dirname(os.path.abspath(__file__))
out = subprocess.run([sys.executable, os.path.join(W, 'binstats.py'), tsv] + ([rj] if rj else []), capture_output=True, text=True, check=True).stdout
M = 4 * len(json.load(open(rj))['REF']) if rj else 28
ARMS = sys.argv[7] if len(sys.argv) > 7 else os.path.join(W, 'binrun-arms.txt')
SFX = '' if rnd == 'r1' else '-' + rnd

NAME = {'mmvq-batch1-knobs': 'mmvq-q8-fastpath'}
CLAIM = {
    'norm-add-fusion': ('terms 02 03', 'the fused RMS_NORM+MUL+ADD also emits the Q8_1 activations and computes the residual ADD inside the fusion'),
    'gdn-producer-fold': ('terms 04 07 08 10 12 (on top of 02 03)', 'fold the gated-delta-net producers and the q/k L2 norms into the fused add-norm'),
    'mmvq-batch1-knobs': ('terms 01 05 09 15, one unit', 'gfx906 MMVQ: 16-column width (MMID stays 8), batch-1 rows/warps-per-block knobs, whole-block vdr-8 load at one column, MUL_MAT_ID sync fix'),
    's1b-repacked-matvec': ('terms 17 18 19', 'repacked Q8_0 narrow-batch mat-vec with launch bounds, keeps 4 waves'),
    'fa-head256-rows': ('terms 22 27', 'gfx906 GCN row for the head-256 flash-attention tile table'),
    'dpp-warp-reductions': ('terms 23 28', 'DPP-based warp reductions on GCN, generic reductions kept where DPP does not apply'),
    'max-ilp': ('build flag', 'compile with -mllvm -amdgpu-sched-strategy=max-ilp (mixa3607/ML-gfx906), base code unchanged'),
    'combo-both': ('terms 02 03 04 07 08 10 12 23 28 + max-ilp flag', 'stack of norm-add-fusion, gdn-producer-fold, dpp-warp-reductions and max-ilp (round 1b)'),
    'combo-multi': ('terms 01 02 03 04 05 07 08 09 10 12 15 23 28 + max-ilp flag', 'combo-both plus mmvq-q8-fastpath (round 1b)'),
}
SRC = {'max-ilp': 'https://github.com/mixa3607/ML-gfx906'}
arms = {}
for line in open(ARMS):
    f = line.rstrip('\n').split('|')
    if f[0]: arms[f[0]] = f[1]

# binstats table rows
T = {}
for line in out.splitlines():
    m = re.match(r'^(\S+)\s+(multi|single)\s+(decode|prefill)\s+(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)%\s+([\d.]+)\s+([\d.]+)$', line)
    if m:
        a, ax, me, ref, n, rv, av, eff, p, q = m.groups()
        T[(a, ax, me)] = dict(ref=ref, n=int(n), rv=float(rv), av=float(av), eff=float(eff), p=float(p), q=float(q))
BIN = {}
for line in out.split('\nBINS\n', 1)[1].splitlines():
    m = re.match(r'^\s+(.+?)\s+multi=(\S+)\s+single=(\S+)\s+->\s+(.+)$', line)   # labels may contain spaces
    if m:
        BIN[m.group(1)] = (m.group(2), m.group(3), m.group(4).strip())
LABEL2ARM = {'mmvq-q8-fastpath(01,05,09,15)': 'mmvq-batch1-knobs'}
if rj:
    LABEL2ARM.update({v: k for k, v in json.load(open(rj)).get('LABEL', {}).items()})
BIN = {LABEL2ARM.get(k, k): v for k, v in BIN.items()}
bad = [a for a, v in BIN.items() if v[2].startswith('NOT BINNED')]
_ref = json.load(open(rj))['REF'] if rj else None
missing = [x for x in (_ref or []) if x not in BIN]
if missing:
    sys.exit(f"refusing: binstats BINS has no row for {missing} (label parse?)")
if bad:
    sys.exit(f"refusing: not binned: {bad} — incomplete or mixed (mixed goes to the lead)")

runs = collections.defaultdict(list)
for line in open(tsv):
    f = line.rstrip('\n').split('\t')
    try: runs[(f[0], f[1])].append((float(f[3]), float(f[4])))
    except (ValueError, IndexError): pass
comp = {}
for line in open(fnc):
    f = line.rstrip('\n').split('\t')
    if len(f) >= 6: comp[f[0]] = f
def p10(xs):
    s = sorted(xs); k = 0.1 * (len(s) - 1); i = int(k)
    return s[i] + (k - i) * (s[min(i + 1, len(s) - 1)] - s[i])

AXN = {'multi': 'multi-user', 'single': 'single-user'}
CELL = {'multi': '4x64K (4 x 65536 prompt, 1024 generated)', 'single': '1x255K (1 x 260864 prompt, 1024 generated)'}
STAT = {'multi': 'mean', 'single': 'median'}
written = []
for a, (vm, vs, b) in BIN.items():
    if not WOULD.get(a, '').strip():
        sys.exit(f"refusing: no 'would change if' text for {a}")
    if a not in comp:
        sys.exit(f"refusing: no Flash-Next compat row for {a}")
    c = comp[a]
    for ax, verdict in (('multi', vm), ('single', vs)):
        d, pf = T[(a, ax, 'decode')], T[(a, ax, 'prefill')]
        ref = d['ref']
        R = runs[(ax, a)]; RR = runs[(ax, ref)]
        dv = [r[0] for r in R]; pv = [r[1] for r in R]
        spread = 100 * (max(dv) - min(dv)) / st.mean(dv)
        # the metric that decided the verdict goes first, so 'stats' leads with its q
        order = [d, pf] if (verdict == 'neutral' or abs(d['eff']) >= abs(pf['eff'])) else [pf, d]
        mname = {id(d): 'decode', id(pf): 'prefill'}
        name = NAME.get(a, a)
        fn = os.path.join(outdir, f"{name}{'' if ax == 'multi' else '-single'}.md")
        cl = CLAIM[a]
        commits = arms[a] or '(build flag)'
        txt = f"""patch:            {name} | {SRC.get(a, 'https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates')} | {commits} | {cl[1]} ({cl[0]})
axis:             {AXN[ax]}
zero point:       {'build-ps-base = master a23e12438' if ref == 'base' else 'build-ps-' + ref + ' = master a23e12438 + ' + ref} measured 2026-09-24/25 ({rnd}, interleaved in the same blocks)
recipe:           {CELL[ax]} | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level {STAT[ax]} of n={len(dv)}; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode {STAT[ax]} {d['av']:.3f} (median {st.median(dv):.3f} / p10 {p10(dv):.3f} / min {min(dv):.3f}) vs ref {d['rv']:.3f} tok/s | prefill {STAT[ax]} {pf['av']:.1f} (min {min(pv):.1f}) vs ref {pf['rv']:.1f} t/s | n={len(dv)} per arm | decode spread {spread:.2f}%
effect:           decode {d['eff']:+.2f}%, prefill {pf['eff']:+.2f}% vs {ref}
stats:            {'; '.join(f"{mname[id(t)]} p={t['p']:.4f} q={t['q']:.4f}" for t in order)} | two-sided exact permutation {len(RR)} vs {len(dv)}, BH over the declared family m={M} | n={len(dv)} per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          {verdict}
bin:              {b}
would change if:  {WOULD[a]}
notes:            Round {rnd} of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user {vm}, single-user {vs}. Flash-Next compatibility: {c[1]} ({c[2]}, text {c[5]} vs base).
                  Data: data/raw/night-20260919/binrun{SFX}.tsv, br{SFX}-{ax}-{a}-*.md, binstats output in the same folder.
"""
        written.append((fn, txt, a, ax))
os.makedirs(outdir, exist_ok=True)
for fn, txt, a, ax in written:
    open(fn, 'w').write(txt)
    print(fn)
