#!/usr/bin/env python3
# Aggregates the rocprofv3 counter CSVs of fa-counters-2.sh per kernel (all four dies summed) and derives the S4 ratios.
import csv, glob, sys, collections
T = sys.argv[1]
SIMDS = 256  # 64 CUs x 4 SIMDs per gfx906 die
for phase in ('prefill', 'decode'):
    agg = collections.defaultdict(lambda: collections.defaultdict(float)); disp = collections.Counter(); dur = collections.defaultdict(float)
    for p in 'ABC':
        for fn in glob.glob(f'{T}/{phase}-{p}/*counter_collection.csv'):
            for r in csv.DictReader(open(fn)):
                k = r['Kernel_Name'][:64]; agg[k][r['Counter_Name']] += float(r['Counter_Value'] or 0)
                if p == 'A' and r['Counter_Name'] == 'SQ_INSTS_VALU': disp[k] += 1
        for fn in glob.glob(f'{T}/{phase}-{p}/*kernel_trace.csv'):
            if p != 'A': continue
            for r in csv.DictReader(open(fn)):
                k = r['Kernel_Name'][:64]; dur[k] += (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3
    tot = sum(dur.values())
    print(f'\n## {phase}: kernel time share (pass A trace), then counters per dispatch, all dies summed')
    print('| kernel | time % | dispatches | us/dispatch | VALU busy % | waves/dispatch | VALU inst/dispatch | VMEM_RD | LDS inst | LDS wait / GUI_ACTIVE | LDS bank confl | L2 hit % | SALU/VALU |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for k, d in sorted(agg.items(), key=lambda kv: -dur.get(kv[0], 0))[:12]:
        n = disp[k] or 1; gui = d.get('GRBM_GUI_ACTIVE', 0); act = d.get('SQ_ACTIVE_INST_VALU', 0)
        vb = 100 * act * 4 / SIMDS / gui if gui else 0
        hit = d.get('TCC_HIT', 0); miss = d.get('TCC_MISS', 0); l2 = 100 * hit / (hit + miss) if hit + miss else 0
        print(f"| {k} | {100*dur.get(k,0)/tot if tot else 0:.1f} | {n} | {dur.get(k,0)/n:.0f} | {vb:.0f} | {d.get('SQ_WAVES',0)/n:.3g} | {d.get('SQ_INSTS_VALU',0)/n:.3g} | {d.get('SQ_INSTS_VMEM_RD',0)/n:.3g} | {d.get('SQ_INSTS_LDS',0)/n:.3g} | {d.get('SQ_WAIT_INST_LDS',0)/gui if gui else 0:.3f} | {d.get('SQ_LDS_BANK_CONFLICT',0)/n:.3g} | {l2:.0f} | {d.get('SQ_INSTS_SALU',0)/d['SQ_INSTS_VALU'] if d.get('SQ_INSTS_VALU') else 0:.2f} |")
