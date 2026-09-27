patch:            gfx906-mmq-config-reachable | https://github.com/exabit-io/mx-llama.cpp | a22eb39ef | make the fork's gfx906 Q8_0 MMQ config (2335544ab) reachable again: the v0.5.0 merge left it behind upstream's new GCN dispatch (arm patch a22eb39ef on master (ADDITION))
axis:             single-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.698 (median 31.698 / p10 31.632 / min 31.629) vs ref 31.639 tok/s | prefill median 401.5 (min 401.3) vs ref 401.5 t/s | n=6 per arm | decode spread 0.39%
effect:           decode +0.19%, prefill -0.01% vs base
stats:            decode p=0.1364 q=0.3357; prefill p=0.9805 q=1.0000 | two-sided exact permutation 6 vs 6, BH over the declared family m=32 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the same: with q8_repack handling every Q8_0 matmul, restoring the fork's gfx906 table changes nothing (0.00% / +0.19%); it would matter only on the MMQ path, so the merge bug has no measured cost for this service
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Addition arm (not inverted). The v0.5.0 merge bug (fork gfx906 Q8_0 MMQ table unreachable behind upstream's
                  GCN dispatch) has no measured cost here: the 27B's Q8_0 weights are repacked and run through q8_repack, not
                  MMQ. Not applied to master; whether to fix it in PR mxxm-t/mx-llama.cpp#17 for MMQ-path users is the lead's call.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-single-mmq-config-reachable-*.md, binstats output in the same folder.
