# optimize.py output

Generated 2026-09-08 by `tools/gen-results.py` (with the M2 24/32-slot cells, the M3 production draft factors, the phase-separated cap curves of TODO 13, and the review fixes: Q4V lookup, command comment, placement-scoped topology/graph factors, production decode factors from the production cells, depth-aware prefill gain, provenance filtered after composition) from `data/benchmarks.json` (three measurement reports + the 2026-09-07 run-through). Re-run after adding data:
`python3 optimize/optimize.py` (production build), `--build stock`, `--allow-quant --max-kl 0.04`, `--cap 125`, `--kv-q4v`, `--measured-only`; then `python3 review/2026-09-08/test_optimizer_patch.py optimize/optimize.py`.

Provenance: "measured" = every factor of the cell is a measured point; "model" = at least one factor is the reports' fitted
step model, an interpolation between measured slot counts, or a kernel gain measured at 2K applied to the depth-independent
part of the step at depth. Model cells carry a 3% discount (6% when the build factor is interpolated, +1% when the cap curve is).

## Production build (ML-gfx906 fork tile table + fast-path MMVQ kernel), default

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs; build prod; cap 200 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    71.5 tok/s aggregate  (71.5 per stream)   prefill   1128 tok/s   memory/die 8.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~54 tok/s, ~12.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 8K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 64.1 (-10%), prefill 1128 (+0%), 7.1 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 0 Q8_0 -> decode 32.2 (-55%), prefill 586 (-48%), 16.4 GiB/die [measured]

B. one user, 32K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 32K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    66.0 tok/s aggregate  (66.0 per stream)   prefill    975 tok/s   memory/die 9.3 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~49 tok/s, ~11.5 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 32768 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 32K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 58.1 (-12%), prefill 975 (+0%), 7.4 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 29.4 (-55%), prefill 438 (-55%), 17.2 GiB/die [model]

C. one user, 128K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (65.0 per stream)   prefill    623 tok/s   memory/die 10.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~42 tok/s, ~10.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 131072 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 128K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 50.1 (-23%), prefill 623 (+0%), 8.6 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 23.7 (-64%), prefill 314 (-50%), 20.2 GiB/die [model]

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  placement tp4     slots/instance 16  streams 16  ctx/slot 16K (up to 84K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   201.0 tok/s aggregate  (12.6 per stream)   prefill   1132 tok/s   memory/die 13.3 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.8 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -c 262144 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 12 (12 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 198.0 (-2%), prefill 1132 (+0%), 12.2 GiB/die [model]
    runner-up: tp4 np 14 (14 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 202.3 (+1%), prefill 1132 (+0%), 12.7 GiB/die [model]

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement tp4     slots/instance 32  streams 32  ctx/slot 32K (up to 40K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   205.0 tok/s aggregate  (6.4 per stream)   prefill   1116 tok/s   memory/die 25.9 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 32 -c 1048576 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 194.1 (-5%), prefill 1116 (+0%), 17.3 GiB/die [model]
    runner-up: tp4 np 12 (12 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 191.2 (-7%), prefill 1116 (+0%), 15.2 GiB/die [model]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 4K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q8_0  build prod
  decode   268.6 tok/s aggregate  (8.4 per stream)   prefill   1253 tok/s   memory/die 30.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~129 tok/s, ~30.1 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 259.7 (-3%), prefill 1253 (+0%), 29.1 GiB/die [measured]
    runner-up: tp4 np 32 (32 streams) ctx 4K kv f16 ub 2048 draft 0 Q8_0 -> decode 213.8 (-20%), prefill 1128 (-10%), 11.9 GiB/die [measured]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  placement tp4     slots/instance 8   streams 8   ctx/slot 128K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    91.8 tok/s aggregate  (11.5 per stream)   prefill    623 tok/s   memory/die 25.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~53 tok/s, ~12.3 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1048576 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 4 (4 streams) ctx 128K kv f16 ub 2048 draft 3 Q8_0 -> decode 89.8 (-2%), prefill 623 (+0%), 16.9 GiB/die [model]
    runner-up: tp4 np 7 (7 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 86.4 (-6%), prefill 623 (+0%), 23.0 GiB/die [model]

H. full 256K context, as many slots as fit
  placement tp4     slots/instance 2   streams 2   ctx/slot 256K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (32.5 per stream)   prefill    425 tok/s   memory/die 16.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~37 tok/s, ~8.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 2 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 4 (4 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 63.2 (-3%), prefill 425 (+0%), 24.9 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 55.3 (-15%), prefill 425 (+0%), 12.8 GiB/die [model]

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  placement tp4     slots/instance 12  streams 12  ctx/slot 32K (up to 104K fits)  kv f16       ub 4096  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   158.6 tok/s aggregate  (13.2 per stream)   prefill    983 tok/s   memory/die 17.5 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~87 tok/s, ~20.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 12 -c 393216 -b 2048 -ub 4096 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 4096 draft 0 Q8_0 -> decode 160.6 (+1%), prefill 983 (+0%), 19.6 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 32K kv f16 ub 4096 draft 1 Q8_0 -> decode 147.2 (-7%), prefill 983 (+0%), 15.3 GiB/die [model]

J. eight slots at the memory ceiling: 8 streams, 160K each
  placement tp4     slots/instance 8   streams 8   ctx/slot 160K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    81.3 tok/s aggregate  (10.2 per stream)   prefill    557 tok/s   memory/die 29.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~47 tok/s, ~10.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1310720 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 8 (8 streams) ctx 160K kv q8_0 ub 2048 draft 1 Q8_0 -> decode 54.1 (-33%), prefill 557 (+0%), 23.6 GiB/die [model]
```

## Stock b10288 (the reference every report number is measured on)

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs; build stock; cap 200 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build stock
  decode    69.3 tok/s aggregate  (69.3 per stream)   prefill    846 tok/s   memory/die 8.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~49 tok/s, ~11.5 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 8K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 62.1 (-10%), prefill 846 (+0%), 7.1 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 0 Q8_0 -> decode 31.2 (-55%), prefill 440 (-48%), 16.4 GiB/die [measured]

B. one user, 32K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 32K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build stock
  decode    64.1 tok/s aggregate  (64.1 per stream)   prefill    739 tok/s   memory/die 9.3 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~44 tok/s, ~10.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 32768 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 32K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 56.5 (-12%), prefill 739 (+0%), 7.4 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 28.5 (-56%), prefill 332 (-55%), 17.2 GiB/die [measured]

C. one user, 128K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build stock
  decode    63.4 tok/s aggregate  (63.4 per stream)   prefill    519 tok/s   memory/die 10.8 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~39 tok/s, ~9.2 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 131072 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 128K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 49.1 (-23%), prefill 519 (+0%), 8.6 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 23.1 (-64%), prefill 262 (-50%), 20.2 GiB/die [model]

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  placement tp4     slots/instance 8   streams 8   ctx/slot 16K (up to 172K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build stock
  decode   161.7 tok/s aggregate  (20.2 per stream)   prefill    844 tok/s   memory/die 11.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~82 tok/s, ~19.2 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 131072 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 7 (7 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 154.2 (-5%), prefill 844 (+0%), 10.7 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 16K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 151.1 (-7%), prefill 844 (+0%), 8.9 GiB/die [model]

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement tp4     slots/instance 8   streams 8   ctx/slot 32K (up to 172K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build stock
  decode   152.9 tok/s aggregate  (19.1 per stream)   prefill    823 tok/s   memory/die 13.0 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~79 tok/s, ~18.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 262144 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 24 (24 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 157.2 (+3%), prefill 823 (+0%), 21.6 GiB/die [model]
    runner-up: tp4 np 7 (7 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 150.1 (-2%), prefill 823 (+0%), 12.5 GiB/die [model]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 4K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q8_0  build stock
  decode   206.0 tok/s aggregate  (6.4 per stream)   prefill    929 tok/s   memory/die 30.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~97 tok/s, ~22.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 200.5 (-3%), prefill 929 (+0%), 29.1 GiB/die [measured]
    runner-up: tp4 np 32 (32 streams) ctx 4K kv f16 ub 2048 draft 0 Q8_0 -> decode 180.1 (-13%), prefill 846 (-9%), 11.9 GiB/die [measured]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  placement tp4     slots/instance 4   streams 4   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build stock
  decode    84.9 tok/s aggregate  (21.2 per stream)   prefill    519 tok/s   memory/die 16.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~46 tok/s, ~10.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 4 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 8 (8 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 83.5 (-2%), prefill 519 (+0%), 25.0 GiB/die [measured]
    runner-up: tp4 np 7 (7 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 83.1 (-2%), prefill 519 (+0%), 23.0 GiB/die [model]

H. full 256K context, as many slots as fit
  placement tp4     slots/instance 2   streams 2   ctx/slot 256K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build stock
  decode    63.6 tok/s aggregate  (31.8 per stream)   prefill    369 tok/s   memory/die 16.8 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~34 tok/s, ~8.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 2 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 4 (4 streams) ctx 256K kv f16 ub 2048 draft 1 Q8_0 -> decode 60.3 (-5%), prefill 369 (+0%), 24.9 GiB/die [measured]
    runner-up: tp4 np 1 (1 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 54.2 (-15%), prefill 369 (+0%), 12.8 GiB/die [measured]

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  placement tp4     slots/instance 8   streams 8   ctx/slot 32K (up to 156K fits)  kv f16       ub 4096  draft 0  topo fixed16  graphs on  quant Q8_0  build stock
  decode   130.6 tok/s aggregate  (16.3 per stream)   prefill    746 tok/s   memory/die 15.3 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~69 tok/s, ~16.2 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 262144 -b 2048 -ub 4096 -cb
    runner-up: tp4 np 4 (4 streams) ctx 32K kv f16 ub 4096 draft 1 Q8_0 -> decode 115.4 (-12%), prefill 746 (+0%), 13.2 GiB/die [measured]
    runner-up: tp4 np 8 (8 streams) ctx 32K kv q8_0 ub 4096 draft 0 Q8_0 -> decode 111.4 (-15%), prefill 746 (+0%), 12.8 GiB/die [measured]

J. eight slots at the memory ceiling: 8 streams, 160K each
  placement tp4     slots/instance 8   streams 8   ctx/slot 160K (up to 172K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build stock
  decode    74.3 tok/s aggregate  (9.3 per stream)   prefill    471 tok/s   memory/die 29.0 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~41 tok/s, ~9.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_STOCK/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1310720 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 8 (8 streams) ctx 160K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 50.1 (-33%), prefill 471 (+0%), 23.6 GiB/die [model]
```

## Model file as a decision variable, quality-bounded (--allow-quant --max-kl 0.04: Q8_0, Q6_K, Q4_K_M)

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs x quant; build prod; cap 200 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    71.5 tok/s aggregate  (71.5 per stream)   prefill   1128 tok/s   memory/die 8.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~54 tok/s, ~12.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 3 Q4_K_M -> decode 74.0 (+4%), prefill 796 (-29%), 6.0 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 3 Q6_K -> decode 71.2 (-0%), prefill 829 (-26%), 7.3 GiB/die [model]

B. one user, 32K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 32K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    66.0 tok/s aggregate  (66.0 per stream)   prefill    975 tok/s   memory/die 9.3 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~49 tok/s, ~11.5 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 32768 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 32K kv f16 ub 2048 draft 3 Q4_K_M -> decode 68.4 (+4%), prefill 688 (-29%), 6.3 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 32K kv f16 ub 2048 draft 3 Q6_K -> decode 65.7 (-0%), prefill 717 (-26%), 7.6 GiB/die [model]

C. one user, 128K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (65.0 per stream)   prefill    623 tok/s   memory/die 10.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~42 tok/s, ~10.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 131072 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 128K kv f16 ub 2048 draft 3 Q4_K_M -> decode 67.3 (+4%), prefill 440 (-29%), 7.8 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 128K kv f16 ub 2048 draft 3 Q6_K -> decode 64.7 (-0%), prefill 458 (-26%), 9.1 GiB/die [model]

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  placement tp4     slots/instance 16  streams 16  ctx/slot 16K (up to 84K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   201.0 tok/s aggregate  (12.6 per stream)   prefill   1132 tok/s   memory/die 13.3 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.8 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -c 262144 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 12 (12 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 198.0 (-2%), prefill 1132 (+0%), 12.2 GiB/die [model]
    runner-up: tp4 np 14 (14 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 202.3 (+1%), prefill 1132 (+0%), 12.7 GiB/die [model]

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement dp4     slots/instance 8   streams 32  ctx/slot 32K (up to 36K fits)  kv q8_0      ub 2048  draft 0  topo default  graphs on  quant Q4_K_M  build prod
  decode   278.8 tok/s aggregate  (8.7 per stream)   prefill    856 tok/s   memory/die 28.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~105 tok/s, ~24.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q4_K_M.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 262144 -b 2048 -ub 2048 -cb -ctk q8_0 -ctv q8_0
    runner-up: tp4 np 32 (32 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 205.0 (-26%), prefill 1116 (+30%), 25.9 GiB/die [model]
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 194.1 (-30%), prefill 1116 (+30%), 17.3 GiB/die [model]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 24K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q4_K_M  build prod
  decode   327.7 tok/s aggregate  (10.2 per stream)   prefill    881 tok/s   memory/die 19.2 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~113 tok/s, ~26.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q4_K_M.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q4_K_M -> decode 316.9 (-3%), prefill 881 (+0%), 17.4 GiB/die [model]
    runner-up: dp4 np 8 (32 streams) ctx 4K kv f16 ub 2048 draft 0 Q8_0 -> decode 268.6 (-18%), prefill 1253 (+42%), 30.9 GiB/die [measured]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  placement tp4     slots/instance 8   streams 8   ctx/slot 128K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    91.8 tok/s aggregate  (11.5 per stream)   prefill    623 tok/s   memory/die 25.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~53 tok/s, ~12.3 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1048576 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 4 (4 streams) ctx 128K kv f16 ub 2048 draft 3 Q8_0 -> decode 89.8 (-2%), prefill 623 (+0%), 16.9 GiB/die [model]
    runner-up: tp4 np 12 (12 streams) ctx 128K kv f16 ub 2048 draft 0 Q4_K_M -> decode 97.7 (+6%), prefill 440 (-29%), 30.2 GiB/die [model]

H. full 256K context, as many slots as fit
  placement tp4     slots/instance 2   streams 2   ctx/slot 256K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (32.5 per stream)   prefill    425 tok/s   memory/die 16.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~37 tok/s, ~8.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 2 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 4 (4 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 63.2 (-3%), prefill 425 (+0%), 24.9 GiB/die [model]
    runner-up: tp4 np 2 (2 streams) ctx 256K kv f16 ub 2048 draft 3 Q4_K_M -> decode 67.3 (+4%), prefill 300 (-29%), 13.9 GiB/die [model]

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  placement tp4     slots/instance 12  streams 12  ctx/slot 32K (up to 104K fits)  kv f16       ub 4096  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   158.6 tok/s aggregate  (13.2 per stream)   prefill    983 tok/s   memory/die 17.5 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~87 tok/s, ~20.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 12 -c 393216 -b 2048 -ub 4096 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 4096 draft 0 Q8_0 -> decode 160.6 (+1%), prefill 983 (+0%), 19.6 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 32K kv f16 ub 4096 draft 1 Q8_0 -> decode 147.2 (-7%), prefill 983 (+0%), 15.3 GiB/die [model]

J. eight slots at the memory ceiling: 8 streams, 160K each
  placement tp4     slots/instance 8   streams 8   ctx/slot 160K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    81.3 tok/s aggregate  (10.2 per stream)   prefill    557 tok/s   memory/die 29.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~47 tok/s, ~10.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1310720 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 8 (8 streams) ctx 160K kv f16 ub 2048 draft 1 Q4_K_M -> decode 84.2 (+4%), prefill 393 (-29%), 26.1 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 160K kv f16 ub 2048 draft 1 Q6_K -> decode 80.9 (-0%), prefill 409 (-26%), 27.4 GiB/die [model]
```

## Production cap for the hyperconverged fleet (--cap 125)

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs; build prod; cap 125 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    63.1 tok/s aggregate  (63.1 per stream)   prefill    930 tok/s   memory/die 8.9 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~47 tok/s, ~11.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 8K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 56.6 (-10%), prefill 930 (+0%), 7.1 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 8K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 28.3 (-55%), prefill 483 (-48%), 13.7 GiB/die [model]

B. one user, 32K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 32K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    58.2 tok/s aggregate  (58.2 per stream)   prefill    804 tok/s   memory/die 9.3 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~43 tok/s, ~10.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 32768 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 32K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 51.3 (-12%), prefill 804 (+0%), 7.4 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 25.9 (-55%), prefill 361 (-55%), 17.2 GiB/die [model]

C. one user, 128K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    57.4 tok/s aggregate  (57.4 per stream)   prefill    514 tok/s   memory/die 10.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~37 tok/s, ~8.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 131072 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 128K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 44.2 (-23%), prefill 514 (+0%), 8.6 GiB/die [model]
    runner-up: tp2 np 1 (1 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 20.9 (-64%), prefill 259 (-50%), 20.2 GiB/die [model]

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  placement tp4     slots/instance 12  streams 12  ctx/slot 16K (up to 116K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode   163.9 tok/s aggregate  (13.7 per stream)   prefill    934 tok/s   memory/die 12.2 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~87 tok/s, ~20.3 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 12 -c 196608 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 13 (13 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 166.2 (+1%), prefill 934 (+0%), 12.5 GiB/die [model]
    runner-up: tp4 np 12 (12 streams) ctx 16K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 155.4 (-5%), prefill 934 (+0%), 9.8 GiB/die [model]

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement tp4     slots/instance 16  streams 16  ctx/slot 32K (up to 84K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode   160.6 tok/s aggregate  (10.0 per stream)   prefill    921 tok/s   memory/die 17.3 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~85 tok/s, ~20.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -c 524288 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 12 (12 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 158.3 (-1%), prefill 921 (+0%), 15.2 GiB/die [model]
    runner-up: tp4 np 14 (14 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 161.6 (+1%), prefill 921 (+0%), 16.2 GiB/die [model]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 4K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode   222.3 tok/s aggregate  (6.9 per stream)   prefill   1033 tok/s   memory/die 30.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 214.9 (-3%), prefill 1033 (+0%), 29.1 GiB/die [measured]
    runner-up: tp4 np 32 (32 streams) ctx 4K kv f16 ub 2048 draft 0 Q8_0 -> decode 176.9 (-20%), prefill 930 (-10%), 11.9 GiB/die [measured]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  placement tp4     slots/instance 4   streams 4   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    79.2 tok/s aggregate  (19.8 per stream)   prefill    514 tok/s   memory/die 16.9 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~44 tok/s, ~10.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 4 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 8 (8 streams) ctx 128K kv f16 ub 2048 draft 1 Q8_0 -> decode 76.0 (-4%), prefill 514 (+0%), 25.0 GiB/die [model]
    runner-up: tp4 np 7 (7 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 76.2 (-4%), prefill 514 (+0%), 23.0 GiB/die [model]

H. full 256K context, as many slots as fit
  placement tp4     slots/instance 2   streams 2   ctx/slot 256K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    57.3 tok/s aggregate  (28.7 per stream)   prefill    350 tok/s   memory/die 16.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~31 tok/s, ~7.3 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 2 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 4 (4 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 55.7 (-3%), prefill 350 (+0%), 24.9 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 48.8 (-15%), prefill 350 (+0%), 12.8 GiB/die [model]

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  placement tp4     slots/instance 12  streams 12  ctx/slot 32K (up to 104K fits)  kv f16       ub 4096  draft 0  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode   131.3 tok/s aggregate  (10.9 per stream)   prefill    811 tok/s   memory/die 17.5 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~72 tok/s, ~16.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 12 -c 393216 -b 2048 -ub 4096 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 4096 draft 0 Q8_0 -> decode 132.9 (+1%), prefill 811 (+0%), 19.6 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 32K kv f16 ub 4096 draft 1 Q8_0 -> decode 121.8 (-7%), prefill 811 (+0%), 15.3 GiB/die [model]

J. eight slots at the memory ceiling: 8 streams, 160K each
  placement tp4     slots/instance 8   streams 8   ctx/slot 160K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod  cap 125 W (~131 gen tok/kJ at 16 clients)
  decode    67.3 tok/s aggregate  (8.4 per stream)   prefill    459 tok/s   memory/die 29.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~39 tok/s, ~9.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  cap 125 W per die: set with settings/powercap.sh 125; caps below ~85 W are accepted by the driver and not honoured
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1310720 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 8 (8 streams) ctx 160K kv q8_0 ub 2048 draft 1 Q8_0 -> decode 44.8 (-33%), prefill 459 (+0%), 23.6 GiB/die [model]
```

## With the q8_0-keys / q4_0-values cache allowed (--kv-q4v; needs a GGML_CUDA_FA_ALL_QUANTS build)

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs; build prod; cap 200 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    71.5 tok/s aggregate  (71.5 per stream)   prefill   1128 tok/s   memory/die 8.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~54 tok/s, ~12.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 8K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 64.1 (-10%), prefill 1128 (+0%), 7.1 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 8K kv q8_0/q4_0 ub 2048 draft 3 Q8_0 -> decode 63.8 (-11%), prefill 1128 (+0%), 8.8 GiB/die [model]

B. one user, 32K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 32K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    66.0 tok/s aggregate  (66.0 per stream)   prefill    975 tok/s   memory/die 9.3 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~49 tok/s, ~11.5 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 32768 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 32K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 58.1 (-12%), prefill 975 (+0%), 7.4 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 32K kv q8_0/q4_0 ub 2048 draft 3 Q8_0 -> decode 57.5 (-13%), prefill 975 (+0%), 9.0 GiB/die [model]

C. one user, 128K working context
  placement tp4     slots/instance 1   streams 1   ctx/slot 128K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (65.0 per stream)   prefill    623 tok/s   memory/die 10.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~42 tok/s, ~10.0 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 131072 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 1 (1 streams) ctx 128K kv q8_0 ub 2048 draft 3 Q8_0 -> decode 50.1 (-23%), prefill 623 (+0%), 8.6 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 128K kv q8_0/q4_0 ub 2048 draft 3 Q8_0 -> decode 48.9 (-25%), prefill 623 (+0%), 9.6 GiB/die [model]

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  placement tp4     slots/instance 16  streams 16  ctx/slot 16K (up to 84K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   201.0 tok/s aggregate  (12.6 per stream)   prefill   1132 tok/s   memory/die 13.3 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.8 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -c 262144 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 12 (12 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 198.0 (-2%), prefill 1132 (+0%), 12.2 GiB/die [model]
    runner-up: tp4 np 14 (14 streams) ctx 16K kv f16 ub 2048 draft 0 Q8_0 -> decode 202.3 (+1%), prefill 1132 (+0%), 12.7 GiB/die [model]

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement tp4     slots/instance 32  streams 32  ctx/slot 32K (up to 40K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   205.0 tok/s aggregate  (6.4 per stream)   prefill   1116 tok/s   memory/die 25.9 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~106 tok/s, ~24.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 32 -c 1048576 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 194.1 (-5%), prefill 1116 (+0%), 17.3 GiB/die [model]
    runner-up: tp4 np 12 (12 streams) ctx 32K kv f16 ub 2048 draft 0 Q8_0 -> decode 191.2 (-7%), prefill 1116 (+0%), 15.2 GiB/die [model]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 4K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q8_0  build prod
  decode   268.6 tok/s aggregate  (8.4 per stream)   prefill   1253 tok/s   memory/die 30.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~129 tok/s, ~30.1 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 259.7 (-3%), prefill 1253 (+0%), 29.1 GiB/die [measured]
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0/q4_0 ub 2048 draft 0 Q8_0 -> decode 258.8 (-4%), prefill 1253 (+0%), 29.8 GiB/die [model]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  placement tp4     slots/instance 8   streams 8   ctx/slot 128K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    91.8 tok/s aggregate  (11.5 per stream)   prefill    623 tok/s   memory/die 25.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~53 tok/s, ~12.3 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1048576 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 4 (4 streams) ctx 128K kv f16 ub 2048 draft 3 Q8_0 -> decode 89.8 (-2%), prefill 623 (+0%), 16.9 GiB/die [model]
    runner-up: tp4 np 7 (7 streams) ctx 128K kv f16 ub 2048 draft 0 Q8_0 -> decode 86.4 (-6%), prefill 623 (+0%), 23.0 GiB/die [model]

H. full 256K context, as many slots as fit
  placement tp4     slots/instance 2   streams 2   ctx/slot 256K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    65.0 tok/s aggregate  (32.5 per stream)   prefill    425 tok/s   memory/die 16.8 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~37 tok/s, ~8.6 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 2 -c 524288 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp4 np 4 (4 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 63.2 (-3%), prefill 425 (+0%), 24.9 GiB/die [model]
    runner-up: tp4 np 1 (1 streams) ctx 256K kv f16 ub 2048 draft 3 Q8_0 -> decode 55.3 (-15%), prefill 425 (+0%), 12.8 GiB/die [model]

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  placement tp4     slots/instance 12  streams 12  ctx/slot 32K (up to 104K fits)  kv f16       ub 4096  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   158.6 tok/s aggregate  (13.2 per stream)   prefill    983 tok/s   memory/die 17.5 GiB   [decode model, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~87 tok/s, ~20.4 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 12 -c 393216 -b 2048 -ub 4096 -cb
    runner-up: tp4 np 16 (16 streams) ctx 32K kv f16 ub 4096 draft 0 Q8_0 -> decode 160.6 (+1%), prefill 983 (+0%), 19.6 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 32K kv f16 ub 4096 draft 1 Q8_0 -> decode 147.2 (-7%), prefill 983 (+0%), 15.3 GiB/die [model]

J. eight slots at the memory ceiling: 8 streams, 160K each
  placement tp4     slots/instance 8   streams 8   ctx/slot 160K (up to 172K fits)  kv f16       ub 2048  draft 1  topo fixed16  graphs on  quant Q8_0  build prod
  decode    81.3 tok/s aggregate  (10.2 per stream)   prefill    557 tok/s   memory/die 29.0 GiB   [decode model, prefill model]
  request-level for 1300-token prompts / 256 generated: ~47 tok/s, ~10.9 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 1310720 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 1
    runner-up: tp4 np 8 (8 streams) ctx 160K kv q8_0 ub 2048 draft 1 Q8_0 -> decode 54.1 (-33%), prefill 557 (+0%), 23.6 GiB/die [model]
    runner-up: tp4 np 8 (8 streams) ctx 160K kv q8_0/q4_0 ub 2048 draft 1 Q8_0 -> decode 51.5 (-37%), prefill 557 (+0%), 17.2 GiB/die [model]
```

## Measured cells only (--measured-only)

```
gfx906 x4 / Qwen3.8-27B: MILP over 18x10x3 core cells x ubatch x draft x topo x graphs; build prod; cap 200 W; VRAM budget 31 GiB/die

A. one user, short context (chat / code, <= 8K)
  placement tp4     slots/instance 1   streams 1   ctx/slot 8K (up to 256K fits)  kv f16       ub 2048  draft 3  topo fixed16  graphs on  quant Q8_0  build prod
  decode    71.5 tok/s aggregate  (71.5 per stream)   prefill   1128 tok/s   memory/die 8.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~54 tok/s, ~12.7 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave 1.88x)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 1 -c 8192 -b 2048 -ub 2048 -cb --spec-type draft-mtp --spec-draft-n-max 3
    runner-up: tp2 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 0 Q8_0 -> decode 32.2 (-55%), prefill 586 (-48%), 16.4 GiB/die [measured]
    runner-up: layer4 np 1 (1 streams) ctx 8K kv f16 ub 2048 draft 0 Q8_0 -> decode 20.7 (-71%), prefill 293 (-74%), 8.9 GiB/die [measured(server, per-request)]

B. one user, 32K working context
  infeasible: no feasible core configuration for this workload

C. one user, 128K working context
  infeasible: no feasible core configuration for this workload

D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot
  infeasible: Infeasible

E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot
  placement tp4     slots/instance 8   streams 8   ctx/slot 32K (up to 172K fits)  kv f16       ub 2048  draft 0  topo fixed16  graphs on  quant Q8_0  build prod
  decode   165.6 tok/s aggregate  (20.7 per stream)   prefill   1116 tok/s   memory/die 13.0 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~94 tok/s, ~22.1 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -c 262144 -b 2048 -ub 2048 -cb
    runner-up: tp4 np 8 (8 streams) ctx 32K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 155.3 (-6%), prefill 1116 (+0%), 10.5 GiB/die [measured]

F. offline batch generation, short prompts, no latency floor
  placement dp4     slots/instance 8   streams 32  ctx/slot 4K (up to 4K fits)  kv f16       ub 2048  draft 0  topo default  graphs on  quant Q8_0  build prod
  decode   268.6 tok/s aggregate  (8.4 per stream)   prefill   1253 tok/s   memory/die 30.9 GiB   [decode measured, prefill measured]
  request-level for 1300-token prompts / 256 generated: ~129 tok/s, ~30.1 requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)
  $LLAMA_PROD/llama-server -m Qwen3.8-27B-Q8_0.gguf --device rocmN   (one server per die, N=0..3) -fa on -np 8 -c 32768 -b 2048 -ub 2048 -cb
    runner-up: dp4 np 8 (32 streams) ctx 4K kv q8_0 ub 2048 draft 0 Q8_0 -> decode 259.7 (-3%), prefill 1253 (+0%), 29.1 GiB/die [measured]
    runner-up: tp4 np 32 (32 streams) ctx 4K kv f16 ub 2048 draft 0 Q8_0 -> decode 213.8 (-20%), prefill 1128 (-10%), 11.9 GiB/die [measured]

G. long-context server: 128K per slot, max aggregate, >= 4 streams
  infeasible: no feasible core configuration for this workload

H. full 256K context, as many slots as fit
  infeasible: no feasible core configuration for this workload

I. prompt ingestion (RAG indexing): maximise prefill, 32K documents
  infeasible: no feasible core configuration for this workload

J. eight slots at the memory ceiling: 8 streams, 160K each
  infeasible: no feasible core configuration for this workload
```
