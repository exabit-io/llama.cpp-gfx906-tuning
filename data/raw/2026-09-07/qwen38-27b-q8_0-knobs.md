# qwen38-27b-q8_0-knobs: llama-bench 4 dies tensor -p 2048 -n 256 -d 0 -r 3, then batched-bench b8 pp2048/tg128; baseline env = NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16  2026-09-07T03:06:01+00:00
| variant | extra env / host | pp2048 t/s | tg256 t/s | b8 pp t/s | b8 tg t/s |
|---|---|---:|---:|---:|---:|
| base-1 | (baseline) | 846.47 | 47.57 | 847.82 | 159.22 |
| proto-ll | NCCL_PROTO=LL | 811.40 | 47.43 | 812.14 | 164.94 |
| proto-simple | NCCL_PROTO=Simple | 848.50 | 40.46 | 849.05 | 158.88 |
| proto-ll128 | NCCL_PROTO=LL128 | 848.41 | 40.47 | 848.99 | 158.71 |
| algo-ring | NCCL_ALGO=Ring | 848.40 | 47.49 | 849.08 | 158.58 |
| ch16-16 | NCCL_MAX_NCHANNELS=16 NCCL_MIN_NCHANNELS=16 | 848.60 | 47.58 | 849.15 | 152.55 |
| ch32 | NCCL_MIN_NCHANNELS=32 | 846.95 | 47.38 | 847.60 | 155.95 |
| ch8 | NCCL_MIN_NCHANNELS=8 | 849.80 | 47.50 | 850.42 | 157.80 |
| base-2 | (baseline) | 848.75 | 47.60 | 849.15 | 155.10 |
| dev-kernarg | HIP_FORCE_DEV_KERNARG=1 | 848.79 | 47.47 | 849.07 | 158.59 |
| hwq1 | GPU_MAX_HW_QUEUES=1 | 849.00 | 47.45 | 849.13 | 156.29 |
| hwq8 | GPU_MAX_HW_QUEUES=8 | 848.47 | 47.40 | 849.26 | 157.99 |
| no-sdma | HSA_ENABLE_SDMA=0 | 844.25 | 47.53 | 844.74 | 155.74 |
| ll+kernarg | NCCL_PROTO=LL HIP_FORCE_DEV_KERNARG=1 | 812.17 | 47.51 | 812.64 | 160.78 |
| base-3 | (baseline) | 848.44 | 47.56 | 849.25 | 156.22 |
| c6-off | (baseline) host: C6 disabled (cpupower idle-set -d 3) | 847.79 | 47.24 | 847.44 | 155.94 |
| c1e-c6-off | (baseline) host: C1E+C6 disabled (idle-set -d 2 -d 3) | 847.45 | 47.39 | 846.12 | 155.11 |
| ll+kernarg+c6off | NCCL_PROTO=LL HIP_FORCE_DEV_KERNARG=1 host: C1E+C6 disabled | 811.22 | 47.19 | 808.82 | 162.61 |
| base-4 | (baseline) | 849.04 | 47.69 | 849.16 | 157.59 |
# done 2026-09-07T03:34:56+00:00
