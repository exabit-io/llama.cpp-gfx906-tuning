# qwen38-27b-q8_0-opt A: llama-bench 4 dies, -p 2048 -n 256 -r 2, tensor split unless noted; then batched-bench b8 pp2048  2026-09-06T18:37:01+00:00
| variant | env / flags | pp2048 t/s | tg256 t/s | b8 pp t/s | b8 tg t/s |
|---|---|---:|---:|---:|---:|
| base | OPT=base NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH | 834.15 | 45.60 | 834.72 | 154.63 |
| topo-fixed | NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH | 849.11 | 46.10 | 850.39 | 155.06 |
| topo-fixed-16ch | NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 | 848.00 | 47.30 | 848.87 | 155.36 |
| internal | GGML_CUDA_ALLREDUCE=internal | 671.28 | 34.93 | 672.56 | 141.61 |
| internal-p2p | GGML_CUDA_ALLREDUCE=internal GGML_CUDA_P2P=1 | 666.87 | 34.67 | 672.24 | 139.87 |
| none | GGML_CUDA_ALLREDUCE=none | 671.16 | 35.02 | 672.50 | 141.90 |
| nccl-p2p | GGML_CUDA_P2P=1 | 835.04 | 45.48 | 835.80 | 154.52 |
| no-graphs | GGML_CUDA_DISABLE_GRAPHS=1 | 834.97 | 42.31 | 835.88 | 150.24 |
| row-split | OPT=row -sm row| fail | fail | fail | fail |
# done 2026-09-06T18:47:54+00:00
