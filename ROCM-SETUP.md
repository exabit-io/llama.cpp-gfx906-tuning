# ROCm 7.14 (TheRock) + PyTorch + llama.cpp on the Mac Pro 7,1 gfx906 dies

Set up 2026-09-03. Ubuntu 24.04.4, kernel 7.0.0-30-generic, 4x Vega 20
(2x Radeon Pro Vega II Duo MPX, PCI 0b/0e/1b/1e:00.0), every die BAR0 = 32 GiB
via resize-amdgpu-bars.service. All links Gen3 x16 (the box's ceiling).

## What was installed and from where

Source of truth: https://github.com/mixa3607/ML-gfx906 (clone in /root/ML-gfx906)
(for **ROCm** only: mixa3607's ML-gfx906 has no llama.cpp source. The llama.cpp code is `exabit-io/mx-llama.cpp`
(`master` = the substrate), see REQUIREMENTS R3.7.)
and https://arkprojects.space/wiki/AMD_GFX906. AMD dropped gfx906 from ROCm;
mixa3607 builds ROCm from AMD's TheRock with gfx906 enabled and publishes debs.

    apt source : /etc/apt/sources.list.d/gfx906.sources  (key /etc/apt/keyrings/apt-gfx906.asc)
    build      : 7.14.0-gfx906+20260802001858
    packages   : amdrocm7.14 amdrocm-core-sdk7.14 amdrocm7.14-gfx906
                 amdrocm-core-sdk7.14-gfx906 amdrocm-hpc-sdk7.14-gfx906
                 rocm-validation-suite amdrocm7.14-transferbench
    prefix     : /opt/rocm  (-> /opt/rocm/core-7.14)
    env        : /etc/profile.d/rocm.sh  (ROCM_PATH, HIP_PATH, PATH)
    ldconfig   : /etc/ld.so.conf.d/rocm.conf
    groups     : nbritton added to render + video (re-login needed)

DO NOT add AMD's official ROCm apt repo on top of this; the packages conflict.

Rebuild from source if ever needed: /root/ML-gfx906/rocm/BUILD-PACKAGES.md
(TheRock, -DTHEROCK_AMDGPU_FAMILIES=gfx906, hipBLASLt/rocWMMA off).

## PyTorch

    venv    : /root/pytorch-gfx906/venv      (source venv/bin/activate)
    wheels  : /root/pytorch-gfx906/wheels    torch 2.13.0 / torchvision 0.27.0 / torchaudio 2.11.0, +gfx906, cp312
    check   : /root/pytorch-gfx906/venv/bin/python /root/pytorch-gfx906/torch_check.py

## llama.cpp

    source  : /root/llama.cpp  (ggml-org tag b10288, same as ML-gfx906 preset b10288-rocm-7.14-ggml)
    install : /opt/llama.cpp/bin  (llama-server, llama-cli, llama-bench, ...)
    build   : /root/llama.cpp/build-native  (installed one)
    flags   : GGML_HIP GGML_HIP_GRAPHS GGML_HIP_RCCL GGML_RPC, GGML_NATIVE=ON,
              CMAKE_C/CXX_FLAGS=-march=native (Xeon W-3245: AVX512 + AVX512_VNNI),
              GGML_BACKEND_DL=OFF (GGML_NATIVE requires it; backends linked statically),
              AMDGPU_TARGETS=gfx906, CMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp"
    rebuild : cmake --build build-native -j32 && cmake --install build-native
    also    : /root/llama.cpp/build = the project's exact recipe (GGML_BACKEND_DL +
              GGML_CPU_ALL_VARIANTS, runtime-picks libggml-cpu-cascadelake which also has VNNI)
    env     : /opt/llama.cpp/bin on PATH via /etc/profile.d/rocm.sh; /etc/ld.so.conf.d/llama.cpp.conf
    model   : /root/models/Qwen2.5-1.5B-Instruct-Q8_0.gguf (smoke-test model)

The wiki's ROCm comparison (6.3.3 vs 7.2.3, Qwen 3.5 9B Q8_0) found no
meaningful difference, so 7.14 is fine for llama.cpp. The project's alternate
"mxxm" preset (mxxm-t/mx-llama.cpp b10254 + patch
/root/ML-gfx906/llama.cpp/build-context/patch/mxxm-gfx906-kcase.patch) tunes
Q5_K/Q6_K MMQ tiles for MI50 by 15-35%. Built 2026-09-07 as /opt/llama.cpp-mxxm
(bench/build-mxxm.sh): its own gfx906 Q8_0 tile table reads prompts +33% on four
dies and +35% on one, perplexity identical to stock. /opt/llama.cpp-mxxm-fh adds
the gfx906 MMVQ Q8_0 fast path (llama.cpp-benchmarking/patches/, bench/build-mxxm-fh.sh)
and is the PRODUCTION build: decode +62% at 12 slots / +34% at 16 on the split,
+31% at 8 slots on one die (llama.cpp-benchmarking/reports/2026-09-07-todo-runthrough.md
s.10c, s.11; settings/launch.sh defaults to it).

## Verified results (2026-09-03)

rocBLAS (/root/rocm-tests/gemm_bench, hipcc-built), per die:

    FP32 GEMM 4096^3        ~11.3 TFLOPS   (peak 14.1)
    FP16 in / FP32 acc 8192 ~18.5 TFLOPS   (peak 28.2)
    plain hgemm (FP16 acc)  ~17.5 TFLOPS   at 4096, but few-% error at K=4096: use f32 accumulate

PyTorch 4096^3 matmul per die: fp32 ~11 TF, fp16 ~11 TF, bf16 ~6 TF (gfx906 has
no bf16 dot instructions; prefer fp16). MIOpen conv fwd/bwd OK. P2P between all
dies OK. RCCL 4-GPU all_reduce of 1 GiB: 65 ms, ~15 GiB/s algbw.
RVS gst fp16 4096 30 s: PASS on all 4 (11.9 TFLOPS each), GPU0 hit 197 W /
87 C junction at 1540 MHz.

llama.cpp b10288, Qwen2.5-1.5B Q8_0, -fa 1 (small model, so multi-GPU tg does
not scale; pp does):

    1 die : pp512 3684 t/s  pp2048 3575 t/s  tg128 237 t/s
    4 dies (-sm layer): pp2048 7386 t/s  tg128 222 t/s

## Inter-GPU topology

All four dies are one XGMI hive (dmesg "XGMI: Add node N, hive 0x9190d9ff..."):
the two dies on each Vega II Duo are IF-linked on-card, and the MPX Infinity
Fabric Link bridge joins the two cards. rocm-smi --showtopo reports XGMI for
every pair. TransferBench p2p (256 MiB, GFX engine):

    GPU->GPU unidirectional  ~28 GB/s   (PCIe Gen3 x16 alone would cap ~12-13)
    GPU<->GPU bidirectional  ~52 GB/s on 4 of 6 pairs, ~21 GB/s on the other 2 (2-hop)
    CPU->GPU / GPU->CPU      ~10-11 GB/s (PCIe Gen3)
    local HBM2 copy          ~315-322 GB/s

## Known limits on gfx906

- No hipBLASLt gfx906 device library (see /root/ML-gfx906/rocm/HIPBLASLT-GFX906.md).
  torch._int_mm and other INT8 paths that require hipBLASLt fail. rocBLAS covers GEMM.
- No Triton in the wheel: torch.compile fails with TritonMissing. Eager only.
- No MFMA / matrix cores (that starts at gfx908). "Tensor" throughput is
  packed-FP16 dot instructions through rocBLAS Tensile kernels.
- amd-memory-tweak / amd-tuning (HBM timing + PowerPlay table edits from the
  wiki perf-tuning page) are available in the apt repo but NOT installed: they
  target MI50 firmware tables and the Vega II MPX cards are a different board.

## Handy commands

    rocminfo | grep -E 'Name:|gfx'          # agents
    rocm-smi ; amd-smi monitor              # clocks/temp/power
    rocm-smi --showrasinfo                  # ECC/RAS
    rvs -c /root/rocm-tests/rvs-gst-fp16-30s.conf
    TransferBench                            # inter-GPU bandwidth
    HIP_VISIBLE_DEVICES=0,1 ...              # pick dies

## Amendments (2026-09-07, technical-lead review)

- Kernel is now 7.0.0-31-generic +barfix1 (was -30 when this was written); the resize-amdgpu-bars service still gives every die a 32 GiB BAR0.
- Host CPU was swapped to a Xeon W-3275M (28c/56t) on 2026-09-05 (was W-3245). The llama.cpp builds were compiled with -march=native on the W-3245; both parts are Cascade Lake (AVX512 + VNNI), so the binaries are unchanged and valid.
- The 1000 MHz "clamp" seen from 2026-09-03 on is the SMC's 1228 W DC power envelope, not thermal and not a driver fault (see /root/llama.cpp-benchmarking/reports/2026-09-07-todo-runthrough.md, "Interruption 3"). Rules: host CPU RAPL-capped at 150 W beside four loaded dies (gpu-test-env.sh does it), no host work beside GPU jobs, a cold power cycle after any clamp. The GPU numbers in "Verified results" above were taken before the clamp was understood and were not re-verified against the five-second clamp test.
- Follow-on builds beside /opt/llama.cpp (b10288): /opt/llama.cpp-faq (GGML_CUDA_FA_ALL_QUANTS), /opt/llama.cpp-mmvq16 (MMVQ_MAX_BATCH_SIZE 16), /opt/llama.cpp-cublas (FORCE_CUBLAS, retired), and /opt/llama.cpp-b10837 (upstream update for BENCHMARKS-TODO item 4, built by /root/rocm-tests/bench/build-b10837.sh). The tuning guide's numbers are all b10288.
- Serving model is Qwen3.8-27B (Q8_0 and four other quants in /root/models); the Qwen2.5-1.5B file above is only the smoke-test model.
