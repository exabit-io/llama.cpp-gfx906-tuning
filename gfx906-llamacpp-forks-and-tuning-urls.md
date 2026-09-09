# gfx906 (AMD Vega 20) — llama.cpp forks and tuning resources

Target silicon: AMD Vega 20 / "Vega 7nm" ISA `gfx906` — Instinct MI50, Instinct MI60, Radeon VII, Radeon Pro VII, Radeon Pro Vega II (Mac Pro 7,1 MPX), Hygon Z100 (licensed derivative).
Scope: Linux + ROCm/HIP only. macOS, Windows and PowerPC material deliberately excluded.
Compiled: 2026-09-09 from GitHub repository search (incl. `fork:true`), GitHub PR search, and web search. Every URL below was checked on 2026-09-09 and returned HTTP 200, except the two marked `(dead)` (kept because other sources still reference them) and the two Discord invites (not checkable). A companion file `gfx906-urls.txt` holds just the live URLs, one per line, in the same order.
Star counts / dates are as of 2026-09-09. Tags in brackets are for filtering.

Tag key: [KERNELS] fork with gfx906-specific kernel/code changes · [FIX] narrow bug-fix fork · [BUILD] build/packaging/Docker of llama.cpp for gfx906 · [MIRROR] unmodified fork, listed for completeness · [UPSTREAM] ggml-org/llama.cpp PR/issue · [GUIDE] tuning/setup knowledge · [ROCM] ROCm-stack fixes for gfx906 · [ENGINE] non-llama.cpp gfx906 inference engine (kernel lessons transfer) · [POWER] clocks/power/PCIe/firmware tools · [HUB] link aggregator / community

---

## 1. Customized llama.cpp forks with gfx906-specific code changes [KERNELS]

Ordered roughly by depth of gfx906-specific work. Read these first.

- https://github.com/iacopPBK/llama.cpp-gfx906 — [KERNELS] The reference gfx906 fork (143★, 18 forks). Warp-cooperative MMVQ kernels for Q4_0/Q4_1/Q8_0, DPP-based wave64 reductions replacing shuffles, Q8 FlashAttention + RoPE kernels, fused norm+quantize, software-pipelined MMQ loads, MXFP4 conversion. Tracks upstream build b7924; README says "not updated, will be updated asap". Author also submitted upstream PRs #16291, #21168, #21698 (see §2). Links a GFX906 Discord: https://discord.gg/ZEcgt3dAw
- https://github.com/milpster/gfx906-llama-cpp — [KERNELS] (17★) Tracks upstream master (synced 2026-09-03 @ 95ef7fc16). Wave64 kernel tables, Vega20-specific tile configuration, native q8_0 tile FA kernel, mixed F16-K/q8_0-V tile kernel, selective native/convert FATTN paths for Q≤16, cost-based split mode for ROCm+Vulkan hybrid rigs, controllable pipeline parallelism, frugal buffer management ("250k context for 27B Q6_K on 40 GB"). Reported +23% prefill (16k), +14% 120k fill, +11% TG at 120k depth vs upstream. Findings in `journal/` and `bench/`. Key commits: 604fca5 (cost-based split), 468c164 (Vega20 tile config), d14628d (q8_0 tile kernel), eaca6d4 (mixed F16-K/q8_0-V). Discussion thread: https://forum.level1techs.com/t/glm-and-i-created-a-llama-cpp-fork-optimized-for-amd-gfx906-mi50-mi60-radeon-vii-gcn-hip/254257
- https://github.com/mxxm-t/mx-llama.cpp — [KERNELS] (46★, pushed 2026-09-09, very active) Multi-GPU-oriented fork with gfx906 kernel-level tuning; keeps single-GPU and speculative decoding working. See FEATURES.md. Docker: https://hub.docker.com/r/mxxm/mx-llama.cpp and https://hub.docker.com/r/mxxm/llama-cpp-gfx906 . This is the base that exabit-io/llama.cpp (branch gfx906) builds on.
- https://github.com/arte-fact/llamacpp-gfx-906-turbo — [KERNELS] (24★) iacopPBK wave64 kernels + TurboQuant turbo3 KV-cache compression (claims 4.6× KV compression, 300K ctx vs 90K f16 on same VRAM, ~18% TG cost) + HIP graphs. ~1,800 lines of patches; Build33 rebased onto upstream b9549. Write-up: https://machinesoftworks.com/posts/custom-gfx906-llamacpp-engine . Cites https://github.com/Madreag/turbo3-cuda (TurboQuant CUDA reference) and https://github.com/TheTom/llama-cpp-turboquant (CPU/Metal variant), paper https://arxiv.org/abs/2504.19874
- https://github.com/moriyasujapan/llamacpp-gfx-906-turbo-gemma4 — [KERNELS] arte-fact turbo fork + Gemma 4 support (Gemma 4 31B Q4_0 ≈21.6 t/s, dense 9B turbo3 ≈57 t/s per MachineSoftworks post).
- https://github.com/moriyasujapan/llama-cpp-gfx906-multibackend — [KERNELS][BUILD] CUDA sm_120 + HIP gfx906 + Vulkan in one binary for MI50/Radeon Pro VII + RTX 50 mixed rigs.
- https://github.com/sixvolts/llamacpp-gfx906-furnace — [KERNELS] (12★) Work on branch `gfx906-perf` as reviewable commits on top of upstream master: three-plane K-quant weight repack for bandwidth, DPP/ds_swizzle warp reductions instead of LDS, MoE MUL_MAT_ID with in-kernel expert routing. MI50: +18–24% dense, +12% MoE. Techniques ported from the author's reinstinct engine (see §6).
- https://github.com/eslowney/llama.cpp-gfx906 — [KERNELS] (7★) Custom flash-attention tile kernel `fattn-tile-f16-gfx906.cu`: 64-thread wavefront, register blocking, native V_DOT2_F32_F16, targeted at D=128 heads (Qwen3-30B). MI50: ~1,224 t/s pp512, ~63 ms/token TG with KV quant. Derived from skyne98's work. Mirror forks: https://github.com/anikifoss/llama.cpp-gfx906 , https://github.com/thickprogrammer/llama.cpp-gfx906
- https://github.com/alex4300/llama.cpp-gfx906-opt — [KERNELS] (branch `gfx906`, pushed 2026-09-09) "Measured kernels for Vega20", Qwen3.8 optimisations. See README-gfx906.md and docs/gfx906/.
- https://github.com/Luna-AI-Infra/Mi50-Qwen38-27B — [KERNELS] HIP kernel overlay library for MI50 16 GB running Qwen3.8-27B: Q3 hybrid storage + specialised GEMV (I2), two-wave reduction with fused residuals (R19), exact-QX kernels for Q5/Q6/Q8 (R4b). Claims ~2.13× baseline throughput. Profiling/validation reports in docs/reports/.
- https://github.com/THEman6989/llama.cpp-gfx906-wrapper — [KERNELS] Patch-set wrapper (`apply-turbo.sh`) that applies wave64 kernels + TurboQuant KV + HIP graphs onto upstream commit acd604fb without maintaining a fork. Companion forks: https://github.com/THEman6989/llama.cpp-gfx906-turbo-mtp (3★, turbo + MTP) and https://github.com/THEman6989/llama.cpp-gfx906-turbo-mtp-old
- https://github.com/stevio2d/GFX906-MI50-optimisations — [KERNELS][GUIDE] (10★) Patch collection for MI50 on ROCm 7.x: flash attention integration, tq3_0 (TurboQuant) KV compression, validation logs, hardware-config guidance. Code branch: https://github.com/stevio2d/llama.cpp-gfx906/tree/tq3_0-mi50-slim-pr ; validation: https://github.com/stevio2d/GFX906-MI50-optimisations/blob/main/validation-tq3-fa-2026-03-27.md ; related research repo: https://github.com/stevio2d/GFX906-mixed-cuda-rocm-research
- https://github.com/Kausik-A/gemma4-mi50-optimizations — [KERNELS] Q8_1 reuse and fused-QKV optimisation for MI50/gfx906 (personal-use).
- https://github.com/seekkii/llama.cpp — [KERNELS] Fork described as "optimized for AMD Instinct MI50 (gfx906) via ROCm/HIP".
- https://github.com/362132718/llamacpp-gfx906-slot-save — [KERNELS] Built on sixvolts furnace; merges upstream PRs #20819/#20822 for slot-state persistence / auto save-restore on SIGTERM (fast model hot-swap on Radeon Pro VII).
- https://github.com/DByte308/ornith-mtp-gfx906 — [KERNELS][GUIDE] Finetuning the MTP speculative-decoding head of Ornith-1.0-35B on 2× gfx906; includes llama.cpp patches and tuning findings.
- https://github.com/Wizard815/mx-llama.cpp-Rocm10 — [KERNELS][BUILD] mx-llama.cpp adapted to ROCm 10.
- https://github.com/renlililoli/hygon-z100-qwen38-llamacpp — [KERNELS][FIX] Tested llama.cpp HIP patch + config for Hygon Z100 (gfx906 derivative) on DTK 26.04.
- https://github.com/bowmanjd/franken-llama — [BUILD][KERNELS] llama.cpp flake.nix supporting MI50 + CUDA, includes TurboQuant.
- https://github.com/FreesoSaiFared/llama.cpp-gfx906-semantic — [KERNELS] Attempt to derive "semantic (prompt) patches" from the iacopPBK diff so the changes can be re-applied to newer upstream.
- https://github.com/exabit-io/llama.cpp/tree/gfx906 — [KERNELS] (your own) upstream master + mx-llama.cpp + Exabit kernel series; companion guide https://github.com/exabit-io/llama.cpp-gfx906-tuning (4× Vega 20 tuning guide, 31 benchmark tables, LP optimiser, kernel patches: +20% single-stream, +33% prefill vs b10288/b10912 baselines).
- (dead) https://github.com/skyne98/llama-labs-gfx906 — referenced by iacopPBK README as the kernel-experiments lab; now 404 (private or renamed). The published results live in the skyne98 wiki studies (§3).

### 1a. Narrow bug-fix forks [FIX] — mostly the SOLVE_TRI / rocblas_strsm segfault on ROCm 7.x with Qwen3.5/3.6/Qwen3-Next Gated-DeltaNet models
- https://github.com/Luskendeilder/llama.cpp-gfx906-solve-tri — SOLVE_TRI crash fix for gfx906 + ROCm 7.x.
- https://github.com/ollo12-prog/llama.cpp-gfx906-solve-tri-fix — Qwen3.5 crash fix on MI50 (mirror: https://github.com/mt4wss/llama.cpp-gfx906-solve-tri-fix ).
- https://github.com/MarcoTool/gfx906-ROCm7.0.2-llama.ccp-Qwen3-next-Support — Docker deployment on ROCm 7.0.2 with rocBLAS 6.3 patch for SSM/Mamba (Qwen3-Next) — SOLVE_TRI fix.
- https://github.com/Wuqiyang312/gfx906-llama-cpp — llama.cpp for gfx906 on ROCm 7.0.2 (mirrors: mt4wss/, incode0-debug/).
- https://github.com/taobaoww2010-alt/llama-cpp-rocm-gfx906 — ROCm 6.3.0 + gfx906 compile patches and binaries. Same author: https://github.com/taobaoww2010-alt/llama-cpp-gfx906_20260413 (Radeon Pro VII), https://github.com/taobaoww2010-alt/llama-cpp-to-amd-mi50 , https://github.com/taobaoww2010-alt/llama-cpp-gfx906
- https://github.com/abubakerkhidir/llama.cpp — fork "with support for AMD Radeon VII".
- https://github.com/blockfeed/llama.cpp-hip-gemma4-mtp — Arch PKGBUILD: llama.cpp ROCm/HIP + Gemma 4 MTP (PR #23398); not gfx906-exclusive.

### 1b. Builds, packaging, Docker and deployment wrappers of llama.cpp for gfx906 [BUILD]
- https://github.com/mixa3607/ML-gfx906 — (337★) The main build farm: llama.cpp, vLLM, ComfyUI, PyTorch, ROCm toolkit for gfx906. Docker Hub namespace `mixa3607` (`rocm-gfx906`, `llama.cpp-gfx906`, `pytorch-gfx906`, `comfyui-gfx906`, `rocm-toolkit-gfx906`), APT repo https://s3.arkprojects.space/apt-gfx906/ubuntu (repo root has no index page; use as an apt source, see README), includes ROCm validation suite, bandwidth tests, AMD Memory Tweak, Vega20 Prometheus exporter. Docker: https://hub.docker.com/r/mixa3607/llama.cpp-gfx906 . Wiki: https://arkprojects.space/wiki/AMD_GFX906 . Discord: https://discord.gg/EgsTWBqPr . Releases: https://github.com/mixa3607/ML-gfx906/releases
- https://github.com/phantomic12/ML-gfx906 — gfx906 builds incl. ROCm TheRock; GitHub-hosted apt repo proof-of-concept.
- https://github.com/kyuz0/mi50-gfx906-toolboxes — (26★) Toolbox/container images for MI50.
- https://github.com/anng-phtk/rocm-vega-llama.cpp — llama.cpp on Vega 64 (gfx900) and Radeon VII/MI50 (gfx906) with ROCm 7.2, incl. mixed-GPU inference.
- https://github.com/monstro-das-bolachas/Radeon-Instinct-MI50-with-ROCm-6.2.4-running-llama.cpp — compatibility workarounds, multi-GPU config, troubleshooting.
- https://github.com/kinchahoy/gfx906-llama-docker — ROCm Docker with llama.cpp for MI60.
- https://github.com/cyber-will-3/qwen-on-gfx906 — Benchmarks + runnable Docker setup for Qwen on Radeon VII via llama.cpp + ROCm.
- https://github.com/amstel8/truenas-llamacpp-vega20 — llama.cpp on TrueNAS for Vega 20; Docker https://hub.docker.com/r/amstel8/llama-rocm
- https://github.com/logicalor/llama.cpp_rocm_mi50 — llama.cpp Docker with multi-GPU incl. MI50.
- https://github.com/xxDoman/combo-llama-rpc-hybrid — llama.cpp RPC hybrid cluster: RTX 40xx + MI50 (gfx906) VRAM pooling.
- https://github.com/Scottcjn/ollama-dual-arch-amd — one runtime across MI50 (gfx906) + RDNA4 (gfx1200).
- https://github.com/spoto-team/pve-mi50-qwen3.6 — Proxmox LXC + MI50 + Qwen3.6 deployment guide.
- https://github.com/janit/viiwork — LLM inference load balancer optimised for Radeon VII GPUs.
- https://github.com/arte-fact/llama-monitor — llama-server management/monitoring web UI (same author as the turbo fork).
- https://github.com/Abiaselli/homeassistant-amd-pipeline — Whisper + LLM + Functionary on an MI50 for Home Assistant.
- https://github.com/A1M918/LocalAI-Custom — Dockerfiles to build/run LocalAI on MI50.
- Docker Hub images (gfx906 llama.cpp): https://hub.docker.com/r/aceshigh1/llama-mi50 , https://hub.docker.com/r/arkahnat/llama.cpp-gfx906 , https://hub.docker.com/r/mxxm/mx-llama.cpp , https://hub.docker.com/r/mxxm/llama-cpp-gfx906
- (dead) knguyen298/llama-swap-gfx906 — Docker image with llama-swap router, mentioned in the Level1Techs thread; GitHub URL 404 as of 2026-09-09.

### 1c. Unmodified / mirror forks [MIRROR] — no distinct gfx906 work found; listed only so the downstream agent can de-duplicate
Forks of iacopPBK/llama.cpp-gfx906: https://github.com/keyz182/llama.cpp-gfx906 (branch `keyz182`) · https://github.com/daniHagl/llama.cpp-gfx906 · https://github.com/TheGreaterWatt/llama.cpp-gfx906 · https://github.com/unverbraucht/llama.cpp-gfx906 · https://github.com/stevio2d/llama.cpp-gfx906 (has the tq3_0 branch, see §1) · https://github.com/ZanMax/llama.cpp-gfx906 · https://github.com/yashrajgandhi95/llama.cpp-gfx906 · https://github.com/pl161187smi/llama.cpp-gfx906 · https://github.com/aaron-harvey/llama.cpp-gfx906 · https://github.com/philmcneely/llama.cpp-gfx906 · https://github.com/jtjames/llama.cpp-gfx906 · https://github.com/randomizedcoder/llama.cpp-gfx906 · https://github.com/kamali-lab/llama.cpp-gfx906 · https://github.com/logicalor/llama.cpp-gfx906 · https://github.com/ikantkode/llama.cpp-gfx906 · https://github.com/nltbinhaavn/llama.cpp-gfx906 · https://github.com/fuutott/llama.cpp-gfx906 · https://github.com/bearqq/llama.cpp-gfx906
Forks of milpster/gfx906-llama-cpp: https://github.com/TuringRepeater/gfx906-llama-cpp · https://github.com/jozefkun/gfx906-llama-cpp
Forks of arte-fact/llamacpp-gfx-906-turbo: https://github.com/dexter-super-work/llamacpp-gfx-906-turbo · https://github.com/airnsk/llamacpp-gfx-906-turbo · https://github.com/rjohny55/llamacpp-gfx-906-turbo
Forks of mxxm-t/mx-llama.cpp: https://github.com/jgbrblmd/mx-llama.cpp · https://github.com/Te-eMster/mx-llama.cpp · https://github.com/BurnWW/mx-llama.cpp · https://github.com/0FL01/mx-llama.cpp · https://github.com/renlililoli/mx-llama.cpp · https://github.com/JCraigWasTaken/mx-llama.cpp · https://github.com/assistmeister/mx-llama.cpp · https://github.com/amplos-ai/mx-llama.cpp · https://github.com/DENEB1312/mx-llama.cpp · https://github.com/stanus74/mx-llama.cpp · https://github.com/mixa3607/mx-llama.cpp
Forks of mixa3607/ML-gfx906 (~35, e.g. unverbraucht/, MOVZX/, nbritton/, larkinwc/ (3★), sscchan/ML-gfx906-with-cuda, kyuz0/mi25-gfx900-ai-toolboxes (gfx900 variant), bryan-fund/ML-gfx906 (rewritten as a portfolio project), lifb168/ML-gfx906) — enumerate live via https://github.com/mixa3607/ML-gfx906/forks

---

## 2. Upstream ggml-org/llama.cpp PRs, issues and discussions relevant to gfx906 [UPSTREAM]

HIP/GCN kernel work (the gfx906-specific ones first):
- https://github.com/ggml-org/llama.cpp/pull/16291 — hip: substitute bpermute ops with swizzle ops (gfx906, maybe all AMD) — merged
- https://github.com/ggml-org/llama.cpp/pull/21168 — ggml-cuda: ds_read_b128 for q4_0 and q4_1 MMQ kernels — merged
- https://github.com/ggml-org/llama.cpp/pull/21698 — ggml-cuda: better VRAM→LDS loading pipeline in load_tiles_q8_0 (iacopPBK) — open
- https://github.com/ggml-org/llama.cpp/pull/21643 — "Gfx906" — closed
- https://github.com/ggml-org/llama.cpp/pull/26466 — ggml-cuda: HIP replace __shfl_xor_sync with DPP instructions — closed
- https://github.com/ggml-org/llama.cpp/pull/27841 — ggml-cuda: hip: add missing AMD GCN MMQ config — open
- https://github.com/ggml-org/llama.cpp/pull/15884 — HIP: use v_dot2_f32_f16 instruction for FA — merged
- https://github.com/ggml-org/llama.cpp/pull/15927 — CUDA: larger SRAM reads for tile FA, AMD FP16 dot — merged
- https://github.com/ggml-org/llama.cpp/pull/15769 — CUDA: faster tile FA (Pascal/AMD), headsize 256 — merged
- https://github.com/ggml-org/llama.cpp/pull/15982 — CUDA: fix FA occupancy, optimize tile kernel
- https://github.com/ggml-org/llama.cpp/pull/16492 — CUDA: faster tile FA, oob checks, more head sizes
- https://github.com/ggml-org/llama.cpp/pull/11831 — HIP: remove GCN from list of devices that avoid MMQ
- https://github.com/ggml-org/llama.cpp/pull/11519 — CUDA/HIP: selectable warp size for mmv
- https://github.com/ggml-org/llama.cpp/pull/11619 — HIP: doc on small default launch bounds
- https://github.com/ggml-org/llama.cpp/pull/11080 — only call rocblas_initialize for versions < 4 (VRAM allocation on some AMD cards)
- https://github.com/ggml-org/llama.cpp/pull/11244 — AMD: parse the architecture as supplied by gcnArchName
- https://github.com/ggml-org/llama.cpp/pull/24588 — HIP: use hipBLAS for dense prefill on gfx900, keep MMQ for MoE
- https://github.com/ggml-org/llama.cpp/pull/22094 — hip: bypass memory pool for flash attention f16 temp buffers
- https://github.com/ggml-org/llama.cpp/pull/20282 — ggml-cuda: GDN use shared mem for HIP
- https://github.com/ggml-org/llama.cpp/pull/19978 — CUDA: extend solve_tri fast kernel to k ≤ 64 (SOLVE_TRI path)
- https://github.com/ggml-org/llama.cpp/pull/20831 — cuda: dynamic MMVQ nwarps for narrow matrices — open
- https://github.com/ggml-org/llama.cpp/pull/15802 — CUDA: fastdiv, launch bounds for mmvq + q8_1 quant
- https://github.com/ggml-org/llama.cpp/pull/22933 — vulkan: opt mul_mat_vecq for MI50
- https://github.com/ggml-org/llama.cpp/pull/10498 — minimal optimizations for CDNA (background)
- https://github.com/ggml-org/llama.cpp/pull/7011 — fix flash attention for ROCm
- https://github.com/ggml-org/llama.cpp/pull/1087 — original ROCm port (history)
Multi-GPU / scheduling / speculative (MI50-tested or MI50-relevant):
- https://github.com/ggml-org/llama.cpp/pull/19922 — multi-GPU pipeline parallelism (xdev host staging) + faster model loading
- https://github.com/ggml-org/llama.cpp/pull/20793 — Sched: fewer synchronizations between tokens, fixed pipeline parallelism
- https://github.com/ggml-org/llama.cpp/pull/20395 — --sched-n-copies for pipeline parallelism
- https://github.com/ggml-org/llama.cpp/pull/24219 — [RFC][PoC] intra-prompt pipeline scheduling for multi-GPU prefill
- https://github.com/ggml-org/llama.cpp/pull/19378 — backend-agnostic tensor parallelism (experimental)
- https://github.com/ggml-org/llama.cpp/pull/23792 — TP: quantized KV cache support
- https://github.com/ggml-org/llama.cpp/pull/24554 — TP: allow 4–10 GPUs (stepfun/laguna) — open
- https://github.com/ggml-org/llama.cpp/pull/22466 — async pinned upload for -sm tensor model load — open
- https://github.com/ggml-org/llama.cpp/pull/22673 — llama + spec: MTP support
- https://github.com/ggml-org/llama.cpp/pull/27210 — spec: adaptive MTP draft depth — open
- https://github.com/ggml-org/llama.cpp/pull/16000 — --numa mirror: mirror model weights to every NUMA node — open
- https://github.com/ggml-org/llama.cpp/pull/14969 — GGML_NUMA_MIRROR implementation
- https://github.com/ggml-org/llama.cpp/pull/16653 — auto-set parameters to maximize GPU utilization
Tooling / build:
- https://github.com/ggml-org/llama.cpp/pull/19434 — tools: quant-bench for profiling raw kernel performance — open
- https://github.com/ggml-org/llama.cpp/pull/16039 — llama-bench: --devices / --list-devices
- https://github.com/ggml-org/llama.cpp/pull/19418 — update ROCm docker container to 7.2
- https://github.com/ggml-org/llama.cpp/pull/19433 — build target for ROCm 7.2 artifacts
- https://github.com/ggml-org/llama.cpp/pull/19594 — build target for ROCm 7.11 artifacts
- https://github.com/ggml-org/llama.cpp/pull/9641 — Docker ROCm builds, AMDGPU_TARGETS vs GPU_TARGETS
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md — official HIP build docs
Issues / discussions:
- https://github.com/ggml-org/llama.cpp/issues/12369 — 32B model eats too much memory on 5× MI50 (gfx906)
- https://github.com/ggml-org/llama.cpp/issues/19880 — ROCm support for newer Qwen models broken (→ SOLVE_TRI fixes in §1a)
- https://github.com/ggml-org/llama.cpp/discussions/9139 — GGML_HIPBLAS build problems on AMD/Ubuntu
- https://github.com/ggml-org/llama.cpp/discussions/11960 — GGML_HIP_UMA performance
- https://github.com/ggml-org/llama.cpp/discussions/21526 — TurboQuant KV compression full HIP/ROCm port (gfx1100; the ROCm reference for the turbo forks)
Re-run these searches for anything newer: https://github.com/ggml-org/llama.cpp/pulls?q=gfx906 · https://github.com/ggml-org/llama.cpp/pulls?q=MI50 · https://github.com/ggml-org/llama.cpp/issues?q=gfx906

---

## 3. gfx906 knowledge bases, tuning guides and benchmark write-ups [GUIDE][HUB]

- https://skyne98.github.io/wiki-gfx906/ — "Wiki GFX906" (repo https://github.com/skyne98/wiki-gfx906 , 16★). Micro-architecture studies that directly inform kernel design:
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/mi50-mi60-architecture-baseline.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-dot4-dot8-exploration.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-special-isa-quant-dequant.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-latency-hiding-ops.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-lds-layout-standard-llm.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/gfx906-kv-cache-read-write-study.html
  - https://skyne98.github.io/wiki-gfx906/studies/2026-02-21/fp32-vs-qdq-dot-gfx906.html
  - https://skyne98.github.io/wiki-gfx906/installing_ROCm_7.x.html · https://skyne98.github.io/wiki-gfx906/resources.html · https://skyne98.github.io/wiki-gfx906/reference.html
- https://arkprojects.space/wiki/AMD_GFX906 — mixa3607's wiki (specs, cache hierarchy). Sub-pages: https://arkprojects.space/wiki/AMD_GFX906/perf-tuning (upp PP-table mods: mem 1000–1150 MHz, GPU 1725–1850 MHz, 140–300 W; TDC −200 W ≈ stock perf, −10 °C; Gemma4-31B scaling tables) · https://arkprojects.space/wiki/AMD_GFX906/llamacpp (ROCm 6.3.3 vs 7.2.3, HIP graphs on/off, 1/2/4 GPU, 0/16K/32K ctx) · https://arkprojects.space/wiki/AMD_GFX906/pcie-lnk-speed · https://arkprojects.space/wiki/AMD_GFX906/vllm · https://arkprojects.space/wiki/AMD_GFX906/tools · https://arkprojects.space/wiki/AMD_GFX906/k8s-gpu-operator
- https://phantomic12.github.io/gfx906-infohub/ — Astro wiki built from Discord exports: 174+ gfx906 GitHub projects, installation, hardware, FAQ, resources (repo https://github.com/phantomic12/gfx906-infohub ).
- https://fankserver.github.io/gfx906-LLM-Inference/ — benchmarks, tuning results, production notes for MI50 on ROCm (Ornith-1.0-35B, Gemma-4, Qwen3.6; MTP, DFlash, EAGLE3); raw JSON in https://github.com/fankserver/gfx906-LLM-Inference
- https://github.com/DKingAlpha/gfx906-setup — (14★) Arch Linux gfx906 revival: ROCm 6.3.3 downgrade, package rebuilds, flash-attention/xformers/vLLM patches, kernel params, attention benchmarks.
- https://github.com/joe2gaan/localaiservers — (15★) gfx906 runtime maintenance, reproducible benchmarks, QC methods, source-level kernel research; site https://localaiservers.com
- https://github.com/larkinwc/mi50grad — "finding secrets of gfx906": custom HIP kernels, kernel P2P allreduce, fused ops, tensor parallel Qwen3.5-27B at 53.74 t/s on 4× MI50; see RESEARCH.md.
- https://github.com/exabit-io/llama.cpp-gfx906-tuning — (your own) 4× Vega 20 tuning guide + LP optimiser + kernel patches.
- https://forum.level1techs.com/t/glm-and-i-created-a-llama-cpp-fork-optimized-for-amd-gfx906-mi50-mi60-radeon-vii-gcn-hip/254257 — Level1Techs thread for milpster's fork (numbers, replies, related projects).
- https://machinesoftworks.com/posts/custom-gfx906-llamacpp-engine — "Building a custom gfx906 llama.cpp engine: Turbo3 KV, MMVQ, Build33 merge" (arte-fact fork; MMVQ Q4_0 55–70 t/s vs 20–30 generic; Llama-3.1-8B Q4_K_M 84 t/s TG / 301 t/s PP with amdgpu.noretry=0).
- https://www.ywian.com/blog/amd-mi50-llm-benchmark-the-budget-vram-king — MI50 LLM benchmark blog.
- https://portegi.es/blog/running-llama-cpp-on-rocm-on-amd-instinct-mi50 — "Running llama.cpp on ROCm on AMD Instinct MI50".
- https://github.com/ai-infos/guidances-setup-8-mi50-llm (14★) · https://github.com/ai-infos/guidances-setup-16-mi50-deepseek-v32 (29★) · https://github.com/ai-infos/guidances-setup-16-mi50-qwen35-397b · https://github.com/ai-infos/guidances-setup-8-mi50-glm47-minimax-m21 (14★) · https://github.com/ai-infos/guidances-setup-32-mi50-kimi-k26 — multi-MI50 rig setup guides (8/16/32 cards).
- https://github.com/nullkalahar/mi50-rocm7 — (7★) complete ROCm 7.0.2 install guide for MI50 (Portuguese).
- https://github.com/noshitcoding/rocm7-mi50 — ROCm 7.x on MI50 upgrade & patch docs (German); companions https://github.com/noshitcoding/vllm-mi50 , https://github.com/noshitcoding/vllm-rocm-mi50 , https://github.com/noshitcoding/ollama-rocm-mi50
- https://github.com/dcruver/MI60 — notes on running an MI60 in a standard PC.
- https://github.com/Igneous/mi50-16g_gfx906_benchmarks — assorted MI50 16 GB benchmarks.
- https://github.com/Hermann-SW/1.0003-POPS — 1 peta-op INT4 synthetic benchmark on 8× MI50 + Radeon VII + Radeon Pro VII.
- https://github.com/dralexsys-b/mi50-snapshot — disaster-recovery toolkit for MI50/Radeon VII ROCm inference stacks.
- https://github.com/luna-niemitalo/rocm_tinkering_gfx906 — ROCm tinkering notes for gfx906.
- https://github.com/mobius/gfx906-perf-test — gfx906 perf test scripts.
- https://github.com/newplayman/amd_mi50 · https://github.com/AETS-MAGI/vega-hbmx-pages — small MI50/Vega note repos (undocumented).
- https://github.com/CypherpunkSamurai/awesome-llama-cpp — index of llama.cpp community forks (lists iacopPBK).
- Community chat: GFX906 Discord https://discord.gg/ZEcgt3dAw (iacopPBK) · https://discord.gg/EgsTWBqPr (mixa3607/ML-gfx906).
- Reddit r/LocalLLaMA threads exist for most of the forks above but Reddit could not be crawled from this session; search there directly: https://www.reddit.com/r/LocalLLaMA/search/?q=gfx906 and https://www.reddit.com/r/LocalLLaMA/search/?q=MI50+llama.cpp

---

## 4. ROCm-stack fixes and builds for gfx906 (rocBLAS/Tensile, sramecc, TheRock, Triton, flash-attention) [ROCM]

- https://github.com/Intermountainh8ter/rocm-gfx906-sramecc-fix — Fixes "No compatible code objects found for gfx906:sramecc-:xnack-" kernel-launch segfault on sramecc- cards (Radeon Pro Vega II / MI50 / Radeon VII): rebuilt rocBLAS, rocSPARSE, rocALUTION with Tensile for the exact ISA; prebuilt for ROCm 7.2.3, no root needed.
- https://github.com/sheldonrong/rocblas-radeon-vega — rocBLAS + TensileLibrary files for gfx900/gfx906 injectable into ROCm 6.4 and 7.2.
- https://github.com/MTLoser/ollama-mi50-rocm71-build — ROCm 7.1 runtime + ROCm 6.3 Tensile libs + SOLVE_TRI kernel patch, full build guide (Ollama, but the Tensile-lib recipe applies to llama.cpp).
- https://github.com/Wizard815/TheRock-gfx906 — gfx906-focused fork of ROCm/TheRock build system.
- https://github.com/c-harris-crux/rocm-libraries-gfx906 — rocm-libraries monorepo with gfx906 fixes.
- https://github.com/lamikr/rocm_sdk_builder — (444★) ROCm SDK builder with gfx906 support; https://github.com/CuteSC2/rocm_sdk_builder_advanced — fork adding CDNA1/GCN5.1 (Vega7/MI50) + Zen optimisations.
- https://github.com/ROCm/rockbuilder — ROCm's own build tool (gfx906 readme mention).
- https://github.com/ROCm/legacy-rocm-build — AMD legacy-architecture build repo.
- https://github.com/ROCm/TheRock/issues/1844 — ROCm and torch failures on gfx906 (TheRock) · https://github.com/ROCm/TheRock/issues/5149 — gfx906 issue thread · https://github.com/ROCm/ROCm/discussions/4276 — ROCm device support wishlist.
- https://therock-nightly-tarball.s3.amazonaws.com — TheRock nightly tarballs (gfx90x incl. gfx906 targets).
- https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html · https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html — ROCm 7.x release notes (support-status changes for gfx906).
- https://github.com/YFrite/torch-builds-gfx906-rocm7.2 — (8★) PyTorch wheels for gfx906 on ROCm 7.2.
- https://github.com/moeKiwiSAMA/rocm-6.3.1-ubuntu-22.04-gfx906 — Dockerfile ROCm 6.3.1 for MI50.
- https://github.com/DKingAlpha/miopen-hip — Arch package MIOpen ROCm 6.3.3 for gfx906.
- https://github.com/RomuloSK/MI50-ROCm-10.x — MI50 on ROCm 10.x notes.
- https://github.com/ravillamanoj/rocm-trim — trims unused GPU-arch bitcode from ROCm installs.
- https://github.com/hboyd2003/ROCm-tensorflow-gcn5.0 — TensorFlow ROCm port for gfx900/902/906.
- Triton for gfx906: https://github.com/nlzy/triton-gfx906 (48★) · https://github.com/ai-infos/triton-gfx906 · https://github.com/gengchaogit/triton-3.2.0-mi50 (14★, adds mi25/mi50/mi60) · https://github.com/NatTuck/triton-mi50 · https://github.com/triton-lang/triton/issues/8992 (upstream gfx906 support issue)
- Flash-attention for gfx906: https://github.com/ROCm-6-4-5/flash-attention/tree/gel-crabs-headdim512 ("Optimized Flash Attention for GFX906 and GFX11") · https://github.com/ai-infos/flash-attention-gfx906 · https://github.com/Lowkey-Loki-SN/noflash-attention (pure-PyTorch O(N) attention for GCN, MI50 benchmarked)
- https://github.com/ai-infos/aiter-gfx906 — AITER (AI Tensor Engine for ROCm) gfx906 fork.
- https://github.com/NnnHU/comfy-kitchen-int8-rocm-906-patch — torch._int_mm fallback for GPUs without XDL (MI50) — INT8 path.
- https://github.com/namnguyen0503/mi50-gfx906-unsloth-bnb4bit-lab — ROCm/Unsloth/bitsandbytes 4-bit lab + VRAM benchmarks for MI50.
- https://github.com/larkinwc/gpu-operator-gfx906 — gfx906-optimised Kubernetes GPU operator for bare metal.

---

## 5. Power, clocks, memory timings, PCIe, firmware and telemetry (Linux) [POWER]

- https://github.com/sibradzic/upp — PowerPlay-table editor for Vega/Navi (the tool used by iacopPBK's power-scaling graphs and arkprojects perf-tuning); memory-clock discussion https://github.com/sibradzic/upp/issues/20
- https://github.com/Eliovp/amdmemorytweak — AMD Memory Tweak: live HBM2 timing adjustment (bundled by ML-gfx906).
- https://github.com/DvDlVs/vega20-unlock — soft-unlock of overdrive (clock/voltage) on Vega 20 via PowerPlay-table injection.
- https://github.com/srvr-farm/mi50-gaming — flashing a Radeon VII vBIOS onto MI50 with display output, unlocked power limits and overclocking.
- https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13 — MI50 32 GB VBIOS dump.
- https://github.com/lightcatcher/amd_mi50_chip_dumps — MI50 ROM/VBIOS backups.
- https://github.com/guedesite/RadeonVII_bioschecker_esp32 — read/verify/repair Radeon VII VBIOS flash with an ESP32.
- https://github.com/corundum/corundum — contains the PCIe link-speed forcing script referenced by arkprojects perf-tuning; see also https://arkprojects.space/wiki/AMD_GFX906/pcie-lnk-speed
- https://github.com/kingbone2006/MI50-Fan-Control — fan controller + real-time telemetry for MI50 / Radeon Pro VII.
- https://github.com/grumpy-kittens/mi50-metrics-exporter — Prometheus exporter for MI50 (ML-gfx906 also ships a Vega20 exporter).
- https://github.com/OhGodAPet/foxscotch — driver for the IR35217 VRM controller (Radeon VII and later).

---

## 6. Other gfx906 inference engines whose kernel work transfers to llama.cpp [ENGINE]

- https://github.com/sixvolts/reinstinct — (12★) custom HIP engine for MI50/MI60: wave64-native kernels, repacked quant formats for cache-line coalescing, fused dequant+matmul, Q8 FlashAttention KV, HIP graph capture; claims 7–43% decode gain over llama.cpp (31B dense 28 vs 21 t/s). Add-ons: https://github.com/GaryJS3/reinstinct-and-more
- https://github.com/JCraigWasTaken/ninfer-gfx906 — HIP/ROCm port of NInfer (single-GPU Qwen engine) to MI50/MI60, branch `gfx906-port`.
- https://github.com/nlzy/vllm-gfx906 — (433★) the vLLM gfx906 fork (branch `gfx906/main`); https://github.com/ai-infos/vllm-gfx906-mobydick (86★, current development line); https://github.com/ttdxq/gfx906-vllm (34★); https://github.com/nick413-bit/gfx906-fa-vllm (14★, FlashAttention-style attention backend for gfx906); https://github.com/Igneous/vllm-gfx906-mobydick-kvarn (KVarN kv dtype); https://github.com/David-Lzy/vllm-gfx906 (evidence archive); https://github.com/PowerfulGhost/vllm-mi50 (15★); https://github.com/PowerfulGhost/vllm-gfx906 (11★); https://github.com/Wulfsta/vllm-flake (15★, Nix); https://github.com/JackDanger/vllm-gfx906 (prebuilt 0.23.1rc0, ghcr.io/jackdanger/vllm-gfx906); https://github.com/cassettesgoboom/vllm-gfx906 (branch fix-gfx906-gdn-attention-deadlock); https://github.com/Kausik-A/qwen3.6-27b-mi50-vllm (6★); https://github.com/li-clement/Homelab-gfx906 ; https://arkprojects.space/wiki/AMD_GFX906/vllm
- https://github.com/NnnHU/sglang-mi50 — SGLang on MI50: self-built Triton 3.4, W4A16 Triton kernels, Qwen3.8-27B / Qwen3.5 hybrid GDN.
- https://github.com/mayor686/ds4-gfx906 — DeepSeek 4 Flash local engine with optimised gfx906 support.
- https://github.com/Llaminar/llaminar — (44★) C++ LLM inference engine with gfx906 in its README (listed by the infohub).
- https://github.com/wxr123-wxr/MetaInfer-cpp — C++ Qwen3 runtime with paged KV, continuous batching, HIP kernels, tensor parallelism (gfx906 in README).
- https://github.com/sourabhuday/gpu-gemm-transformers — HIP GEMM kernels for attention/MLP with LDS tiling and register blocking (gfx906 in README).
- https://github.com/fsword73/SGEMM_on_VEGA — alternative SGEMM implementation for Vega (GCN5 assembly-level reference).
- https://github.com/Aalanli/AMDGPUExperiments — AMD GPU kernel experiments, ISA docs gfx803–gfx1102, Omniperf/Omnitrace notes.
- https://github.com/mrowan137/mnist-from-scratch-hip — small HIP classifier explicitly tuned for Radeon Pro VII (gfx906).
- Ollama builds for MI50 (same ROCm/Tensile problems, useful build recipes): https://github.com/xxDoman/ollama-mi50-rocm7.2-optimized-gfx906 (8★, fixes garbage-text corruption) · https://github.com/xxDoman/ollama-amd-mi50 (10★) · https://github.com/xxDoman/ollama_mi50 · https://github.com/xxDoman/ollama-amd-rocm71-vl · https://github.com/taozebra/ollama-0.13.1_enableAmdMi50 · https://github.com/likelovewant/ollama-for-amd (1,790★, general AMD)

---

## 7. Search queries to re-run for freshness

- https://github.com/search?q=gfx906+llama+fork%3Atrue&type=repositories&s=updated&o=desc
- https://github.com/search?q=gfx-906+llama&type=repositories (hyphenated spelling used by the "turbo" family)
- https://github.com/search?q=gfx906&type=repositories&s=updated&o=desc (321 repos on 2026-09-09)
- https://github.com/search?q=MI50+llama&type=repositories · https://github.com/search?q=MI60+llama&type=repositories · https://github.com/search?q=%22radeon+vii%22+llama&type=repositories · https://github.com/search?q=vega20+llama&type=repositories
- https://github.com/mixa3607/ML-gfx906/forks · https://github.com/iacopPBK/llama.cpp-gfx906/forks · https://github.com/mxxm-t/mx-llama.cpp/forks
- GitHub API equivalent (100/page, include forks): https://api.github.com/search/repositories?q=gfx906+llama+fork:true&per_page=100&sort=updated
