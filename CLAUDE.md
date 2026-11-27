# CLAUDE.md — llama.cpp on 4 × gfx906 (Vega 20), tuning-guide folder

**Read `REQUIREMENTS.md` first. It outranks everything else in this folder, including this file.** The service is a queued, batched multi-user server at the largest context per slot (4–12 slots, ≥ 12 tok/s per request, 64K the floor); single-stream, single-user and 2K cells are out of scope and never gates. Items marked TBC there need the lead's answer before work that depends on them.

**Code: one repository, [`exabit-io/mx-llama.cpp`](https://github.com/exabit-io/mx-llama.cpp). Since 2026-11-14 the substrate** is `merge-v1.0.0` (`a73c601e6`) = mxxm-t's fork at `cdc81dae4` + llama.cpp v1.0.0 (`b1b770d`, b12359) + two HIP fixes (grouped top-k ballot, `PAD_REFLECT_1D` launch size), offered to mxxm-t as [PR #24](https://github.com/mxxm-t/mx-llama.cpp/pull/24) (#17, #22 and #23 were closed unmerged). Branches: `master` = `gfx906-single` = `gfx906-multi` = `224b73e24` (tag `gfx906/v1.0.0/r2/master`: substrate + RCCL default + both-profile patches + the LLVM 24 rule for the head-256 tile instance at threshold 85; r1 was `290fc642a` with the rule at 90, r0 `df2c6e5c5` without it), `gfx906-candidates` (unbinned), `merge-v1.0.0` (pure substrate). **Taking the next upstream release through the stack: `RELEASE-RUNBOOK.md`** (procedure, scripts in `tools/release/`, and the silent-break classes to check). `exabit-io/llama.cpp` is retired history. mixa3607's `ML-gfx906` is a different project (ROCm builds + Docker presets, no llama.cpp source). **Installed: the deb `mx-llama-cpp-gfx906` 1.0.0 in `/opt/mx-llama.cpp` (version `1.0.0+git20261116.224b73e24.rocm10.1...-1` = `-r2`, master `224b73e24`; runtime ROCm 10.1.0-gfx906+20261018172427 through its DT_RPATH, compiled by ROCm 10.1's toolchain and without a global `-march=native` since 2026-11-15; run its binaries directly, no `LD_LIBRARY_PATH`) from the rolling apt channel `mx-llama-cpp-gfx906` on `exabit-io/resize-amdgpu-bars`, and `python3-torch-gfx906` 2.15.0 (ROCm 10.1 build) for the system python** from the rolling apt channel `python3-torch-gfx906` there (with the `python3-typing-extensions` 4.16.0 and `python3-sympy` 1.14.0 backports it depends on). **ROCm: the 10.1 stream since 2026-11-14 (`/opt/rocm/core-10.1`, HIP 7.16, LLVM 24), with 10.0 still installed beside it (the reference builds of the report run on it; the llama package no longer needs its compiler). LLVM 24 generates a slower head-256 flash-attention tile kernel (prefill -4.1 % at 1x255K) unless that object is compiled with `-mllvm -amdgpu-vgpr-threshold-percent=85`, which the tree does (since `290fc642a` at 90, which cost MTP decode at depth 1.8 %; since `224b73e24` at 85). `reports/2026-11-14-rocm-10.1.md`, scripts and the record of the system changes in `tools/rocm101/`.** Plan: `RE-RE-SURVEY-ACTION-PLAN.md`.

## What this folder is
A tuning guide for llama.cpp on macpro2019-01 (2 × Radeon Pro Vega II Duo = 4 × Vega 20 / gfx906, 32 GB HBM2 each, one XGMI ring; Ubuntu 24.04, ROCm 10.1 since 2026-11-14 (10.0 since 2026-09-19, still installed; 7.14 purged); model Qwen3.8-27B Q8_0). Current code: the substrate and bin branches at the top of this file. *Builds of 2026-09-07..09, history (ROCm 7.14, no longer run):* **stock** (upstream b10288, `/opt/llama.cpp`, the reference every report number is measured on), the **2026-09-07 production** ("mxxm+fh", `/opt/llama.cpp-mxxm-fh`: ML-gfx906 fork tile table + the two MMVQ patches), and the **production** build since 2026-09-08 (`/opt/llama.cpp-prod` → `/opt/llama.cpp-mxxm-fh-nq`: the same plus `patches/0001–0009`, built from the `fusion` worktree `/root/mx-llama.cpp-fusion`; run with `settings/gfx906.env`). Start with `README.md` (its Status section is the current state). This folder is the repository https://github.com/exabit-io/llama.cpp-gfx906-tuning (commit as Joshua <joshua@exabit.io>, push key `/root/.ssh/id_ed25519_exabit`); the code is https://github.com/exabit-io/mx-llama.cpp. The measurements live in `reports/` (`reports/README.md` indexes them with the published artifact links) and their raw tables in `data/raw/<day>/` (copied from `/root/rocm-tests/bench`; `data/raw/README.md`), and are transcribed into `data/benchmarks.json`; `optimize/optimize.py` is a MILP over that data that picks a launch configuration for a workload and build (`--build prod|stock`, `--cap W`, `--allow-quant --max-kl`); `settings/` holds the chosen launches and the power-cap script; `ISA-NOTES.md` is the hardware review; `BENCHMARKS-TODO.md` is what to measure next; `NEXT-STEPS.md` is the software roadmap. `tools/` holds copies of the box-side scripts (test environment, watchdog, SMC log, server clients, power-cap and HBM probes, M1 trace, the queue/`post*` runners, bisect tooling, report generators); `ROCM-SETUP.md` is how the box was set up. The three AMD reference texts are git-ignored (copyright); `reference/README.md` has the sources.

## Conventions
- **Every number in the docs traces to a report cell or to `optimize.py` output.** Extrapolations are labelled "model"; do not present them as measurements. The run-to-run spread on this box is ±2% (report 1 s.6), so differences below that are noise.
- **Context floor RAISED TO 64K (lead, 2026-09-23): "64k is absolute floor."** Nothing below 64K is measured,
  gated or recommended, for any model or profile. A thinking model can spend 2K tokens reasoning before it
  answers, so 2K cells represent nothing. Load and smoke tests may use any depth; their numbers are never
  results. Supersedes the 16K/32K rule below.
- ~~**Context floor (user rule, 2026-09-09): nothing below 16K matters and 32K is the realistic floor.**~~ *Superseded by the 64K floor above; kept for the record.* Measure, gate and recommend at 32K depth first (a 128K row where it applies); 2K cells are sanity rows only, and where the guide still quotes a 2K-only number say so. `optimize.py` and `launch.sh` default to 32K per slot.
- Units as llama.cpp reports them: tok/s, ms per decode step, GiB as reported at load, GB/s = 10⁹ bytes/s.
- Die naming: HIP devices 0–3. Their PCI numbers move when cards are added (since the 2026-10-23 card install: 0b, 0e, 19, 1c; before: 0b, 0e, 1b, 1e), so never hard-code them: derive them from `/sys/class/drm/card[0-9]`. Physical ring HIP 0-1-3-2 (each module's two dies, plus the bridge between the same switch port of each module); the firmware reports the bridge pairs crossed. **RCCL topology is system configuration, not a script setting:** package `gfx906-rccl-topology` (Exabit apt channel `gfx906-rccl-topology` on exabit-io/resize-amdgpu-bars; source `tools/gfx906-rccl-topology/`): its unit regenerates `/var/cache/gfx906-rccl-topology/topology.xml` from sysfs on every boot, and its conffile `/etc/rccl.conf` points every RCCL process at it. Scripts do not set `NCCL_TOPO_FILE`; nothing system-wide lives in `/root`.
- "tp4" = `-sm tensor` over all four dies; "tp2" = two dies of one module; "dp4" = four independent single-die instances; "layer4" = `-sm layer`.
- Slots (`-np`): on the stock build they matter in stairs of 8 (MMVQ ≤ 8 columns, then MMQ tiles of 16/24/32) — never recommend 9–15 or 17–23 on stock. On the production build MMVQ runs to 16 columns and 12–16 slots are the peak on the split; single-die instances stay at ≤ 8.
- Power: the box has a 1228 W DC envelope; four dies at 200 W + host above ~150 W crosses it and the SMC latches every die at 1000 MHz until a cold power cycle. Caps below ~85 W are accepted and ignored (DPM floor). Production cap on hyperconverged nodes: 125 W.

## Adding measurements
1. Put the new report (HTML or MD) in `reports/` and add a row to `reports/README.md`; copy the raw tables/JSON it cites into `data/raw/<day>/`.
2. Add the cells to `data/benchmarks.json` under a new key, with a `_desc` and the command that produced them.
3. If the cell is one `optimize.py` models (see `decode_tps` / `prefill_tps` / `memory_gib`), wire it in so the cell becomes "measured".
4. Re-run: `cd optimize && python3 optimize.py`, `--build stock`, `--allow-quant --max-kl 0.04`, `--cap 125`, `--kv-q4v`, `--measured-only`, and paste into `results.md` (its header lists the six commands). Update the table in README section 3 if a winner changed.
5. `pip install pulp` is the only dependency (CBC ships with it). After editing `optimize.py`, run `python3 review/2026-09-08/test_optimizer_patch.py optimize/optimize.py` (six regression tests from the 2026-09-08 review) and the six commands that regenerate `results.md` (its header lists them).

## Reference documents (`reference/`)
Large text extractions with `.toc.md` section maps giving line numbers. Grep the `.txt`, or jump with the TOC. Key locations in the ISA: new Vega 7nm instructions lines 293–330; packed math 2326; VOP3P / DOT instructions 7814–7962; LDS 503; GPRs 793; DPP/SDWA limits 11494. LLVM guide: gfx906 features line 505; gfx9-generic exclusions 798–812; target IDs 2030.

## Things not to do
- Do not recommend `-sm layer`, `-sm row` (fails to load), `GGML_CUDA_ALLREDUCE=internal`, `GGML_CUDA_DISABLE_GRAPHS=1`, or a 4-bit V cache — all measured losers or non-functional here.
  **CORRECTED 2026-09-21 on the 4-bit V cache: it is NOT non-functional, and the reason it is a loser is specific.**
  `q8_0`-K / `q4_0`-V runs and is quality-free (PPL 5.5771 ± 0.062 on 16K/6, inside the reference cluster;
  `data/raw/2026-09-06/qwen38-27b-q8_0-ctx-ppl-q8q4-layer.md`). What it needs is an FA build that compiles the
  combination — `GGML_CUDA_FA_QUANTS` must include `q8_0-q4_0`, which the fork's own recipe has and our v0.4.1 builds
  did not, so "non-functional" was a build-config artifact **of the v0.4.1 campaign builds only**.
  **RETRACTED 2026-09-22:** I also claimed the historical "slower" figure timed that fallback. It did not.
  It came from `/opt/llama.cpp-faq`, an explicit `GGML_CUDA_FA_ALL_QUANTS=ON` build, at **8 sequences x 32K
  with ntg=128**, where q8_0/q4_0 decoded 107.4 against q8_0/q8_0's 111.2 aggregate — a **3% cost**, not 18%
  (the 18% was against f16). Today's gain is measured at **4 slots and 1 slot with ntg=1024 on v0.4.1 +
  RCCL + ROCm 10.0**. The sign difference is a difference of SHAPE and PLATFORM, not a build artefact, and
  **q4_0-V is NOT established at 8 slots** — R2.2's other design point, where the old data says it loses 3%. It saves **23.5%** of KV bytes (26.0 vs 34.0 KiB/token
  over the four dies, survey/q4v-cache.md) but its decode slope is **0.092 vs 0.086** ms per sequence per 1K of depth — i.e. **7% SLOWER**,
  not faster. Fewer bytes read and yet a higher slope means the FA kernel's dequant cost exceeds the bandwidth it
  saves on gfx906. Since capacity here is **decode-bandwidth-bound, not memory-bound** (memory allows ~3.1x more
  than R3.1 does), trading decode for memory is the wrong direction at the current design points. Keep it as a
  capacity lever for a future memory-bound case (a larger model such as Flash-Next), not as a throughput option.
- Do not recommend q8_0 KV for speed; it is a capacity trade (slower at depth: 43.4 vs 73.9 tok/s at the 8-slot ceiling).
- Do not recommend `--kv-unified` for speed on any build: on b10288 the server reads a lone prompt through the pool at half speed (188 vs 366 tok/s); the lone-prompt path is fixed in b10837, but 8 × 32K still loses a third of decode and half of prefill there (M4, 2026-09-08). It is a capacity mode.
- The XGMI topology is the Apple A2326 bridge ring (two Duo modules; each die: one port to its on-card partner, one across the bridge; ~33 GB/s per direction per link). The A2339 bridge would make two isolated two-link pairs and is deliberately not installed: models above 64 GB (Flash-Next) need all four dies on one fabric. A pair's link matters little anyway: with no direct link a pair loses 3–5% of prefill and nothing at decode (`pair_link_sensitivity`, 2026-09-08).
- Do not offer `-b 1024/512` as a fix for head-of-line blocking; measured, it is not one (report 2 s.5).
- Do not recommend `-np 32` for interactive traffic: it is +4–5% on the decode-only bench (M2) but 80–85 tok/s against 83 at the server level with half the per-user rate (`server_final_prod`, 2026-09-08); `launch.sh busy` is 16 slots. The two pairs stay the request-traffic answer (101.9 vs 83.4 tok/s at 16 clients on the final build).
- The MTP flag is `--spec-type draft-mtp --spec-draft-n-max N` (report 2 s.12).
- Do not re-test the nineteen environment knobs of run-through s.1 (NCCL protocol/algorithm/channels, `HIP_FORCE_DEV_KERNARG`, `GPU_MAX_HW_QUEUES`, `HSA_ENABLE_SDMA`, host C-states): all within 0.3% or losers.
- Do not re-try the measured MMVQ losers (rows 1, rows 8, rows 2/warps 2, whole-block loads, LDS staging, aligned dword loads) or `GGML_CUDA_FORCE_CUBLAS`; see `patches/README.md`.
- Do not call the 1000 MHz clamp thermal or DPM; it is the power supply (run-through). Never run host work beside four pinned dies without the 150 W RAPL cap.
- **Source of truth for the code: https://github.com/exabit-io/mx-llama.cpp** (local clone `/root/exabit-llama.cpp`, remote `exabit-mx`; push key `/root/.ssh/id_ed25519_exabit`; commits as Joshua <joshua@exabit.io>). Branches, commits and tags: `RE-RE-SURVEY-ACTION-PLAN.md` sections 2 and 9. *History:* https://github.com/exabit-io/llama.cpp (retired 2026-09-24) holds everything before — the `gfx906` branch (2026-09-08..09: upstream master merged into the fork at b10912 + the Exabit series, built at `/opt/llama.cpp-gfx906*` on ROCm 7.14, purged 2026-09-19), the v0.4.1 squash campaign of 2026-09-20..23, and the tags `gfx906/v0.4.1/*`, `gfx906/v0.5.0/*`, `import/*`.
- **Reference perplexity moved on upstream's side (2026-09-08 evening):** 16K/6-chunk ppl of Qwen3.8-27B Q8_0 is 5.5969 on every build up to the fork base (production included) and 5.6216 on upstream from 2026-09-06 (5.6118 for 09-02..09-06): two upstream fidelity changes (the GDN q/k norm with eps inside the root, 5fdfa6282, and one at or before the sparse-FA commit 8e93a9773), not our kernels (`upstream_ppl_bisect`, `reports/2026-09-08-ppl-bisect.md`). Compare new builds against 5.62, and never read the 0.4% as a regression.
- The Qwen3.8-Flash-Next model (architecture `qwen4exp`, 512 experts) needs upstream support after 2026-09-07 plus mxxm-t's PLE-table sharding (`LLAMA_PLE_SHARD=1`): both are in the substrate (loads and generates at `--n-cpu-moe 41 -lm mlock`, verified 2026-09-24); the 2026-09-07/08 production binaries cannot load it. Quant on the box: `/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/` (103.7 GiB in four shards, of which 26.8 GiB is the IQ4_NL per-layer n-gram table `per_layer_token_embd` and 71.7 GiB the 512 experts; `tools/gguf-tensors.py` tallies it) plus the 4.1 GB Q8_0 MTP head; measured 27.7–28.1 GiB per die with the table sharded (`LLAMA_PLE_SHARD=1`); first numbers 365 tok/s prefill / 20.6 ± 8 decode on four dies, wikitext ppl 1.93 (memorised text, build-to-build use only), key `flash_next_trial`.
- *History (2026-09-09, ROCm 7.14 builds purged 2026-09-19):* **two builds served by profile** (`settings/launch.sh`): `/opt/llama.cpp-gfx906` (the `gfx906` branch, tag `gfx906-20260909`) for team/busy/pairs/ingest/batch, `/opt/llama.cpp-prod` for single/long/long8/ceiling. Gates are at 32K (`tools/night-0909-*.sh`). Do not re-test alex4300's tile rows or DPP inside the attention kernels (measured losers, `reports/2026-09-09-night-report.md`). The name `s1b-a` is a tag (spilling form) and a branch (a3): merge `refs/heads/s1b-a`. `llama-batched-bench -pps` fails on Qwen3.8 (M-RoPE position check): use one prompt per sequence.
- Six-head GQA packing in the attention tile is withdrawn (alex4300 measured it slower; the kernel is instruction-bound, `reports/2026-09-09-fork-survey.md`). The community forks were surveyed on 2026-09-09; do not re-survey iacopPBK, milpster, furnace, arte-fact/TurboQuant, eslowney, Luna — the survey has the verdicts and the reasons.
- Kernel-fusion lesson (2026-09-08): the small decode kernels are latency-bound, not launch-bound; fusing consecutive ones or running them on more streams returned ~3% and 0. Do not re-plan S3 as "fuse the small kernels".
- The fork's custom allreduce (`GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1`) is +14% single-stream but −8% at 8–16 slots unless `GGML_TP_AR_MAX_NE=20481` (4 rows; the knob exists only in the fusion tree and later builds) gates it; `settings/gfx906.env` sets all three. Never `GGML_TP_AR_NO_GATE=1` (prefill −36%). See the 2026-09-08 report.
- **CORRECTED 2026-09-21, measured: the prefill credit belongs to the AR SIZE GATE, not the tile table.**
  The guide has said since 2026-09-08 that the fork tile table is worth "+33% prefill" — attributed *by
  inspection only*, as the line below still notes. Five configurations, measured at 125 W on the v0.4.1
  base with `--cache-ram 49152`, say otherwise: stock 522.9 t/s; substrate with custom AR **ungated**
  510.2; substrate with custom AR **off** 633.1; bundle with the gate **on** 633.3; bundle with the gate
  **off** 511.4 (4×64K, n=4 per arm). So the substrate's prefill machinery really is worth ~+21% over
  stock — and **ungated custom AR was masking all of it at −19%**. Confirmed fresh at n=4 on both axes:
  gate prefill **+23.84% multi-user (q=0.0761), +28.36% single-user (q=0.0761)**, decode flat on both
  (q=0.94 / 0.85). Mechanism: prefill works on large tensors, which the gate routes to the standard
  path; decode works on ≤4-row tensors, which stay on custom AR. **Never run custom AR without
  `GGML_TP_AR_MAX_NE=20481`** — and note the knob is NOT compiled into every build, so setting it in
  `gfx906.env` is not evidence that it is in effect: check with `strings <build>/bin/libggml-hip.so*`.
  Verdict records: `survey/tp-ar-size-gate.md`, `survey/custom-allreduce-ungated.md`.
- **q8 weight repack is confirmed to improve BOTH axes** (2026-09-21, n=4): decode +5.41% multi-user /
  +3.08% single-user, prefill +5.24% / +2.79%, all q=0.0761. It is on by default in the substrate, so
  the operational rule is simply: never pass `--no-repack`. `survey/q8-repack.md`.
- `review/2026-09-08/` is another contributor's review (optimizer patches, source corrections, modeling findings); `review/2026-09-08/RESPONSE.md` records what was applied, superseded or declined and why. Facts it established: gfx906 attention already uses `v_dot2_f32_f16`; the Q8_0×Q8_1 dot ignores the block sum; the fork differs from upstream across 81 files, so the +33% prefill is attributed to the tile table by inspection only.
- Do not treat the Instinct tuning guide's EPYC BIOS items as applicable; the host is an Intel Xeon W.
- **ROCm 10.1's toolchain (LLVM 24) needs the tile rule** (2026-11-14, revised 2026-11-15, cause bisected 2026-11-16). Without it ROCm 10.1's
  compiler costs prefill 1.4 % at 4x64K and 4.1 % at 1x255K (decode unchanged, output identical), all of it in `fattn-tile-instance-dkq256-dv256`:
  vector registers are spilled inside the kernel's inner loop under `max-ilp`. **Cause (report s.4.2): a compiler bug that upstream has already
  fixed.** LLVM `80df471b75d0` (#213584, 2026-08-03) made machine LICM take its register-pressure limit from `RegisterClassInfo`, whose unsigned
  subtraction wraps on AMDGPU (VGPR limit 128 becomes 4.29e9 in every kernel), so LICM hoists without a limit. The fix is `1d7d16b94c61` (#216372,
  2026-08-19). ROCm 10.1's compiler (`d6f6cb691863`, the tip of ROCm/llvm-project `release/therock-10.1`, upstream base 2026-08-13) has the first
  and not the second; `therock-10.2`/`10.3` have both; ROCm 10.0's has neither. Do not bisect it again, and there is nothing to report to LLVM
  (a draft for ROCm with a one-function reproducer is in `tools/rocm101/llvm-bisect/`; filing is the lead's decision).
  **The rule stays when the compiler is fixed** (s.4.3): on ROCm 10.1's compiler commit with the fix applied, the kernel without the rule is 0.6 %
  under clang 23's and with the rule 0.4 % above (1x128K prefill, n=3). `master` compiles that one object with `-mllvm -amdgpu-vgpr-threshold-percent=85`
  when the HIP compiler is clang 24 or newer (`290fc642a` introduced the rule at 90 = package `-r1`; `224b73e24` moved it to 85 = `-r2`, next bullet),
  and both packages passed the design-point acceptance (against the 10.0 source build, n=5: `-r1` prefill +0.27 % / +0.28 %, decode -0.52 % at 4x64K
  and +0.05 % at 1x255K; `-r2` prefill +0.34 % / +0.84 %, decode -0.40 % / +0.06 %). The option does not exist in clang 23 (ROCm 10.0). **Do not compile a source WITHOUT the rule (`df2c6e5c5` and older, any
  other branch) with 10.1's toolchain for a build that is measured or shipped**: use ROCm 10.0's compiler for it
  (`tools/rocm101/build-rocm.sh SRC BUILD LOG /opt/rocm/core-10.0`; the deb script refuses such a source and names `COMPILERSTREAM=10.0`). The rule is
  tied to this kernel as LLVM 24 compiles it: a later LLVM or a changed tile kernel needs the two-cell check again (`tools/rocm101/gpu-opt-confirm.sh`),
  then the acceptance. Already tried and not a fix: `iterative-ilp` or the default scheduler for the tile kernels, tile-table nbatch_K 64, occupancy 1,
  the driver's offload pipeline with or without LTO (not the cause). **Never `-mllvm -amdgpu-use-amdgpu-trackers=1`**: it removes the spills too and
  costs decode 3.1 % at 4x64K and 5.0 % at 1x128K on both compilers. `reports/2026-11-14-rocm-10.1.md` s.4, 4.1-4.3, 5.5, 5.6.
  A code-generator question like this one is answered fastest with `llc` alone: `tools/rocm101/llvm-bisect/` (blobless clone of ROCm/llvm-project
  in `/root/llvm-bisect`, `llc` per commit in 4-10 min, the kernel's bitcode from `codegen/ircross.py`, reloads counted by `codegen/loopstat.py`,
  and `codegen/llccross.py` to put an `llc` object into a library for timing; `llc` at ROCm 10.0's commit reproduces clang 23's object byte for byte).
- **A per-object compile option changes EVERY kernel of the object: read it on all of them, and time the ones the service runs, before it ships**
  (2026-11-16, same report s.5.10, 5.11). Each batch shape runs another tile kernel (Qwen3.8-27B: head 256, `ncols2` = 2): one token per stream
  `<1,2>`, two `<2,2>`, three or four `<4,2>`, five to eight `<8,2>`, nine or more `<16,2>`. `llama-batched-bench` runs only `<16,2>` (prefill) and
  `<1,2>` (decode); **MTP verification with `--spec-draft-n-max 3` (what `settings/launch.sh` uses, R3.9) runs `<4,2>`**, which no design-point cell
  touches. `-r1`'s threshold 90 was chosen and accepted on those two cells and put 20 scratch reloads into the one loop of `<4,2>` (no rule: 1,
  clang 23: 1): MTP decode at 117.7K depth 45.48 tok/s against 46.33 without the rule and 46.28 for clang 23 (-1.84 %). **85 leaves no reload in a
  loop of any of the object's 34 kernels**: 48.36 tok/s (88: 48.59), prefill 612.2 against 610.1 (n=3, same text and draft counts in all 15 cells);
  at the design points 85 is +0.34 % / +0.40 % prefill over 90 in every sample and level on decode (`<1,2>` is instruction-identical with no rule,
  90, 88 and 85). The installed `-r2` package against `-r1`'s GPU library: MTP decode 48.25 against 45.14 tok/s (+6.89 %, n=3). Tools, all in
  `tools/rocm101/`: `codegen/tilescan.py OBJ...` (reloads in loops per tile kernel), `codegen/loopscan.py BUILD_A BUILD_B` (every device function of
  two builds), `mtp-depth2.sh` (five swap libraries, MTP at depth; server cells go through `/v1/chat/completions`, a raw `/completion` prompt ends
  at 1 token), `tile-thr-confirm.sh` (design-point kernels), `mtp-accept.sh` + `rocm-ab5.sh` (the two acceptances of a package), `tbo-r2.sh`
  (`test-backend-ops` on a package's GPU library), `codegen/maskcmp.py` (GPU code of two builds from different source directories). **Any future
  change to a tile object, its options or its compiler repeats all of it**, and an MTP cell at depth is part of every package acceptance.
- **An FP8 KV cache is not a throughput configuration** (2026-11-16, s.5.12): `f8_e4m3`/`f8_e4m3` at 4x64K decodes 13.13 tok/s per slot against
  24.76 with f16 (-46.9 %, n=3), prefill +0.76 %; clang 24 costs it nothing against clang 23 (13.135 / 13.143) although its FP8-V vec kernels carry
  623 reloads in loops against 146 (not the LICM bug: they stay with `-disable-machine-licm` and at every threshold). Quality (2026-11-17,
  `tools/rocm101/fp8-ppl.sh`, 16K x 6 on the installed `-r2`, KL divergence against the f16 cache's logits): FP8 mean KLD 0.0090, PPL +0.9 %, same top
  token 98.43 %; q8_0 0.0060, +0.4 %, 98.84 %. Usable as a capacity lever (under the optimizer's 0.04 bound), but q8_0 is the better compact cache on
  quality; do not re-measure either.
- **The host compiler stays gcc 13** (2026-11-16, s.5.13): the six host libraries built by Ubuntu clang 19 or ROCm 10.1's clang 24 decode 0.70 % and
  0.82 % slower at 4x64K than gcc 13.3's (24.587 / 24.556 against 24.760 tok/s per slot, n=3, every run below every gcc run); OpenMP is worth 0.06 %;
  prefill equal. Do not re-screen host compilers or host flags for this code.
- **Do not re-survey GPU-side compile options** (2026-11-15, same report s.5.4): sixteen single changes timed at 4x64K with the 10.0 compiler, none
  faster than the tree's flags (`-O3`, `max-ilp`). Losers: load-store vectorizer off -19 % decode, `-unroll-threshold=3000` -75 % prefill (through
  `q8_repack` mul-mat, not attention), `iterative-maxocc` -53 % prefill, `max-memory-clause` -2.4 % decode, `-O2` -3.5 % prefill, the default scheduler
  -0.4 % decode / -1.7 % prefill. Level: target ID `gfx906:sramecc-:xnack-`, vectorizers off, `-unroll-threshold=300`, function calls off, relaxed
  occupancy, scalarize-global-loads off, promote-alloca limits, divergent register indexing. Byte-identical GPU code (nothing to time):
  `-amdgpu-early-ifcvt`, `-funroll-loops`, `-inline-threshold=1000`, `-split-spill-mode=speed`. `iterative-ilp` on every source crashes clang 23.
  **Repeated with the shipping compiler (10.1's clang 24) at BOTH design points, 2026-11-16/17, s.5.8: closed for this compiler too.** 24 variants, 50
  cells, two positive controls sorted as expected in both passes; no variant faster on decode at both design points. The two "up" on prefill went
  to a confirmation pass against the shipped rule (n=3 shuffled, both design points) and neither holds: `-O2` on the 85 tree +0.20 % of 4x64K prefill
  but -0.28 % decode and -0.14 % prefill at 1x255K; threshold 80 on every source -0.53 % of 1x255K prefill. What a compiler change moved: `-O2` changed
  sign (-3.5 % prefill under clang 23, level to slightly up under clang 24, and it costs decode at depth); vectorizers off are level at 4x64K but
  -0.46 % / -0.61 % decode at 1x255K; the trackers option -2.3 % decode at 4x64K, -6.9 % at 1x255K; the four byte-identical options now change 1 to 3
  objects (level); `iterative-ilp` crashes clang 24 as well (post-RA scheduler, `softmax.cu`). Not tried: machine LICM off on single objects (on every
  source decode +0.32 % / prefill -1.05 % at 4x64K, one sample). Lessons that outlive the numbers: an option's reading does not carry over a
  compiler change; a survey at one design point does not cover the other; one sample per variant ranks variants, it does not resolve 0.3 %; and when
  the shipped flags change during a survey, the confirmation's bar is the shipped build, not the survey's reference
  (`tools/rocm101/gpu-opt24-screen.sh`, `gpu-opt24-confirm.sh`). A new compiler (ROCm 10.2) is a reason to repeat the survey; a new idea on this
  compiler is not, unless it is a variant not in s.5.4 or s.5.8.
- **No build and no other host load while the dies are loaded, also in correctness runs; read the clock column after
  every GPU job** (2026-11-26, `reports/2026-11-26-v110-rebase.md` s.8): the dies latched at 1000 MHz within seconds of a
  56-thread build starting beside a GPU test and every speed of the next two hours was void. `grep -oE 'card0:[0-9]+MHz/[0-9]+%' <run>/watchdog.out`
  must show 1204-1730 MHz at 100 % busy; all samples at 1000 is the clamp (reboot, cold power cycle if that does not
  clear it). Do not rewrite a source while a build compiles it.
- **The fork is its own project (lead, 2026-11-26).** It follows upstream llama.cpp releases and pulls in the code that
  serves its goals; mxxm-t's gfx906 patches stay and his later changes are still read and taken on merit. Do NOT make
  a cycle wait on his tree, do NOT pick a resolution only to stay mergeable for him, do NOT open a pull request to him
  unless the lead asks (none of #17, #22, #23, #24 was taken; #25 is open). Never propose dropping his patches. The
  performance condition and its measurement: `reports/2026-11-26-v110-rebase.md` section 10.
- **llama.cpp v1.1.0 cycle is IN PROGRESS** (2026-11-26): merged onto mxxm-t's own b12500 merge with three fix commits
  (pushed, PR mxxm-t#25, #24 closed), 22 Exabit commits on top (local `master-v110`), DeepSeek-V4.1's upstream
  layout ported on `wip/ds41-upstream-layout` (tested, joins master at package time), nothing published; the tile rule's threshold must be chosen again on the changed kernel
  (85 is not clean any more). State and the open list: `tools/release/v110/README.md`. Until it ships, the tree and the
  package in this file are the v1.0.0 ones.
- **Do not chase the ROCm 10.1 runtime's decode cost** (0.3 % at 4x64K, same report s.5.3): it is ROCr (`libhsa-runtime64`), whose async-events
  thread sleeps between polls where 10.0's spun one host core for the whole run. No environment switch, not a defect, nothing to ship.
- **No global `-march=native` on the llama deb** (same report s.5.1, 5.2): on `libllama`/`libllama-common` it cost 0.41 % of 4x64K decode; of ten
  other host flag sets none beats CMake Release defaults + `GGML_NATIVE` (512-bit arms -0.6 to -0.8 %, `-O2` -1.29 %, LTO and PGO no gain).
  With two ROCm streams installed, every build and run names its ROCm tree: `/opt/rocm` follows the alternatives, a 10.0-toolchain build has no ROCm
  path of its own and loads whatever `LD_LIBRARY_PATH`/`/opt/rocm/lib` gives it, and the 10.1 link driver embeds `/opt/rocm/core-10.1/lib` as DT_RPATH.
- There is no `rocm-smi` in ROCm 10.1; `amd-smi` replaces it. Scripts set the DPM level through sysfs (`tools/gpu-test-env.sh`).
- Do not move or rename `reports/*.html` — external notes link to them by name.
- **Every build is configured with `-DGGML_HIP_RCCL=ON` (REQUIREMENTS R3.11, lead 2026-09-21).** Upstream's default is
  **OFF**, and this campaign inherited it without audit: with RCCL absent, `GGML_USE_NCCL` is undefined, nothing links
  `librccl`, and the collective falls back to f16 conversion, with a warning nccl -> internal (needs `n_devices==2`) -> none -> **meta-backend
  butterfly**. Every AllReduce measurement before 2026-09-21 was therefore against butterfly, not RCCL. The fleet is a
  multi-node cluster on Mellanox ConnectX IB, so RCCL is retained alongside the fork's intra-node custom AllReduce and a
  survey verdict may never bin it away. Check any build with `tools/assert-build-config.sh <build-dir>` before measuring
  it; `GGML_CUDA_ALLREDUCE=nccl|internal|none` selects the path at runtime for A/B work.
  **MEASURED 2026-09-22, three-way on one binary, n=4 per arm per cell:** RCCL instead of butterfly is
  **+18.59% prefill at 4x64K, +11.34% at 1x254K, +18.21% at 1x64K** (q=0.0857), decode unchanged. Custom AR
  is **+6.93% / +5.02% / +9.24% decode** (q=0.0857) with prefill untouched (±0.06%, n.s.). The two are
  **orthogonal and ship together**: custom AR owns decode, the collective owns prefill. This is the largest
  confirmed effect in the survey and it is a build flag, not a patch.
  **CORRECTION that follows:** the "+23.84% prefill from the AR size gate" reported on 2026-09-21 was a
  misattribution. The gate has no prefill effect of its own — it prevents custom AR from taking large
  tensors, which otherwise collapses prefill to ~510 t/s. Its apparent gain was recovery to the butterfly
  baseline. The gate stays required whenever custom AR is on; the prefill credit belongs to RCCL.
  **Also invalidated:** gate 1's "substrate prefill machinery is worth ~+21% over stock" (633 vs 523)
  compared a butterfly substrate against a butterfly stock. Both were off the shipping collective, so the
  tile table's value is still unmeasured; the stock zero point is being rebuilt with RCCL to redo it.
- **Every build is configured with `-DGGML_CUDA_FA_QUANTS=all` (REQUIREMENTS R3.11, lead 2026-09-22).** Upstream
  compiles four K/V pairs; `all` compiles all 49. A combination that is not compiled **does not fail** -- it
  converts K and V to f16 and logs `ggml_cuda_flash_attn_ext_vec: no FlashAttention vector kernel compiled for
  K/V types <k>-<v>, converting K and V to f16 instead (slow)`. A reading taken that way is an f16 result
  wearing a quantised label. Check builds with `tools/assert-build-config.sh` and run logs with
  `tools/assert-no-fa-fallback.sh`. The KV type is a **per-model** choice, not a global default: KV is 31.5% of
  per-step bytes on Qwen3.8-27B and 3.1% on Flash-Next, so the right V quantisation differs by model.
- Never run a build from `/opt/llama.cpp-*` without `LD_LIBRARY_PATH=<that prefix>/lib`: the binaries have no rpath and the loader path is the stock `/opt/llama.cpp/lib`, so the production binary would load stock kernels. `settings/launch.sh` sets it; `llama-bench` needs `-dev rocm0/rocm1/rocm2/rocm3` (slashes) for one four-die test, commas mean separate tests.
