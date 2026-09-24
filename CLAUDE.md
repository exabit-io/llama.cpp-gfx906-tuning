# CLAUDE.md — llama.cpp on 4 × gfx906 (Vega 20), tuning-guide folder

**Read `REQUIREMENTS.md` first. It outranks everything else in this folder, including this file.** The service is a queued, batched multi-user server at the largest context per slot (4–12 slots, ≥ 12 tok/s per request, 64K the floor); single-stream, single-user and 2K cells are out of scope and never gates. Items marked TBC there need the lead's answer before work that depends on them.

**Code (lead, 2026-09-24): one repository, [`exabit-io/mx-llama.cpp`](https://github.com/exabit-io/mx-llama.cpp); the substrate** (`528384980`, branch `merge-v0.5.0`) = mxxm-t's fork (`mxxm-t/mx-llama.cpp` @ `eefc4e732`, its 182 commits kept individually) merged with llama.cpp v0.5.0 (`7fe450e`) + RCCL on by default, offered to mxxm-t as [PR #17](https://github.com/mxxm-t/mx-llama.cpp/pull/17). Branches: `master` (the build: substrate + patches binned for both profiles — AR size gate, FA_QUANTS=all), `gfx906-single`/`-multi` (+ per-profile winners), `gfx906-candidates` (our unbinned patches), and `merge-v0.5.0` (the pure substrate, PR #17's branch, until merged). `exabit-io/llama.cpp` is retired history (archived 2026-09-24, read-only). mixa3607's `ML-gfx906` is a different project (ROCm builds + Docker presets, no llama.cpp source). Plan: `RE-RE-SURVEY-ACTION-PLAN.md`.

## What this folder is
A tuning guide for llama.cpp on macpro2019-01 (2 × Radeon Pro Vega II Duo = 4 × Vega 20 / gfx906, 32 GB HBM2 each, one XGMI ring; Ubuntu 24.04, ROCm 10.0 since 2026-09-19 (7.14 purged); model Qwen3.8-27B Q8_0). Current code: the substrate and bin branches at the top of this file. *Builds of 2026-09-07..09, history (ROCm 7.14, no longer run):* **stock** (upstream b10288, `/opt/llama.cpp`, the reference every report number is measured on), the **2026-09-07 production** ("mxxm+fh", `/opt/llama.cpp-mxxm-fh`: ML-gfx906 fork tile table + the two MMVQ patches), and the **production** build since 2026-09-08 (`/opt/llama.cpp-prod` → `/opt/llama.cpp-mxxm-fh-nq`: the same plus `patches/0001–0009`, built from the `fusion` worktree `/root/mx-llama.cpp-fusion`; run with `settings/gfx906.env`). Start with `README.md` (its Status section is the current state). This folder is the repository https://github.com/exabit-io/llama.cpp-gfx906-tuning (commit as Joshua <joshua@exabit.io>, push key `/root/.ssh/id_ed25519_exabit`); the code is https://github.com/exabit-io/mx-llama.cpp. The measurements live in `reports/` (`reports/README.md` indexes them with the published artifact links) and their raw tables in `data/raw/<day>/` (copied from `/root/rocm-tests/bench`; `data/raw/README.md`), and are transcribed into `data/benchmarks.json`; `optimize/optimize.py` is a MILP over that data that picks a launch configuration for a workload and build (`--build prod|stock`, `--cap W`, `--allow-quant --max-kl`); `settings/` holds the chosen launches and the power-cap script; `ISA-NOTES.md` is the hardware review; `BENCHMARKS-TODO.md` is what to measure next; `NEXT-STEPS.md` is the software roadmap. `tools/` holds copies of the box-side scripts (test environment, watchdog, SMC log, server clients, power-cap and HBM probes, M1 trace, the queue/`post*` runners, bisect tooling, report generators); `ROCM-SETUP.md` is how the box was set up. The three AMD reference texts are git-ignored (copyright); `reference/README.md` has the sources.

## Conventions
- **Every number in the docs traces to a report cell or to `optimize.py` output.** Extrapolations are labelled "model"; do not present them as measurements. The run-to-run spread on this box is ±2% (report 1 s.6), so differences below that are noise.
- **Context floor RAISED TO 64K (lead, 2026-09-23): "64k is absolute floor."** Nothing below 64K is measured,
  gated or recommended, for any model or profile. A thinking model can spend 2K tokens reasoning before it
  answers, so 2K cells represent nothing. Load and smoke tests may use any depth; their numbers are never
  results. Supersedes the 16K/32K rule below.
- **Context floor (user rule, 2026-09-09): nothing below 16K matters and 32K is the realistic floor.** Measure, gate and recommend at 32K depth first (a 128K row where it applies); 2K cells are sanity rows only, and where the guide still quotes a 2K-only number say so. `optimize.py` and `launch.sh` default to 32K per slot.
- Units as llama.cpp reports them: tok/s, ms per decode step, GiB as reported at load, GB/s = 10⁹ bytes/s.
- Die naming: HIP devices 0–3 = PCI 0b, 0e, 1b, 1e. Physical ring 0b–0e–1e–1b (HIP 0-1-3-2); the firmware reports the bridge pairs crossed. RCCL needs `/root/rccl_topo_fixed.xml`.
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
  **q4_0-V is NOT established at 8 slots** — R2.2's other design point, where the old data says it loses 3%. It saves **23.5%** of KV bytes (105.6 vs 138.1 KiB/token
  for this model) but its decode slope is **0.092 vs 0.086** ms per sequence per 1K of depth — i.e. **7% SLOWER**,
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
- **Since 2026-09-09 evening two builds serve by profile** (`settings/launch.sh`): `/opt/llama.cpp-gfx906` (the `gfx906` branch, tag `gfx906-20260909`) for team/busy/pairs/ingest/batch, `/opt/llama.cpp-prod` for single/long/long8/ceiling. Gates are at 32K (`tools/night-0909-*.sh`). Do not re-test alex4300's tile rows or DPP inside the attention kernels (measured losers, `reports/2026-09-09-night-report.md`). The name `s1b-a` is a tag (spilling form) and a branch (a3): merge `refs/heads/s1b-a`. `llama-batched-bench -pps` fails on Qwen3.8 (M-RoPE position check): use one prompt per sequence.
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
