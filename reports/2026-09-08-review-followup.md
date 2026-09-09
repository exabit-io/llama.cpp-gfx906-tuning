# Review follow-up, 2026-09-08 evening: MTP gate, width isolation, cold start, Flash-Next, S1b candidate 1, S1 patch

Written against the external review of the night report (`review/`, received 2026-09-08 evening). The box was warm-rebooted at 20:57 UTC
(clean; dies at 1730 MHz, fans under `t2fanrd`); every row below is post-reboot unless it says otherwise. Runners: `tools/post28.sh`
(A–D), `tools/s1b-test.sh` (A0, B2, S1b, S1). Raw tables in `data/raw/2026-09-08/`. Environment `settings/gfx906.env`, host RAPL 150 W,
dies at 200 W / `high`, clamp watchdog armed.

Builds: production `/opt/llama.cpp-prod` (fork b10254 + patches 0001–0009); branch `/opt/llama.cpp-gfx906-master-r2` (`gfx906` at
5d36d6fc8, repack on by default, `--no-repack` = `nr1`); S1b candidate 1 `/opt/llama.cpp-gfx906-s1b-{a,a2,a3}` (worktree
`/root/exabit-llama.cpp-s1b`, tags `s1b-a`, `s1b-a2`, head = a3); S1 `/opt/llama.cpp-s1-upstream` (master 5d806aa25 + the tile table)
against `/opt/llama.cpp-master` (pristine).

## Summary

- **The review's core recommendation stands and is now measured:** the production hold is right. The `gfx906` branch **fails the MTP gate** in both variants (warmed, paired 95% intervals against production: draft 3 at 32K −7.1% [−12.2, −1.9] with the repack, −10.4% [−15.4, −5.4] without; drafting off −4.5 to −6.4% at both depths; acceptance identical), so the deficit is the per-step cost of the small verify batch, not drafting.
- **The "8-column kernel loss" is withdrawn.** After the reboot the branch matches production at 7 / 8 / 9 / 12 rows in three rounds and in twelve homogeneous 8-row cells; what the night report saw was the branch's warm-up, which is per process *and per batch shape*, occasionally recurring for one cell (−4 to −11%, ~20 s) early in a process. Not thermal (clocks identical), mechanism open, prewarm covers it.
- **Cold start closed:** warm-up inside the first 128–192 tokens, ~5 s per process, no batch-shape first-use penalty, load-to-ready 19–23 s, first token 2.0–2.3 s at 2K. Prewarm 256 tokens per serving shape before readiness.
- **Flash-Next closed for now:** steady single-stream decode 41.6 tok/s (the 20.6 ± 8 was two warm-up samples); wikitext 2.00 vs 9.43 on a held-out corpus of our own documents, against 5.07 / 8.83 for the 27B — wikitext is memorised, the held-out number is the metric and it is sane.
- **S1b candidate 1 delivered (build a3):** widths 9–16 on the repacked narrow-batch kernel with a width-dependent launch bound (the 64-VGPR build spills and collapses to 70–80 tok/s; a bare bound halves occupancy). 12 / 16 rows at parity with production (196 / 206), 17 rows +15%, 24 / 32 rows +15 / +20%, one die +11 / +9% at 12 / 16, prefill +20 / +28% kept, perplexity and `test-backend-ops` clean, and **+1–3% at 8 clients, +5.5% at 16 clients at the server level** — the first branch build to beat production there. It leaves 1–4 rows untouched (−5%), which is where the MTP gate fails; candidate 2 is therefore redefined as the port of production's one-column kernel to the two-plane layout.
- **S1 upstream patch ready as v2:** the tile table alone on master 5d806aa25, applied from J = 32 (the J = 16 entry read −7.5% at 16 rows on master): +29% / +34% prefill, +15% at 32 rows, parity at 1 / 8 / 16 rows, numerics identical. Authorship is Marko Tombak's; sending is a decision for him and you.
- **Not done:** the FA controlled-LDS experiment (S4) and the tenant-side shared-host check (needs the hyperconverged workload).

## 1. MTP gate (post28 A): production / branch repack / branch `--no-repack`, drafting off and draft 3, 2K and 32K, three rotated rounds

Single user, `-np 1 -c 34816`, 300 generated tokens, wave 2 = decode only on the cached prompt (`mtp-depth-client.py --waves 2`).
Each cell is a fresh server, so round 1's 2K cells sit inside the after-load warm-up (section 3) — read rounds 2–3, or the warmed
repeat in section 1b.

| build | spec | ctx | wave-2 tok/s, rounds 1 / 2 / 3 | accepted / drafted |
|---|---|---:|---|---|
| production | off | 2K | 57.8 / 57.7 / 57.6 | – |
| production | off | 32K | 53.2 / 53.1 / 53.0 | – |
| production | draft 3 | 2K | 52.9 / 76.1 / 76.7 | 200 / 294 = 0.68 |
| production | draft 3 | 32K | 67.3 / 71.4 / 72.2 | 199 / 300 = 0.66 |
| branch, repack | off | 2K | 54.0 / 52.8 / 53.6 | – |
| branch, repack | off | 32K | 49.4 / 49.9 / 50.1 | – |
| branch, repack | draft 3 | 2K | 56.2 / 75.6 / 76.5 | 204 / 284 = 0.72 |
| branch, repack | draft 3 | 32K | 56.5 / 66.9 / 67.7 | 200 / 297 = 0.67 |
| branch, `--no-repack` | off | 2K | 54.3 / 54.5 / 54.7 | – |
| branch, `--no-repack` | off | 32K | 48.9 / 50.7 / 51.0 | – |
| branch, `--no-repack` | draft 3 | 2K | 70.8 / 71.4 / 71.7 | 200 / 294 = 0.68 |
| branch, `--no-repack` | draft 3 | 32K | 64.6 / 63.8 / 64.4 | 200 / 297 = 0.67 |

Steady rounds (2–3): with draft 3 the repacked branch matches production at 2K (76.0 vs 76.4) and reads **−5.6% at 32K** (67.3 vs
71.8); `--no-repack` reads **−6.4% at 2K and −10.7% at 32K** (71.6 / 64.1). With drafting off both branch variants are 4–7% under
production at both depths (53.2–54.6 vs 57.7 at 2K, 50.0–50.9 vs 53.1 at 32K). Acceptance is the same everywhere (0.66–0.72; the
repacked build drafts slightly fewer tokens, 284 vs 294, its numerics differ). So the MTP deficit is real, it is larger without the
repack, and it is not an acceptance effect: it is the branch's slower verify step (a 4-token batch on the split, where the branch's
single-stream and small-batch decode already reads −3 to −7%). The gate as declared in NEXT-STEPS is failed by both variants on this pass.

## 1b. MTP gate, warmed (s1b-test A0): the gate as declared

Same design, but every server generates 600 tokens on a throw-away prompt before the measured waves (section 3 puts the warm-up inside
the first 192 tokens). Three rotated rounds; wave-2 (decode-only) tok/s per round, then the paired 95% t-interval of the per-round
difference against production (unit = round, n = 3):

| build | spec | ctx | rounds 1 / 2 / 3 | acc. | paired diff vs production [95% CI] | relative [95% CI] |
|---|---|---:|---|---|---|---|
| production | off | 2K | 57.5 / 57.6 / 57.5 | – | | |
| production | off | 32K | 53.0 / 53.2 / 53.1 | – | | |
| production | draft 3 | 2K | 77.6 / 75.8 / 76.6 | 0.68 | | |
| production | draft 3 | 32K | 73.5 / 71.6 / 72.4 | 0.66 | | |
| branch, repack | off | 2K | 54.0 / 53.8 / 53.8 | – | −3.67 [−4.05, −3.29] | **−6.4% [−7.0, −5.7]** |
| branch, repack | off | 32K | 50.2 / 50.0 / 49.9 | – | −3.07 [−3.64, −2.49] | **−5.8% [−6.9, −4.7]** |
| branch, repack | draft 3 | 2K | 77.8 / 78.8 / 77.6 | 0.72 | +1.40 [−2.18, +4.98] | +1.8% [−2.8, +6.5] |
| branch, repack | draft 3 | 32K | 66.8 / 67.9 / 67.4 | 0.67 | −5.13 [−8.87, −1.40] | **−7.1% [−12.2, −1.9]** |
| branch, `--no-repack` | off | 2K | 54.6 / 54.8 / 54.6 | – | −2.87 [−3.01, −2.72] | **−5.0% [−5.2, −4.7]** |
| branch, `--no-repack` | off | 32K | 50.7 / 50.9 / 50.6 | – | −2.37 [−2.65, −2.08] | **−4.5% [−5.0, −3.9]** |
| branch, `--no-repack` | draft 3 | 2K | 71.7 / 73.6 / 73.0 | 0.68 | −3.90 [−8.54, +0.74] | **−5.1% [−11.1, +1.0]** |
| branch, `--no-repack` | draft 3 | 32K | 64.4 / 65.4 / 65.0 | 0.67 | −7.57 [−11.19, −3.95] | **−10.4% [−15.4, −5.4]** |

Verdict against the declared margin (lower bound of the paired interval above −2%): **the branch fails the MTP gate in both variants.**
Seven of the eight branch cells have their whole interval below −2%; the one exception (repack, 2K, draft 3: +1.8% [−2.8, +6.5]) merely
fails to exclude it. The warmed numbers agree with the steady rounds of section 1 to within 1%, so the warm-up explained round 1's
scatter and nothing else. Acceptance is identical across builds (the repacked build drafts 284 instead of 294 tokens for the same 200
accepted — its numerics differ slightly — and still lands at the same speed), so the deficit is the per-step cost of the 4-row verify
batch on the split, which is the same 4–7% the branch loses at 1–4 rows without drafting. The S1b candidates in section 5 address 9–16
columns and cannot move this; the single-stream / 1–4-row gap is a separate item (the fork's cached views and buffer type on the
tensor-split path are the suspects — the same code that produces the warm-up and the shape-transition effect).

## 2. Width isolation (post28 B): is the 8-column loss a width-8 kernel cost?

`llama-batched-bench` tp4 at 2K, `-npl 8,6,7,8,9,8,12,16,8` in one process (the 8-cell first, after 7, after 9 and last, after 16), branch
`--no-repack` against production, three interleaved rounds:

| width (position) | branch r1 / r2 / r3 | production r1 / r2 / r3 |
|---|---|---|
| 8 (first) | 174.0 / 173.8 / 173.5 | 174.3 / 174.2 / 174.0 |
| 6 | 138.2 / 150.3 / 150.2 | 152.7 / 152.2 / 152.5 |
| 7 | 162.4 / 163.1 / 162.4 | 164.5 / 164.3 / 164.7 |
| 8 (after 7) | 174.2 / 173.9 / 173.8 | 173.7 / 173.5 / 173.3 |
| 9 | 175.4 / 181.1 / 180.1 | 182.7 / 183.1 / 182.7 |
| 8 (after 9) | 174.4 / 173.9 / 173.6 | 173.4 / 173.3 / 173.2 |
| 12 | 195.9 / 195.4 / 195.4 | 196.5 / 196.3 / 196.5 |
| 16 | 197.3 / 200.1 / 199.7 | 202.5 / 202.5 / 202.4 |
| **8 (last, after 16)** | **167.3 / 161.6 / 162.6** | 173.6 / 173.3 / 173.2 |

Two results. (1) The 8-column loss the night report carried (−6.5% to −28% on the first cell, seven runs between 18:00 and 20:20)
**did not reproduce after the reboot**: the first 8-cell and the two middle ones are at parity (±0.3%) in all three rounds, as are 7, 9
and 12; 6 and 16 read −1.5% (round 1's 138 at width 6 is the one outlier). Whatever produced it was process- or box-state dependent, and
the reboot cleared it; it was never a fixed kernel cost, and the night report's sentence about "the merged kernel body at exactly eight
columns" is withdrawn. (2) A reproducible **−6% remains for an 8-cell that follows the 16-cell** in the same process (161.6–167.3 vs
173.2–173.6), on the branch only. That is a shape-transition effect, and a server changes its batch width every step, so it is the
first candidate for the server-level −3 to −4%; `s1b-test.sh` B2 runs a transition sequence (16,8,16,8,12,8,4,8,1,8,32,8) on the
branch with and without repack, the S1b build and production.

## 3. Cold start (post28 C): four fresh processes per build

`llama-server -np 8 -c 65536`, load-to-ready by 1-s polling of `/health`; then one 2K-prompt request (64 tokens), five decode-only
64-token continuations, and two 8-way batches of distinct 1300-token prompts (first and second use of the batch-8 shape).

| build | load-to-ready | TTFT (2K prefill) | decode tok/s, requests 1 / 2 / 3 | steady (requests 3–6) | batch-8 aggregate, 1st / 2nd use |
|---|---:|---:|---|---:|---|
| production, 4 processes | 18.6–22.6 s | 2.03–2.07 s | 52.9–53.0 / 56.4–56.6 / 56.2–56.4 | 56.0–56.6 | 102.4–102.8 / 103.5–103.8 |
| branch `--no-repack`, 4 | 19.6–20.8 s | 2.10–2.26 s | 50.6–51.8 / 50.6–52.2 / 54.9–55.2 | 54.5–55.3 | 97.2–99.3 / 98.8–99.4 |
| branch repack, 4 | 11.3–25.6 s | 1.82–2.81 s | 26–50 / 26–51 / 54.1–54.6 | 53.7–54.6 | 99.4–100.7 / 101.3–101.6 |

Per process (repack): p1 50.0 / 50.8 / 54.6; p2 26.3 / 25.9 / 54.1; p3 25.9 / 43.1 / 54.3; p4 29.8 / 41.6 / 54.2 — three of four processes
start at 26–30 tok/s and one at 50; all are at steady state by the third request (192 tokens).

The warm-up is a per-process cost paid inside the first 128–192 generated tokens: production is at steady state from its second request
(53.0 then 56.4), the branch without repack from its third (≈51 → 55), the branch with repack from its third as well but from a lower and
variable start (26 or 50 tok/s at requests 1–2; one process in four skipped it). At 26 tok/s two 64-token requests cost about five
seconds of wall time per process start. The first use of the batch-8 shape carries no extra penalty on any build (first and second use
within 2%). Load-to-ready is 19–23 s for production and `--no-repack`, 11–26 s with the repack (page-cache dependent). Time to first
token at 2K is 2.0–2.1 s (production), 2.1–2.3 s (`--no-repack`), 1.8–2.8 s (repack). **Prewarming covers it:** 256 generated tokens
before the server reports ready (a `--warmup`-style request in the launcher) removes the user-visible part; nothing shape-specific is
needed. The steady single-stream decode inside a server is 56.0–56.6 (production), 54.5–55.3 (`--no-repack`), 53.7–54.6 (repack).

## 4. Flash-Next (post28 D): per-sample decode curve and a held-out corpus

`llama-bench -p 0 -n 128 -r 8` on the branch build, four dies `-sm tensor`, `LLAMA_PLE_SHARD=1`, UD-Q4_K_XL (103.7 GiB + MTP head):

| run | mean ± sd | per-sample tg128 tok/s |
|---|---:|---|
| repack on | 36.4 ± 9.9 | 24.3 17.0 42.0 41.9 41.6 41.6 41.6 41.6 |
| repack off | 37.2 ± 8.7 | 26.7 20.0 42.1 41.9 41.7 41.7 41.6 41.6 |

Steady single-stream decode is **41.6 tok/s** (the night report's 20.6 ± 8.0 was two samples inside this warm-up; the "above 20" claim
is withdrawn in favour of the measured curve). The warm-up is two samples (256 tokens) deep and identical with the repack off, so for
this model it is not the Q8_0 repack (a Q4_K model carries few Q8_0 tensors): first touch of the 512 experts and the sharded PLE table
is the candidate, and the same 256-token prewarm covers it.

Perplexity, `-c 2048 --chunks 8`, the same binary and settings for both corpora (`held-out` = `/root/models/heldout-exabit-docs.txt`,
251 KB of this project's own 2026-09 reports and notes, which cannot be in any training set):

| model | wikitext-2 test | held-out |
|---|---:|---:|
| Flash-Next UD-Q4_K_XL (branch, PLE sharded) | 2.004 ± 0.037 | 9.43 ± 0.28 |
| 27B Q8_0 (production) | 5.072 ± 0.130 | 8.83 ± 0.25 |

On the held-out corpus Flash-Next reads 7% above the 27B Q8_0 — the ordinary distance between a 4-bit mixture and an 8-bit dense model
on unseen technical prose — while on wikitext it reads 2.5× *below* it. wikitext is memorised by this model and is not a quality metric
for it; the held-out number says the build computes it correctly. Sharded vs host-resident correctness, the 8-slot / 32K memory
headroom and MTP on this model remain to be measured before any capacity planning (they are not in this pass).

## 5. S1b candidate 1: the repacked narrow-batch mat-vec through 16 columns (s1b-test)

The change (worktree `/root/exabit-llama.cpp-s1b`, branch `s1b-a`, three commits): the fork's `mul_mat_vec_repacked_nc<2,1,N,2,64>` is
generic in its column count, so widths 9–16 are eight explicit instantiations with the width-8 geometry, plus a Q8_0-only crossover
(`rp_mmv_max_tokens`; MXFP4/IQ4_NL/K-quants keep 8 until measured) — 82 lines. The other repacked kernels, the MoE path and every
non-gfx906 target are untouched. Three builds differ only in the launch bound (code-object metadata in
`data/raw/2026-09-08/s1b-kernel-metadata.md`):

| build | bound | VGPRs at 8 / 12 / 16 cols | spills at 12 / 16 |
|---|---|---|---|
| a | none (the compiler assumes a 1024-thread block: 64-VGPR cap) | 59 / 64 / 64 | 4 / 18 (20 / 52 B scratch) |
| a2 | `__launch_bounds__(NWAVES*64)` | 129 / 181 / 228 (one wave per SIMD from 8 up) | 0 / 0 |
| **a3** | `__launch_bounds__(NWAVES*64, NCOLS > 10 ? 2 : 4)` | 59 / 67 / 75 (2–10 byte-identical to a; 12–16 at three waves per SIMD) | 0 / 0 |

`test-backend-ops -o MUL_MAT` on a3: 1288 / 1288. Perplexity 16K: 5.6173 (= the branch). llama-bench: pp2048 1359 tp4 / 424 one die
(= the branch, +20% / +28% over production's 1131 / 332); tg128 55.1 tp4 (production 58.0, the branch 55.5), 22.1 one die (production 21.7).

**Batched decode, tp4 at 2K** (`-npl 7,8,9,12,16,17,24,32`, two interleaved rounds; 7 and 8 are the first two cells of each process and sit
inside the repack warm-up on the repacked builds — read them from the transition table below):

| build | 7 | 8 | 9 | 12 | 16 | 17 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 163.6–164.6 | 174.3 | 183.1–183.3 | 196.6–196.7 | 202.4–202.5 | 153.3 | 191.9–192.1 | 214.6 |
| branch r2 (repack; tile at 9–16) | 139–167 | 166–174 | **108.7–114.2** | **141.2–141.4** | **169.0–169.1** | 170.6–175.2 | 219.5–219.7 | 258.2–258.3 |
| a (spills at 12 / 16) | 131–148 | 153–166 | 170.9–176.6 | **80.0** | **70.1** | 160.1–169.1 | 219.4–220.0 | 258.0–258.1 |
| a2 (one wave per SIMD) | 113–123 | 123–129 | 132.0–132.8 | 146.9 | 178.0 | 171.6–173.7 | 219.5–219.9 | 257.9–258.1 |
| **a3** | 126–127 | 158–161 | 174.0–174.6 | **196.3–196.4** | **205.8–206.2** | **176.2–176.6** | 219.5–219.8 | 258.2–258.3 |

**Shape transitions in one process** (`-npl 16,8,16,8,12,8,4,8,1,8,32,8`, after the warm-up), and **one die at 512** (`-npl 4,8,12,16`):

| build | 16 | 8 | 16 | 8 | 12 | 8 | 4 | 8 | 1 | 8 | 32 | 8 (after 32) | ‖ one die 4 / 8 / 12 / 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| production | 203.2 | 174.1 | 202.6 | 173.9 | 196.7 | 174.0 | 121.0 | 174.8 | 56.4 | 174.1 | 214.5 | 173.8 | 53.4 / 70.3 / 59.6 / 58.7 |
| branch `--no-repack` | 201.0 | 173.2 | 200.8 | 172.8 | 195.8 | 173.0 | 119.7 | 174.4 | 56.1 | 174.4 | 212.8 | **154.6** | – |
| branch repack | 159.6 | 163.4 | 166.0 | 164.5 | 141.0 | 175.4 | 119.8 | 175.4 | 51.4 | 174.3 | 256.2 | 167.2 | 56.2 / 68.7 / 58.1 / 63.6 |
| **a3** | 188.2 | 171.1 | 203.2 | 173.3 | 196.5 | 175.9 | 121.8 | 175.4 | 53.7 | 174.8 | 258.0 | 170.3 | 56.3 / 68.9 / **66.1** / **63.8** |

Reading: the register analysis decided it — the 64-VGPR build's scratch traffic in the inner loop collapses 12 and 16 columns to 70–80
tok/s, the bare bound's one-wave occupancy costs a third at 7–9, and the width-dependent bound gives **12 at parity with production, 16 at
+2%, 17 at +15%** (the tile no longer starts at 9), 8 and 4 at parity, 24 / 32 at the repack's +15 / +20%, and on one die +11% / +9% at
12 / 16. The 9-column cell reads −4.5% against production's canonical 16-column MMVQ, the only width where the two-row-per-lane repacked
kernel is behind it. Not in S1b's range and unchanged by it: single stream (a3 53.7–55.1 vs production 56.4–58.0, −5%; the fork's
one-token repacked mat-vec against production's whole-block one-column load, patch 0008) — that is candidate 2's real target now (port the
one-column fast path, not the multi-column one, to the two-plane layout), and it is also where the MTP verify deficit of section 1b lives.

**The after-32 effect (open):** an 8-cell that follows the 32-cell reads −11% on the branch without repack, −4% with, −2% on a3, and 0 on
production; every other transition (16→8, 12→8, 4→8, 1→8) is at parity on all builds. It is tied to the widest batch, which is also the
one that fills the whole 69632-token cache; `tools/b3-transition.sh` (32,8,8,8,16,8,24,8,32,8,8,8, and a 35K-cache variant) tells whether
it persists over later 8-cells and whether 24 triggers it — result in section 5b.

**Server level** (team profile `-np 16`, `server-bench.py` 1300 / 256, 8 and 16 clients; a3 with the repack on; three rounds, a3 first
in each pair):

| round | a3, 8 / 16 clients | production, 8 / 16 clients |
|---|---|---|
| 1 (s1b-test E) | 80.5 / 86.6 | 64.9 / 81.9 (the 8-client cell is a 15% outlier against every earlier production run, 76–78) |
| 2 (b3) | 80.2 / 86.2 | 78.3 / 81.9 |
| 3 (b3) | 79.2 / 86.5 | 78.3 / 81.7 |

Against production's two clean rounds a3 reads **+1.2 to +2.8% at 8 clients and +5.5% at 16** (86.2–86.6 vs 81.7–81.9), the first
time any branch build has beaten production at the server level; the night report's branch numbers (62.0 / 76.0 / 72.6 / 79.6 with the
repack, 66.3 / 73.5 / 79.9 / 81.2 without) were the 9–16-row cliff and the warm-up. **Gate status for a3:** serving gate passed;
numerics passed; the MTP / single-stream gate fails as for the branch (a3 changes nothing at 1–4 rows: tp4 single stream 53.7–55.1 vs
production 56.4–58.0). Promotion therefore still waits on the one-token path — candidate 2 in its corrected form (port production's
whole-block one-column load to the two-plane layout) — and on the per-shape warm-up being prewarmed.

## 5b. The after-32 effect is the warm-up in a longer form (b3)

`-npl 32,8,8,8,16,8,24,8,32,8,8,8` in one process (`-c 69632`), decode tok/s per cell:

| build | 32 | 8 | 8 | 8 | 16 | 8 | 24 | 8 | 32 | 8 | 8 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 215.0 | 174.0 | 174.0 | 174.0 | 202.4 | 174.1 | 192.0 | 173.9 | 214.7 | 173.7 | 173.9 | 173.9 |
| branch `--no-repack` | 210.9 | **156.1** | **168.0** | **161.3** | 200.9 | 173.0 | 194.4 | 174.3 | 213.0 | 174.0 | 174.1 | 173.0 |
| a3 | 240.1 | **166.2** | **165.9** | **169.0** | 206.2 | 174.4 | 219.4 | 175.5 | 258.0 | 174.3 | 175.8 | 174.3 |

The three 8-cells after the *second* 32-cell are at parity on both branch builds, so a 32→8 transition costs nothing. The loss sits in the
first three 8-cells of the process (about twenty seconds of wall time after a first cell that ran 65K prefill tokens and 4096 decode
tokens in the 32-row shape) and then never returns. That is the after-load warm-up of section 3 once more — per process, and evidently
per shape, since 4096 tokens of the 32-row shape did not pay it for the 8-row one — and it is the same thing the night report's 8-slot
runs saw whenever an 8-cell came early in a process. The clock sampler (`b3-clocks.txt`) shows the branch's slow window and
production's equivalent window at the same 1608–1730 MHz, 80–84 °C and ~200 W per die, so it is not thermal or power state.
Consequences: (a) the branch's per-shape warm-up must be prewarmed for every serving shape the launcher expects (1, 4, 8, 16 rows,
i.e. a short batched prewarm, not a single-stream one); (b) every earlier "8-slot" verdict on the branch was measured inside it; (c) the
mechanism (the fork's lazy first-use on its cached views / buffer types, both present with `--no-repack`) is the open item — S1b does
not touch it. A 35K-cache variant of the sequence is in `qwen38-27b-b3.md` (section "same on a 35K cache").

## 5c. Twelve homogeneous 8-row cells across the fork's history (bisect8)

`-npl 8,8,8,8,8,8,8,8,8,8,8,8` (tp4, 2K, `-c 69632`) per build; the fork positions are the bisect builds of the warm-up study (first-parent
b10254..b10912, pristine fork code without the Exabit series, `--no-repack` from position 33 on), now with `llama-batched-bench` added:

| build | twelve 8-row cells, tok/s | min / max | cells < 171 |
|---|---|---|---:|
| production | 174.4 174.0 173.8 174.0 173.9 174.1 173.9 174.0 173.9 173.6 173.7 173.7 | 173.6 / 174.4 | 0 |
| branch `--no-repack` | 174.4 174.1 173.1 173.4 174.6 173.2 174.4 174.2 173.4 174.4 174.4 173.3 | 173.1 / 174.6 | 0 |
| fork b10912 pristine `--no-repack` | 152.3 153.1 153.1 153.8 154.0 154.1 154.2 154.3 153.9 154.1 154.3 153.6 | 152.3 / 154.3 | – |
| fork position 16 (803f00bb7) | 162.2 161.6 162.0 162.0 162.1 162.3 162.0 162.0 161.6 161.7 161.7 161.8 | 161.6 / 162.3 | – |
| position 32 (8253e3fbf) | 158.8 157.8 158.8 159.2 159.4 159.4 159.2 159.5 159.5 159.5 159.5 159.6 | 157.8 / 159.6 | – |
| position 33 (e21ccb704, repack default on) | 160.8 160.4 160.5 160.1 160.4 160.2 160.3 159.9 160.1 160.1 160.3 160.4 | 159.9 / 160.8 | – |
| position 34 (97e14020c) | 159.2 159.3 159.3 159.6 159.7 159.6 159.0 159.4 159.5 159.4 159.6 159.6 | 159.0 / 159.7 | – |
| position 48 (115a32c1e) | 161.9 161.5 160.5 161.5 161.7 161.9 161.8 161.9 162.4 162.3 162.4 162.4 | 160.5 / 162.4 | – |
| position 64 (a65e71eb2) | 152.8 153.8 **148.1** 153.9 153.7 154.5 154.3 154.0 154.2 154.0 154.2 154.3 | 148.1 / 154.5 | – |
| position 80 (def64b335) | 154.9 155.2 155.2 **152.1** 155.8 156.1 156.1 156.0 156.2 156.1 156.2 156.1 | 152.1 / 156.2 | – |
| position 96 (9435cfcc4) | 151.8 152.4 **149.7** 153.6 153.1 153.8 153.5 153.3 153.5 153.2 153.1 153.2 | 149.7 / 153.8 | – |
| position 112 (16628b027) | 153.6 154.2 153.5 **149.1** 155.9 156.1 156.3 156.3 156.2 156.1 156.3 155.9 | 149.1 / 156.3 | – |
| production, repeat | 174.9 174.2 174.3 174.2 174.3 174.2 174.1 174.1 174.1 174.1 174.1 174.1 | 174.1 / 174.9 | 0 |
| branch `--no-repack`, repeat | 174.0 173.8 **167.7** 173.7 174.6 173.2 174.2 174.2 173.2 174.1 174.2 173.2 | 167.7 / 174.6 | 1 |

Result: **in twelve homogeneous 8-row cells the branch is production** (173.1–174.6 vs 173.6–174.4 in the first run; the repeat shows one
cell at 167.7 in the third position, −4%, and eleven at parity), so the intermittent episodes of §2, §5 and §5b are rare on a steady shape
and short — one cell of twelve, early in the process — and a history bisect on this shape cannot locate them. The
pristine fork positions read 152–162 because they carry the fork's own 8-column kernel, not the Exabit 16-column series (production's 174
is the series); the only structure in them is a single 3–4% dip in cell 3 or 4 at positions 64, 80, 96 and 112 and none before 48 — a weak
pointer at the fork's late-August runtime changes (cached views, the graph work), the same era the warm-up bisect named, and not proof.
The mechanism stays open; the operational answer does not depend on it: a per-shape prewarm at start (1, 4, 8, 16 rows) and the
acceptance runs measured after it.

## 6. S1 upstream candidate: the tile table alone on pristine master (s1b-test F, s1v2-test)

`patches/upstream-S1/` = the fork's tile-table commit f48d37902 (author Marko Tombak) rebased onto master 5d806aa25 with the lost brace
restored, 36 lines: `mmq-config-gfx906.cuh` (Q8_0: 8 warps, J up to 128, else the rdna2 table), host selection on `GGML_CUDA_CC_VEGA20`,
device selection on `__gfx906__`, the `J > 64` occupancy gate (never for MoE ids, never with fewer tiles than CUs). No repack, no MMVQ
change, no compiler flag. `test-backend-ops -o MUL_MAT`: 1288 / 1288. Against `/opt/llama.cpp-master` (pristine, same recipe):

| | pristine master | + tile table (v1, J ≥ 8) |
|---|---:|---:|
| pp2048 tp4 (two rounds) | 844.0 / 844.8 | 1087.1 / 1088.1 (**+28.8%**) |
| pp2048 one die | 234.1 | 314.5 (**+34.3%**) |
| tg128 tp4 / one die | 45.8 / 20.0 | 45.8 / 20.0 |
| batched tp4 2K, 1 / 8 / 16 / 32 slots | 44.8 / 153.3 / 151.6 / 180.3 | 44.7 / 145.8 / **140.2** / 207.9 (+15%) |
| perplexity 16K | 5.6216 | 5.6216 |

The prefill gain is the ablation's (+30% / +37% on b10288 → +29% / +34% on master) and numerics are bit-for-bit, but on master the
**16-row decode reads −7.5%**: upstream's MMQ has changed since b10288 (where the same table gave +16% at 16 slots), and its rdna2 J=16
tile now beats the gfx906 table's J=16 entry. The 8-row cell (−4.9%) is inside pristine master's own round-to-round spread at that
width (15 tok/s in the paired data; 8 rows go to MMVQ and never see the table). **v2 of the patch applies the gfx906 config from J = 32
up** (tag `s1-v1` keeps v1); measured after the transition test — section 6b. Authorship stays Marko Tombak's; sending needs his
agreement (`patches/upstream-S1/README.md`).

## 6b. S1 patch v2 (J ≥ 32): clean on master (s1v2-test)

Batched tp4 2K, `-npl 1,8,16,32`, two interleaved rounds (pristine master / v1 / v2), and llama-bench:

| build | 1 | 8 | 16 | 32 | pp2048 tp4 | pp2048 one die | tg128 tp4 / one die |
|---|---:|---:|---:|---:|---:|---:|---|
| pristine master | 44.7 / 44.8 | 156.3 / 156.4 | 151.9 / 151.7 | 180.1 / 180.2 | 844 | 234 | 45.8 / 20.0 |
| v1 (J ≥ 8) | 44.8 / 44.8 | 156.6 / 157.3 | **140.4 / 140.3** | 208.0 / 208.1 | 1087–1088 | 314 | 45.8 / 20.0 |
| **v2 (J ≥ 32)** | 44.8 / 44.8 | 157.4 / 157.3 | 151.7 / 151.7 | 208.1 / 207.9 | 1082 | 314 | 45.9 / 20.0 |

v2 keeps the whole prefill gain (+29% tp4, +34% one die) and the 32-row gain (+15%) and is at parity at 1, 8 and 16 rows; section 6's 8-row
dip (145.8) was pristine master's own round-to-round scatter at that width (it reads 153–157 across four runs tonight). `patches/upstream-S1/`
now carries v2 (tag `s1-v2` in the s1 worktree; v1 kept as `s1-v1`), numerics identical, `test-backend-ops` clean. Technically ready to
send; authorship and the decision to send are Marko Tombak's and yours.
