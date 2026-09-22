# gfx906 re-survey — action plan

**Authoritative kick-off document** for the per-profile gfx906 patch survey. Restructured holistically
2026-09-20 after the multi-user baseline campaign and the move to a common v0.4.1 base; before that it
had grown by accretion and read as an append log.

Read in this order:
1. `/root/llama.cpp-benchmarking/REQUIREMENTS.md` — outranks everything, including this file
2. **Part 1 of this document — the method.** Applied algebra: every patch is a term against the common
   base v0.4.1, bins are branches, and §1.8 says where the algebra stops being valid. Do not start the
   survey without it.
3. `/root/night-20260919/REPORT.md` — the multi-user baseline campaign, which settled the design point
   and every test condition the survey uses
4. the rest of this document, then the memory index

**Status 2026-09-20.** Stage A (baseline) and Stage B (torch) are **complete**. Stage C (rebase to
v0.4.1) is **substantially done** — the substrate and 17 of our 29 patches are on the common base, with
12 conflicts and the build/gate/push outstanding. Stage D (the survey) is **gated on four items in
Part 6**. Nothing has been binned; no patch has a verdict.

---

## Part 0 — State of the world, 2026-09-20

### The box

ROCm 10.0 only; 7.14 is **purged** (`/opt/rocm/core-10`, `core-10.0`). Kernel 7.0.0-31-generic with the
local PCI BAR-resize patch, four Vega 20 dies at 32 GiB BAR, one XGMI ring (physical order
0b-0e-1e-1b; the firmware hop table mislabels the bridge pairs, corrected topology in
`/root/rccl_topo_fixed.xml`). PyTorch 2.13.0+gfx906 installed from the published deb. Xeon W-3275M,
378 GiB RAM.

### Source trees — six, collapsing to one

| tree | role | state |
|---|---|---|
| `/root/exabit-llama.cpp` | **the working repo** (`origin` = exabit-io/llama.cpp) | branches on v0.4.1, see Part 2 |
| `/root/llama.cpp` @ b10288 | pristine upstream reference | 0 local commits — no rebase needed |
| `/root/llama.cpp-b10837`, `-b10859` | pristine upstream references | 0 local commits — no rebase needed |
| `/root/mx-llama.cpp-b10254` | mxxm substrate snapshot | read-only provenance |
| `/root/mx-llama.cpp-b10912` | single-user lineage (10 of our patches on mxxm b10912) | superseded by `gfx906-single` |

The three pristine checkouts need no rebase: zero local commits, nothing to preserve. The two
`mx-llama.cpp-*` trees become read-only provenance once `gfx906-single` is complete.

### Installed builds

`/opt/llama.cpp-gfx906-rocm10` is the build every baseline number was measured on (reports as
"build 11067" — **that is this branch's own commit count, `b10913` + 154, NOT upstream b11067; the two
are not comparable**). `/opt/llama.cpp-gfx906` (multi-user) and `/opt/llama.cpp-mxxm-fh` (single-user)
are the pre-10.0 production builds; both were built against the purged 7.14 runtime and **cannot run**.

### The measured baseline — see `/root/night-20260919/REPORT.md` for the full campaign

- **Design point: 4 x 64K q8_0 at 125 W.** 16.45 tok/s per request on the ladder (full-slot worst
  case), **18.24 median / 17.04 p10 / 17.02 min** under true chat shape, TTFT 1.63 / 1.92 s. Passes
  R3.1 on median, p10 **and** min — 37% above the floor at its worst.
- **Both design points named in R2.2 FAIL R3.1**: 8 x 192K = 5.15 tok/s/req (43% of floor),
  4 x 256K = 7.56 (63%). Both fit memory comfortably.
- **Capacity is decode-bandwidth-bound, not memory-bound.** Memory permits ~1.63 Mtok of KV; R3.1
  permits ~0.52 Mtok. Nothing OOMed in 16 cells.
- **Decode model**, 12 cells, mean error 0.5%, worst 1.6%:
  `per-slot decode time = 17.6 ms + 3.49 ms x slots + 94.6 ms x total_KV_Mtok`. Use it to price a
  configuration instead of running one.
- Prefill depends only on per-sequence depth, not slot count:
  `prefill t/s = 1/(0.000732 + 0.004856 x depth_Mtok)`.
- **Per-die memory itemisation (§5.2, first time recorded):** model 6517 MiB = 6.36 GiB; q8_0 KV
  **8.5 KiB/token/die** (34 KiB/token over four dies, constant to +/-0.3%); compute buffer
  6.3-12.5 KiB/token/die, reaching 9.7 GiB/die at 8 x 192K. `offloaded 66/66 layers` in every cell.
  Only **16 of the 65 blocks carry KV** (qwen35 hybrid attention) — which is why KV is far cheaper
  than a naive 64-layer calculation predicts.
- **Production fan control costs ~0.3%** at the design point (junction 68-72 C at ~800 rpm versus
  47-49 C maxed, ~25 C below throttle). Valid at 125 W only — at 200 W under PWM it would be much
  hotter.
- Peak DC across the campaign **1144 W** of the 1228 W envelope; `PZ0T` never left zero.

### Phase 3 correctness gate — results worth carrying forward results worth carrying forward

- Perplexity 16K/6 on `Qwen3.8-27B-Q8_0`, 4 rotated runs, **bit-identical** 7.14 vs 10.0:
  `PPL = 5.6171 +/- 0.06236`, all six chunks matching to the last digit. Lands inside the
  historical corpus cluster (5.5969–5.6448). **ROCm 10.0 is numerically neutral.**
- `test-backend-ops`: 10.0 = 12667/12667 on ROCm0/1/3, **12666/12667 on ROCm2**;
  7.14 = 12667/12667 on all four. One `TOPK_MOE` case, 10.0-specific, off the production
  model's path. Not chased, per the lead. See `FINDING-topk-moe.md`.
- Power: peak 831 W across four dies, `PZ0G` 1088.9 W of a 1228 W envelope, `PZ0T = 0.0`
  throughout. No clamp. The 150 W host RAPL cap is what makes this fit.

---

### Retired, deliberately

The old "Measured on ROCm 10.0" table (all `service-client.py` shape, D6) has been **removed** rather
than carried forward: every service-axis figure in it is shape-invalid, and D9 later showed the same
numbers were additionally distorted by an undersized prompt cache. The capacity-ladder figures it held
are superseded by the 16-cell campaign. Nothing in it should be quoted.

### C5 correctness gate on the rebased tree — 2026-09-20

`gfx906-substrate-v041` (v0.4.1 + substrate) **builds clean** after five passes and four defect classes,
all of them invisible to git (recorded as T9).

**`test-backend-ops`, all four dies, zero failures:**

| die | result |
|---|---|
| ROCm0 | 16198/16198 |
| ROCm1 | 16198/16198 |
| ROCm2 | **16198/16198** — 416 TOPK_MOE cases, 0 failed |
| ROCm3 | 16198/16198 |

Two things worth carrying forward. First, coverage **widened**: the pre-rebase build ran 12,667 cases,
v0.4.1 runs 16,198, so upstream added ~3,500 op cases in the interval. Second, **the ROCm2 TOPK_MOE
failure recorded in `FINDING-topk-moe.md` is gone** — pre-rebase ROCm2 was 12666/12667 with one TOPK_MOE
case failing; it is now clean across 416 cases. Upstream PR 28313 ("ROCm: resolve TOP_K kernels") falls in
v0.4.1's range and is a plausible cause, unverified.

A methodological note, because it nearly produced a false pass: the first attempt ran with `-b ROCm0`,
which made `test-backend-ops` **skip** ROCm1/2/3 and CPU while reporting "5/5 backends passed". Skipped
backends count as passed. The all-dies result above is the re-run without the filter.

Perplexity 16K/6 is the second half of the gate; reference PPL = 5.6171 +/- 0.06236, historical cluster
5.5969-5.6448.

### What is NOT done

- **Nothing is binned.** No patch has a verdict record. 177 candidates await the survey.
- Stage C: 12 conflicts unresolved, no build, no C5 gate, nothing pushed.
- `gfx906-required` and `gfx906-both` do not exist yet.
- The four Part 6 gating items.
- The fan-power delta (first attempt failed — see T8).

---

### State at 2026-09-20 09:40 UTC — readiness pass

**The first algebraic measurement on the common base is done.** R (substrate) vs R+ours (our 20 terms
as a bundle), n=4 per arm, ABBA-interleaved, 125 W, `--cache-ram 49152`, `-ngl all`:

| cell | metric | R | R+ours | effect | p (exact permutation) |
|---|---|---:|---:|---:|---:|
| 4x32K | decode tok/s/slot | 20.774 +/- 0.058 | 21.441 +/- 0.021 | **+3.21%** | 0.0286 |
| 4x32K | prefill t/s | 566.0 | 726.5 | **+28.37%** | 0.0286 |
| 4x64K | decode tok/s/slot | 15.424 +/- 0.034 | 15.872 +/- 0.017 | **+2.90%** | 0.0286 |
| 4x64K | prefill t/s | 510.2 | 632.9 | **+24.04%** | 0.0286 |

p=0.0286 is the FLOOR for n=4 vs n=4 (2 of 70 arrangements): the arms do not overlap at all. BH FDR at
q=0.10 rejects all four; all four also clear the >=2% effect half of the dual criterion. Scope limit,
stated so it is not over-read later: this validates the **bundle** on the **multi-user axis only**. It
bins no individual term, and says nothing about single-user until gate 2 exists.

**A real defect found and fixed while clearing Stage C.** `gfx906-multi` and `gfx906-single` both forked
at 4167bb280 — *before* the five v0.4.1 build-failure resolutions — so:
- neither compiled (two copies of `fast_bf16_hardware_available`, Trap T9 again: clean merge, broken build);
- worse for the method, neither shared the substrate R that the comparison above measured, so any number
  taken from them would have violated the common-base requirement silently.

Resolution: both are now **empty destination bins** at the substrate head, which is what the
branch-per-bin topology always meant (their own tip commit said they held "pre-survey, not binned,
contents"). Verified first that nothing is lost: all **12/12** code commits that sat on them are
represented in `terms/*.patch` as distinct terms (11 patches; two commits are the same tp-allreduce term
ported twice). Pre-survey tips preserved as `backup/gfx906-{multi,single}-presurvey-20260920`. The 29
terms are committed at `f04198b3f`. `gfx906-{required,both,single,multi}` and `gfx906-substrate-v041` now
all point at one identical, green substrate.

**Gate status.** Gate 3 (variance) PASSED: CV 0.187%/0.245%, so n=2 detects 2% — that is what makes the
~21 h budget real rather than ~51 h. Gates 1 and 2 are scripted as `gate12.sh`, blocked only on the
stock v0.4.1 build now in flight (the zero point did not exist as a binary at all). Gate 2 adopts an
explicit instrument definition:

> **single-user axis instrument** = `llama-batched-bench -npl 1 -npp 32768 -ntg 1024`, four dies,
> `-sm tensor`, q8_0 KV, 125 W, n=4; metrics decode tok/s (one stream) and prefill t/s.

Four dies and one stream because that is what the single-user profile actually runs; 32K because of the
context floor rule. Three arms (stock / substrate / bundle) so the axis gets its zero point, its baseline
and the bundle in one job.

**Gate 4 scoped down by inspection.** PR #24549 is *not* in v0.4.1 by number. What is present is
3f7c29d31 (`graph_reused`) plus per-input `can_reuse()` checks, and 57819b8d4, which disables graph reuse
for **pipeline parallelism only** ("TODO: figure out a way to make graph reuse work with pipeline
parallelism"). Tensor split is not covered by that guard and is exactly what we run. v0.4.1 also exposes
a runtime kill switch, `LLAMA_GRAPH_REUSE_DISABLE`, which makes the gate cheap. Instrument
(`gate4-graphreuse.sh`): determinism across slots — four slots, temperature 0, identical prompt, every
completion must be byte-identical to the others and to a 1-slot reference, with reuse ON vs OFF. A
throughput benchmark cannot see this class of bug at all. The test is vacuous unless reuse actually
happens, so arm A must report `graphs reused` > 0; if it is 0, that fact is the finding.

**Scope decision taken (not blocking).** The two MTP terms `17dfa2336` (adaptive MTP draft depth, PR
#27210) and `c9ce0c4e0` depend on the fork's `process_decode`, which v0.4.1 no longer has (upstream
replaced it with a virtual `process(const llama_batch&)`). Reimplementing that plus per-draft
`tensor_parallel_size` would put real work on the critical path for 2 terms out of 186, so both are
binned **technique-requires-implementation** and excluded from this campaign. R3.9 only requires MTP
measured on/off, which the existing mxxm-fh build can satisfy separately after binning. Reversible on
the lead's call.


### Gates 1 and 2 — RESULT, 2026-09-20 11:02 UTC (20/20 cells, 0 failures)

The zero point exists and the chain B -> R -> R+ours is now measured end to end at the pinned recipe
(125 W, `--cache-ram 49152`, `-ngl all`, q8_0 KV, `-sm tensor`, n=4, exact permutation, BH FDR q=0.10):

| axis | arm | cell | decode tok/s/slot | sd | prefill t/s | sd |
|---|---|---|---:|---:|---:|---:|
| multi | B stock v0.4.1 | 4x32K | 16.945 | 0.016 | 586.5 | 0.20 |
| multi | R substrate | 4x32K | 20.774 | 0.058 | 566.0 | 2.50 |
| multi | R+ours | 4x32K | 21.441 | 0.021 | 726.5 | 0.13 |
| multi | B stock v0.4.1 | 4x64K | 13.181 | 0.014 | 522.9 | 0.14 |
| multi | R substrate | 4x64K | 15.424 | 0.034 | 510.2 | 2.00 |
| multi | R+ours | 4x64K | 15.872 | 0.017 | 632.9 | 0.27 |
| single | B stock v0.4.1 | 1x32K | 30.065 | 0.050 | 586.8 | 0.43 |
| single | R substrate | 1x32K | 39.795 | 0.159 | 557.4 | 5.85 |
| single | R+ours | 1x32K | 39.582 | 1.229 | 720.7 | 7.39 |

**The substrate trades prefill for decode, and that is a survey target, not a footnote.** Against stock
it is +22.60% / +17.02% decode at 4x32K / 4x64K and +32.36% single-stream — all p=0.0286, all through BH
— but **-3.50% / -2.42% prefill on the multi-user cells and -5.02% single-stream**, also significant. So
somewhere in the 157 substrate commits sits a prefill regression that the fork's own tile table then
more than repays: our terms take prefill to +23.87% / +21.04% / +29.30% over stock. Binning the substrate
is therefore not box-ticking; there is a measured regression inside it to isolate.

**Our bundle on the single-user axis: prefill +29.30% (p=0.0286), decode -0.53% (p=1.0000, not
significant).** That is consistent with the earlier S1b finding of single-stream parity, and it is the
first time the single-user axis has had a zero point to say so against.

### AMENDMENT to the gate 2 instrument — single-stream decode is heavy-tailed

Gate 2's own data invalidates the estimator I specified for it. The `R+ours` 1x32K reps were
40.18 / 40.19 / 40.22 / **37.74**. The outlier is not noise:
- prefill in that cell was normal (722 t/s); the whole loss is in the TG phase, T_TG 27.13 s vs 25.48 s;
- the clock trace catches all four dies at the 1000 MHz DPM floor at ~34 W for one sample **inside** the
  cell, while DC was 770 W of the 1228 W envelope, PZ0T = 0, temps 53-56 C and fans steady — a stall,
  not a power or thermal event;
- stock is tight (30.06/30.00/30.08/30.12) and the four-slot arms stay at CV 0.10-0.28%; only
  single-stream on the fork-substrate builds does this (CV 3.10%).

This is the **intermittent four-die single-stream stall** already recorded for the fork's tensor-parallel
state, reproducing on the v0.4.1 base. Consequence for the method: single-stream decode is tight (~0.05%
CV) *plus* rare ~6% stalls, so **one stall moves a mean of n=4 by 1.5% — most of the 2% effect floor.**
Gate 3's n=2 sizing was measured on the FOUR-SLOT cell and does not transfer to this axis.

`stallprobe.sh` (n=12 per arm, substrate and bundle, interleaved, run-level replication) measures the
stall RATE so n and the estimator for the single-user axis are chosen on evidence. Until it reports, the
single-user axis is **not** cleared for screening, and no single-user verdict may be written from n=4
means. The multi-user axis is unaffected and cleared.


### Gate 4 — PASS, and gate 2's estimator amended. ALL FOUR GATES NOW CLEAR (2026-09-20 11:20 UTC)

**Gate 4: graph reuse is bitwise-neutral under `-sm tensor`.**

| arm | slots | `graphs reused` | completions |
|---|---:|---:|---|
| reuse ON | 4 | 191 | `46e0e1803a12` x4, identical |
| reuse OFF (`LLAMA_GRAPH_REUSE_DISABLE=1`) | 4 | 0 | `46e0e1803a12` x4, identical |
| reuse ON | 1 | 191 | `2070fac26968` |
| reuse ON, separate server process 18 min earlier | 4 | 191 | `46e0e1803a12` — reproducible |

Reuse is **active** (191 graphs, and the kill switch verifiably takes it to 0), so the test is not
vacuous — the failure mode flagged in advance. Reuse ON and OFF are **bitwise identical at the same
batch shape**: if reuse were going stale, disabling it would change the output.

**My first verdict criterion was wrong and the script has been corrected.** It also required the 4-slot
output to match a 1-slot reference, and printed FAIL when it did not. But that difference *persists with
reuse disabled*, which proves reuse is not its cause: batching changes matmul widths (MMVQ vs MMQ paths),
so batched inference is not bitwise equal to unbatched even at temperature 0. The gate's real question is
whether reuse changes anything at a FIXED batch shape, and the answer is no. Cross-shape difference is
now reported as information, with the true failure condition being reuse-ON != reuse-OFF. Consequence for
the survey: every comparison must hold the batch shape fixed — which the pinned cells already do.

**Gate 2 amended: the fix is the ESTIMATOR, not more reps.** `stallprobe.sh`, n=12 per arm:

| arm | n | decode mean | median | sd | CV | min | max | stalls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| substrate | 12 | 39.871 | 39.910 | 0.114 | 0.287% | 39.58 | 39.96 | 0 |
| substrate+ours | 12 | 40.258 | 40.275 | 0.059 | 0.146% | 40.16 | 40.34 | 0 |

**Zero stalls in 24 runs.** Combined with gate 2 the rate is 1 in 16 single-stream runs of R+ours and
0 in 16 of the substrate: the stall is real but rare, and not a property of a build. And the two
distributions do not overlap at all (substrate max 39.96 < ours min 40.16), giving
**+0.97%, p<1e-5 (200k permutations)** — significant, but **below the 2% materiality floor, so parity**
on single-user decode, with prefill the real single-user gain (+29.3%).

Why gate 2's n=4 mean said -0.53% and the truth is +0.97%:

| estimator on the gate-2 reps 40.18 / 40.19 / 40.22 / 37.74 | value |
|---|---:|
| mean | 39.582 — one stall drags it 1.5% low, most of the 2% floor |
| median | 40.185 — lands on the n=12 truth of 40.258 |

**Rule adopted for the single-user axis: MEDIAN of n=4, not mean, plus an explicit stall count per arm.**
Median at n=4 already survives one stall, so this costs nothing in GPU hours — the earlier worry that the
axis would need 3x the reps was wrong. The multi-user axis keeps the mean (four-slot CV 0.10-0.28%, no
stalls ever observed).

**Gate summary: 1 PASS (zero point measured), 2 PASS (instrument defined, baseline measured, estimator
fixed on evidence), 3 PASS (variance), 4 PASS (reuse bitwise-neutral). The survey is unblocked.**

A new checker rule came out of this: **T11, bare `wait`**. `wait` with no arguments waits for every
background child of the shell, so `gate4-graphreuse.sh` deadlocked after writing all four completions --
it waited forever on the `llama-server` it had started itself, GPUs at 0%. It reads exactly like "wait for
the requests". Encoded with fixtures; `scriptcheck.sh` self-test now 27 assertions.


## Part 1 — The method: applied algebra

Lead, 2026-09-20: *"We are using applied techniques of algebra here, so everything needs a common base,
and that is v0.4.1."* This part states that method precisely, and — just as importantly — states where
it stops being valid.

### 1.1 Terms

Let **B = v0.4.1**, the common base (D8-D11, D16). Every candidate patch **P** is expressed as a diff
against B. That is what makes P a **term**: an object that can be added to, and subtracted from, any
tree that shares B.

177 candidates: 148 from the mxxm substrate (D14) plus 29 of ours, all now expressed against B.

### 1.2 Composition

A build is
```
    T = B + R + Σ Pᵢ          R = the required substrate (bin neutral-required-substrate, D16)
```
Because every term shares B, the sum is well defined. For terms touching disjoint code it is also
order-independent — which is what lets a bin be a *set* rather than a sequence.

### 1.3 Measurement is a function of four arguments, not one

```
    m(T, axis, metric, cap)
```
- **axis** ∈ {single-user, multi-user} — never averaged (D1/R2.7)
- **metric** — the ONE metric line, per rule 1; for multi-user, per-request decode with median/p10/min
- **cap** — 125 W by default (D10). **A term's effect is a function of the cap**, so a Δ measured at
  one cap may not hold at another. The ISA review's roofline-ridge finding (D15) makes this concrete
  for flash-attention patches, which must be benched at both caps.

Dropping any argument produces the class of error that lost the >300 tok/s configuration (D4).

### 1.4 Two attribution operators

```
    add-one-in     Δ⁺(P) = m(B+R+P) − m(B+R)
    leave-one-out  Δ⁻(P) = m(F)     − m(F−P)        F = the full candidate set
```

Both are valid; they answer different questions. Δ⁺ is "what does P buy on its own?"; Δ⁻ is "what does
P still buy in the presence of everything else?" — which is how it ships, so **Δ⁻ is the one that
decides adoption.**

### 1.5 The independence test, and how it detects the conflicts bin

If **Δ⁺(P) ≈ Δ⁻(P)**, P is independent of the other terms and its verdict is safe.

If they **differ materially**, P interacts with at least one other term. That is not noise — it is the
signature of the **`conflicts-with-another-patch`** bin, and the verdict must name the other term. We
have already found 12 such cases before running a single benchmark: our patches and the substrate both
modify `mmvq.cu`, the `ggml-cuda.cu` fusion path and `gated_delta_net.cu`.

### 1.6 Why a common base is REQUIRED, not merely tidy

`F − P` is undefined unless P's diff context exists in F. A term expressed against base A therefore
**cannot** be subtracted from a tree built on base B. Mixed bases do not make the algebra harder — they
make it **invalid**, so no leave-one-out verdict can be computed at all.

The cost was measured, not assumed: the single-user set is 10 commits, every one with a multi-user
twin, and **9 of the 10 require the identical conflict resolution as 9 of the 12 multi-user ones**. On
the old split bases (multi on b10913+merges, single on b10912) that resolution had to be done twice, on
two different bases, producing two patch sets that could not be compared. On the common base it is done
once and both profiles inherit it.

### 1.7 Binning is a partition; the branches make it queryable

Each of the 177 terms receives **exactly one** of the nine bins (D12, D16). The branch-per-bin topology
(D16) means each bin's term set is a git query rather than hand-kept bookkeeping:

```
    git log gfx906-both   ^gfx906-required   ==  the `both` bin
    git log gfx906-single ^gfx906-both       ==  the `single-user-only` bin
    git log gfx906-multi  ^gfx906-both       ==  the `multi-user-only` bin
```

Bins therefore cannot drift from branches, and the verdict records (`survey/*.md`, enforced by
`survey/survey-lint.py`) are the ledger that says *why* each term sits where it does.

### 1.8 Where the algebra STOPS — the honest limits

This is bookkeeping discipline, not a claim that performance is linear. Four limits, each of which has
already bitten or is expected to:

1. **Terms that touch the same code do not commute.** Their composition depends on resolution order, so
   `Σ` is not well defined for them. These are exactly the conflicts bin (1.5) and each needs a *chosen*
   resolution, recorded per profile.
2. **Δ is not additive when terms share a bottleneck:** `Σ Δ(Pᵢ) ≠ Δ(Σ Pᵢ)`. D15 makes this concrete —
   if decode is memory-traffic-bound and moving ~4x the necessary bytes, then two patches that each
   remove traffic will **not** sum, because the second acts on a smaller remainder. Expect the composed
   build to under-perform the sum of its parts, and do not read that as a measurement error.
3. **A Δ is only valid at the cap and depth it was measured at** (1.3). The 4 x 32K A/B cell is a
   *proxy*; its rank-correlation with the 4 x 64K design point must be validated once before the sweep
   (Part 5, cost section), and any marginal verdict re-checked at the design point.
4. **Therefore the composed build must be MEASURED, never predicted from its terms.** That is what the
   §5 acceptance chain is for, and it is the last gate before promotion. A build assembled from
   individually-winning terms is a hypothesis until the chain passes.

### 1.9 What this buys

Each of the 177 candidates ends with: a bin, a branch, a verdict record naming its axis, its metric, the
cap, the evidence level (individual or group), and what would overturn it. That is the structural answer
to the failure this whole re-survey exists to correct — a verdict recorded as "regression" with no axis,
which discarded a multi-user gain because it looked bad on a single-user cell (D4).

### 1.10 Statistical inference — a Δ is not a result until it survives a test

Lead, 2026-09-20: *"I would also hope you plan on using statistical methods to p-value test the
patches."* Correct, and the naive version of it would be wrong in three separate ways.

#### The unit of replication is the RUN, not the request

A run yields 12-24 per-request decode figures, and they are **not independent**: requests share the
server and interfere — the campaign measured decode falling from ~13 to ~6.5 tok/s when another slot
prefills. Treating 24 requests as 24 samples is **pseudo-replication** and would inflate significance
enormously. So the run-level statistic (median per-request decode) is one observation, and
**n = number of runs per arm**.

#### n = 3 is statistically incapable of significance — n = 4 minimum, n = 5 preferred

For a two-sided **exact permutation test** on run-level statistics:

| n per arm | arrangements | minimum achievable two-sided p | can reach 0.05? |
|---:|---:|---:|---|
| 2 | 6 | 0.333 | no |
| 3 | 20 | **0.100** | **no** |
| 4 | 70 | 0.029 | yes |
| 5 | 252 | 0.008 | yes |

**This retires the "triplicate" plan**: with three runs per arm the smallest p an exact test can return
is 0.10, so no verdict could ever be called significant. Earlier cost estimates in this document that
assumed triplicate were therefore measuring something that could not have been concluded.

#### BLOCKER 3 sets the budget, so it runs first

Runs needed to detect the §5 noise band (2%) at 80% power, two-sided 0.05:

| run-to-run sigma | n per arm |
|---:|---:|
| 0.3% | 2 |
| 0.5% | 2 |
| 1.0% | 4 |
| 2.0% | 16 |
| 3.0% | 35 |

Indirect hint: prefill at 64K read 954.95 / 955.31 / 955.83 t/s across three *different* cells — 0.09%
spread. If decode is comparably tight, n = 4-5 is ample. If sigma is nearer 2%, the campaign quadruples.
**Measure the variance before committing GPU hours** (Part 6, BLOCKER 3).

#### The test, and the design that makes it valid

- **Test:** two-sided exact permutation on the run-level statistic. Report the observed effect, its
  confidence interval, and p. Permutation rather than t because n is small and normality is unevidenced.
- **Blocked, interleaved, randomised order.** Baseline and candidate runs alternate within a block so
  thermal drift, ccache state and page-cache warmth affect both arms equally. Never all-baseline then
  all-candidate — that confounds the patch with drift, which is the same class of error as starting all
  clients at t=0 (D7).
- **Shared baseline.** For add-one-in, the baseline arm is measured once at n=5 and reused, so the cost
  is `177 x n + n`, not `177 x 2n`.

#### Multiplicity: 177 tests will manufacture ~9 false positives

At uncorrected alpha = 0.05, 177 tests yield **8.8 expected false positives** — indistinguishable from
nine genuinely good patches. Counting both axes it is 354 questions and **17.7** expected.

**DECIDED by the lead 2026-09-20: Benjamini-Hochberg FDR at q = 0.10**, applied across the whole family.
The alternative was Holm/FWER at 5% (proposed by the external handoff's DR-B and DR-C). FDR and FWER are
**not interchangeable**, and the choice was presented rather than assumed. FDR is the right fit here
because this is a *discovery* campaign whose winners are each re-confirmed individually by the §5
acceptance chain before promotion — so a controlled proportion of false discoveries is caught downstream,
whereas FWER's conservatism would discard true positives the campaign exists to find.

The verdict record stores the **raw p and the BH-adjusted q**. Two conditions on validity:
- BH is applied to the **confirmation** statistics only, never to screen statistics (see escalation
  below) and never to LASSO- or shrinkage-derived pseudo-p-values, which selection bias makes invalid.
- The family is declared **before** looking: all 177 terms x 2 axes. Adding hypotheses afterwards and
  re-running BH is not FDR control.

#### Significance is necessary, not sufficient — the dual criterion

A patch with p = 0.01 and a +0.3% effect is real and useless. A verdict of `improves` or `regresses`
therefore requires **both**:

1. **BH-adjusted q < 0.10**, and
2. **|effect| >= the noise band (2%, §5)** on the named axis

Anything that fails either test is `neutral-drop`. This keeps the bins decidable from the numbers
rather than from judgement.

**Structural necessity is recorded separately from performance evidence** (handoff §4.2: *"required is
not synonymous with neutral"*). `neutral-required-substrate` was a bad name because it implies a measured
equivalence. A required patch can cost performance and still be mandatory, and when removing it makes the
build **unrunnable there is no performance number to record at all** — that absence must not be written
down as "neutral". So the verdict record carries an independent field:

    structural: required-by:<patch-id,...> | standalone

A term with `structural: required-by:...` belongs in `gfx906-required` on structural grounds whatever its
performance verdict says, and its performance field may legitimately read `not-measurable (removal breaks
the build)`.

#### Screening is triage; confirmation uses FRESH data (corrected 2026-09-20)

**An earlier version of this section was wrong.** It proposed screening at n=2 and escalating the
ambiguous band to n=5 by *adding* three runs to the two screen runs. Pooling them biases the estimate
upward for exactly the patches selected *because* their screen looked good — the winner's curse. The
external handoff's §8.2 puts it correctly: *"Discovery data are not automatically confirmation data."*

Corrected protocol:

1. **Screen** every term at n=2 against the round's frozen base. Purpose: **triage and ordering only.**
   No p-value, no bin, no verdict is derived from screen data. A screened-only term's state is
   `not-prioritised` or `unresolved`, never `neutral`.
2. **Freeze the shortlist** — candidate identities, axes, metrics, margins and the hypothesis family —
   *before* collecting confirmation data.
3. **Confirm** each finalist on **n>=4 fresh, randomised, blocked paired runs**. Only these enter the
   permutation test and BH.
4. If the recipe changes after confirmation results are inspected, the affected claims need **fresh**
   confirmation. A holdout reused indefinitely is not a holdout.

Cost: screening 177 terms x 2 axes at n=2 is ~12 GPU h; confirming a shortlist of ~25 at n=4 fresh is
~7 h. **~19-25 GPU hours**, which is *cheaper* than the earlier (invalid) pooled scheme and statistically
sound. The remaining ~150 terms retain honest provisional states rather than fabricated verdicts — which
is what the original request for per-patch p-values eventually needs, just not in the first pass.

---

## Part 2 — Repo and branch topology

*(Decision D16. Lead-approved 2026-09-20.)*

Lead approved: *"Yes that repo structure works for me too."*

```
exabit-io/llama.cpp                  (the one code repo; remotes: upstream=ggml-org, mxxm=mxxm-t)
│
v0.4.1                               common base and zero point (D8-D11, and the algebra: patches
│                                    against one base are composable terms)
│
├─ gfx906-substrate-v041             UNBINNED POOL. Test source only, never deployed.
│                                    All 177 candidates live here awaiting verdicts.
│
└─ gfx906-required                   bin `neutral-required-substrate` — enabling infrastructure that
   │                                 wins nothing alone but that winners depend on
   │
   └─ gfx906-both                    its own commits = bin `both`   <- THE tracked delta
      │
      ├─ gfx906-single               its own commits = bin `single-user-only` -> /opt/llama.cpp-mxxm-fh
      └─ gfx906-multi                its own commits = bin `multi-user-only`  -> /opt/llama.cpp-gfx906
```

**Each branch's own commits equal exactly one bin**, so every delta is a query rather than hand-kept
bookkeeping: `git log gfx906-both ^gfx906-required` IS the both-bin set. Bins cannot drift from
branches. And because the profile branches descend from `gfx906-both`, both are **directly buildable** —
no compose-at-build-time step that nobody tested.

Patches in no branch, but still requiring a verdict record: `regresses-both`, `upstream-already-has-it`,
`technique-requires-implementation`, `neutral-drop`.
`conflicts-with-another-patch` is expressible — the same patch may appear in both profile branches with
**different resolutions**, since they are separate branches.

**This collapses six source trees to one repo.** The three pristine upstream checkouts
(`/root/llama.cpp` @ b10288, `llama.cpp-b10837`, `llama.cpp-b10859`) have zero local commits and need no
rebase — they are historical reference checkouts. The two `mx-llama.cpp-*` trees become read-only
provenance. Measured justification: the single-user patch set is 10 commits, all with multi-user twins,
and **9 of the 10 are the same patches that conflict with the substrate** — so on a shared base that
resolution happens once instead of twice.


Lead approved: *"Yes that repo structure works for me too."*

```
exabit-io/llama.cpp                  (the one code repo; remotes: upstream=ggml-org, mxxm=mxxm-t)
│
v0.4.1                               common base and zero point (D8-D11, and the algebra: patches
│                                    against one base are composable terms)
│
├─ gfx906-substrate-v041             UNBINNED POOL. Test source only, never deployed.
│                                    All 177 candidates live here awaiting verdicts.
│
└─ gfx906-required                   bin `neutral-required-substrate` — enabling infrastructure that
   │                                 wins nothing alone but that winners depend on
   │
   └─ gfx906-both                    its own commits = bin `both`   <- THE tracked delta
      │
      ├─ gfx906-single               its own commits = bin `single-user-only` -> /opt/llama.cpp-mxxm-fh
      └─ gfx906-multi                its own commits = bin `multi-user-only`  -> /opt/llama.cpp-gfx906
```

**Each branch's own commits equal exactly one bin**, so every delta is a query rather than hand-kept
bookkeeping: `git log gfx906-both ^gfx906-required` IS the both-bin set. Bins cannot drift from
branches. And because the profile branches descend from `gfx906-both`, both are **directly buildable** —
no compose-at-build-time step that nobody tested.

Patches in no branch, but still requiring a verdict record: `regresses-both`, `upstream-already-has-it`,
`technique-requires-implementation`, `neutral-drop`.
`conflicts-with-another-patch` is expressible — the same patch may appear in both profile branches with
**different resolutions**, since they are separate branches.

**This collapses six source trees to one repo.** The three pristine upstream checkouts
(`/root/llama.cpp` @ b10288, `llama.cpp-b10837`, `llama.cpp-b10859`) have zero local commits and need no
rebase — they are historical reference checkouts. The two `mx-llama.cpp-*` trees become read-only
provenance. Measured justification: the single-user patch set is 10 commits, all with multi-user twins,
and **9 of the 10 are the same patches that conflict with the substrate** — so on a shared base that
resolution happens once instead of twice.

#### Implied approval of the ninth bin

`gfx906-required` **is** the `neutral-required-substrate` bin, so approving this topology approves
splitting `neutral` into `neutral-drop` and `neutral-required-substrate`. Stating that inference
explicitly rather than assuming it. Nine bins now; `survey/survey-lint.py` must be updated to accept
the two new values before the first verdict is written.

#### Creation order — `gfx906-required` must exist BEFORE the campaign

Per-patch binning needs each patch to apply and build on a baseline, and some provably do not: our three
repacked narrow-batch mat-vec patches cannot apply without `ggml-cuda/q8_repack/`. So:

1. **Populate `gfx906-required` first**, from evidence rather than guesswork. Known-required today:
   `ggml-cuda/q8_repack/` (three of our patches cannot apply without it). The meta/TP backend is a
   *hypothesis* until a dependent patch is shown to need it.
2. The per-patch A/B baseline is **`gfx906-required`**, not bare v0.4.1.
3. Where a patch still will not stand alone, test it with its dependencies and record in the verdict
   that the evidence is group-level (`would change if`).

`gfx906-required` and `gfx906-both` do not exist yet — they are created as the campaign populates them.
The existing `gfx906-multi` / `gfx906-single` hold **pre-survey** contents (the rebased equivalents of
what the two production builds ship today) and get rebuilt from verdicts once binning is done. They are
legitimate as non-regression baselines, and are not binned results.

---

---

## Part 3 — Decisions of record

All sixteen are equally binding. Numeric order.

### D1. Two build profiles, gated independently (REQUIREMENTS **R2.7**)

Two maintained gfx906 builds, because the patch sets are **mutually exclusive between modes**:

- **multi-user / service** — the primary deliverable. Batched `llama-server`. Judged on §3 + the §5 chain.
- **single-user** — batch-1 / single-stream / MTP. Judged on single-stream decode at the R2.2 context floor.

Each is measured, gated and promoted **on its own axis**. A change is never rejected for
regressing the other profile — it goes to the other build, or to neither.

Still true: never gate or headline the *multi-user* build on a single-user cell; never average
the two axes; 2K-context cells remain out of scope for both (floor 16K, realistically 32K).

R2.6 is superseded. §4 no longer forbids single-stream/MTP work.


### D2. Baseline means *stock upstream*, not "the current build"

The lead's definition, and the correct anchor for attribution. Two distinct things, do not
conflate them:

- **Zero point / reference** — de jure upstream stock llama.cpp. What you measure *from*.
- **Non-regression baseline** — the currently promoted build. What you must not get worse than.

**gfx906 needs ZERO llama.cpp patches.** `/root/llama.cpp` is pristine `ggml-org/llama.cpp`
at `b10288` — clean tree, no extra commits, no diff vs the tag, and upstream contains no
gfx906 references at all. It builds and runs with nothing but `-DAMDGPU_TARGETS=gfx906`.
Every gfx906 patch in this project is **performance, not enablement**.

Caveat: that is only true because ROCm itself is patched one layer down — AMD dropped gfx906
and mixa3607 rebuilds TheRock with it re-enabled.


### D3. Rebase BEFORE the survey — target **v0.4.1**, DONE 2026-09-20 (see Stage C)

Lead: "before we do the new survey we are going to rebase on b11046 or whatever is the
current latest build."

| Tree | Base | Behind `b11046` | Series on top |
|---|---|---|---|
| `/root/exabit-llama.cpp` (multi-user) | `b10913` | ~133 builds | **154 commits** |
| `/root/mx-llama.cpp-b10254` (single-user) | `b10254` | ~792 builds | fork + local patches |
| `/root/llama.cpp` (zero point) | `b10288` | ~758 builds | 0 |

Why the order matters: the 154-commit series **already contains upstream merges and adopted
PRs** (`d4abd573f` = upstream #28552, `17dfa2336` = PR #27210 by stew675, plus merges from
`mxxm/master` and `upstream/master`). Some of what we carry as "our patches" may already be
upstream or superseded by it. Rebasing tells us which patches still apply — that is a survey
output in itself. Surveying first would measure patches upstream has made redundant.

Expect real conflicts in `ggml-cuda`/`ggml-hip`, where both we and upstream are active.


### D4. The >300 aggregate tok/s figure is unrecoverable (REQUIREMENTS **R3.3**)

Lead: it was never logged properly; it was discarded during the single-user optimisation work
because it read as a regression on the single-user axis — which R2.7 now makes a category
error. It is **not** a gate and **not** a baseline. Removed from the §7 TBC list. The
multi-user baseline is re-established from scratch by the current campaign.


### D5. ROCm 10.0 is SETTLED — there is no rollback (lead, 2026-09-19)

> "resistance is futile, we'll just have to work through debugging any possible performance /
> regressions with 10.0, we won't be going back to 7.14, the arrow of time moves forward."

The LLM tooling ecosystem targets 10.0; maintaining a rollback we will never take is dead weight.
**Consequences for this plan:**

- Stage **A5 (decide promotion) is deleted.** 10.0 is promoted by decision, not by measurement.
- Phase 4 is no longer a promotion gate — it is **baseline establishment**. The 7.14 arm still runs
  ONCE, for reference only: knowing *whether* 10.0 regressed is what distinguishes "a bug to chase"
  from "these are simply the numbers". That reference cannot be reconstructed after 7.14 is purged,
  and losing an unrepeatable measurement is exactly what cost the >300 tok/s configuration (D4).
- A regression found against 10.0 is **debugged, not reverted**.

**7.14 removal is sequenced, not deferred.** Two things still bind it today:
1. The PyTorch venv hard-codes `/opt/rocm/core-7.14/lib` in its RUNPATH — 45 of 48 ROCm libs.
   Not repointable; the binary must be rebuilt (Stage B). No ROCm 10.0 wheel exists upstream.
2. Phase 4 has not yet banked the 7.14 reference numbers (it died in run 1 of 4).

Purge once BOTH are satisfied. Cost of waiting is 5.3 GB against 2.9 TB free — no pressure.
Fallback if Stage B stalls: purge the 7.14 *packages* but keep `/opt/rocm/core-7.14/lib` as a
plain directory so the venv keeps working until torch is rebuilt.


### D6. The service workload shape was WRONG — and this invalidates the corpus's service axis

Lead, 2026-09-19: *"I tell you very little (prefill) and you generate lots of tokens and then prompt
after prompt the session context grows. If I have to tell you 192K tokens to get you to produce only a
small fraction of that something isn't right."*

REQUIREMENTS **R2.4 rewritten** accordingly. A real session per turn: user says **100-500 tokens**;
model generates **2K-8K** (thinking + answer — Qwen3.8's thinking alone exceeds 4K, so any gen budget
under ~4K misrepresents the model); **tool calls** inject results mid-session; context accumulates from
the **model's own output**. That is how a session reaches R2.2's 128K-192K — not a document pasted in
at turn 0. Once a session outgrows its slot, context must be compressed / shifted / checkpointed —
**cost unmeasured, new TBC.**

| | old harness | reality |
|---|---|---|
| opens with | 16K-56K injected document | ~300-token question |
| generated/turn | 512 | 2K-8K |
| **prefill : decode** | **~33 : 1** | **~1 : 15** |

Inverted by ~500x. **Every service-axis number in the corpus taken with `tools/service-client.py`
measures document-QA / RAG, not interactive chat, and must be re-taken.** The capacity-ladder numbers
(`llama-batched-bench`) SURVIVE — they measure decode at depth directly and do not depend on session
shape.

Replacement instrument: **`/root/rocm-tests/bench/chat-client.py`** (written 2026-09-19). Models the
above, reports `pre:dec` as a first-class column so the shape cannot silently invert again, supports
`--tool-rate` / `--tool-tokens` and `--ctx-limit` (the compression point).

**Retractions forced by this** — do not carry these forward:
- "TTFT p90 of 168 s is brutal" — artifact of document injection; a real turn prefills ~300 tokens.
- "The service is prefill-bound, so the fork tile table's **+33% prefill** is free multi-user
  performance" — unsound. In a decode-bound reality that patch's **+62% decode at 12 slots** is the
  relevant figure. (The underlying cross-profile point still stands: a patch binned "single-user" may
  be the multi-user win. That is what D-survey below is for.)


### D7. Harness defects found and fixed 2026-09-19

Both in `bench/service-client.py`; both silently corrupted results rather than failing loudly.

1. **No arrival model.** `ThreadPoolExecutor.map` started every client at t=0 — an N-way simultaneous
   prefill no real server sees. **Measured cost: median decode 11.05 -> 15.15 tok/s** at 4 x 64K q8_0
   from staggering alone, i.e. the artifact pushed a *passing* configuration under the R3.1 floor.
   Added `--arrival-rate` (Poisson) and `--ramp`; default 0 preserves old behaviour so prior results
   stay reproducible. §5.1 now REQUIRES a non-degenerate arrival model.
2. **An absolute `TAG` crashed the client after the whole workload ran.** `B` was hard-coded and the
   tag joined onto it, so `/abs/path` produced `/root/rocm-tests/bench//abs/path` and threw
   `FileNotFoundError` at write time — losing every aggregate. Phase 4 would have produced **zero**
   usable numbers across all four runs. Fixed in the client (absolute/relative tags now honoured).

Backups: `service-client.py.orig-20260919`.


### D8. The design point is **4 x 64K q8_0 at 125 W** (measured 2026-09-19/20)

The multi-user baseline campaign measured twelve (slots x depth) cells at 200 W and four at 125 W.

    per-slot decode time = 17.6 ms + 3.49 ms x slots + 94.6 ms x total_KV_Mtok
    12 cells, mean error 0.5%, worst 1.6%. Use it to PRICE a configuration instead of running one.

- Per-slot decode depends on **total** KV over all slots, so `slots x depth` is one budget; but slot
  count is *also* an independent cost, so the shape matters and not only the product. At an identical
  0.393 Mtok: 6x64K = 13.20, 8x48K = 12.18, 12x32K = 10.30 tok/s per request.
- At 200 W, four of twelve cells clear R3.1's 12 tok/s: 4x64K (17.75), 6x64K (13.20), 4x128K (12.40),
  8x48K (12.18).
- **At the 125 W production cap, exactly one does: 4 x 64K at 16.45 tok/s, 65.8 t/s aggregate.**
  6x64K falls to 11.71, 4x128K to 10.92, 8x48K to 10.67.
- Prefill depends only on per-sequence depth, not slot count (955 t/s at 64K for 4, 6 and 8 slots).

**Stage D3's "service run at the design point" therefore means 4 x 64K q8_0 at 125 W.**
R2.2 (depth per user) and R3.3 (aggregate throughput) point opposite ways along this frontier and
R3.2's own objective is flat along it, so the spec does not break the tie — open question for the
lead, along with whether R3.1 is median or p10.


### D9. `--cache-ram 49152` is MANDATORY, and its absence invalidates service measurements

`llama-server`'s `--cache-ram` defaults to **8192 MiB**, but one parked 32-44K conversation is
1.5-2.1 GiB, so 12 clients need ~30 GiB and get 8. The cache then evicts constantly and each
displaced client re-prefills its whole conversation. Same hardware, same config, same client:

| 8 x 48K, 12 clients | default 8 GiB | `--cache-ram 49152` |
|---|---:|---:|
| cache misses | 15/24 | **0/24** |
| prefill tokens | 510,305 | **14,168** |
| prefill:decode | 5.22:1 | **0.14:1** |
| TTFT med / p90 | 32.6 / 37.9 s | **1.36 / 2.84 s** |
| decode median | 7.45 | **12.10** |

**This extends D6.** Workload shape is a property of client **x configuration**, not of the client
alone: `chat-client.py` doing correct R2.4 chat against a default server produced **28.56:1** at
4 x 128K — worse than the 33:1 document harness R2.4 was written to replace. Any service-axis A/B run
without a sized cache buries every patch under a 60-100% eviction penalty and mis-ranks all of them.

With the cache sized, chat-shape **median** decode matches the ladder's per-slot figure to within 1%
at all three configs tested, and `slots x median` reproduces the ladder aggregate to within 1%. So the
cheap `llama-batched-bench` cell is a valid predictor of the median — but not of the spread.


### D10. The survey runs at **125 W**, because that is where compute efficiency is under pressure

Lead's decision and reasoning, 2026-09-19: HBM2 runs at full clock regardless of the cap, so a fixed
125 W envelope puts selection pressure on the compute die's logic efficiency. Confirmed:

- `hbm-bw-cap-sweep.md`: HBM bandwidth is **cap-invariant from 110 W down to 50 W** — 880.6 GB/s
  read, ~715 GB/s copy, sclk at the 999 MHz DPM floor, **die drawing 115 W throughout**.
- So at 125 W only **~10 W/die** sits above that streaming floor, versus ~85 W at 200 W: an **8.5x
  smaller marginal budget for compute**.
- The workload split confirms it: **prefill loses a uniform 20.2% at 125 W (0.5 pp spread over four
  configs differing 2x in slots and 2x in depth); decode at depth loses only 10.7%.**
- Dies were pegged at the cap in **every** cell at both caps (198-199 W at 200; 124 W at 125), so
  nothing was clock- or thermally limited. Performance-per-watt *is* performance here.

**Consequence for D2: every old survey verdict was taken at 200 W and therefore systematically
understates compute-efficiency patches.** An independent reason to re-derive, on top of D6 and D1.
**First prediction to test:** the fork tile table was adopted for +33% *prefill*, and prefill is what
the cap punishes hardest — so it should be worth MORE at 125 W, and its ranking against the Q8_0 MMVQ
fast path (a decode patch, in the -11% regime) may invert versus 2026-09-08.

**Caution:** absolute t/s deltas shrink at 125 W, so more repetitions may be needed; but relative
deltas should grow for compute patches, so SNR may improve instead. Verify on the first patch rather
than assume, and spot-check anything marginal at 200 W in case a ranking inverts between caps.


### D11. `-ngl all` is mandatory under `-sm tensor`

Build 11067 defaults `-ngl` to **`auto`**, and auto has **no implementation** for tensor split —
every start logs `common_fit_params: failed to fit params to free device memory:
llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`. So a tensor-split run without an
explicit `-ngl` relies on undefined offload behaviour, and at depth a partial offload would report a
plausible-looking decode number instead of an honest OOM. `-ngl all` resolves to `n_gpu_layers = -2`;
the probe confirmed **`offloaded 66/66 layers to GPU` in every cell**. Belongs in
`settings/gfx906.env` and `launch.sh`, not just in benchmark scripts.


### D12. The verdict record is MANDATORY, and the taxonomy has nine bins (lead, 2026-09-20; ninth bin added via D16)

Lead: *"Yes adopt the verdict schema and both new bins."*

**A candidate is not "surveyed" until a verdict record exists with every field populated.** The record
lives at `/root/llama.cpp-benchmarking/survey/<patch-id>.md`, one file per candidate **per axis**
(two files if a patch is surveyed on both). Template: `survey/TEMPLATE.md`.

**Enforcement is a script, not a convention:** `survey/survey-lint.py` validates every record and
exits non-zero on any violation. It refuses a missing or empty field, an unreplaced template
placeholder, an `axis` outside {multi-user, single-user}, an unrecognised verdict or bin, and — the
specific D4 failure — a `regresses` verdict that does not carry median/p10/min on its named axis.
Self-tested against a deliberately broken record: 8 violations caught, exit 1.

The two anti-recurrence fields:
- **`axis`** has no default, so an axis-free verdict is literally unwritable. The >300 tok/s
  configuration was lost to exactly that (D4).
- **`would change if`** forces the author to state what evidence was *not* collected, which is what
  turns a verdict into something a later reader can overturn rather than inherit.

Seeded with two real records, one per new bin: `pr27841-gcn-mmq-config.md`
(**upstream-already-has-it**) and `tech-gemv-gemm-threshold.md`
(**technique-requires-implementation**). Both pass the linter.


### D13. Cross-engine technique sweep, and the forge coverage matrix (lead, 2026-09-20)

Lead: *"if you are going to look at vLLM and SGlang patches for the gfx906, you need to check for
those on GitHub and other forges too."* Done 2026-09-20.

**Forge coverage — this is the reusable result; do not redo it blindly.**

| forge | control query | gfx906 llama.cpp | gfx906 vLLM/SGLang/Triton |
|---|---|---|---|
| GitHub | works | many | many |
| **Gitee** | **search unusable** (v5 API returns `[]`; `so.gitee.com` is an Indexea JS widget) | **unknown** | **2 confirmed via lead-supplied URLs** |
| AUR | works (86 for "rocm") | 0 | 0 |
| Codeberg / Gitea | works | 0 | 0 |
| GitLab | works | 0 | 0 |
| Hugging Face (models + spaces) | works | 0 | 0 |

**Conclusion: gfx906 work lives on GitHub and Gitee only.** Gitee **repository pages read fine** — only
its *search* is blocked, so the blind spot is **discovery, not access**. Closing it needs a Gitee token
or ~2 minutes of a human running `https://so.gitee.com/?q=gfx906`. Compounding: the available web
search is **US-only**, so CSDN / Zhihu / Chinese forums are structurally invisible, and the MI50 is
predominantly a Chinese second-hand-market card.

**The real gap was not enumeration — it was mining.** The 2026-09-09 list *already* contains the
vLLM/SGLang/Triton ecosystem; the GitHub sweep returned almost entirely `known` entries. But D1's
process only ever binned **llama.cpp patches**, so none of these were ever read for transferable
technique. The `technique-requires-implementation` bin is what makes them actionable. Highest-signal
entries to mine first, all already in the list:

| source | why |
|---|---|
| `nick413-bit/gfx906-fa-vllm` | a **FlashAttention backend written for gfx906** — we run `-fa on` with our own tile tables and head-256 rows |
| `NnnHU/sglang-mi50` | **Qwen3.8-27B + "Qwen3.5 hybrid GDN"** — the same model, and it must handle the same hybrid attention that makes only **16 of 65 blocks carry KV** (found 2026-09-20) |
| `ai-infos/vllm-gfx906-mobydick` (86*) / `Igneous/…-kvarn` | current gfx906 vLLM development line; KVarN **KV dtype** work, adjacent to our q8_0-vs-q8/q4 KV decision |
| `nlzy/vllm-gfx906` (434*) | the canonical gfx906 vLLM fork |
| `nlzy/triton-gfx906` (48*) | gfx906 Triton — kernel-level arch facts |

New-but-low-signal since 2026-09-09: `bryan-fund/ML-gfx906`, `ajunlonglive/vllm-gfx906-6.3.4`,
`frieddeli/mi50-vllm-rocm-runtime` (all 0-2 stars, mirrors or personal configs).


#### New candidates found 2026-09-20 (the re-survey the lead asked for)

New repos since 2026-09-09: **`bespokeontology/qwen3.8-27b-mi50-cpp-engine`** (2026-09-13 — same
model, same 4x MI50 hardware; a custom C++/HIP engine, so technique source rather than a patch) and
`atretador/gfx906_MI50_16GB_scripts` (scripts). Everything else returned by search was already in
the 2026-09-09 list.

Upstream PRs worth binning (GCN/HIP-relevant, not RDNA-gated):

| PR | state | why it matters here |
|---|---|---|
| **24549** | open | graph reuse under SPLIT_MODE_TENSOR — **our exact config**, see GAP 10 |
| **20831** | open | dynamic MMVQ nwarps for **narrow** matrices — same territory as our Q8_0 MMVQ fast path |
| 28709 | open | MMQ N-tiles heuristic for CDNA — adjacent arch, pattern likely portable |
| 28613 / 28195 | open | MMVQ batch thresholds / 64-row MMQ tiles (RDNA3.5-gated, method relevant to our batch-size staircase at 8/16/24) |
| 28943 | open | skip fully masked KV tiles in FA (WMMA/RDNA path; idea applies to deep-context decode) |
| 27962 / 28616 | open | branch-free SWAR `__vsub4`/`__vcmpne4` — gfx906 lacks the native ops; check arch gating |
| 28313 | open | ROCm TOP_K kernels — possibly related to our `FINDING-topk-moe.md` |
| **28846** | **merged** | BF16 -> F32 fallback on pre-CDNA/pre-RDNA3 — **gfx906 has no BF16**; confirm build 11067 contains it |
| **27841** | **merged** | GCN MMQ config (our S1 patch) — **upstream took it**; confirm whether our local copy is now redundant |

27841 being merged is a concrete instance of D3's rationale: we may be carrying a patch upstream
already has. The rebase will reveal how many others.

#### Gitee: access works, DISCOVERY does not (2026-09-20)

Correction to GAP 8: I can read Gitee **repository** pages normally (verified on two). Only Gitee's
**search** is unreachable — the v5 API returns `[]` even for a control query, and `so.gitee.com` is an
Indexea-backed JS widget. **So the blind spot is discovery, not access:** given URLs (or a Gitee
token) these repos survey normally. Cheapest fix remains 2 minutes of a human with a browser.

Two repos the lead supplied, both **vLLM** forks — not llama.cpp, so **nothing here is an applicable
patch.** They are technique sources, and two of their findings independently corroborate ours:

**`gitee.com/bjxamo_admin/vllm-gfx906`** (mirrors `Said-Akbar/vllm-rocm` lineage; last update
2025-05-02, so old):
- **quantization GEMM-vs-GEMV threshold changed from 50/24 to 8/8.** This is our batch-size
  staircase arrived at independently: we measured MMVQ in use at <= 8 and decode-cost steps at
  8/16/24. Two different engines converging on 8 makes the switchover worth an explicit sweep in
  llama.cpp rather than an assumption.
- **GGUF kernels: "doubling the number of thread blocks" for batched requests, "10%~20% performance
  improve".** Same family as our S1b width-dependent launch bounds, and we are batched multi-user.
  Testable hypothesis.
- **GPTQ kernels: dot-product accumulator FP16 -> FP32 "to avoid calculation overflow"** — a
  *correctness* claim about gfx906 FP16 accumulation. Confirm llama.cpp's HIP MMVQ/MMQ accumulate in
  FP32 (they appear to, via int8 dot + FP32 scale); the failure mode would be silent.
- Flash attention: raising Triton `num_stages` benefits gfx906 — principle may map to our FA tiles.

**`gitee.com/wj7927/gfx906-vllm`** (mirrors `github.com/ttdxq/gfx906-vllm`, active to 2026-08):
- shape-aware kernel dispatch: single-token FP16 -> LLMM1, small-batch non-gfx9 -> Triton, remaining
  gfx906 -> PyTorch/ROCm GEMM fallback. Same idea as our tile table / MMVQ dispatch.
- claims official AMD RCCL lacks gfx906 kernels and a custom rebuild is needed for TP.
  **Checked on this box: not a problem for us** — `librccl` (package `amdrocm-rccl-host10.0`,
  mixa3607's gfx906-enabled rebuild) lists gfx906 among its architectures, and empirically RCCL beats
  llama.cpp's internal allreduce by 23% in decode, which a broken fallback would not.

### D14. mxxm is SUBSTRATE — all 148 of its commits get binned (lead, 2026-09-20)

Lead, correcting Claude's framing: *"mxxm is a substrate and we have to test all 148 of his patches
and bin them accordingly."*

**The correction matters and Claude had it wrong.** The v0.4.1 rebase triage separated our 29 commits
from 148 authored by mxxm-t / Marko Tombak and treated the latter as third-party history to step
around. That is backwards: `/opt/llama.cpp-mxxm-fh` — the **single-user production build** — *is* that
fork plus our patches, so the substrate is already load-bearing in production. And three of our own
patches (the repacked narrow-batch mat-vec family, i.e. the Q8_0 MMVQ fast path that REQUIREMENTS §6
requires stay active) will not even rebase without `ggml-cuda/q8_repack/`, which exists only in the
fork. The substrate is not optional and it is not background.

#### What the 148 actually are

Concentrated exactly where performance lives — 195 file-touches in `ggml-cuda`, only 8 housekeeping
commits, and several quoting their own measured gains ("fuse the K-quant and IQ4_NL MoE up and gate
mat-vec on gfx906 (+6 to +8% decode)"). One touches `qwen4exp` — Flash-Next, our R3.10 second model.

| group | commits |
|---|---:|
| meta / tensor-parallel / allreduce | 28 |
| q8_repack | 27 |
| spec / MTP | 18 |
| graph / lane / scheduling | 13 |
| MoE fusion + top-k router | 11 |
| model support | 10 |
| mat-vec / mmq / mmf | 9 |
| flash attention | 3 |
| warp / DPP reductions | 3 |
| server / API | 2 |
| housekeeping | 6 |
| unclassified (needs a read) | 18 |

Out of scope on our config (R2.5: Q8_0 model quant, q8_0 KV; models Qwen3.8-27B + Flash-Next):
only ~5 commits naming quants we do not run and ~5 naming models we do not run. **The substrate is
overwhelmingly applicable** — it cannot be dismissed on scope.

#### Cost — CORRECTED 2026-09-20 after the lead asked "why do you need 50 minutes for each test?"

**The 50 min/test figure was inflated about 8x. Per-commit binning of all 177 is affordable.**

Three errors in the original estimate:

1. **A chat run (23 min) was double-counted per patch.** Unnecessary: the campaign established that
   the ladder's per-slot decode predicts the chat-shape **median** to within 1% at all three configs
   tested. The chat run belongs in §5 acceptance on the *composed* build, not in every A/B. Single-user
   was also put at 15 min when `llama-bench` at the 32K floor is ~35 s.
2. **The ladder cell was 98% prefill for a 2% measurement.** The 4 x 64K cell at 125 W spends
   **345.9 s prefilling to measure 7.8 s of decode**. `-ntg 1024` gives **8x the decode sample for
   +15% wall time** (cell 354 s -> 408 s). `-ntg 128` should never have been the A/B setting.
   Note `-ntg` lists do NOT amortise — `batched-bench` re-prefills every (npp,ntg,npl) combination
   (triple nested loop, tools/batched-bench/batched-bench.cpp:132-147) — and `-pps` is not a valid
   shortcut: sharing the prompt shares the KV, so total KV drops 4x and the decode-at-depth
   characteristic being measured changes.
3. **The depth is a choice.** A 4 x 32K cell costs 2.8 min against 5.6 min at 4 x 64K, and 32K is
   exactly R2.2's floor, so it is a defensible measurement rather than an out-of-scope shortcut.

**Per-patch, per-axis, with the 4 x 32K cell at `-ntg 1024`:**

| item | cost |
|---|---:|
| build (ccache, few files; the record holds an 80 s FULL build) | 1-2 min |
| multi-user cell (4 x 32K, -ntg 1024) | ~3 min |
| single-user `llama-bench` at the 32K floor | ~1 min |
| **single rep, both axes** | **~5-6 min** |
| **triplicate, both axes** | **~13-15 min** |

| campaign | n=1 (screen only) | n=5 (full test) |
|---|---:|---:|
| 11 groups, leave-one-out | 1.1 h | 3.7 h |
| **all 177 commits, per-commit** | **17.7 h** | **~65 h** |

**Superseded by §1.10.** A flat n=3 cannot establish significance at all (exact-test floor 0.10) and a
flat n=5 costs ~65 h. The adopted design is **adaptive**: screen at n=2, escalate only the ambiguous
band to n=5, landing near **35-40 GPU hours**. The true n is set by the harness sigma, which Part 6
gate 3 measures first.

**So Option B — per-commit binning of every one of the 177, which is what the lead asked for — costs
~18 h single-rep or ~41 h triplicate and is the right choice.** Option A (grouping) is retained only
as a fallback if the proxy validation below fails and the budget doubles. Grouping remains useful for
*presentation* — reporting 177 verdicts organised by the D14 groups — but is no longer needed to make
the campaign affordable.

**Two preconditions, both cheap:**
- **Validate the 4 x 32K proxy once** for rank correlation against the 4 x 64K design point: measure
  2-3 patches at both depths. If rankings diverge, fall back to 4 x 64K and double the budget.
- **Measure harness variance (BLOCKER 3)** to decide single-rep versus triplicate. If the cell repeats
  to better than ~0.5%, single-rep resolves any effect above ~1% and the campaign is ~18 h.
  Indirect hint: prefill at 64K read 954.95 / 955.31 / 955.83 t/s across three *different* cells
  (0.09% spread). That is suggestive, not a repeatability test.

#### Consequence for the rebase target

If the substrate must be carried, the rebase is not "our 29 onto v0.4.1" — it is **the substrate plus
our 29 onto v0.4.1**, i.e. ~177 commits. That is the C3 decision the plan already flagged as the
lead's, now forced: either rebase the fork forward (a large recurring cost on every upstream move, and
R3.7 wants the series on our own branch), or extract the substrate's *applicable* patches onto our
branch once and own them. **The survey's Option A output is exactly what tells us which subset is
worth owning** — so the survey should run BEFORE committing to a rebase strategy for the substrate,
inverting D3's order for the fork specifically.

Claude's 29-commit rebase onto v0.4.1 stands as useful work regardless: it proved the eight
gfx906 kernel patches apply clean across a 45-day jump, and it identified the q8_repack dependency.
Branch `gfx906-v041`, record in `docs/REBASE-v041-TRIAGE.md`.

---

### D15. ISA re-review — what the survey should actually hunt for (2026-09-20)

Full document: `/root/llama.cpp-benchmarking/reference/ISA-REVIEW-20260920.md` (947 lines, every claim
labelled DOCUMENTED / INFERENCE / NOT IN THESE DOCUMENTS with ISA cites). Commissioned to re-read the
Vega 7nm ISA against the campaign's measurements. Five conclusions that change survey priorities:

#### 1. The decode kernel is AT the memory roofline but carrying ~4x the necessary bytes

94.6 ms per Mtok of KV works out to only **90 GB/s/die** on KV bytes alone (8.5 KiB/tok/die x 1e6 =
8.70 GB in 94.6 ms) against the **880.6 GB/s** measured — apparently 9.6x off. But add the scratch
traffic the campaign itself measured (8.5 KV + 2 x 12.5 compute-buffer KiB/tok/die) and it becomes
**354 GB/s/die, within 4% of the project's own 369 GB/s FA-vec figure.**

**So the kernel is not slow — it is moving about four times the bytes it needs to.** The VALU roofline
for the same work is 3.5-7.1 ms, i.e. compute is nowhere near binding.

**Consequence: the win is traffic elimination, not a faster inner loop.** Specifically: killing
scratch / split-K round-trips, and consuming q8_0 KV *in quantised form* rather than dequantising to
scratch. The compute-buffer figure from §5.2 of the report — which had looked like mere bookkeeping —
is the explanation for the decode bottleneck. **Any candidate patch should be judged first on whether
it reduces bytes moved per token.**

#### 2. The 3.49 ms/slot term is not dispatch overhead

Kernel count does not scale with slot count, so it is not launch cost. It is **45% of a full weight
re-read per row** (7.76 ms/pass), which matches the project's existing counter data (29 -> 96
loads/wave going from batch 1 to batch 8). The constant 17.6 ms is ~9.2 ms launch + RCCL, and
17.6 + 3.49 = 21.1 ms lines up with the M1 trace's 21.5 ms single-stream token.

#### 3. head_dim = 256 exactly — so the head-256 FA tile row is the row this model uses

Confirmed twice from the campaign's own numbers (q8_0 8704 B and f16 16384 B per token per die;
34816/136 = 256). RoPE is 64/256 = **25% partial rotary**. That puts the **alex4300 head-256 tile
row** at the top of the flash-attention candidate list rather than somewhere in the middle — it is not
a speculative row, it is the one this model dispatches to.

#### 4. MFMA confirmed absent; the 125 W optimisation ranking

Zero MFMA hits (gfx908 = CDNA1; no AGPRs or accum_offset before gfx90a). Ranked for a fixed 125 W
envelope:

1. **bytes moved** (cap-free — HBM is not governed by the cap)
2. `V_DOT4_I32_I8` / `V_PK_*` — 4x / 2x fewer lane-instructions for the same work
3. 32-bit instruction encodings, and `GLOBAL` with `SADDR` + immediate offset instead of `FLAT`
4. SALU / SMEM offload
5. DPP last

**And explicitly: do not raise occupancy.** DPP gotcha worth knowing before adopting any DPP patch —
it rides only VOP1/VOP2/VOPC, never `V_DOT*`, `V_PK_*` or `V_FMA_F32`, and `ds_swizzle` cannot cross
lane 32.

#### 5. Contradictions flagged — treat these as open, not settled

- The KV coefficient is **4.1x worse** than the project's own 369 GB/s FA figure. The two readings
  imply *different* patches, so **trace before choosing an FA patch.**
- **The 715 GB/s copy figure may be impossible**: on a one-direction convention it implies 1430 GB/s of
  DRAM traffic against a 1024 GB/s peak. Pin down the convention before quoting it again.
- "Everything is power limited" holds only between **125 and 200 W**. Below ~115 W the DPM floor binds,
  not the cap. **Do not extrapolate the cap slopes downward.**
- Decode attention intensity of 11.3 FLOP/byte **straddles the roofline ridge** (15.7 at 1730 MHz, 9.1
  at 999 MHz), so **every FA patch must be benched at both caps** — a patch can win at one and lose at
  the other.
- **The AMD reference documents contain no performance data at all.** The Infinity Fabric guide is an
  MI100 *installation* manual; the Instinct tuning guide is EPYC/PCIe with image-only figures. There is
  no HBM peak, clock-domain map, power model or instruction throughput in any of the four. **Every such
  number in our reports is project-measured and must be cited that way**, not as vendor-documented.

---

### D17. Round-based promotion into `gfx906-both` (lead, 2026-09-20), and four corrections from the external handoff

Lead: *"the sooner we can commit genuinely known good performance improvement patches to gfx906-both,
that speeds up the subsequent testing of additional gfx906-both patches and also subsequent testing
that we need to do to for properly binning gfx906-single and gfx906-multi patches."*

Source reviewed: `/root/gfx906_patch_discovery_complete_ai_handoff.md` (1431 lines, a handoff from a
separate AI conversation carrying three proposed designs DR-A/B/C). It states its own limits —
*"a proposal, not a proven result on the user's system"*, and no patch list, costs, or variance were
available to it. Per memory `public-artifact-no-internal-refs`, outside reviews are **checklists to
verify, not authority**. Most of what it lists as missing, tonight's work supplies.

#### The lead's point is a methodological necessity, not just a speed-up

The handoff's §9 states the required stack comparisons:

```
R      vs R+B        the shared delta
R+B    vs R+B+S      incremental single-user delta
R+B    vs R+B+M      incremental multi-user delta
R      vs R+B+S      full single-user gain
R      vs R+B+M      full multi-user gain
```

> *"These are precisely how the highlighted `gfx906-both` delta retains a measured identity instead of
> becoming merely the overlap of two patch lists."*

**So `R+B` must EXIST before the single-user or multi-user increments can be measured at all.** Without
it, `both` degenerates from a measurement into an inference (the intersection of two winner lists).
Populating `gfx906-both` early is therefore a precondition for correctly binning S and M, and the
lead's instruction is adopted.

#### But nothing enters `gfx906-both` unmeasured

The §6 "must remain active" list (Q8_0 MMVQ fast path, fork tile table, custom allreduce) was measured
at **200 W** (D10: that mis-ranks compute patches), on the **old base**, and partly through the
**shape-invalid harness** (D6) with an **undersized cache** (D9). Committing it as "known good" would be
exactly the error this re-survey exists to correct. "Genuinely known good" must mean *measured on the
current recipe*.

**The fast path is to measure those first, in priority order** — not to assume them:

| # | seed candidate | prior |
|---:|---|---|
| 1 | Q8_0 MMVQ fast path + 16-column width | R6; +62% decode at 12 slots claimed |
| 2 | fork tile table / head-256 FA rows | R6; **D15 confirms head_dim = 256 is this model's row** |
| 3 | custom allreduce, four-row gate | R6; +14.5% single-stream; `gfx906.env` depends on it |
| 4 | DPP warp reductions on GCN | applied clean; D15 ranks DPP last, so a good falsification test |
| 5 | MoE K-quant/IQ4_NL fused mat-vec | the substrate claims +6 to +8% decode |
| 6 | `q8_repack` paths | **structural** — three of our patches cannot apply without it |
| 7 | meta/TP backend (28 commits) | **structural** for `-sm tensor`; verify before assuming |
| 8 | graph/lane whole-token graph | HIP graphs are worth 7% single-stream |

At n=4 fresh confirmation on both axes, ~18 min per candidate: **~2.4 GPU hours to a measured
`gfx906-both`**, plus BLOCKER 3 (variance, ~2 h, which sets n) and the zero point (~1 h).
**~5 GPU hours to a defensible seeded base, not 40.**

#### Round-based promotion, freezing between rounds

A growing base breaks cross-patch comparability (§1.2). So promote in **rounds**:

1. Establish `gfx906-required` from **structural necessity** (a patch enters only because a candidate
   provably cannot build or run without it).
2. **Round 1:** screen candidates against `R`. Confirm the shortlist on **fresh** data. Promote the
   confirmed `both` winners into `gfx906-both`. **Freeze.**
3. **Round 2:** measure S and M increments against the frozen `R+B`. A patch's effect may have changed
   — that is the point, and it is §4.1's *deployed-contribution* estimand, the one that decides adoption.
4. Iterate while any round still produces promotions.

Within a round the base is fixed, so Δ values are comparable; across rounds each verdict records its
base, so nothing is silently compared against a different tree.

#### Four corrections the handoff forces on Part 1

1. **§8.2 — discovery data are not confirmation data.** My adaptive escalation (screen n=2, escalate the
   ambiguous band to n=5) **pooled the screen runs into the confirmation**, which biases the estimate
   upward for exactly the patches selected *because* they looked good. Corrected: the screen is
   **triage only**; finalists are confirmed on **fresh** paired runs. §1.10 is amended accordingly.
2. **§4.2 — "required is not synonymous with neutral."** The bin name `neutral-required-substrate`
   implies a *measured* equivalence. It must not. A required patch can cost performance and still be
   mandatory, and when its removal makes the build unrunnable **no performance number exists to record**.
   Structural necessity is recorded **separately** from performance evidence.
3. **§4.3 — mutually exclusive implementations are a CATEGORICAL treatment**, not two binary terms:
   `reference | single-oriented | multi-oriented`. R2.7 says the single- and multi-user patch sets may be
   mutually exclusive, so those are levels of one factor. `A+B` must never be attempted for such a pair.
   Also: where `B requires A`, the state `A=0,B=1` **does not exist**; estimate A with B absent and the
   *incremental* effect of B given A.
4. **§8.3 — the error-control policy is an OPEN CHOICE, not settled.** I proposed Benjamini-Hochberg
   FDR at q=0.10; DR-B and DR-C propose Holm/FWER at 5%; DR-A proposes 5% FDR. **FDR and FWER are not
   interchangeable**, and the handoff is right that this must be presented rather than assumed.
   **RESOLVED 2026-09-20: the lead chose BH FDR at q = 0.10.** Rationale recorded in §1.10 — this is a
   discovery campaign whose winners are each re-confirmed by the §5 acceptance chain before promotion, so
   a controlled proportion of false discoveries is caught downstream, whereas Holm's conservatism would
   discard true positives the campaign exists to find. Applied to **confirmation** statistics only, over a
   family declared in advance (177 terms x 2 axes).

#### Where our approach is stronger than the handoff's, and why

The handoff's §5.1 notes that 177 coefficients need at least 178 independent treatment rows for full
identification, so its 64-128-row pooled screen **requires** sparsity assumptions, grouping, or priors,
and remains "vulnerable to interactions and aliasing".

**Per-patch A/B is fully identified** — every patch gets its own contrast, with no aliasing and no
sparsity assumption. The handoff asks for exactly this comparison: *"A direct cheap single-patch triage
baseline should not be dismissed without a like-for-like cost comparison against pooled screening on the
actual code."* Here it is, from **measured** costs on this box rather than a one-minute-per-execution
assumption:

| approach | cost | assumptions required |
|---|---:|---|
| **per-patch A/B, adaptive n** (ours) | **~18-40 GPU h** | none beyond run-level independence |
| DR-A / DR-B pooled screen | 37-42 nominal h | effect sparsity, legal design geometry, no strong aliasing |
| DR-C narrow screen | 10.7-15 h *excluding* validation | as above, plus reduced workload coverage |

So the assumption-heavy pooled screen buys little or nothing here. That is a consequence of the harness
being cheap (2.8 min proxy cell) — which was not knowable without tonight's measurements.

**Still missing, from the handoff's own §11.1 list:** measured noise (BLOCKER 3), the lead's hard budget
cap, and the FWER-versus-FDR choice above. Everything else it lists — patch IDs, dependency facts, build
and run durations, hardware isolation, practical margins — tonight's campaign supplies.

---

## Part 4 — Traps that have already bitten; do not re-learn these

### T1. Library shadowing — the one that produces a *perfect false pass*

`/etc/ld.so.conf.d/llama.cpp.conf` puts `/opt/llama.cpp/lib` in the global ldconfig cache, and
**no llama.cpp binary here has a RUNPATH**. So any build loads the *stock* ggml unless
`LD_LIBRARY_PATH` says otherwise. 107 of the 231 project harness scripts already do this — it
is the invocation contract. An A/B that forgets it loads the same library on both sides, agrees
perfectly, and green-lights a bad result.

Second layer, now that two ROCm trees exist: builds link `/opt/rocm/lib`, an
`update-alternatives` symlink. Flipping the alternatives silently moved every **existing**
build onto the 10.0 runtime. Pin both layers:

```
LD_LIBRARY_PATH=$PREFIX/lib:/opt/rocm/core-7.14/lib     # hold a build on 7.14
LD_LIBRARY_PATH=$PREFIX/lib:/opt/rocm/core-10.0/lib     # hold a build on 10.0
```

Assert before measuring:
```
LD_LIBRARY_PATH=... ldd $PREFIX/bin/llama-server | grep -E 'libggml-hip|libamdhip64'
```
Both must point where you intend. See memory `llamacpp-build-library-shadowing`.


### T2. Pre-2026-09-19 builds have no test binaries

They were configured `LLAMA_BUILD_TESTS:BOOL=OFF`, so `test-backend-ops` has **never** been run
against the 7.14 production build. Build one into a separate dir with `-DLLAMA_BUILD_TESTS=ON`
(~55 s) if a baseline is needed.


### T3. A bash trap handler must `exit`

Hit at 06:14 UTC. `phase4-service.sh`'s `cleanup()` restored the box and logged "STOPPED" but
did **not** exit — so bash cleared the traps and **resumed the script**, starting the next run
with the RAPL cap back at 413 W, the fan curve normal and no watchdog. That is precisely the
four-dies-plus-uncapped-host case that trips the 1228 W clamp. Now fixed: `cleanup()` ends in
`exit 1` and also kills the watchdog and SMC logger.

Related: bash defers a trap until the foreground child returns. SIGTERM to the script did
nothing while it blocked on the python client; the handler only ran once the client was killed.


### T4. `pkill -f` self-kill (exit 144)

An unanchored `pkill -f <pattern>` inside a tool command matches the tool's own `bash -c`
shell, killing it (exit 144). Anchor to `^/bin/bash /abs/path`, use `pkill -x`, or kill by PID.
Also: `pgrep -x` silently matches nothing for names over 15 chars — `test-backend-ops` is 16.


### T5. The power envelope is the hard physical limit

`PZ0G` is the DC total that clamps, envelope **1228 W**. `PZ0T` goes nonzero the sample *before*
the dies latch to 1000 MHz — it is the leading indicator, watch it, not the clock. Only a **cold
power cycle** (unplug 30 s) clears a clamp; a warm reboot re-clamps. Driver GPU reset FAILS and
leaves dies EBUSY — never retry it. Host CPU must be RAPL-capped to 150 W during any four-die
run (`gpu-test-env.sh` does this); never run a build beside a loaded four-die job.


### T5b. A benchmark harness can be wrong in ways nothing reports

Three independent harness defects in one day, none of which raised an error at the time: an inverted
workload shape (D6), no arrival model, and a crash-after-run that destroyed aggregates (D7). Before
trusting ANY service number, check: what is the prefill:decode ratio, when do clients arrive, and did
the client actually write its results? A harness that runs to completion is not a harness that
measured the right thing.


### T4b. `pgrep -f <pattern>` in a WAIT loop blocks forever — self-*block*, not self-kill

T4's third instance, 2026-09-20, and a distinct failure mode. A chain script waited with:

```bash
while pgrep -f 'test-backend-ops' >/dev/null 2>&1; do sleep 30; done
```

It never exited. The matches were not the benchmark — they were **the monitoring shells themselves**,
whose command text contained the literal string (one from the heredoc that wrote the script, one from a
reporting command that grepped its log). `pgrep -f` matches full command lines, so *any* shell that
mentions the pattern looks like a running job. The gate silently never started; no log, no error, and
the dependent chain waited on it in turn.

**Rule: never chain by pattern.** Chain by **sequence** (both steps in one script) or by **PID** (`wait
$pid`, or `kill -0 $pid`). If a pattern wait is unavoidable, anchor it to an absolute binary path
(`^/abs/path/binary`) and confirm no shell will ever quote that path — which is hard to guarantee, so
prefer sequence or PID.

Related and already recorded: T4 (unanchored `pkill -f` kills the tool's own shell, exit 144) and the
`pgrep -x` 15-character `comm` truncation.

### T10. A single `local a=$1 b=$W/x-$a` leaves `$a` EMPTY — and is fatal under `set -u`

Bash expands **every word** of a `local` (or `declare`/`export`) statement **before performing any
assignment**, so a later assignment that references an earlier one in the same statement sees the
variable **unset**:

```bash
local tag=$1 rep=$2 out=$W/f-$tag-$rep.md      # WRONG: $tag and $rep are empty in `out`
local tag=$1; local rep=$2; local out=$W/f-$tag-$rep.md   # right
```

Hit **twice in one night**, the second time in a script written an hour after fixing the first:

1. `fanpower.sh` — both samples silently wrote to the same file `fanpower-.log`, truncating it each
   time, and the run reported **0.0 W** instead of failing.
2. `variance-gate.sh` — under `set -u` the same shape aborted the gate **15 seconds** in with
   `line 33: tag: unbound variable`, after the surrounding chain had already logged "exit=0".

**Two rules.** Always use separate `local` statements when one value derives from another. And when a
gate finishes implausibly fast, **check its output before believing its exit code** — `run-gates.sh`
recorded `variance exit=0` for a script that had produced no measurements at all.

A multi-assignment `local` with **no** cross-reference is fine (`ladder.sh:82`
`local slots=$1 depth=$2 kv=q8_0` ran all night correctly). The defect is specifically self-reference.

### T9. A hunk resolved to one side, inside a file whose context came from the other side, is NOT a resolution

Hit three times in one hour on 2026-09-20 while rebasing onto v0.4.1, each time producing a tree that
merged cleanly and did not compile.

1. **`ggml-cuda/common.cuh`** — no textual conflict at all, because upstream and the fork define
   `fast_bf16_hardware_available` at *different places in the file*. Git merged both.
   `error: redefinition of 'fast_bf16_hardware_available'`.
2. **`ggml-cuda/mmq.cu`** — two hunks resolved to upstream inside a file whose surrounding function was
   the fork's. Upstream's hunks referenced `s12`, `s13`, `ne_get_rows`, which only the fork's context
   declares. `error: use of undeclared identifier`.
3. **`common/speculative.cpp`** — same pattern, worse: `process_decode` left **referenced twice and
   declared zero times**, plus 11 orphan references to `verify_h` and 9 to `pending_h`.

**The rule:** when two branches have diverged *architecturally* in a file, resolve at **file
granularity** — take one side whole — or merge the semantics deliberately and test. Choosing a side
per hunk silently mixes two incompatible designs, and `git` will not warn you because there is no
textual conflict to report.

**The corollary that matters more:** a clean merge is not evidence of correctness, and neither is
"all conflicts resolved". Only a build is, and only a test is evidence the build is right. The 11
resolved hunks of the substrate squash looked complete and produced a tree with three compile errors
in three separate files. **Build before claiming a rebase is done**, and treat "resolved N conflicts"
as a progress note, never a result.

Applies directly to the 12 conflicts still open between our patches and the substrate (Stage C4):
each overlaps the substrate in the same regions, so per-hunk side-picking will reproduce this defect.

### T6. Verify a suspiciously good or suspiciously bad result before reporting it

This session: an 80 s build looked too fast (it was genuine — 550 objects, ccache); a
"hung" `llama-cli` was actually sitting in an interactive prompt spinning on EOF because
`-no-cnv` did not take (it reproduced identically on 7.14, which proved it was the test, not
ROCm); and bit-identical perplexity looked like a shadowing bug until the per-chunk values and
the distinct binaries/runtimes confirmed it was real.

---


### T7. `kill ${VAR:-0}` signals the ENTIRE PROCESS GROUP

`kill 0` means "every process in my process group", so a guard written as `kill ${SAMP:-0}` kills the
script itself when the variable is unset. Present in `phase4b-q8kv.sh` and in the first draft of
`ladder.sh`. Use a guarded helper:

```bash
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
```


### T8. The SMC logger accumulates, and corrupted reads silently disable the clamp guard

`ladder.sh` and `serve.sh` kill the clock sampler on normal exit but **not** `smc-log.sh` — only the
INT/TERM trap does. So every run leaves its logger polling, and concurrent `smc-read.py` access
corrupts the applesmc interface. It fails by returning **garbage, not an error**:

- `JSONDecodeError` tracebacks written into the SMC log instead of readings. On 2026-09-19/20 three
  runs ended with **zero** valid `PZ0G` samples and therefore a silently dead DC-envelope guard.
- Spurious values: `PZ0G=5100` appeared alongside `PZ0F=1044.8` and zone sums of ~907 W on the same
  line (real DC ~1045 W).

Two consequences, both still owed as fixes:

1. Kill the SMC logger on the normal exit path, or start **one per session** rather than per run.
   Serialise `smc-read.py` with `flock`.
2. **`clamp-watchdog-v2.sh` kills the chain on a SINGLE `PZ0G >= 1228` sample.** Given reads
   demonstrably corrupt, that is a false-positive path that can kill a healthy multi-hour run. It
   should require two consecutive samples, as its `>= 1200 W` rule already does, and it should report
   a log it cannot parse instead of skipping the DC test in silence.

Nothing was at risk on 2026-09-19/20 (peak DC 1144 W of 1228; the 200 W runs had a valid guard
throughout; the three degraded runs were all inherently low-power) — but the guard degraded without
anything reporting it, which is T5b in a new place.

---

## Part 5 — The action plan

Rules that govern every stage (from memory `rca-requirement-traceability`):

1. Write the objective as ONE metric line with acceptance criteria, and have the lead confirm it.
2. Objective cells first; every other cell needs a one-sentence justification or is not run.
3. **Before any multi-hour GPU run, send the lead the plan:** which requirement each stage serves, GPU
   hours, electricity. Non-negotiable.
4. A lead correction re-derives the plan from the requirement; it is never a parameter edit.

### Stage A — establish the multi-user baseline — **DONE 2026-09-19/20**

Ran as a 10-hour autonomous campaign. 16 capacity cells, chat-shape validation at three configs with
and without a sized prompt cache, the §5.2 memory itemisation, the context-compression answer, the
125 W power pass and the production-fan validation. Outcome: the design point of D8, the decode model,
and D9/D10/D11. Full report `/root/night-20260919/REPORT.md`; raw chronological log with every decision
and defect `NIGHT-LOG.md`. Superseded scripts `phase4-service.sh` and `phase4b-q8kv.sh` were discarded
rather than run — see D14 and the report's §9 for why.

Still owed from Stage A: the fan-power delta (first attempt failed, T8), and the clean re-run of the
context-compression rates with a sized cache (`ctxcompress2.runlist`, staged). Neither gates Stage D.

### Stage C — rebase everything onto v0.4.1 — **SUBSTANTIALLY DONE, 4 items open**

Target is **v0.4.1** (`b29c606e2`, 2026-09-14), the last official release, per the lead — a tagged
release rather than a moving master, so the base is reproducible.

Done:
- **C1.** Target chosen and fetched. Upstream master had moved to `e613ef2c8`; v0.4.1 is 92 commits
  behind it and 204 ahead of the fork's base `b10760`.
- **C2.** Substrate rebased by **squash-merge**, not sequential replay. `git merge --squash mxxm/master`
  onto v0.4.1 gave **11 conflict hunks in 9 files**, against 30 hunks in the fork's *first commit alone*
  under replay — sequential replay fights every intermediate state against upstream's churn. This is the
  single biggest efficiency finding of the rebase. Record: `docs/REBASE-v041-TRIAGE.md`.
- **C3 (partial).** Our 29 applied on top: **17 clean, 12 conflicting**. On bare v0.4.1 only 8 were
  clean, which confirms the substrate dependency — the three `q8_repack` patches went from *inapplicable*
  to clean.
- Single-user lineage mapped: 10 commits, all with multi-user twins, **9 of them the same patches that
  conflict**. This is the measured justification for the common base (§1.6).
- One patch binned from the rebase alone: the fork's BF16->F32 fallback is `upstream-already-has-it`
  (PR 28846 merged a strictly better per-vendor version).

Open:
- **C4. Resolve the 12 conflicts.** 9 are shared between profiles, so resolving once serves both. All 12
  are also `conflicts-with-another-patch` candidates whose interaction must be *measured*, not decided —
  our patches and the substrate independently modify `mmvq.cu`, the `ggml-cuda.cu` fusion path,
  `gated_delta_net.cu` and `speculative.cpp`.
- **C5. Build, then run the Phase-3 gate** (perplexity 16K/6 + `test-backend-ops`) on every branch.
  Nothing has been compiled yet.
- **C6. Measure the stock zero point at the pinned recipe** (BLOCKER 1). Every baseline number so far is
  on the *patched* build; D2 requires the zero point to be stock.
- **C7. Push** all branches to `exabit-io/llama.cpp` — only after C5 passes. Pushing branches that have
  never compiled to a public repo is not acceptable.
- Two subsystems were excluded from the base as architectural divergence and are now candidates in their
  own right (bin `technique-requires-implementation`): the fork's chunked-column MMQ versus upstream's
  NVFP4 `mmq_args`, and the fork's `process_decode` MTP mirroring versus upstream's virtual
  `process(const llama_batch &)`.

### Stage D — the survey: round-based promotion into `gfx906-both`

Every step is the method of Part 1 applied. **Do not start until Part 6's gates clear.**

Structure follows D17: the required stack comparisons are `R vs R+B`, then `R+B vs R+B+S` and
`R+B vs R+B+M`, so **`R+B` must exist before the single- and multi-user increments can be measured at
all.** Populating `gfx906-both` early is a precondition, not an optimisation. Within a round the base is
frozen so Delta values are comparable; across rounds every verdict records its base.

#### D-0. Establish `gfx906-required` on STRUCTURAL grounds

A term enters only because a candidate provably cannot build or run without it — never because it
measured neutral. Known today: `ggml-cuda/q8_repack/` (three of our patches cannot apply without it).
The meta/TP backend is a **hypothesis** until a dependent is shown to need it. Each entry records
`structural: required-by:<ids>`, and its performance field may legitimately read
`not-measurable (removal breaks the build)`.

#### D-1. Seed `gfx906-both` by MEASURING the highest-prior candidates first (~2.4 GPU h)

Priority-ordered, because §6's "must remain active" list was measured at 200 W, on the old base, and
partly through the shape-invalid harness — it is a strong *prior*, not evidence:

| # | candidate | prior |
|---:|---|---|
| 1 | Q8_0 MMVQ fast path + 16-column width | R6; +62% decode at 12 slots claimed |
| 2 | fork tile table / head-256 FA rows | R6; **D15 confirms head_dim = 256 is this model's row** |
| 3 | custom allreduce, four-row gate | R6; +14.5% single-stream; `gfx906.env` depends on it |
| 4 | DPP warp reductions on GCN | applied clean; **D15 ranks DPP last — a deliberate falsification test** |
| 5 | MoE K-quant/IQ4_NL fused mat-vec | the substrate claims +6 to +8% decode |
| 6 | `q8_repack` paths | structural (goes to D-0, not here) |
| 7 | meta/TP backend | structural for `-sm tensor`; verify before assuming |
| 8 | graph/lane whole-token graph | HIP graphs are worth 7% single-stream |

Each gets **fresh n>=4 paired confirmation on both axes**. Confirmed `both` winners are promoted and
`gfx906-both` is **frozen** before round 2.

#### D-2. Enumerate the remaining terms

177 total (148 substrate per D14 + 29 ours), each a diff against the common base. Sources per D13,
including the non-GitHub forges and the open Gitee discovery gap. Record dependencies and mutual
exclusions **before** measuring: where `B requires A` the state `A=0,B=1` does not exist, and mutually
exclusive single/multi implementations are **one categorical factor** (`reference | single-oriented |
multi-oriented`), never two binary terms, and never combined as `A+B`.

#### D-3. Validate the proxy cell once

The 4 x 32K A/B cell costs 2.8 min against 5.6 min at the 4 x 64K design point. Measure 2-3 patches at
both depths and check **rank** correlation and sign stability, not correlation alone. If rankings diverge,
fall back to 4 x 64K and double the budget. Per the handoff's proxy-false-negative warning, include a
long-context condition where that mechanism matters.

#### D-4. Screen the remainder — triage only (~12 GPU h)

n=2 per term per axis against the frozen base, blocked and interleaved. **No verdict, bin or p-value comes
from screen data** (§1.10). Screened-only terms are `not-prioritised` or `unresolved`.

#### D-5. Freeze the shortlist, then confirm on FRESH data (~7 GPU h)

Freeze candidate identities, axes, metrics, margins and the hypothesis family. Collect **new** randomised
paired runs at n>=4. Only these enter the exact permutation test and **BH FDR at q = 0.10** (lead-decided
2026-09-20) across the declared family of 177 x 2.

#### D-6. Bin by the dual criterion

`improves` / `regresses` requires **BH q < 0.10 AND |effect| >= 2%** on the named axis. Everything else is
`neutral-drop`, or carries `structural: required-by:...` independently of its performance verdict.

#### D-7. Round 2 and beyond — the increments that need `R+B`

With `gfx906-both` frozen, measure `R+B vs R+B+S` and `R+B vs R+B+M` for the single- and multi-user
candidates. This is §4.1's **deployed-contribution** estimand — the one that decides adoption, and the
reason D-1 had to come first. A term's effect may differ from its round-1 value; that is the finding, not
an inconsistency. Iterate while any round still produces promotions.

#### D-8. Interaction and false-negative safeguards

Per the handoff's §9, retained deliberately because a cheap design gives up interaction coverage:
- **weak-marginal synergy:** for a compatible pair use `neither / A / B / A+B`; the contrast is
  `y_AB - y_A - y_B + y_0`. Never for a mutually exclusive pair.
- **cancellation:** a neutral *pool* does not make every member neutral.
- **correlated attribution:** identical columns yield a bundle result only; no amount of replication
  invents a missing contrast.
- **final-stack redundancy:** re-ablate important terms in the assembled recipe, and never delete several
  individually-redundant alternatives at once — one may be redundant only because another is present.
- **audit:** re-test a small random sample of discarded candidates.

#### D-9. Write the verdict record for every term

Mandatory, enforced by `survey/survey-lint.py`. A term without a passing record is **not surveyed**.
Operational outcomes — build failure, wrong results, OOM, crash, timeout — are recorded as qualification
outcomes, **never dropped as missing rows**.

**Cost, revised 2026-09-20:** D-1 seeding ~2.4 h + D-4 screen ~12 h + D-5 confirmation ~7 h = **~21 GPU
hours**, plus Part 6's gates (~4 h) and Stage C's host time. Cheaper *and* statistically sounder than the
earlier pooled-escalation scheme, and cheaper than the external handoff's 37-42 h pooled screen while
needing none of its sparsity or aliasing assumptions.

### Stage E — compose the builds and promote

- **E-1.** Build `gfx906-both`, then `gfx906-single` and `gfx906-multi`, from the bins (Part 2). Each
  branch's own commits are its bin, so composition is mechanical.
- **E-2. Measure the composed builds.** Per §1.8, `Σ Δ(Pᵢ) != Δ(Σ Pᵢ)` when terms share a bottleneck —
  and D15 says decode is memory-traffic-bound, so traffic-removing terms will **not** sum. Expect the
  composition to under-perform the sum of its parts; that is physics, not error.
- **E-3.** Full §5 acceptance chain on each composed build against the current production build,
  interleaved, at the design point: service run (chat shape, arrival model, sized cache), capacity
  ladder, quality (perplexity + `test-backend-ops`), power at 125 W.
- **E-4.** Promote only if R3.1 and R3.2 hold at the design point and R3.3/R3.4 do not regress beyond the
  noise band. Nothing else promotes a build.


### Stage B — torch 2.13 from source on ROCm 10.0 — **DONE 2026-09-19**

Built, packaged, published and dogfooded. Took ~24 min, not the 3-6 h estimated.

- Wheels: `torch 2.13.0+gfx906.20260917140126`, `torchvision 0.27.0`, `torchaudio 2.11.0`,
  built from stock upstream v2.13.0 (no patches; gfx906 via `PYTORCH_ROCM_ARCH`) as a bare-metal
  port of upstream's docker-only `build-whl.Dockerfile`. Script `/root/build-torch-rocm10.sh`.
- Flags verified in the committed cmake config, not just intended: `BLAS=mkl`,
  `-march=native` -> cascadelake + AVX512-VNNI, `USE_MKLDNN=ON`/OMP, `USE_ROCM=ON`.
  MKL was added deliberately — the box had only reference netlib BLAS, the worst option on a Xeon.
- Debian package `python3-torch-gfx906 2.13.0+gfx906.20260917140126-1`, installs to
  `/usr/lib/python3/dist-packages` (no venv). Script `/root/build-pytorch-deb.sh`.
- **Signed twice**: the deb itself carries `_gpgorigin` (`debsigs --sign=origin`, Exabit key
  6043AD7B...0F4BE230) AND the flat apt repo metadata is clearsigned with the same key.
- Published to **exabit-io/resize-amdgpu-bars** (the barfix-kernel repo — all custom packages go
  there), tag `pytorch-2.13.0-gfx906-rocm10.0`, as a signed flat apt repo in the release assets.
- **Dogfooded**: added the public source, `apt update`, `apt install python3-torch-gfx906`,
  and `/usr/bin/python3` (no venv) reports torch 2.13.0+gfx906, hip 7.15.26333, 4 devices,
  GPU matmul OK, MKL True, MKLDNN True.
- Consequence: **the PyTorch dependency on ROCm 7.14 is gone.** That was one of the two blockers
  on purging 7.14 (D5); only Phase 4's reference arm remains.

MISPLACED: an earlier wheels-only release went to `llama.cpp-gfx906-tuning` by mistake before the
lead corrected the destination. Superseded by the deb; pending the lead's OK to delete.


---

## Part 6 — Gates that must clear before Stage D

Four items. Nothing else blocks the survey.

| # | gate | why it blocks | cost |
|---|---|---|---|
| **1** | **No stock zero point at the pinned recipe** | D2 defines the zero point as stock upstream; every baseline figure so far is on the *patched* build, so there is nothing to A/B against. Also note tonight's baseline is **pre-rebase** — the non-regression baseline must be re-measured on the rebased tree. | ~1 h GPU (C6) |
| **2** | **Single-user axis has no 10.0 baseline and no defined instrument** | D1 requires a per-axis A/B, but "single-stream decode at the 32K floor" is not yet a runnable recipe — no binary, prompt length, depth or repeat count. Until it exists, Stage D can only run multi-user, which is the single-axis mistake that lost the >300 tok/s config (D4). | ~1 h GPU + spec work |
| **3** | **Harness run-to-run variance unmeasured** | §1.10: sigma **sets n, and therefore the whole budget** — n=2 at sigma 0.5%, n=16 at sigma 2%. Also §5 promotes on a 2% band that has never been shown to exceed noise. | ~2 h GPU |
| **4** | **Upstream PR 24549 may be a correctness bug in our exact configuration** | Stale graph reuse when contexts share memory under `SPLIT_MODE_TENSOR`, found with *Gemma MTP + tensor split*. We run `-sm tensor` with HIP graphs ON, and R3.9 requires enabling MTP — the precise trigger. Check before the MTP A/B, or a crash will be misread as "MTP does not work on gfx906". | ~1 h |

Gate 3 must run **first**, because it determines the cost of everything after it.

**Settled since this table was written:** the error-control policy is **BH FDR at q = 0.10**, and the
**budget cap is ~25 GPU hours** (21 survey + 4 gates) — both lead-approved 2026-09-20. Current best estimate for the whole survey is **~21 GPU hours**
(D-1 seeding 2.4 + screen 12 + fresh confirmation 7) plus ~4 h of gates — against the ~51 h implied by the
lead's earlier objection, and against the external proposals' 37-42 h. Rule 3 requires this to go to the
lead before Stage D starts.

**Progress on the gates as of 2026-09-20 01:23:**
- Gate 4 (PR 24549) — not started.
- Gates 1 and 2 — not started; both need the rebase to be pushed first.
- Gate 3 (variance) — not started. **This is the next thing to run.**
- Unlisted but now done: the rebased substrate **builds** and passes `test-backend-ops` on all four dies
  (16198/16198 each, zero failures), which was a precondition nobody had written down.


## Part 7 — Cold-resume checklist

```bash
uname -r                                     # expect 7.0.0-31-generic (+barfix1)
readlink -f /opt/rocm/lib                    # which ROCm is live
ls -d /opt/rocm/core-*                       # which trees exist
systemctl is-active t2fanrd                  # MUST be active; dies silently on a new kernel
cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw   # 413000000 = normal, 150000000 = test cap left on
grep -om1 'always_full_speed=[a-z]*' /etc/t2fand.conf              # false = normal curve
rocm-smi --showperflevel                     # auto = normal
/opt/rocm/bin/amd-smi monitor | head -5      # 4 dies, idle, 0 VRAM
resize-amdgpu-bars check                     # 4/4 dies at 32 GiB
cat /root/rocm10-migration-20260919/phase4.log                     # where phase 4 stopped
```

If a previous run left the box in test mode (RAPL at 150 W, fans maxed, perf level high), source
`/root/llama.cpp-benchmarking/tools/gpu-test-env.sh` and call `restore`.

### File map

| What | Where |
|---|---|
| Requirements (outranks all) | `/root/llama.cpp-benchmarking/REQUIREMENTS.md` (+ `.bak-20260919`) |
| Migration working dir | `/root/rocm10-migration-20260919/` |
| Phase 4 script (bug fixed) | `…/phase4-service.sh` |
| Phase 3 gate script | `…/phase3-gate.sh` |
| Build script | `…/build-llamacpp-rocm10.sh` |
| TOPK_MOE writeup | `…/FINDING-topk-moe.md` |
| Rollback pin | `…/pin-before.conf` |
| Corpus harness (methodology to mirror) | `/root/rocm-tests/bench/service-A.sh` |
| Service client (CURRENT, R2.4 shape) | `/root/rocm-tests/bench/chat-client.py` |
| Service client (OLD, shape-invalid) | `/root/rocm-tests/bench/service-client.py` (+ `.orig-20260919`) |
| GPU test env (caps, sampler, restore) | `/root/llama.cpp-benchmarking/tools/gpu-test-env.sh` |
| Clamp watchdog | `/root/rocm-tests/bench/clamp-watchdog-v2.sh` |
| Tuned env (custom AR, RCCL topo) | `/root/llama.cpp-benchmarking/settings/gfx906.env` |
| ROCm pin | `/etc/apt/preferences.d/rocm-gfx906` |
| Upstream project clone | `/root/ML-gfx906` |
| Fork/PR source list | `/root/llama.cpp-benchmarking/gfx906-llamacpp-forks-and-tuning-urls.md` |

### Relevant memory notes

`service-is-batched-max-context` (amended for R2.7) · `rca-requirement-traceability` (rule 5
amended) · `gfx906-patch-survey-by-profile` (this plan) · `gfx906-fork-survey-2026-09-09`
(superseded) · `llamacpp-build-library-shadowing` · `macpro-rocm-gfx906-install` ·
`macpro-gpu-clock-clamp` · `macpro-smc-power-zones` · `context-size-floor-32k` ·
`user-is-project-management` · `bash-tool-pkill-self-kill`

---


---

---

## Part 8 — Open questions for the lead

Resolved items are kept with their resolution so a future reader does not reopen them.

### Open

1. **R3.1: median, p10, or min?** It selects the configuration and is worth up to 1.7x aggregate
   throughput. **Less urgent than it was:** the recommended design point 4 x 64K at 125 W passes on **all
   three** readings (18.24 / 17.04 / 17.02), so this only binds if the design point moves. Claude gates on
   p10 and reports all three everywhere.
2. **R2.2 depth-per-user versus R3.3 aggregate throughput.** Opposite directions along the frontier, and
   R3.2's objective is flat along it, so the specification cannot break the tie. A product decision.
3. **Context-compression policy (R2.4's TBC).** Mandatory, not optional: with the server default every
   session dies at the slot boundary with HTTP 400, and `--context-shift` provably does not prevent it.
   Must be solved where the conversation is assembled — four options in the report's §4.
4. **`--cache-ram` of 30-48 GiB per node** on hyperconverged nodes (KVM/DB/Ceph alongside). A fleet budget
   item that did not exist before D9; it buys 1.6x decode and 24x TTFT.
5. ~~**The survey's hard budget cap.**~~ **CLOSED 2026-09-20: the lead approved ~21 GPU hours for the
   survey plus ~4 h of gates (~25 h total).** Against the ~51 h implied by the lead's earlier objection and
   the external proposals' 37-42 h. The phases are capped separately per rule 3: D-1 seeding 2.4 h,
   D-4 screen 12 h, D-5 fresh confirmation 7 h, Part 6 gates 4 h. **If a phase overruns its cap, stop and
   report rather than silently spending the next phase's budget.**
6. **Single-user design point.** The multi-user one is settled (D8). The single-user axis needs its own:
   context depth at or above the 32K floor, and whether MTP is on — itself blocked on reimplementing the
   MTP subsystem against upstream's interface (Stage C).
7. **Power-cap revisit.** Deferred by the lead. When reopened: the 125 W policy budgets for a 413 W host
   that the W-3275M's 205 W TDP and 303 W measured all-core draw say cannot occur; ~305 W would permit
   ~157 W dies. Fan power is now measured at **21.2 W idle** (an upper bound on the delta).

### Resolved

| question | resolution |
|---|---|
| bin taxonomy | **D12/D16.** Nine bins plus a separate `structural:` field; enforced by `survey/survey-lint.py` |
| error-control policy | **BH FDR at q = 0.10** (lead, 2026-09-20). §1.10 and D17 |
| repo topology | **D16.** One repo, branch-per-bin, common v0.4.1 base |
| rebase target | **v0.4.1**, the last official release (lead). D3, Stage C |
| single-user tree (old C3) | **D16.** A `gfx906-single` branch, not a separate tree — its 10 patches are the same logical patches as 10 of our 29 |
| Stage ordering (Stage B last) | Moot — Stage B completed 2026-09-19 |
| mxxm's status | **D14.** Substrate, not third-party history; all 148 commits get binned |
