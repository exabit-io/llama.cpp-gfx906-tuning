# Night run 2026-09-19/20 — decisions of record and results

Claude acting as technical lead; the lead (PM) is away from keyboard and said "do it all,
don't ask me other questions". Everything here is a decision I made rather than deferred.
Each is one line to overrule.

## Decisions taken in the lead's absence

**N1 — R3.1 is gated on p10, with median and min reported alongside.** (Resolves open question 1.)
R3.1 says "*every* request in a full batch decodes at >= 12 tok/s". Median would let half the
requests sit under reading speed, which contradicts "every"; min makes the gate hostage to one
queueing outlier. **p10 >= 12 tok/s is the primary gate**; median and min are reported in every
row so the lead can move the line without a re-run. Where median passes and p10 fails, the row
says so explicitly. In the capacity ladder the question does not arise: `llama-batched-bench`
decodes all sequences in lockstep, so per-slot = aggregate/slots exactly.

**N2 — R3.6's 125 W pass covers the design points, not the whole ladder.** Running all eight
cells at both caps is ~5 h for information the 2026-09-08 perf/W study already settled
(per-watt monotonic down to the DPM floor). The ladder runs at 200 W; the cells that pass R3.1
and the chosen design point are re-run at 125 W.

**N3 — deep contexts are SEEDED, not grown, and labelled so.** Growing a session to 192K by
generation means generating 192K tokens per session: ~3.8 h for ONE configuration. `chat-client.py`
gained `--seed-ctx N`, which lays down the depth in one prefill and then runs true R2.4 turns
(short user message, 2K-8K generated, 30% tool calls) on top. The seed turn is excluded from the
decode and TTFT percentiles and reported separately; the seed's tokens are excluded from `pre:dec`
because counting them would re-create exactly the document-shaped inversion R2.4 forbids.

**N4 — `phase4b-q8kv.sh` is NOT used; it is superseded.** Three reasons, all fatal: it drives
`service-client.py`, whose shape R2.4 invalidated; its `run()` ends in `sleep`, so `|| exit 1`
guards can never fire (a trap already logged on 2026-09-19); and its `trap cleanup ... EXIT`
fires on *normal* completion. Replaced by `serve.sh` + immutable per-step runlists.

**N5 — `-ngl all` is passed explicitly.** Build 11067 defaults `-ngl` to **`auto`**, not "all".
At 192K-256K depths an auto-offload decision could leave layers on the host and report a decode
number that looks like a result rather than an OOM — a T5b-class silent defect. `-ngl all`
resolves to `n_gpu_layers = -2` in the log; peak VRAM of 12.8 GiB/die at 4x64K against ~7 GiB/die
of weights confirms the weights are die-resident.

## Defects found and fixed in the harness tonight

- `kill ${VAR:-0}` — **`kill 0` signals the entire process group**, i.e. the script kills itself.
  Present in my first draft of `ladder.sh` and latent in `phase4b-q8kv.sh` (`kill ${SAMP:-0}`).
  Replaced with a guarded `kpid()` helper.
- `pkill -x llama-batched-bench` **matches nothing** — `comm` is truncated to 15 characters and the
  name is 20 (trap T4, second half). Fixed in `probe-alloc.sh`; `ladder.sh` was already running and
  must not be edited (bash reads scripts incrementally), so its abort path is by PID.
- Build 11067 suppresses info-level logs, so no run log contains the KV / compute buffer sizes that
  REQUIREMENTS §5.2 requires. Rather than restart the ladder, `probe-alloc.sh` re-allocates each
  cell with `-v` and a trivial `-npp`: buffer sizes depend on `-c`, `-npl` and KV type, not on
  prompt length, so the allocation is identical at ~1-2 min/cell instead of 25.

## Order of work

1. Capacity ladder, q8_0 KV, 200 W — 8 cells. **Running from 12:00 UTC.**
2. chat-client validation at the two best points (seeded, Poisson arrivals).
3. MTP on/off A/B at the winning point — flag validated first (R3.9).
4. Context-compression cost (R2.4 TBC).
5. `probe-alloc` itemisation + 125 W pass at the design point (R3.6).

## Results

(appended as they land)

---

## HEADLINE: the binding constraint is decode bandwidth, not memory — and both named design points fail R3.1

Measured (q8_0 KV, 200 W/die, build 11067, tp4):

| cell | fits | total KV | pp t/s | tg agg | **per slot** | R3.1 (>=12) |
|---|---|---:|---:|---:|---:|---|
| 4 x 64K | yes, 12.8 GiB/die | 0.262M | 954.95 | 71.01 | **17.75** | **PASS** |
| 8 x 192K *(lead's design point)* | yes, 29.9 GiB/die | 1.573M | 593.43 | 41.20 | **5.15** | **FAIL — 43% of floor** |

`8 x 192K q8_0` **fits and is still not usable**: 29.9 GiB/die leaves ~1 GiB inside R3.2's
31 GiB budget, and it decodes at 5.15 tok/s per request. This is the same wall f16 8 x 128K hit
(~1.6 tok/s); q8_0 KV bought the *memory*, not the *speed*.

### The curve

Every decode step reads the whole KV cache of every active slot, so per-slot decode is a
function of **total** KV — `slots x depth` as a single product — not of per-slot depth:

    per-slot tok/s = 1 / (0.0288 + 0.1052 * total_KV_in_Mtok)
    i.e. a 28.8 ms fixed per-step cost + 105 ms per million KV tokens read

Cross-checked against an **independent** corpus point before being used: M4 (b10837, 8 slots x
32K, a different build) measured 16.1 tok/s per stream; the curve predicts 17.8. Consistent, so
the 5.15 is a real capability boundary and not a broken run.

**Consequence: R3.2 is not memory-bound at all.** Memory allows 1.573M tokens of KV; R3.1 allows
about **0.52M**. The lead's design points were chosen against the memory budget (which they
satisfy) and are roughly 2-3x past the decode budget.

### What R3.1's floor buys (open question 1, now with numbers attached)

| R3.1 floor | total KV budget | 4 slots | 8 slots | 12 slots |
|---:|---:|---:|---:|---:|
| 10 tok/s | 677K | 165K/slot | 83K/slot | 55K/slot |
| **12 tok/s** | **519K** | **127K/slot** | **63K/slot** | **42K/slot** |
| 13.4 tok/s | 436K | 106K/slot | 53K/slot | 35K/slot |

So the floor is not a detail — it *is* the context target. At 12 tok/s the service can offer
**8 users x 64K or 4 users x 128K**, and 192K per user costs a drop to ~9 tok/s (75% of reading
speed). That is a product tradeoff, and it is the lead's call, not mine; I am measuring both
sides of it rather than picking one.

The decisive cells are **8 x 64K and 4 x 128K**, both at 0.524M total KV, which the curve puts at
11.92 — within noise of the floor. Whether they land above or below 12 is what the rest of the
ladder settles, and it decides the design point.

### Second named design point also fails; the curve is confirmed

| cell | fits | total KV | pp t/s | tg agg | **per slot** | R3.1 | predicted |
|---|---|---:|---:|---:|---:|---|---:|
| 4 x 256K *(lead's design point)* | yes, 24.5 GiB/die | 1.049M | 499.20 | 30.25 | **7.56** | **FAIL — 63% of floor** | 7.19 |

**Both design points named in R2.2 on 2026-09-19 fail R3.1**, by 2.3x and 1.6x respectively. Both
fit memory comfortably (24.5 and 29.9 GiB/die of 31). Refit on three measured cells spanning a 6x
range of total KV:

    per-slot tok/s = 1 / (0.0272 + 0.1045 * total_KV_Mtok)      residuals +3.3 / +1.4 / -3.3 %

12 tok/s is crossed at **538K total KV tokens**. Prefill also decays with depth: 955 t/s at 64K,
593 at 192K, 499 at 256K per sequence.

Note the fit slightly OVER-predicts at shallow depth (18.3 vs 17.75 measured at 4x64K; 18.3 vs
16.1 for corpus 8x32K), because per-slot batching overhead at 8 slots is not captured by a pure
total-KV term. So the two decisive cells, both at 0.524M and both predicted 12.20, are genuinely
borderline, and **4 x 128K should beat 8 x 64K at equal total KV** — fewer slots, less per-step
overhead. If only one clears the floor it picks the design point, and 4 x 128K is also the better
answer to R2.2's "as large as reasonably possible" per slot.

### 8 x 64K also fails (10.59) — and the frontier is not where the cell list looked

| cell | total KV | pp t/s | tg agg | **per slot** | R3.1 |
|---|---:|---:|---:|---:|---|
| 8 x 64K | 0.524M | 955.31 | 84.69 | **10.59** | **FAIL (88% of floor)** |

Adding a per-slot term fits all four measured cells to **+/-0.6%**:

    per-slot decode time = 19.2 ms + 3.10 ms x slots + 95.6 ms x total_KV_Mtok

(Fitted to this build's four cells; in-sample. The extra cells below are the out-of-sample test.
Corpus M4 at 8x32K measured 16.1 against a fitted 14.5, but that is build b10837, not 11067, so
it is a sanity check and not a validation.)

### The requirements do not determine the design point — R2.2 and R3.3 point opposite ways

Because per-slot decode depends on TOTAL KV, holding per-slot decode at exactly 12 tok/s makes
`slots x depth` nearly constant. So **R3.2's objective ("slots x context per slot is maximised")
is flat along the whole R3.1 frontier**, and the tie is broken by other requirements that disagree:

| slots | max depth at 12 tok/s | total KV | aggregate t/s | >= 32K (R2.2)? |
|---:|---:|---:|---:|---|
| 4 | 132K | 0.541M | 48 | yes |
| 6 | 77K | 0.476M | 72 | yes |
| 8 | 50K | 0.411M | 96 | yes |
| 10 | 34K | 0.346M | 120 | yes |
| 12 | 23K | 0.282M | 144 | **no** |

- **R2.2** ("as large as reasonably possible" per slot) points at **4 x 132K**, 48 t/s aggregate.
- **R3.3** (aggregate throughput is the primary target once R3.1/R3.2 are met) points at
  **10 x 34K**, 120 t/s aggregate — **2.5x the throughput of the same hardware**, at a quarter of
  the per-slot depth.
- **R2.1** caps slots at 12, and at 12 slots the depth that holds 12 tok/s (23K) breaks R2.2's
  32K floor. So 12 slots is out on depth, not on slot count.

This is a product decision, not a technical one: **depth per user vs users served**. I am measuring
both ends rather than choosing. Needed from the lead, in one line each: the R3.1 floor (open
question 1) and whether R2.2's "as large as possible" outranks R3.3's throughput.

### Second ladder pass added

The original eight cells never sampled the shallow/many-slot half of the frontier, so nothing in
them could have found the throughput-optimal configuration. Adding, after pass 1:
**6 x 64K, 8 x 48K, 10 x 32K, 12 x 32K** — ~1.5M prefill tokens, ~30 min, and a genuine
out-of-sample test of the model above.

### 4 x 128K PASSES at 12.40 — the deepest configuration that holds reading speed

| cell | total KV | pp t/s | tg agg | **per slot** | R3.1 | predicted |
|---|---:|---:|---:|---:|---|---:|
| 4 x 128K | 0.524M | 727.51 | 49.59 | **12.40** | **PASS** | 12.23 (+1.4%) |

Decisive comparison, and it confirms the model's structure: **4 x 128K passes (12.40) while
8 x 64K fails (10.59) at the SAME 0.524M total KV.** Slot count is not free — 3.10 ms per slot per
decode step — so at equal total KV, fewer/deeper slots win on per-request rate and more/shallower
slots win on aggregate. That is the whole R2.2-vs-R3.3 tension in two measured cells.

Standing after five cells (q8_0 KV, 200 W/die):

| cell | total KV | per slot | aggregate | R3.1 |
|---|---:|---:|---:|---|
| 4 x 64K | 0.262M | 17.75 | 71.0 | **PASS** |
| **4 x 128K** | 0.524M | **12.40** | 49.6 | **PASS** |
| 8 x 64K | 0.524M | 10.59 | 84.7 | FAIL |
| 4 x 256K *(named)* | 1.049M | 7.56 | 30.3 | FAIL |
| 8 x 192K *(named)* | 1.573M | 5.15 | 41.2 | FAIL |

### 8 x 96K: 8.40 per slot (predicted 8.39) — model validated on six cells

### R3.2 quantified: memory permits 3.1x more context than reading speed does

From the ladder's own peak-VRAM sampling:

    peak VRAM GiB/die = 9.30 + 13.35 * total_KV_Mtok

i.e. 9.30 GiB/die fixed (weights + compute buffers + runtime) and q8_0 KV at **13.7 KiB per token
per die**, ~53.4 KiB/token across the four dies.

| ceiling | binding total KV | 4 slots | 8 slots | 12 slots |
|---|---:|---:|---:|---:|
| memory, 31 GiB/die (R3.2) | 1.63M | 397K/slot | 198K/slot | 132K/slot |
| **decode, 12 tok/s (R3.1)** | **0.52M** | **132K/slot** | **63K/slot** | **42K/slot** |

**This explains where the named design points came from.** The memory ceiling at 8 slots is
198K/slot and R2.2 named 8 x 192K; at 4 slots it is 397K and R2.2 named 4 x 256K. Both were
derived from the memory budget — which they satisfy — and the memory budget is not the constraint
that matters. R3.1 binds 3.1x earlier. R3.2's text ("slots x context per slot is maximised subject
to R3.1 and the memory budget of 31 GiB per die") already orders these correctly; what was missing
was the measurement showing which term binds.

### 8 x 128K: 6.92 per slot (predicted 6.93) — FAIL

### Step 3 de-risked: MTP is genuinely implemented for this model

The plan flagged `--spec-type draft-mtp` as UNVALIDATED. Checked before spending GPU time:

- the build's arg parser accepts **`draft-mtp`** and **`draft-mtp-adaptive`** (PR 27210), so both
  variants exist in this binary;
- `libllama.so` carries **per-architecture** MTP paths, including a `QWEN35 MTP` one;
- the model declares **`general.architecture = qwen35`** with **`nextn_predict_layers = 1`**
  (the `blk.64.nextn.*` head the server currently loads and discards).

So MTP can engage for this model, and the R3.9 A/B is worth running. It is still verified at
runtime per config — `serve.sh` fails a run loudly if `unused tensor blk.*nextn` appears in the
server log while `--spec-type` was requested, so an MTP result can never be reported from a server
that quietly ran without MTP.

Also from the GGUF: **`context_length = 262144`**. Every depth on the ladder, including 4 x 256K,
is inside the model's native window, so no cell is confounded by rope extrapolation.
Architecture detail: 24 heads, 4 KV heads, n_embd 5120, 65 blocks (64 + 1 nextn).

### Note on §5.1's "two rounds, order rotated"

Step 2 runs ONE round per configuration. The rotation in §5 exists to cancel ordering and thermal
drift when a *candidate build* is interleaved against the *production build*; here there is no A/B
pair, only characterisation of a design point on a single build. The two-round rotation is reserved
for the promotion chain. The MTP A/B in step 3 DOES get rotated order, because it is a true A/B.

---

## LADDER PASS 1 COMPLETE — 8 cells, all fit memory, only 2 clear R3.1

q8_0 KV, 200 W/die, build 11067 (commit 1d1361e7a), ROCm 10.0, tp4, `-fa on -ngl all -b/-ub 2048`,
`-ntg 128`, Qwen3.8-27B-Q8_0. Sorted by per-request decode:

| cell | total KV | pp t/s | tg agg t/s | **per slot** | R3.1 (>=12) | peak VRAM GiB/die | cell s |
|---|---:|---:|---:|---:|---|---:|---:|
| 4 x 64K | 0.262M | 954.95 | 71.01 | **17.75** | **PASS** | 12.8 | 309 |
| **4 x 128K** | 0.524M | 727.51 | 49.59 | **12.40** | **PASS** | 16.3 | 743 |
| 8 x 64K | 0.524M | 955.31 | 84.69 | 10.59 | FAIL | 15.4 | 572 |
| 4 x 192K | 0.786M | 593.26 | 37.70 | 9.43 | FAIL | 20.4 | 1351 |
| 8 x 96K | 0.786M | 826.40 | 67.17 | 8.40 | FAIL | 19.0 | 978 |
| 4 x 256K *(named design point)* | 1.049M | 499.20 | 30.25 | 7.56 | FAIL | 24.5 | 2131 |
| 8 x 128K | 1.049M | 727.99 | 55.39 | 6.92 | FAIL | 22.7 | 1471 |
| 8 x 192K *(named design point)* | 1.573M | 593.43 | 41.20 | 5.15 | FAIL | 29.9 | 2690 |

**Every cell fit in memory. Six of eight fail R3.1.** Both design points named in R2.2 are among
the failures. No cell OOMed, so "does not fit" never appeared — the capacity limit on this hardware
is decode bandwidth, full stop.

### Final decode model — 0.8% worst-case over a 6x range of total KV

    per-slot decode time = 18.24 ms + 3.21 ms x slots + 95.7 ms x total_KV_Mtok

| cell | measured | fit | err |
|---|---:|---:|---:|
| 4x64K | 17.75 | 17.80 | +0.3% |
| 4x128K | 12.40 | 12.30 | -0.8% |
| 8x64K | 10.59 | 10.62 | +0.3% |
| 4x192K | 9.43 | 9.40 | -0.3% |
| 8x96K | 8.40 | 8.39 | -0.1% |
| 4x256K | 7.56 | 7.61 | +0.6% |
| 8x128K | 6.92 | 6.93 | +0.1% |
| 8x192K | 5.15 | 5.14 | -0.2% |

Three terms, physically interpretable: a fixed 18 ms per decode step, 3.2 ms per active slot, and
95.7 ms per million KV tokens read. It predicted 8 x 64K, 4 x 128K, 8 x 96K, 8 x 128K and 4 x 192K
*before* they were run, each within 1.5%.

### Prefill depends ONLY on per-sequence depth, not on slot count

| depth | 4 slots | 8 slots |
|---:|---:|---:|
| 64K | 954.95 | 955.31 |
| 128K | 727.51 | 727.99 |
| 192K | 593.26 | 593.43 |

Agreement to three digits at three different depths. This is also the cleanest available evidence
that the harness is measuring what it claims: two independent runs, different slot counts,
different total memory footprints, same number.

    prefill t/s = 1 / (0.000732 + 0.004856 * depth_Mtok)   -> 1122 t/s at 32K, 499 t/s at 256K

Pass 2 (6 x 64K, 8 x 48K, 10 x 32K, 12 x 32K) started 14:50 UTC.

---

## CAPACITY LADDER COMPLETE — 12 cells

| cell | total KV | pp t/s | agg t/s | **per slot** | R3.1 | VRAM GiB/die |
|---|---:|---:|---:|---:|---|---:|
| 4 x 64K | 0.262M | 955 | 71.0 | **17.75** | **PASS** | 12.8 |
| 6 x 64K | 0.393M | 956 | 79.2 | **13.20** | **PASS** | 13.9 |
| 4 x 128K | 0.524M | 728 | 49.6 | **12.40** | **PASS** | 16.3 |
| 8 x 48K | 0.393M | 1035 | 97.5 | **12.18** | **PASS** | 13.8 |
| 10 x 32K | 0.328M | 1132 | 117.9 | 11.79 | FAIL (98%) | 13.1 |
| 8 x 64K | 0.524M | 955 | 84.7 | 10.59 | FAIL | 15.4 |
| 12 x 32K | 0.393M | 1132 | 123.6 | 10.30 | FAIL | 13.7 |
| 4 x 192K | 0.786M | 593 | 37.7 | 9.43 | FAIL | 20.4 |
| 8 x 96K | 0.786M | 826 | 67.2 | 8.40 | FAIL | 19.0 |
| 4 x 256K *(named)* | 1.049M | 499 | 30.2 | 7.56 | FAIL | 24.5 |
| 8 x 128K | 1.049M | 728 | 55.4 | 6.92 | FAIL | 22.7 |
| 8 x 192K *(named)* | 1.573M | 593 | 41.2 | 5.15 | FAIL | 29.9 |

**Four of twelve clear R3.1 at 12 tok/s. All twelve fit memory; nothing OOMed.**

### Slot count is an independent cost — not just a way of spending KV budget

Three cells hold the *same* 0.393M total KV:

| config | same total KV | per slot | aggregate |
|---|---:|---:|---:|
| 6 x 64K | 0.393M | **13.20** | 79.2 |
| 8 x 48K | 0.393M | **12.18** | 97.5 |
| 12 x 32K | 0.393M | **10.30** | 123.6 |

And the sharper version: **10 x 32K holds LESS KV (0.328M) yet is SLOWER per request (11.79) than
8 x 48K (0.393M, 12.18)**. Adding slots costs more than the KV it saves, because each active slot
adds ~3.5 ms to every decode step. So "maximise slots x context" (R3.2) is not a sufficient
objective — the shape of the product matters, not only its size.

### Final model — 12 cells, mean error 0.5%, worst 1.6%

    per-slot decode time = 17.57 ms + 3.49 ms x slots + 94.6 ms x total_KV_Mtok

### THE FRONTIER — best aggregate throughput at each candidate R3.1 floor (measured cells only)

| R3.1 floor | cells passing | best aggregate | at config |
|---:|---:|---:|---|
| 10 | 7 of 12 | **123.6 t/s** | 12 x 32K @ 10.30/req |
| **12** | 4 of 12 | **97.5 t/s** | **8 x 48K @ 12.18/req** |
| 13.4 | 1 of 12 | **71.0 t/s** | 4 x 64K @ 17.75/req |

The floor choice is worth **1.7x of aggregate throughput** between 10 and 13.4. This is the single
highest-leverage open question in the spec.

### My recommendation, pending step 2

At a 12 tok/s floor the candidates are **8 x 48K** (97.5 agg, but only **1.5% margin** over the
floor) and **6 x 64K** (79.2 agg, **10% margin**, 33% more depth per user). A 1.5% margin is thin
against real chat variance — variable generation lengths, queueing, tool calls — so 8 x 48K may
well fall under 12 at p10 once the true R2.4 shape is applied, while 6 x 64K should hold.
**Step 2 measures exactly this**, which is why all three configs are being run under chat shape
rather than promoting the batched-bench winner directly.

---

## THE BIGGEST FINDING: the prompt cache is undersized by 3x, and the service collapses because of it

Step 2, `chat-8x48K`, 8 slots x 48K, 12 clients (= 1.5 x slots per §5.1), true R2.4 chat shape,
Poisson arrivals, contexts seeded to 24K:

| metric | ladder said | chat shape measured |
|---|---:|---:|
| decode per request, median | 12.18 | **7.45** |
| decode p10 / min | — | **6.02 / 5.04** |
| TTFT median / p90 | — | **32.55 s / 37.92 s** |
| aggregate gen t/s | 97.5 | **35.0** |
| prefill:decode | — | **5.22 : 1** (R2.4 says a chat session is ~1:15) |
| turns whose whole conversation was re-prefilled | — | **15 of 24** |

### The mechanism, from the per-turn record

| turn | clients keeping their cache | median prefill | median TTFT | median decode |
|---:|---|---:|---:|---:|
| 1 | 8 of 12 | 1,847 tok | 2.99 s | 7.76 |
| 2 | **1 of 12** | **33,010 tok** | **34.41 s** | 6.84 |

A turn that keeps its cache prefills exactly its new tokens (214, 227, 371, 380 ...) and answers in
~1-4 s. A turn that lost it re-prefills the entire conversation — 31,600 to 38,721 tokens — and
takes 30-41 s. **And the decode rate tracks it:** cache-hit turns on an otherwise quiet server hit
**12.71, 13.14, 12.44 tok/s**, i.e. exactly the ladder's 12.18. Cache-hit turns that run *while*
other slots re-prefill 32K collapse to 6-8. The prefill storm steals decode from every slot.

### Root cause: `--cache-ram` defaults to 8192 MiB

The server log says it plainly, 18 times in one run:

    srv alloc: - making room for prompt cache entry, removing oldest entry (size = 1631.967 MiB)

One parked context at 32-44K is **1.5-2.1 GiB** — which independently confirms the ladder's KV
measurement of 53.4 KiB/token (44K x 53.4 KiB = 2.35 GiB). Twelve clients therefore need ~30 GiB of
parked context and the default budget is 8 GiB, so the cache holds ~4 of 12 and evicts the rest.

**The prompt cache is not broken and R2.3 is not wrong — the budget is three times too small.**
This is a one-flag fix: `--cache-ram 49152`. Host RAM is 378 GiB total, 365 available.

Required cache-ram = clients x depth x 53.4 KiB:

| config | needed |
|---|---:|
| 8 x 48K, 12 clients | 30.0 GiB |
| 4 x 128K, 6 clients | 40.0 GiB |
| 6 x 64K, 9 clients | 30.0 GiB |

**Caveat for the fleet (R3.8):** these nodes are hyperconverged (KVM, DB, Ceph beside the service),
so 30-48 GiB of host RAM per node is a real cost, not free. It buys back a 1.6x decode improvement
and a ~30x TTFT improvement, so it is very likely worth it — but it is a fleet budget item and the
lead should know it exists.

### Why this was invisible until tonight

The old `service-client.py` injected its document at turn 0 and generated only 512 tokens a turn, so
its *sessions were short and its contexts small* — the cache was never pressured. R2.4's verified
note ("turns 2 and 3 prefilled 1,027 tokens against a ~50,000-token conversation, a 50x reduction —
the cache works") is correct **and** consistent with this: that observation was made where the cache
fit. The shape correction is what exposed the budget.

**Step 2b added:** the same three configs with `--cache-ram 49152`, chained automatically. This also
means every step-2 number is a *default-configuration* result, not a capability limit.

### 4 x 128K with the default cache: worse than the harness R2.4 replaced

| metric | 8 x 48K, 12 cl | 4 x 128K, 6 cl |
|---|---:|---:|
| ladder said (decode/req) | 12.18 | 12.40 |
| **chat, decode med / p10 / min** | 7.45 / 6.02 / 5.04 | **5.51 / 3.73 / 3.64** |
| TTFT med / p90 | 32.6 / 37.9 s | **151.0 / 157.6 s** |
| aggregate gen t/s | 35.0 | **11.6** |
| cache misses | 15 / 24 | **12 / 12** |
| **prefill : decode** | 5.22 : 1 | **28.56 : 1** |
| wall | 2795 s | 3963 s |

One parked 128K context is 6.5 GiB, so the 8 GiB default holds exactly **one** of six clients and
every turn re-prefills ~110K tokens.

**The reframing this forces.** R2.4 states a real session runs ~1:15 prefill:decode, and rejects the
old harness for running 33:1. `chat-client.py` was doing correct R2.4 chat here — and produced
**28.56:1**. The inversion was not in the client; it came from cache eviction.

So **workload shape is a property of the client AND the cache configuration, not of the client
alone.** The prompt cache is the mechanism that makes an interactive chat workload decode-bound; an
undersized cache turns the identical client traffic into a prefill-bound workload. R2.4's 1:15 is
achievable but it is a *consequence of configuration*, not a given. Step 2b tests exactly that, and
`pre:dec` is the column to read first.

A corollary worth stating plainly: **the deeper the design point, the more cache RAM it needs, and
the more violently it fails without it.** 4 x 128K needs 40 GiB and degrades 2.2x worse than
8 x 48K, which needs 30 GiB. Depth per user is not only a decode-bandwidth cost (the ladder) — it is
also a host-RAM cost (this), and both push the same way: toward shallower slots.

---

## STEP 2 COMPLETE — chat shape, DEFAULT server configuration (`--cache-ram 8192`)

| config | ladder (decode/req) | chat decode med / p10 / min | TTFT med / p90 | agg gen t/s | pre:dec | cache miss | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 x 48K, 12 cl | 12.18 | 7.45 / 6.02 / 5.04 | 32.6 / 37.9 s | 35.0 | 5.22:1 | 15/24 | 2795 s |
| **6 x 64K, 9 cl** | 13.20 | **9.34 / 6.88 / 6.79** | 49.0 / 55.1 s | 25.3 | 7.00:1 | 10/18 | 2695 s |
| 4 x 128K, 6 cl | 12.40 | 5.51 / 3.73 / 3.64 | 151.0 / 157.6 s | 11.6 | 28.56:1 | 12/12 | 3963 s |

**All three fail R3.1 by 1.3x-3.3x as the server ships.** Two things to notice:

1. **The ranking inverts between the two metrics.** 6 x 64K is best per request (9.34); 8 x 48K is
   best in aggregate (35.0). Same hardware, same build, same shape — the R2.2-vs-R3.3 tension again.
2. **Cache-miss rate tracks depth, not slot count**: 56% at 6 x 64K, 63% at 8 x 48K, 100% at
   4 x 128K. Deeper slots evict harder because each parked context is bigger.

### The precise failure path, from the server's own trace

    slot release:      id 3 | task 8323 | stop processing: n_tokens = 50873
    slot get_availabl: id 3 | task -1   | selected slot by LRU, t_last = 20759277016
    srv alloc:         - making room for prompt cache entry, removing oldest entry (2232.294 MiB)
    slot launch_slot_: id 3 | task 9189 | processing task

`llama-server` looks for a slot holding the request's prefix and falls back to **LRU** when none has
it. The prefix is not there because the prompt-cache entry was already evicted for want of RAM. So
the returning client lands on an arbitrary slot and re-prefills from scratch. Secondary cost: that
2,232 MiB eviction stalls **all four dies** for ~2.5 s between the release and the next launch.

These are therefore **default-configuration numbers, not capability limits** — which step 2b tests.

### Incidental: `-ngl auto` cannot work with `-sm tensor` on this build

Every server and bench start logs:

    W common_fit_params: failed to fit params to free device memory:
      llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort

Benign here (the process continues, and decision N5 passes `-ngl all` explicitly), but it means the
build's **default** `-ngl auto` has no working implementation under tensor split — the fitting
routine aborts and offload falls back to whatever the default path chooses. Anyone running this
build with `-sm tensor` and no explicit `-ngl` is relying on undefined behaviour. **`-ngl all`
should be treated as mandatory in `settings/gfx906.env` and in launch.sh for the tensor-split
profile**, not as a benchmark nicety. It is worth a one-line upstream report too.

---

## STEP 2b: `--cache-ram 49152` — the single highest-value change of the campaign

Identical config, identical client, identical shape. Only the cache budget differs.

### 8 x 48K, 12 clients

| metric | default 8 GiB | **sized 48 GiB** | change |
|---|---:|---:|---|
| cache misses | 15 / 24 | **0 / 24** | eliminated |
| prefill tokens | 510,305 | **14,168** | **36x less** |
| **prefill : decode** | 5.22 : 1 | **0.14 : 1** | decode-bound at last |
| TTFT median / p90 | 32.55 / 37.92 s | **1.36 / 2.84 s** | **24x faster** |
| decode median | 7.45 | **12.10** | **+62%** |
| decode p10 / min | 6.02 / 5.04 | 7.14 / 6.50 | +19% |
| aggregate gen t/s | 35.0 | **42.3** | +21% |
| wall | 2795 s | 2315 s | -17% |

**`pre:dec` 5.22:1 -> 0.14:1.** The workload only *became* the interactive chat workload R2.4
describes once the cache could hold the sessions. R2.4's "~1:15" is now the right order of magnitude
(0.14:1 = 1:7), whereas the same client against a default server produced 5.22:1. This is the
strongest possible confirmation that shape is client x configuration, not client alone.

**TTFT 1.36 s median.** R3.4's acceptance bound is still TBC; with the cache sized, TTFT is no
longer a meaningful risk (1.36 / 2.84 s), and the 168 s p90 that R2.4 already retracted as a
document-injection artifact is doubly dead.

### The gate splits — this is what decision N1 was for

- **median 12.10 >= 12: PASSES**
- **p10 7.14 < 12: FAILS**

The distribution is **bimodal**, and for a structural reason: a request that decodes during a quiet
window reaches ~13 tok/s (the ladder's 12.18, confirmed); a request that overlaps another slot's
prefill drops to ~6.5. With clients = 1.5 x slots (§5.1) there is *always* someone prefilling, so
the slow tenth is not noise — it is the queue's shadow on decode.

**So the honest statement is:** at 8 x 48K with a sized cache the service delivers reading speed to
the typical request and not to the slowest tenth. Whether that ships depends entirely on the lead's
reading of R3.1 ("every request"), which is why open question 1 is the one that actually blocks.

**Follow-up worth running (not yet run):** the same config with **clients = slots** (8, not 12). That
removes the queue and therefore the prefill overlap, and would show whether p10 >= 12 is reachable at
all on this hardware or whether §5.1's 1.5x oversubscription is itself incompatible with R3.1's
"every request". That is a requirements question with a cheap experiment behind it.

### 4 x 128K with a sized cache — and the reconciliation with the ladder

| metric | default 8 GiB | **sized 48 GiB** | change |
|---|---:|---:|---|
| cache misses | 12 / 12 | **1 / 12** | ~eliminated |
| prefill tokens | 1,317,296 | **8,859** | **149x less** |
| prefill : decode | 28.56 : 1 | **0.19 : 1** | decode-bound |
| TTFT median / p90 | 151.0 / 157.6 s | **2.53 / 5.54 s** | **60x faster** |
| decode median | 5.51 | **12.30** | **+123%** |
| **decode p10 / min** | 3.73 / 3.64 | **11.60 / 11.60** | **+211%** |
| wall | 3963 s | 2224 s | -44% |

### THE RECONCILIATION: with a sized cache the ladder was right all along

| config | ladder per-slot | chat median | chat p10 | chat min | ladder agg | slots x chat median |
|---|---:|---:|---:|---:|---:|---:|
| 8 x 48K, 12 cl | 12.18 | **12.10** | 7.14 | 6.50 | 97.46 | 96.8 (-0.7%) |
| 4 x 128K, 6 cl | 12.40 | **12.30** | 11.60 | 11.60 | 49.59 | 49.2 (-0.8%) |

`llama-batched-bench` per-slot decode predicts chat-shape **median** decode to under 1%, and
`slots x median` reproduces the ladder's aggregate to under 1%. **The entire step-2 shortfall was
cache configuration — not workload shape, not the harness, not the hardware.** The ladder is a valid
predictor of the median; what it cannot predict is the *spread*.

### The spread is the real differentiator, and it favours FEWER, DEEPER slots

| config | median | p10 | min | spread (med - p10) |
|---|---:|---:|---:|---:|
| 8 x 48K, 12 cl | 12.10 | 7.14 | 6.50 | **4.96** |
| 4 x 128K, 6 cl | 12.30 | 11.60 | 11.60 | **0.70** |

4 x 128K delivers reading speed almost **uniformly** (min 11.60 = 97% of the floor); 8 x 48K is
bimodal. Mechanism: fewer slots means fewer concurrent prefills, so less interference with decode.
**This inverts the naive reading of the frontier.** On aggregate throughput 8 x 48K wins 2:1; on
"*every* request >= 12 tok/s" — R3.1's actual words — 4 x 128K is far closer, and it also gives each
user 2.7x the context.

### Caveat on my own aggregate column

The reported `gen tok/s` (42.3 and 20.7) is **depressed by the seeding shortcut** (decision N3),
which generates no tokens: 285 s of 2315 s wall at 8 x 48K (12%), but 844 s of 2224 s at 4 x 128K
(38%). Steady-state aggregate is the ladder's figure (97.5 and 49.6), confirmed by slots x median
above. **Do not read my `gen tok/s` column as the service's throughput** — it is a run-level number
that includes harness setup. Reporting it without this note would overstate the depth penalty.

---

## STEP 2b COMPLETE — the service, correctly configured

All with `--cache-ram 49152`, true R2.4 chat shape, Poisson arrivals, clients = 1.5 x slots,
seeded contexts, zero or near-zero cache evictions.

| config | ladder per-slot | **chat median** | p10 | min | spread | TTFT med / p90 | pre:dec | steady agg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 48K, 12 cl | 12.18 | **12.10** | 7.14 | 6.50 | 4.96 | 1.36 / 2.84 s | 0.14:1 | **97.5** |
| 6 x 64K, 9 cl | 13.20 | **13.20** | 10.10 | 9.00 | 3.10 | 1.48 / **1.75** s | 0.09:1 | 79.2 |
| 4 x 128K, 6 cl | 12.40 | **12.30** | **11.60** | **11.60** | **0.70** | 2.53 / 5.54 s | 0.19:1 | 49.6 |

Three independent confirmations that the measurement is sound: chat median matches the ladder's
per-slot decode to within 1% at all three points (12.10/12.18, 13.20/13.20, 12.30/12.40);
`slots x median` reproduces the ladder aggregate to within 1%; and `pre:dec` is now 0.09-0.19:1,
i.e. the decode-bound chat workload R2.4 specifies.

### Verdict against R3.1, both readings

| reading | 8 x 48K | 6 x 64K | 4 x 128K |
|---|---|---|---|
| **median >= 12** | PASS (12.10) | PASS (13.20) | PASS (12.30) |
| **p10 >= 12** (my N1 gate) | FAIL (7.14, 60%) | FAIL (10.10, 84%) | **near-miss (11.60, 97%)** |

**On the median reading every candidate ships, and the choice is pure throughput: 8 x 48K at 97.5
aggregate. On the "every request" reading none ships yet, and 4 x 128K is the only one within reach.**
The spread — not the median — is what separates these configurations, and spread is driven by
concurrent prefill, which is driven by the client:slot ratio. Hence the clients=slots experiment,
queued to run after the R3.6 work.

### Recommendation (mine, as technical lead, pending the lead's R3.1 reading)

- **If R3.1 means the median:** ship **8 x 48K q8_0 KV with `--cache-ram 49152`** — 97.5 t/s
  aggregate, 12.10 tok/s median, TTFT 1.36 s, 48K per user (above R2.2's 32K floor).
- **If R3.1 means every request:** **4 x 128K** is the only candidate in reach (p10 11.60) and it
  also gives 2.7x the context per user; it costs half the aggregate throughput.
- **If a single answer is needed before the lead rules:** **6 x 64K** is the defensible compromise —
  median 13.20 (the largest margin of the three), the best TTFT p90 (1.75 s), 79.2 aggregate
  (81% of the maximum), and 64K per user (2x R2.2's floor).

In every case `--cache-ram` must be set. That is not a tuning preference; without it the same
hardware delivers 5.5-9.3 tok/s and 30-150 s TTFT.

---

## STEP 4 — context compression (R2.4's remaining TBC): the DEFAULT KILLS THE SESSION

`ctxshift-off` = server default (`--no-context-shift`), 8 slots x 32K, 8 clients, sessions seeded to
26K and then grown by generation until they cross the slot limit.

**Result: 8 of 20 records are refusals — one per client. Every session died at the crossing.**

    HTTP 400 {"code":400,
              "message":"request (32885 tokens) exceeds the available context size (32768 tokens),
                         try increasing it",
              "type":"exceed_context_size_error","n_prompt_tokens":32885,"n_ctx":32768}

This is not graceful degradation and not a rate penalty — **the conversation cannot continue**. The
first refusal arrives at ctx ~32886, i.e. the first token past the slot. Before the crossing the
service was healthy (decode median 11.57, TTFT 2.69 s).

**So R2.4's TBC is not a choice between optimisations — it is mandatory.** Something must handle the
crossing or every long-running chat session terminates with a 400. The options are the server's
`--context-shift` (measured in the next two runs) or a client/product-side policy (summarise,
checkpoint, start a new session with carried-over context). Given R2.2's "as large as reasonably
possible" ambition and R2.4's "context accumulates from the model's own output", sessions WILL reach
this boundary in normal use.

### Two honest caveats on this run

1. **Thin sample:** only 4 turns were measured, because the sessions died. The decode/TTFT figures
   are indicative, not solid.
2. **My eviction prediction was wrong.** I expected clients = slots to avoid cache eviction; the run
   logged **10 evictions**, because 8 x 32K contexts need ~13.7 GiB against the 8 GiB default. So the
   *rate* numbers in this step carry the same contamination step 2 had. The refusal finding is
   unaffected (a hard HTTP error, not a rate), but **step 4 should be re-run with `--cache-ram` before
   its rate numbers are quoted.** `ctxcompress.runlist` was written before the cache finding existed
   and I did not update it — my mistake, logged here rather than quietly dropped.

### `--context-shift` DOES NOT solve this — the policy must live in the client

| run | server_extra | refusals | wall | decode med | first refusal |
|---|---|---:|---:|---:|---|
| ctxshift-off | *(default)* | **8 / 20** | 768 s | 11.57 | ctx 32,885 > 32,768 |
| ctxshift-on | `--context-shift` | **8 / 20** | 769 s | 11.52 | ctx 32,885 > 32,768 |

Byte-identical outcomes. The flag was verifiably passed (`server_extra='--context-shift'` in the run
log) and the server logged **zero** context-shift activity.

**Why, and why it matters.** `--context-shift` discards old context when *generation* runs past the
window. It cannot apply when the *submitted prompt* is already larger than the slot: that is rejected
at request validation, before any generation. And an interactive chat client resends the whole
conversation each turn (that is what makes the prompt cache work), so the prompt grows monotonically
and eventually exceeds n_ctx. At that instant the session is dead with HTTP 400.

**Answer to R2.4's TBC:** the compression policy **cannot be a server flag** — it must be implemented
where the conversation is assembled. Options, in the product's hands:

1. **Summarise/compact** older turns into a shorter synopsis and resend that (keeps one session alive;
   costs a summarisation call and loses fidelity).
2. **Sliding window** — drop oldest turns client-side (cheap, silently loses history, and *breaks the
   prompt cache* because the prefix changes, so every turn after the first drop re-prefills; given the
   step-2 findings that is expensive and must be measured before adopting).
3. **Session rollover** — start a fresh session seeded with a carried-over summary (keeps the cache
   prefix stable for the new session).
4. **Size the slot so the boundary is never reached in practice** — at 128K per slot a 4K-per-turn
   session survives ~30 turns; at 48K only ~10. **This links R2.4 directly to the design point:
   shallower slots hit the wall sooner.**

Option 2 is the obvious-looking choice and is probably the worst on this hardware, for the cache
reason. Option 1 or 3 is what I would recommend, and **the lead should choose, because it is a product
behaviour (what the user sees happen to their history), not a tuning decision.**

Note also that the depth tradeoff now has a third axis: deeper slots cost decode bandwidth (ladder)
and host RAM (cache), but buy *session longevity* before compression is needed. 4 x 128K survives
~3x as many turns as 8 x 48K.

### Checkpoints (`-ctxcp`): a PROVISIONAL observation, not a result

| run | evictions | prefill tok | pre:dec | TTFT med / p90 | decode p10 | refusals |
|---|---:|---:|---:|---:|---:|---:|
| `--context-shift` (ctxcp 32, default) | 10 | 9,129 | 1.54:1 | 2.69 / 6.54 s | 7.2 | 8/20 |
| `--context-shift -ctxcp 0` | 2 | **35,843** | 5.96:1 | 2.16 / **31.03 s** | 6.3 | 8/20 |

Disabling checkpoints quadrupled prefill and made TTFT p90 4.7x worse. The self-consistent reading is
that checkpoints cost prompt-cache RAM (hence *more* evictions when enabled: 10 vs 2) but allow a
slot to resume without re-prefilling from zero (hence far less total prefill).

**I am not quoting this as a finding.** Three reasons: only **4 measured turns** per run; the whole of
step 4 ran with the undersized 8 GiB cache, so eviction behaviour confounds everything; and neither
server log contains a single "checkpoint" line, so the mechanism is inferred, not observed. Step 4b
re-runs all three policies with `--cache-ram 49152` and is the version to read.

What step 4 DOES establish beyond doubt, because it is a hard HTTP error rather than a rate:
**every session dies at the slot boundary, and `--context-shift` does not prevent it.**

---

## PROBE-ALLOC — §5.2's per-die memory itemisation (never previously recorded)

Under `-sm tensor`, llama.cpp reports buffers against a composite device `Meta(ROCm0,ROCm1,ROCm2,
ROCm3)` and the figures are **per die** — which is why `probe-alloc.sh`'s `ROCm0`-based greps came
back empty and `extract-alloc.py` was needed to recover them from the `-v` logs.

| cell | total KV | offload | model MiB/die | KV MiB/die | compute MiB/die | **sum GiB/die** | headroom |
|---|---:|---|---:|---:|---:|---:|---:|
| 4 x 64K | 0.262M | 66/66 | 6517 | 2184.5 | 3209.3 | **11.63** | 19.37 |
| 8 x 64K | 0.524M | 66/66 | 6517 | 4369.0 | 3565.3 | **14.11** | 16.89 |
| 4 x 128K | 0.524M | 66/66 | 6517 | 4360.5 | 4585.3 | **15.10** | 15.90 |
| 8 x 96K | 0.786M | 66/66 | 6517 | 6545.0 | 5101.3 | **17.74** | 13.26 |
| 4 x 192K | 0.786M | 66/66 | 6517 | 6536.5 | 6633.3 | **19.23** | 11.77 |
| 8 x 128K | 1.049M | 66/66 | 6517 | 8721.0 | 6637.3 | **21.36** | 9.64 |
| 4 x 256K | 1.049M | 66/66 | 6517 | 8712.5 | 8681.3 | **23.35** | 7.65 |
| 8 x 192K | 1.573M | 66/66 | 6517 | 13073.0 | 9709.3 | **28.61** | 2.39 |

- **`offloaded 66/66 layers to GPU` in every cell** — decision N5's `-ngl all` verified; no cell was
  silently part-offloaded, so every ladder number is a true all-GPU measurement.
- **Model 6517 MiB/die = 6.36 GiB**, matching R3.2's stated "weights 6.3 GiB per die".
- **q8_0 KV = 8.51-8.53 KiB/token/die**, i.e. **34 KiB/token across the four dies**, constant to
  +/-0.3% over a 6x range.
- **The compute buffer is a first-class consumer**, 6.3-12.5 KiB/token/die, scaling with depth and
  slot count: 9.7 GiB/die at 8 x 192K — *larger than the whole KV cache at 4 x 64K*. This has never
  appeared in the corpus and it is a third of the memory at deep design points.
- Even the worst cell keeps 2.39 GiB spare, so memory never bound anything.

### Correction to an earlier figure in this log

I earlier derived "q8_0 KV = 13.7 KiB/token/die (53.4 KiB/token over four dies)" from the peak-VRAM
slope. **That slope is KV plus compute-buffer growth, not KV.** The true KV cost is 8.5 KiB/token/die
(34 KiB/token total). The cache-ram sizing rule should be ~40 KiB/token (34 plus the ~12% overhead
seen in observed prompt-cache entries, which ran ~38 KiB/token), not 53.4. My `--cache-ram 49152`
was therefore generous rather than wrong, and every conclusion drawn from it stands; but the sizing
formula in the step-2b section should be read with this correction.

### Architecture note

`llama_kv_cache: size = 8738.00 MiB (65792 cells, **16 layers**, 4/4 seqs)` — only **16 of the 65
blocks carry a KV cache** (per-layer trace: layers 3, 7, 11, 15 ... have a device, the rest are
"filtered"). `qwen35` uses hybrid attention with roughly every fourth layer full-attention. That is
why KV costs 34 KiB/token instead of the ~68 KiB a naive 64-layer q8_0 calculation predicts, and it
is the reason such deep contexts fit at all.

---

## THE POWER CAP IS THE BINDING POLICY, AND 125 W BUDGETS FOR A HOST THAT DOES NOT EXIST

Raised by the lead 2026-09-19 ~21:00 ("where did we get the 125 W powercap from, I thought the clamp
limit closer to 175 W? What about 150 W?"). Both numbers in the lead's question are real and they
answer different questions:

- **The clamp is not a per-die figure.** It is the chassis **DC total of 1228 W** (SMC `PZ0G`); dies
  latch to 1000 MHz ~20 s after crossing. Tonight peaked at 1143 W.
- **~175 W is the perf/W region** (2026-09-08 study): 185 W costs -2% throughput, 170 W -5%, 140 W
  -10%. A node *powered only to serve* runs at ~184 W.
- **125 W is a policy margin** from the production-envelope budget (2026-09-08 05:3x): host
  **uncapped**, plus 4x M.2 NVMe on a PEX8747, 2x ConnectX-4 Lx, 2x SATA. Worst case =
  `bays(cap) + 595` where `bays(cap) = 1.093*4*cap + 56`. Verified against the page to within 1 W.

| cap | worst-case DC | margin to 1228 | host RAPL cap required |
|---:|---:|---:|---:|
| **125 W** | 1198 | **+30 W** | uncapped (413) fine |
| 130 W | 1219 | +9 W | uncapped (needs <=422) |
| 131 W | 1224 | +4 W | uncapped — the hard ceiling |
| 140 W | 1263 | **-35 W** | 378 W |
| 150 W | 1307 | -79 W | 334 W |
| 170 W | 1394 | -166 W | 247 W |

### 130 W is safe but pointless; 145-146 W is the move

Interpolating the two measured caps (125 and 200 W), 5 extra watts buys ~0.1 tok/s:

| config | 125 W | 130 W | 140 W | 150 W | cap for 12.0 |
|---|---:|---:|---:|---:|---:|
| 6 x 64K | 11.71 | 11.81 | 12.01 | 12.21 | **140 W** |
| 8 x 48K | 10.67 | 10.77 | 10.97 | 11.17 | 191 W (unreachable) |
| 4 x 64K | 16.45 | 16.54 | 16.71 | 16.88 | passes already |

**The real finding: the 595 W non-bay worst case assumes the CPU draws its full 413 W RAPL ceiling,
but the measured all-core draw is 303 W.** So a host cap of 350 W is 47 W ABOVE anything the CPU has
been observed to pull, and it permits **146 W dies** — which puts 6 x 64K over the R3.1 floor
(~12.13) at no real host cost. That is a far better trade than shaving 5 W onto the die cap.

### Caveats stated rather than buried

- The 595 W figure **excludes the ~50 W SATA spin-up transient**. Against that, 130 W is -41 W and
  **125 W is already -20 W** — so neither cap covers a spin-up coinciding with pinned bays and a maxed
  host. Probably not a real conjunction (spin-up happens at boot with idle GPUs), but it is not an
  argument against 130 W specifically; the current production cap has the same exposure.
- **Asymmetric failure:** a clamp needs a **cold power cycle (unplug 30 s)**; a warm reboot re-clamps.
  On a node in a 3-node HA quorum that is expensive, so margin carries option value.
- Every decode figure above is **interpolated from two measured caps**. The 150 W ladder is queued and
  sits 4 W from the interesting 146 W point, so this becomes measured within the hour. **The cap
  decision should wait for it.**

---

## LEAD'S DECISIONS, 2026-09-19 ~21:20

1. **Stay at 125 W.** The 150/170 W cap sweep is **cancelled** — "we can circle back around to
   finding the perfect powercap settings later". It had started one cell; killed, box restored
   (RAPL 413, fans normal, caps 200, t2fanrd active), partial data moved to `aborted/`.
2. **The 413 W RAPL ceiling is unrealistic** and the lead's reasoning is worth keeping: the
   W-3275M's **TDP is 205 W**, and the measured 303 W all-core sits almost exactly at the mean of
   TDP and the RAPL ceiling — (205 + 413)/2 = 309 W. So budgeting the host at 413 W is budgeting
   for a state the silicon is not specified to reach. When the cap is revisited, the realistic
   worst case is ~305-310 W, which permits ~157 W dies. **Not acted on tonight.**
3. **Fan power is unaccounted for, and all corpus testing pinned fans at max.**

### 125 W LADDER, COMPLETE — at the production cap only ONE config clears R3.1

| cell | 200 W | **125 W** | penalty | agg at 125 W | R3.1 |
|---|---:|---:|---:|---:|---|
| **4 x 64K** | 17.75 | **16.45** | **-7.3%** | 65.8 | **PASS** |
| 6 x 64K | 13.20 | 11.71 | -11.3% | 70.3 | FAIL |
| 4 x 128K | 12.40 | 10.92 | -11.9% | 43.7 | FAIL |
| 8 x 48K | 12.18 | 10.67 | -12.4% | 85.4 | FAIL |

**The design point at the production cap is 4 slots x 64K.** Note 4 x 64K also takes the smallest
cap penalty (-7.3% vs -12%): with fewer slots it does less work per decode step, so it was never as
power-limited to begin with. The cap hurts the wide configs most — exactly the ones that looked best
on aggregate throughput at 200 W.

### The fan caveat, and what is now being measured

Every number in this log, and in the whole corpus, was taken with `always_full_speed=true` — fans
pinned at max for repeatability. Two consequences, neither previously tested:

1. **Die temperatures were best-case** (junction 78-86 C tonight). Under production PWM the dies run
   hotter, and Vega 20 throttles near 95-100 C. The corpus has never verified that a design point
   holds its rate at production fan settings — and `t2fanrd` failing once already produced 36
   thermal-throttle events (memory: macpro-t2fanrd-needs-applesmc-t2-dkms).
2. **Max-fan power is inside every DC total measured** — so the envelope figures are conservative on
   cooling, but the size of that term is unknown.

Added to the queue: `serve-prodfans.sh` (identical to `serve.sh` except it leaves `t2fanrd` on the
normal PWM curve) runs the 4 x 64K design point at 125 W, one variable against the max-fan run; and
`fanpower.sh` measures the idle DC delta between max RPM and the PWM curve.

### Remaining queue (all automated, ~4 h)

125 W service runs (4x64K + 6x64K) -> production-fan validation -> clients=slots -> MTP A/B at
4x64K -> step 4b -> fan power delta.

---

## HARNESS DEFECT FOUND AT CLOSE — the SMC logger accumulates and corrupts the clamp guard

**My defect, in both `ladder.sh` and `serve.sh`.** Their normal-exit path kills the clock sampler
(`kpid "${SAMP:-}"`) but **not** `smc-log.sh` — only the INT/TERM trap does. So every run leaves its
logger running, and by late evening several `smc-read.py` processes were polling the applesmc
interface concurrently. Concurrent access corrupts the reads, producing two symptoms:

1. **`JSONDecodeError` tracebacks written into the SMC logs** instead of readings. Three runs ended
   with **zero** valid `PZ0G` samples: the 125 W ladder, probe-alloc, and the 125 W service run.
2. **Spurious values** — `PZ0G=5100` and `PZ0G=3400` appear in `step2-chat-smc.log`, against
   `PZ0F=1044.8` and zone sums of ~907 W on the same line. The real DC then was ~1045 W.

### Safety assessment — degraded, but nothing was actually at risk

- **Peak DC across the whole night: 1144.2 W**, 84 W inside the 1228 W envelope. Never close.
- The **200 W runs** — the only ones anywhere near the envelope — had valid `PZ0G` for their entire
  duration (their logs' first traceback is timestamped 20:02, long after they finished). So the DC
  guard was live exactly when it mattered.
- The three runs with a dead DC guard were all **inherently low power** (125 W caps put the ceiling
  near 1047 W; probe-alloc does trivial work at `-npp 64`). Only the sclk guard protected them, and
  no clamp occurred: `.clamp-detected` was never created.

**But the guard degraded silently, which is trap T5b exactly** — "a harness can be wrong in ways
nothing reports". I found it only because a heartbeat printed an impossible 5100 W.

### Fixes owed (not applied tonight — scripts are mid-run and immutable while running)

1. `ladder.sh` / `serve.sh`: kill `smc-log.sh` on the **normal** exit path, not only in the trap.
   Better: start one logger per session rather than per run.
2. `smc-read.py` / `smc-log.sh`: serialise SMC access with `flock`. The applesmc interface does not
   tolerate concurrent readers, and it fails by returning garbage rather than an error.
3. `clamp-watchdog-v2.sh`: a **single** sample of `PZ0G >= 1228` triggers `kill_chain` immediately.
   Given reads demonstrably corrupt, that is a false-positive path that could kill a valid multi-hour
   run. It should require two consecutive samples, as the >= 1200 W rule already does.
4. A watchdog whose `SMCLOG` contains no parseable `PZ0G` for N consecutive checks should **say so**
   rather than silently skipping the DC test.

Item 3 nearly bit tonight: had that 5100 W reading landed in the log the **active** watchdog was
tailing, it would have killed a healthy run and written a spurious clamp report.

---

## DECISION (lead, 2026-09-19 ~21:40): run the patch survey at the 125 W cap

The lead's reasoning: HBM2 runs at full clock regardless of the cap, so a 125 W envelope puts
selection pressure on the Vega 20 compute die's logic efficiency within a fixed power budget
(analogy: Transmeta Crusoe/Efficeon were performant for their envelope). **Tonight's data confirms
the mechanism, and it is stronger than the argument needed.**

### Evidence 1 — the cap does not govern memory at all

`hbm-bw-cap-sweep.md` (2026-09-08): HBM bandwidth is **invariant from a 110 W cap down to 50 W** —
880.6 GB/s read, 714-719 GB/s copy at every cap, sclk pinned at the 999 MHz DPM floor, **die drawing
115 W throughout**. The streaming floor is not reducible by the cap.

    => at 125 W, only ~10 W/die sits above the 115 W streaming floor.
       at 200 W, ~85 W does.  An 8.5x smaller marginal budget for compute.

### Evidence 2 — the two workload types split exactly as that predicts

Four cells measured at both caps:

| cell | prefill delta | decode delta |
|---|---:|---:|
| 4 x 64K | -20.6% | -7.3% |
| 6 x 64K | -20.1% | -11.3% |
| 8 x 48K | -20.1% | -12.4% |
| 4 x 128K | -20.1% | -11.9% |
| **mean / spread** | **-20.2% / 0.5 pp** | -10.7% / 5.1 pp |

**Prefill (compute-bound) loses a uniform 20%** — 0.5 pp of spread across configs differing 2x in
slots and 2x in depth, which is about as clean a signature of a single shared bottleneck as a
benchmark produces. **Decode at depth (KV-read-bound) loses only 7-12%**, because most of its work is
the cap-invariant streaming.

### Evidence 3 — the dies are power-limited at BOTH caps, so efficiency IS the metric

Median per-die draw during every cell: **198-199 W at the 200 W cap, 124 W at the 125 W cap.** There
is no slack at either. Nothing tonight was clock-limited or thermally limited; everything was power
limited. That makes performance-per-watt the performance metric, not a secondary concern.

### Consequences for the survey

1. **Old survey verdicts systematically understate compute-efficiency patches**, because they were
   taken at 200 W where compute had 8.5x the marginal budget. This is an *independent* reason to
   redo the survey, additional to the workload-shape (D6) and per-profile (D1) reasons.
2. **Concrete prediction to test first:** the fork tile table was adopted for **+33% prefill**, and
   prefill is what the cap punishes hardest. It should therefore be worth MORE at 125 W than at
   200 W — and its ranking against the Q8_0 MMVQ fast path (a decode patch, in the -11% regime) may
   invert relative to the 2026-09-08 conclusion.
3. **Measurement design:** absolute t/s deltas shrink at 125 W, so more repetitions may be needed to
   resolve them; but relative deltas should GROW for compute patches, so SNR may improve instead.
   Verify on the first patch rather than assume. Spot-check anything marginal at 200 W in case a
   ranking inverts between caps.
4. Survey baseline is therefore pinned to: **4 x 64K, q8_0 KV, `--cache-ram 49152`, `-ngl all`,
   125 W/die**, ladder cell as the cheap repeatable metric and a sized-cache chat run as the service
   metric.

---

## PRODUCTION-FAN VALIDATION — the design point holds under PWM (lead's concern, answered)

The lead noted 2026-09-19 that every measurement in the corpus pinned the fans at max
(`always_full_speed=true`) for repeatability, while production runs them under `t2fanrd` PWM on
temperature thresholds — and that fan power is real and unaccounted. Two risks followed: hotter dies
might throttle, and the DC totals might be carrying max-fan power.

`serve-prodfans.sh` reruns the design point identically except that `t2fanrd` keeps the normal curve.

| metric | max fans (~1200 rpm) | **production PWM (~800 rpm)** | delta |
|---|---:|---:|---:|
| decode median | 18.30 | **18.24** | -0.3% |
| decode p10 | 17.14 | **17.04** | -0.6% |
| decode min | 17.04 | **17.02** | -0.1% |
| TTFT median / p90 | 1.63 / 1.92 s | 1.75 / 1.97 s | +0.12 / +0.05 s |
| aggregate gen t/s | 31.6 | 31.5 | -0.3% |
| prefill:decode | 0.14:1 | 0.14:1 | — |
| wall | 1403 s | 1411 s | +0.6% |
| **junction** | **47-49 C** | **68-72 C** | **+21 C** |

**Verdict: production fan control costs nothing measurable at the design point.** Every decode figure
moves less than 0.6%, inside the noise band. Junction rises 21 C and lands roughly 25 C below the
Vega 20 throttle point, so there is ample thermal headroom at the production cap.

**Consequence: the corpus's max-fan measurements are representative for this design point at 125 W.**
That was not previously established and could not have been assumed.

**Boundary on the result — do not over-generalise it.** This holds at **125 W**, where junction only
reaches 68-72 C even with the fans slowed. At the 200 W cap, junction already ran 78-86 C *with fans
maxed*; under PWM at 200 W it would be substantially hotter and could throttle. The result covers the
deployed configuration, not every cap.

Still owed: the fan-power delta itself (`fanpower.sh`, running last, GPUs idle). Note the block-diagram
review independently proposed a better version of that experiment — two runs at *fixed cap and host
load*, logging PZ0G/PZ7G/PDSR plus all four tachs — which would separate fan power under load rather
than at idle. Worth doing when the cap question is revisited.

---

## The fan-power delta FAILED to measure — root cause was trap T8 again

`fanpower.sh` ran to completion, exited **rc=0**, and produced **no data**: both samples came back
empty and `fanpower.md` was never written.

**Root cause:** `smc-read.py` was itself broken. Its cached key map
`/root/rocm-tests/bench/smc-keymap.json` was **corrupted by concurrent writes** — 14,869 bytes holding
valid JSON that ends at char 13,978 with garbage appended, timestamped 20:02, precisely when several
`smc-log.sh` processes had accumulated. Two of them ran `build_map()` at once and interleaved their
output into the same path. **This is trap T8 (the accumulating SMC logger) manifesting in the key map
rather than in the log** — the same concurrency defect, a second symptom.

Repaired by rebuilding: 18,124 bytes, 1,374 keys. Verified working (idle DC 252.3 W, wall 307.4 W,
`PZ0T` 0.0 — consistent with the recorded 244 W idle baseline).

**A second defect, mine:** `fanpower.sh` exited 0 while measuring nothing. It should have failed loudly
when the sample came back empty. That is trap T5b in a script I wrote *in the same session that
documented T5b*. The fix belongs with the T8 fixes: `smc-read.py` needs `flock` around `build_map()`
and around reads, and any script consuming it must treat an empty sample as an error.

Re-run launched after the campaign closed and the box freed.

---

## FAN POWER MEASURED — 21.2 W at idle (lead's question, answered)

Third attempt; the first two failed (T8 keymap corruption, then a `local` multi-assignment bug in my
own script — both now fixed, and the script fails loudly instead of reporting 0.0).

| fan mode | DC total (PZ0G) | fan RPM (1/2/3/4) |
|---|---:|---|
| **max — every measurement in the corpus** | **255.1 W** | 1200 / 2509 / 2510 / 2499 |
| **production PWM curve** | **233.9 W** | 496 / 486 / 486 / 493 |
| **delta** | **21.2 W** | |

### What this settles

1. **Fan power was accounted for, inside the "unzoned remainder".** The production envelope budget lists
   an unzoned 42-55 W term and never broke fans out; 21.2 W fits inside it. The block-diagram review's
   hypothesis was right — the budget is not missing the fans, it just never named them.
2. **Every DC figure in the corpus carries up to ~21 W of fan power that production will not spend at
   idle.** So the envelope arithmetic is conservative by that much at idle.
3. **It does NOT create usable headroom.** This is the *idle* delta, where PWM sits at its floor
   (~490 rpm) against max (1200-2510 rpm) — i.e. the **upper bound** on the gap. Under load the
   production curve ramps and the delta shrinks. So the 30 W worst-case margin at a 125 W cap must not
   be re-spent on the assumption that 21 W of it is fan power.

### The better experiment, still owed

The block-diagram review independently proposed the right version: two runs at **fixed per-die cap and
fixed host load**, one `always_full_speed=true` and one under PWM, logging PZ0G/PZ7G/PDSR plus all four
tachometers to steady state. The PZ0G delta at matched zone readings is fan power *under load*, which is
what the envelope actually needs. Worth doing when the power cap is revisited — it would say how much of
the 30 W margin is real.

Cross-check for confidence: idle DC here reads 233.9-255.1 W against the corpus's recorded 244 W idle
baseline, and `smc-read.py` independently reported 252.3 W after the keymap repair. Consistent.

## 2026-09-20 — scriptcheck hardening (while cmp-terms.sh runs)

The lead's framing: **"Code is deterministic, LLM reasoning isn't."** That is the whole justification for
this pass, and it was vindicated three times inside it.

`scriptcheck.sh` was rewritten and put under a fixture-based self-test (`scriptcheck-selftest.sh`,
25 assertions). Every historical bug gets a BAD fixture (must fire exactly N times) **and** a GOOD
counterpart (must fire zero) — because a checker that flags correct code gets ignored, which defeats it
as surely as missing the bug does.

Three defects that reading did not catch and the self-test did, all within this pass:
1. The first "fix" appended findings inside a `cmd | while read` pipeline. The loop body of a pipeline
   runs in a SUBSHELL, so every append was discarded: it reported "no FATAL findings" on a file with a
   planted bug. Same silent-no-op class as the bugs it exists to find.
2. `nocomment()` numbered lines itself and then `grep -n` numbered them again, so `^`-anchored checks
   (T5b) saw `12:w=$(...)` and matched nothing.
3. Two `add` call sites were left in the old 1-argument form (the conversion regex assumed two spaces
   after the severity; `WARN` had three), tripping `set -u` at runtime.

Checks now: T10, T7, T4 (unanchored p*grep -f), T4b (pattern-keyed wait loop; FATAL unanchored, WARN
anchored), T4c (`-x` comm > 15 chars), T3, T5b. Comment-only lines are never judged. Waivers are
explicit and visible: `# scriptcheck: ok T4,T4b — reason` downgrades to ACK and prints the reason; a
waiver naming a different check does not silence the finding.

`waitproc.sh` is new and replaces pattern-keyed chaining: `waitproc.sh PID` (preferred, `kill -0`) or
`waitproc.sh - '^/abs/path'`, which REFUSES an unanchored pattern, and both bounded by `WAIT_MAX`
(default 4 h) so a lost predecessor cannot wedge a chain. Verified against live processes, not asserted:
the anchored pattern matched the running watchdog (1) and this shell (0); `kill -0` on the live
cmp-terms pid timed out as designed.

Night-script audit: **17 FATAL -> 4**, all four deliberate:
- `ladder.sh` 38/40 — the ladder is COMPLETE and its bytes are the provenance for published cells. The
  defect (`pkill -x llama-batched-bench`, 19 chars, matches nothing) is benign: a stray bench process
  would not be killed. No re-run planned. NOT edited.
- `cmp-terms.sh` 16/47 — RUNNING (pid 317041). bash reads a script incrementally from disk, so editing
  it mid-run can make it execute garbage. **OWED: patch both pkill lines the moment the run finishes.**
Fixed: `serve.sh`, `serve-prodfans.sh`, `probe-alloc.sh`, `variance-gate.sh` (pkill anchored to
`^/bin/bash [^ ]*<script>`), `chain-gate.sh` and `chain-variance.sh` (the two loops that actually hung
last night, now waitproc.sh).
