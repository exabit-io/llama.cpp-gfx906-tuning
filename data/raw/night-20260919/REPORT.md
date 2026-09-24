# Multi-user baseline campaign — 2026-09-19/20

Claude, technical lead, acting autonomously overnight. Raw log: `NIGHT-LOG.md`. Requirements:
`/root/llama.cpp-benchmarking/REQUIREMENTS.md`. Build: `/opt/llama.cpp-gfx906-rocm10`
(build 11067, commit `1d1361e7a`), ROCm 10.0, Qwen3.8-27B-Q8_0, tp4, `-fa on`, `-ngl all`,
`-b/-ub 2048`, q8_0 KV, `gfx906.env` (custom AR + corrected XGMI ring).

---

## 1. The three findings that matter

### 1.1 Both design points named in R2.2 fail R3.1 — and memory was never the constraint

| cell | fits memory | per-request decode | vs 12 tok/s floor |
|---|---|---:|---|
| 8 x 192K q8_0 *(named)* | yes, 28.6 GiB/die | **5.15** | 43% |
| 4 x 256K q8_0 *(named)* | yes, 23.4 GiB/die | **7.56** | 63% |

Twelve cells measured; **all twelve fit in memory, nothing OOMed, and only four clear R3.1.** The
capacity limit on this hardware is decode bandwidth. Memory permits ~1.63 Mtok of total KV; R3.1
permits ~0.52 Mtok — **3.1x less**. The named points were derived from the memory ceiling (8 slots
permits 198K/slot; R2.2 named 192K), and that ceiling is irrelevant.

    per-slot decode time = 17.6 ms + 3.49 ms x slots + 94.6 ms x total_KV_Mtok
    (12 cells, mean error 0.5%, worst 1.6%; predicted 5 cells to within 1.5% before they ran)

Per-slot decode depends on **total** KV across all slots, so `slots x depth` is one budget — but
slot count is *also* an independent cost, so the shape matters and not only the product. At an
identical 0.393 Mtok: 6x64K = 13.20, 8x48K = 12.18, 12x32K = 10.30. And 10x32K holds *less* KV than
8x48K yet is *slower* per request.

### 1.2 The prompt cache is undersized by 3x by default, and the service collapses because of it

`--cache-ram` defaults to **8192 MiB**. One parked 32-44K context is 1.5-2.1 GiB, so 12 clients need
~30 GiB and get 8. Same hardware, same config, same client — only the cache budget differs:

| 8 x 48K, 12 clients | default 8 GiB | **`--cache-ram 49152`** |
|---|---:|---:|
| cache misses | 15 / 24 | **0 / 24** |
| prefill tokens | 510,305 | **14,168** (36x less) |
| prefill : decode | 5.22 : 1 | **0.14 : 1** |
| TTFT median / p90 | 32.6 / 37.9 s | **1.36 / 2.84 s** (24x) |
| decode median | 7.45 | **12.10** (+62%) |

At 4 x 128K it is worse — 12/12 misses, 151 s TTFT, 28.56:1 prefill:decode, i.e. **worse than the
33:1 document harness R2.4 was written to replace.** The client was doing correct R2.4 chat
throughout. **Workload shape is a property of client x configuration, not of the client alone**; the
prompt cache is the mechanism that makes chat decode-bound, and an undersized one turns identical
traffic into a prefill-bound workload.

With the cache sized, chat-shape median decode matches the ladder's per-slot figure to within 1% at
all three configs, and `slots x median` reproduces the ladder aggregate to within 1%. **The ladder
was a valid predictor all along; the entire step-2 shortfall was configuration.**

### 1.3 At the 125 W production cap, exactly one configuration clears R3.1

| cell | 200 W | **125 W** | penalty | agg at 125 W | R3.1 |
|---|---:|---:|---:|---:|---|
| **4 x 64K** | 17.75 | **16.45** | **-7.3%** | 65.8 | **PASS** |
| 6 x 64K | 13.20 | 11.71 | -11.3% | 70.3 | FAIL |
| 4 x 128K | 12.40 | 10.92 | -11.9% | 43.7 | FAIL |
| 8 x 48K | 12.18 | 10.67 | -12.4% | 85.4 | FAIL |

---

## 2. Recommended design point

**4 slots x 64K, q8_0 KV, `--cache-ram 49152`, `-ngl all`, 125 W/die.**
16.45 tok/s per request, 65.8 t/s aggregate, 12.8 GiB/die (18 GiB spare), 64K per user (2x R2.2's
floor). It is the only configuration that satisfies R3.1 at the production power cap, and it takes
the smallest cap penalty because with fewer slots it does less work per decode step.

If the R3.1 floor moves, the answer moves with it — the floor *is* the capacity target:

| R3.1 floor | best aggregate (200 W, measured cells) | config |
|---:|---:|---|
| 10 | 123.6 t/s | 12 x 32K |
| **12** | 97.5 t/s | 8 x 48K |
| 13.4 | 71.0 t/s | 4 x 64K |

---

## 3. Decisions needed from the lead

1. **R3.1's floor, and whether it is median or p10.** This is the single highest-leverage open item;
   it is worth 1.7x aggregate throughput and it selects the configuration. With a sized cache at
   8 x 48K, median 12.10 **passes** and p10 7.14 **fails** — the distribution is bimodal because a
   request overlapping another slot's prefill drops to ~6.5. I have gated on **p10** (decision N1)
   as the reading closest to "every request", and reported median and min everywhere so the line can
   move without a re-run.
2. **R2.2 depth-per-user vs R3.3 aggregate throughput.** They point opposite ways along the frontier
   and R3.2's own objective is flat along it, so the spec does not break the tie.
3. **The context-compression policy (R2.4's TBC).** Not an optimisation choice — see §4.
4. **Whether `--cache-ram` of 30-48 GiB per node is acceptable on hyperconverged nodes** (KVM/DB/Ceph
   alongside). It buys 1.6x decode and 24x TTFT, so it is very likely worth it, but it is a fleet
   budget item that did not exist before tonight.

---

## 4. Context compression is mandatory, and no server flag provides it

With the server default (`--no-context-shift`), **every session died** at the slot boundary:

    HTTP 400 exceed_context_size_error: request (32885 tokens) exceeds the available
    context size (32768 tokens)

8 of 8 clients hit it. **`--context-shift` does not prevent this** — byte-identical outcome, 8
refusals, zero context-shift activity logged. It discards old context when *generation* runs past the
window; it cannot help when the *submitted prompt* already exceeds the slot, which is what happens
when a chat client resends a growing conversation. So the policy must live where the conversation is
assembled:

1. **summarise/compact** older turns — keeps the session, costs a call, loses fidelity;
2. **sliding window** — cheap and obvious, and probably the worst choice here: it changes the prefix
   and therefore **destroys the prompt cache**, whose cost §1.2 quantifies;
3. **session rollover** with a carried-over summary — keeps the cache prefix stable;
4. **size slots so the wall is rarely reached** — 128K survives ~30 turns at 4K/turn, 48K only ~10.

Recommendation: 1 or 3. This is a product behaviour (what the user sees happen to their history),
so it is the lead's call. Note it adds a **third axis** to the depth tradeoff: deeper slots cost
decode bandwidth and host RAM, but buy session longevity.

---

## 5. Memory itemisation (§5.2, never previously recorded)

Per die, tensor-split buffers (reported by llama.cpp against a composite `Meta(ROCm0..3)` device):
**model 6517 MiB = 6.36 GiB** (confirms R3.2's stated 6.3), **q8_0 KV 8.5 KiB/token/die** = 34
KiB/token over four dies, constant to +/-0.3% across a 6x range, and a **compute buffer of
6.3-12.5 KiB/token/die** that reaches 9.7 GiB/die at 8 x 192K — larger than the whole KV cache at
4 x 64K, and absent from the corpus until now. `offloaded 66/66 layers to GPU` in every cell.

Only **16 of the 65 blocks carry a KV cache** (`qwen35` hybrid attention, roughly every fourth layer
full-attention), which is why KV costs 34 KiB/token rather than the ~68 KiB a naive 64-layer
calculation predicts — and why contexts this deep fit at all.

---

## 6. Power

Peak DC across the night **1144.2 W**, 84 W inside the 1228 W envelope; `PZ0T` never left zero.
Junction 78-86 C. Dies were pegged at the cap in **every** configuration at both 200 W (198-199 W)
and 125 W (124 W) — nothing was clock- or thermally limited, so performance-per-watt is the
performance metric.

**Provenance of the 125 W cap** (asked by the lead): it is not the clamp. The clamp is the chassis DC
total of 1228 W. 125 W is the last per-die cap whose absolute worst case fits — all bays pinned *and*
the host at its 413 W RAPL ceiling *and* the planned 4x NVMe + 2x ConnectX-4 + 2x SATA = 1198 W.
130 W also fits (+9 W) but buys ~0.1 tok/s and crosses no threshold. The lead's observation stands
for the later revisit: the W-3275M's TDP is 205 W and measured all-core is 303 W, so budgeting 413 W
reserves headroom the silicon is not specified to reach; a realistic ~305 W worst case would permit
~157 W dies. **Deferred by the lead; staying at 125 W.**

---

## 7. What was deferred, and why

- **MTP on/off A/B (R3.9)** — deferred *into* the patch survey. Which build carries MTP is a
  per-profile patch question, which is exactly what the survey decides. Pre-work done: the build
  accepts `draft-mtp` and `draft-mtp-adaptive`; `libllama` carries a `QWEN35 MTP` path; the model is
  `general.architecture = qwen35` with `nextn_predict_layers = 1`. `serve.sh` already fails a run
  loudly if `--spec-type` was requested and `blk.*nextn` is still reported unused, so an MTP result
  can never come from a server that quietly ran without MTP.
- **clients = slots** — staged (`noqueue.runlist`). It would show whether p10 >= 12 is reachable at
  all, or whether §5.1's 1.5x oversubscription is itself in tension with R3.1's "every request".
- **Step 4b** (context compression with a sized cache) — staged (`ctxcompress2.runlist`). The refusal
  finding is secure; only the *rate* numbers need the clean re-run.
- **150/170 W cap sweep** — cancelled by the lead; revisit with the TDP argument above.

---

## 8. Handoff to the patch survey

**Baseline recipe, pinned.** 4 x 64K, q8_0 KV, `--cache-ram 49152`, `-ngl all`, **125 W/die**,
`gfx906.env`, libs pinned `LD_LIBRARY_PATH=$PREFIX/lib:/opt/rocm/core-10.0/lib`. Cheap repeatable
metric: the `llama-batched-bench` ladder cell (367 s at 125 W). Service metric: `bench/chat-client.py`
with `--seed-ctx 40960 --turns 3 --gen 4096 --tool-rate 0.3 --arrival-rate 0.05`, 6 clients.

**Why 125 W is the right test bed** (lead's argument, confirmed tonight). HBM bandwidth is
cap-invariant — 880.6 GB/s read at every cap from 110 W down to 50 W, die drawing 115 W — so at
125 W only ~10 W/die sits above the streaming floor versus ~85 W at 200 W, an **8.5x smaller
marginal budget for compute**. The workload split confirms it: **prefill loses a uniform 20.2%
(0.5 pp spread across four configs), decode only 10.7%.** A fixed envelope makes logic efficiency
the objective function.

**Consequence: every old survey verdict understates compute-efficiency patches**, because they were
taken at 200 W. This is an independent reason to redo the survey, on top of D6 (workload shape) and
D1 (per-profile).

**First prediction to test:** the fork tile table was adopted for **+33% prefill**, and prefill is
what the cap punishes hardest. It should be worth *more* at 125 W than at 200 W, and its ranking
against the Q8_0 MMVQ fast path (a decode patch, in the -11% regime) may invert versus the
2026-09-08 conclusion.

**Measurement caution:** absolute t/s deltas shrink at 125 W, so more repetitions may be needed; but
relative deltas should grow for compute patches, so SNR may improve instead. Verify on the first
patch rather than assume, and spot-check anything marginal at 200 W in case a ranking inverts.

**Mandatory settings, not preferences.** `-ngl all` (build 11067's default `-ngl auto` has **no
implementation** under `-sm tensor` — `llama_params_fit is not implemented for SPLIT_MODE_TENSOR,
abort` — so tensor-split without an explicit `-ngl` relies on undefined behaviour), and
`--cache-ram 49152` (without it the same hardware delivers 5.5-9.3 tok/s and 30-150 s TTFT).
Both belong in `settings/gfx906.env` and `launch.sh`.

---

## 9. Harness defects — fixed and owed

**Fixed tonight:** `kill ${VAR:-0}` signals the whole process group (a script that kills itself;
latent in `phase4b-q8kv.sh`); `pkill -x llama-batched-bench` matches nothing at 20 characters (comm
is truncated to 15); `chat-client.py` crashed on the first server refusal and lost every aggregate;
build 11067 suppresses the info logs carrying §5.2's buffer sizes (recovered by re-allocating with
`-v` at trivial cost rather than re-running the ladder).

**Owed (found at close, not yet applied):** `ladder.sh` and `serve.sh` kill the clock sampler on
normal exit but **not** `smc-log.sh`, so loggers accumulate; concurrent `smc-read.py` access corrupts
the applesmc interface, which emits `JSONDecodeError` into the logs and spurious values
(`PZ0G=5100` against `PZ0F=1044.8` on the same line). Three runs therefore had **zero** valid `PZ0G`
and a silently dead DC-envelope guard — all three inherently low-power, and the 200 W runs (the only
ones near the envelope) had a valid guard throughout, so nothing was at risk. But
`clamp-watchdog-v2.sh` kills the chain on a **single** `PZ0G >= 1228` sample, and reads demonstrably
corrupt: that is a false-positive path that could kill a healthy multi-hour run. It should require
two consecutive samples, as its >= 1200 W rule already does, and should report a log it cannot parse.

**Also owed:** `phase4b-q8kv.sh` should be deleted or rewritten — it drives the shape-invalid
`service-client.py`, its `run()` ends in `sleep` so `|| exit 1` can never fire, and its `EXIT` trap
fires on normal completion.
