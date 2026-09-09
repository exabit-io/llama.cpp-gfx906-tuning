# Engineering requirements specification — llama.cpp serving on the gfx906 fleet

Status: **draft 1, 2026-09-09 evening, written after the fact.** Items marked **TBC** need the product engineering lead's confirmation
or correction; everything else is taken from the project's records (the 2026-09-04 agreement in the preliminary report, the
reports in `reports/`, the fleet notes) or from the lead's statements on 2026-09-09. This document outranks `README.md`,
`NEXT-STEPS.md`, the optimiser and every report: when they disagree with it, they are wrong. Change it only with the lead's
agreement, and log the change at the end.

## 1. Purpose

A production inference service: `llama-server` on Exabit's 2019 Mac Pro nodes (four Vega 20 / gfx906 dies each, 32 GB HBM2 per
die, one XGMI ring), serving a queued, batched, multi-user workload with the largest context window per slot the dies can hold.
The deliverable is the optimized llama.cpp build for gfx906 plus its launch configuration, kept current with upstream, and the
measurements that prove it against this specification.

## 2. The workload (what "the service" means)

- **R2.1** Many independent clients, arriving continuously; the server batches them (continuous batching) into **4 to 12 slots**
  and **queues the rest** at the HTTP layer. Agreed 2026-09-04 (8 slots; more slots only slow every request). **TBC:** the exact
  slot count is chosen by measurement inside 4..12; it is never above 12.
- **R2.2** Requests are long-context: prompts and conversations well above 16K tokens; **32K is the realistic floor** of what a
  request holds, and the service must accept requests **as large as the memory allows** (Qwen3.8's own thinking exceeds 4K on its
  own). **TBC:** the design target per slot — the records show 128K (8 slots × 128K f16 fits), 192K (needs q8_0 KV) and 256K
  (needs q8_0-K / q4_0-V); the lead's statement is "as large as reasonably possible".
- **R2.3** Multi-turn: a client's conversation grows across requests; the server keeps the prompt cache so a turn prefills only
  its new tokens. Prompt-cache behaviour is part of the service, not an option.
- **R2.4** Generation lengths are hundreds to thousands of tokens per turn (thinking model). **TBC:** representative distribution.
- **R2.5** Model: Qwen3.8-27B at Q8_0 (quality is not traded for speed: Q4 file quants are out). **TBC:** Qwen3.8-Flash-Next
  (`qwen4exp`) as a second model on the same service.
- **R2.6** Single-user, single-stream, batch-1 and 2K-context cases are **out of scope**. They are never gates, never objectives,
  never the basis of a build decision, and never reported as headline numbers.

## 3. Hard requirements

- **R3.1 Per-request rate.** Every request in a full batch decodes at or above **reading speed**, taken as **≥ 12 tok/s** per
  request at the service's design depth (the 2026-09-04 agreement: 13.4 tok/s per request at 8 clients was accepted, 2.9 at 32
  was rejected as below reading speed). **TBC:** the exact floor (10, 12 or 13.4).
- **R3.2 Capacity.** Slots × context per slot is maximised subject to R3.1 and the memory budget of **31 GiB per die** (weights
  6.3 GiB per die on the split, the rest KV and compute buffers). The KV-cache type (f16, q8_0, q8_0-K/q4_0-V) is chosen to
  reach the context target; its decode cost at depth is measured, not assumed.
- **R3.3 Throughput.** The service's aggregate rate (generated tok/s over all slots, and total tok/s including prefill) at the
  design point is the primary optimisation target, **after** R3.1 and R3.2 are met. **TBC:** the lead cites a previously
  achieved figure above 300 aggregate tok/s for the intended use case; that configuration is not in the records on the box and
  must be identified so it becomes the non-regression baseline (R6).
- **R3.4 Latency.** Time to first token at the design depth, with the queue non-empty, is reported for every configuration;
  **TBC:** an acceptance bound.
- **R3.5 Quality.** Perplexity and greedy output must match the reference build within the noise band; kernel changes are exact
  or their KL against the reference is measured and reported.
- **R3.6 Power.** The node stays inside the chassis envelope (1228 W DC) with the host uncapped; on the hyperconverged fleet the
  dies run at a **125 W cap** (fleet economics 2026-09-08); every result at the design point is reported at 200 W and at 125 W.
- **R3.7 Currency.** The build tracks upstream `ggml-org/llama.cpp` master; the gfx906 changes live as a maintained series on
  the `gfx906` branch of exabit-io/llama.cpp, rebased or merged on every upstream move, validated by the acceptance chain (R5)
  before promotion.
- **R3.8 Fleet.** Sixteen nodes, 64 dies, hyperconverged (KVM, DB, Ceph beside the service); the configuration must be
  reproducible from the repository on every node (`settings/`, `scripts/gfx906/build.sh`).

## 4. Explicitly not required (and not to be worked on)

Single-stream decode speed; MTP for a single user; 2K-context cells; more than 12 slots; Q4 weight files; any kernel or knob whose
only measured gain is on an out-of-scope cell.

## 5. Acceptance: how a build or setting is judged

The acceptance chain runs on the production build candidate and on the current production build, interleaved, at the design point
(**TBC:** 8 slots × 128K per slot as the first design point; then the largest context R3.2 allows):

1. **Service run:** `llama-server -cb` with the slot count of R2.1, clients = 1.5 × slots (so the queue is never empty), multi-turn
   sessions opening with 16K..(D−8K) documents and three turns of 1K user + 512 generated with the prompt cache on
   (`tools/service-client.py`). Report: per-request decode rate median and p10 (R3.1), aggregate gen and total tok/s (R3.3), TTFT
   median and p90 (R3.4), requests per minute. Two rounds, order rotated.
2. **Capacity ladder:** slots × depth × KV type over the configurations that fit, decode at depth (`llama-batched-bench`, one
   prompt per sequence), with KV and compute buffer sizes recorded (R3.2).
3. **Quality:** perplexity 16K/6 and greedy output against the reference (R3.5); `test-backend-ops` all ops.
4. **Power:** the service run at 125 W (R3.6).

A candidate is promoted only if it meets R3.1 and R3.2 at the design point and does not regress R3.3 or R3.4 by more than the
noise band (±2%) against the current production build. Nothing else promotes it.

## 6. Non-regression baselines (must never get worse)

- The 2026-09-04 agreed configuration: tp4, 8 slots, short prompts, 8 clients — 67.5 tok/s aggregate at 13.4 tok/s per request,
  15.8 requests/min.
- **TBC:** the lead's >300 aggregate tok/s configuration for the intended use case, once identified.
- The measured kernel gains of the project's own series (the Q8_0 MMVQ fast path: +62% at 12 slots, +34% at 16; the fork tile
  table: +33% prefill; the custom allreduce with the four-row gate) must remain active in every promoted build — a base whose
  defaults bypass them (the fork's Q8_0 repack did) is not promotable until they are restored.

## 7. Open items for the lead (TBC list)

R2.1 slot count inside 4..12 · R2.2 context target per slot · R2.4 generation-length distribution · R2.5 Flash-Next · R3.1 exact
per-request floor · R3.3 the >300 tok/s baseline configuration · R3.4 TTFT bound · R5 the first design point.

## Change log

- 2026-09-09 draft 1 (Claude, after the lead's correction). Written after two days of work that optimised out-of-scope cells; see
  `reports/2026-09-09-night-report.md` for what that cost.
