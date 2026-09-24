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
- **R2.2 — FLOOR RAISED TO 64K (lead, 2026-09-23): "64k is absolute floor."** Nothing below 64K is measured,
  gated, optimised or recommended, for any model or profile. The reasoning the lead gave: a thinking model can
  spend 2K tokens reasoning to itself before it answers even a single-sentence request, so a 2K cell does not
  represent this service at all, and 32K was already generous. The earlier ~~16K minimum / 32K realistic floor~~
  is superseded. Load tests and functional smoke tests may use any depth; their numbers may never be quoted as
  results. (A 2K-depth Flash-Next load test on 2026-09-23 produced decode figures that were then used for an
  R3.1 comparison -- that comparison is void, and this clause exists because of it.)
- **R2.2** Requests are long-context: prompts and conversations well above 16K tokens; **32K is the realistic floor** of what a
  request holds, and the service must accept requests **as large as the memory allows** (Qwen3.8's own thinking exceeds 4K on its
  own). ~~**TBC:** the design target per slot~~ — **narrowed 2026-09-19 (lead).** The preferred design points are
  **8 slots × 192K with q8_0 KV** and **4 slots × 256K with q8_0 KV**. f16 KV at 8 × 128K was measured on
  2026-09-19 and is NOT the design point: it was proposed by Claude for comparability with the 7.14 corpus
  rather than derived from this requirement, and early in the run per-request decode sat around 1.6 tok/s,
  far under the R3.1 floor. q8_0 KV halves the dominant memory consumer at these depths, buying depth at
  equal cost. Records also list 256K reachable with q8_0-K / q4_0-V. Lead's framing remains "as large as
  reasonably possible".
- **R2.3** Multi-turn: a client's conversation grows across requests; the server keeps the prompt cache so a turn prefills only
  its new tokens. Prompt-cache behaviour is part of the service, not an option.
- **R2.4 Session shape (lead, 2026-09-19 — resolves the TBC and CORRECTS the harness).** The workload is an
  **interactive chat session that grows by generation, not by injection.** Per turn:
  - the user says **little** — a short message, order 100-500 tokens;
  - the model generates **a lot** — thinking plus answer, order **2K-8K** tokens. Qwen3.8's thinking alone
    exceeds 4K (R2.2), so a per-turn generation budget below ~4K does not represent this model at all;
  - the session may include **tool calls**, whose results inject prefill mid-session and change the pattern;
  - context accumulates from the **model's own output** across many turns. That is how a session reaches the
    128K-192K depths of R2.2 — it is NOT a document pasted in at turn 0;
  - with the prompt cache on, only the **new** tokens of each turn prefill (verified 2026-09-19: turns 2 and 3
    prefilled 1,027 tokens against a ~50,000-token conversation, a 50x reduction — the cache works);
  - once a session outgrows its slot, context must be **compressed / shifted / checkpointed**. What that costs
    is unmeasured. **TBC:** the compression policy.

  **Consequence — the prefill:decode ratio.** A real session runs roughly **1 : 15** (prefill : decode).
  `tools/service-client.py` ran roughly **33 : 1** — inverted by about 500x — because it injects a 16K-56K
  document at turn 0 and generates only 512 tokens per turn. **Every service-axis measurement taken with that
  harness models document-QA / RAG, not interactive chat, and must not be read as a chat result.**
  `bench/chat-client.py` (written 2026-09-19) implements this requirement and reports `pre:dec` as a
  first-class column so the shape can never silently invert again.

  **What this retracts.** Conclusions drawn from the old shape do not carry: a TTFT p90 of 168 s is an artifact
  of document injection (a real turn prefills ~300 tokens); and "the service is prefill-bound, so the fork tile
  table's +33% prefill is the win for multi-user" is unsound — in a decode-bound reality the relevant figure is
  that patch's **+62% decode**, not its prefill gain.
- **R2.5** Model: Qwen3.8-27B at Q8_0 (quality is not traded for speed: Q4 file quants are out).
  ~~**TBC:** Qwen3.8-Flash-Next as a second model~~ — **confirmed 2026-09-19, see R3.10.**
- **R2.6** ~~Single-user, single-stream, batch-1 and 2K-context cases are out of scope.~~ **Superseded 2026-09-19 — see R2.7.**
  Single-user is now its own profile with its own build. What remains out of scope for **both** profiles: 2K-context cells (the
  floor is 16K, realistically 32K, per R2.2) and more than 12 slots. The multi-user service remains the primary deliverable: a
  single-user gain never justifies a multi-user regression, and the two are never averaged into one headline number.

- **R2.7 Two build profiles (lead, 2026-09-19).** The gfx906 changes are maintained as **two separate builds**, because the
  optimisations are mutually exclusive between the modes — a patch set that wins in one loses in the other:
  - **multi-user / service build** — batched `llama-server` and `llama-batched-bench`. Judged on §3 and the §5 chain.
    Currently the `gfx906` branch (tag `gfx906-20260909`), installed at `/opt/llama.cpp-gfx906`.
  - **single-user build** - batch-1 / single-stream, MTP. ~~Judged on single-stream decode at the R2.2 context floor.~~
    **CORRECTED 2026-09-21 (lead): a floor is not a design point.** Judging single-user at 32K against a multi-user
    design point of 4 x 64K is not an apples-to-apples comparison. The single-user design points are:
    - **1 x 255K - PRIMARY.** Same **total KV** as the 4 x 64K multi-user design point (256K), so the two axes differ
      only in concurrency, under equal memory and bandwidth pressure. 255K rather than 256K because `n_ctx_train` is
      262144 and a cell needs room for its generated tokens: depth 260864 + 1280 headroom = 262144 exactly. This also
      matches R2.2's own stated design point of 256K per slot.
    - **1 x 64K - CONTROL.** Same **per-sequence depth** as multi-user, which isolates batching from depth.
    The 32K floor of R2.2 remains a floor: a minimum below which nothing is measured, gated or recommended. It is never
    a comparison depth. Any single-user verdict recorded against a 1 x 32K cell is **provisional** until it reproduces
    at 1 x 255K.
    Currently the fork tile table + Q8_0 MMVQ fast path, installed at `/opt/llama.cpp-mxxm-fh`.

  Each profile is measured, gated and promoted **independently, on its own axis**. A change is never rejected for regressing the
  other profile's axis — it goes to the other build, or to neither. Both builds track upstream per R3.7 and both ship per R3.8.

## 3. Hard requirements

- **R3.1 Per-request rate.** Every request in a full batch decodes at or above **reading speed**, taken as **≥ 12 tok/s** per
  request at the service's design depth (the 2026-09-04 agreement: 13.4 tok/s per request at 8 clients was accepted, 2.9 at 32
  was rejected as below reading speed). **TBC:** the exact floor (10, 12 or 13.4).
- **R3.2 Capacity.** Slots × context per slot is maximised subject to R3.1 and the memory budget of **31 GiB per die** (weights
  6.3 GiB per die on the split, the rest KV and compute buffers). The KV-cache type (f16, q8_0, q8_0-K/q4_0-V) is chosen to
  reach the context target; its decode cost at depth is measured, not assumed.
- **R3.3 Throughput.** The service's aggregate rate (generated tok/s over all slots, and total tok/s including prefill) at the
  design point is the primary optimisation target, **after** R3.1 and R3.2 are met. ~~**TBC:** the lead cites a previously achieved figure above 300 aggregate tok/s...~~
  **Resolved 2026-09-19 (lead): not recoverable.** The run was never logged properly — it was discarded during the single-user
  optimisation work because it read as a regression on the single-user axis, which R2.7 now makes a category error. The >300
  figure is therefore NOT a gate and NOT a baseline. The multi-user baseline is re-established from scratch by the current
  measurement campaign.
- **R3.4 Latency.** Time to first token at the design depth, with the queue non-empty, is reported for every configuration;
  **TBC:** an acceptance bound.
- **R3.5 Quality.** Perplexity and greedy output must match the reference build within the noise band; kernel changes are exact
  or their KL against the reference is measured and reported.
- **R3.6 Power.** The node stays inside the chassis envelope (1228 W DC) with the host uncapped; on the hyperconverged fleet the
  dies run at a **125 W cap** (fleet economics 2026-09-08); every result at the design point is reported at 200 W and at 125 W.
- **R3.7 Currency.** The build tracks upstream `ggml-org/llama.cpp` releases. **Code and substrate (lead, 2026-09-24):**
  all gfx906 code lives in `exabit-io/mx-llama.cpp`. The substrate is mxxm-t's gfx906 fork merged with the latest llama.cpp
  release (v0.5.0, `7fe450e`), offered to mxxm-t as mxxm-t/mx-llama.cpp#17 (after it merges, mxxm-t's master is the substrate).
  `master` = the substrate + the Exabit patches binned for both profiles; `gfx906-single` / `-multi` add the profile-only
  patches; `gfx906-candidates` holds the ones not yet binned; all move onto each new substrate on every upstream release and validated by the acceptance
  chain (R5) before promotion. ~~the `gfx906` branch of exabit-io/llama.cpp~~ — retired 2026-09-24.
- **R3.9 Speculative decoding / MTP (lead, 2026-09-19).** The service uses the model's own multi-token-
  prediction weights where they exist — Qwen3.8-27B ships `blk.N.nextn.*`, and a server that loads and
  discards them is leaving performance unclaimed. MTP is **in scope for BOTH profiles** (R2.7), as draft-1
  on batched slots for the multi-user service and as the single-user build's own axis.
  .
  **It is not assumed to be a win: it is measured.** On 2026-09-09 the `gfx906` branch FAILED the MTP gate
  on verify-step cost, both variants. Speculative decoding pays off when latency-bound with spare compute;
  at 8-12 saturated slots the service is throughput-bound and the verify step competes with real work. So
  every design point is run **MTP on and MTP off**, per profile, and the winner is adopted per profile —
  it may legitimately differ between them. A build that cannot make MTP net-positive for multi-user must
  say so with numbers, not silently run without it.

- **R3.10 Second model (lead, 2026-09-19).** Qwen3.8-Flash-Next (`qwen4exp`) is a supported model on the
  same service, not an optional extra. Promoted from R2.5's TBC.


- **R3.11 Multi-node collectives and RCCL (lead, 2026-09-21).** The fleet is a **multi-node GPU cluster interconnected
  with Mellanox ConnectX InfiniBand**. PCIe bays 5, 6 and 7 are empty today; the cards go in once the llama.cpp patchset
  work lands, and large-model multi-node testing follows. Therefore:
  - **Every build ships with `GGML_HIP_RCCL=ON`.** Upstream's default is OFF, and this campaign inherited that default
    without auditing it against this requirement. With RCCL absent, `GGML_USE_NCCL` is undefined, nothing links
    `librccl`, and the collective falls back to the meta-backend **butterfly** path. Any AllReduce verdict measured
    that way is against the wrong baseline for a shipping build and must be re-based against RCCL.
  - **Every build ships with `GGML_CUDA_FA_QUANTS=all` (lead, 2026-09-22).** Upstream's default compiles four
    K/V combinations; `all` compiles the full 7x7 cross-product of `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16`.
    (`GGML_CUDA_FA_ALL_QUANTS` is deprecated in favour of it.) This is mandatory for the same reason RCCL is:
    **a missing FlashAttention kernel does not fail.** llama.cpp converts K and V to f16 and warns, so a
    reading can look like a quantised-cache result when it is an f16-conversion result. That has already
    produced a wrong verdict in this project -- `/opt/llama.cpp`, `/opt/llama.cpp-prod` and
    `/opt/llama.cpp-mxxm-fh` all lack the `q8_0-q4_0` kernel, and only 1 of 3 historical quantised-V raw
    files even records which build produced it. Measured cost of `all`: 84 fattn objects instead of 38 and
    a 75 MiB `libggml-hip.so` instead of 59 -- 16 MiB and a few minutes of ccache-warm compilation.
    **The KV choice is per MODEL, not global.** KV bytes per token depend on how many blocks hold attention,
    the KV head count and head_dim: for Qwen3.8-27B the KV cache is **31.5%** of the bytes moved per step at
    4x64K, for Qwen3.8-Flash-Next only **3.1%**. So the same V quantisation that buys decode speed on one
    model can be pointless or a net loss on the other. Compiling everything means the KV type is chosen per
    model at launch (via `optimize.py`) without a rebuild, and every model needs its own sweep.
    Note `iq4_nl` is **not** in the FA type list, so an `iq4_nl` cache has no kernel at any setting and
    always converts.
  - **The RCCL AllReduce is retained, not replaced.** The fork's custom AllReduce is an intra-node PCIe/XGMI
    optimisation and is kept for the single-node case on its measured merits; RCCL remains the path that multi-node
    work will use. A survey verdict may never bin RCCL support away.
  - **Known gap, not a flag.** RCCL ON does **not** give multi-node tensor parallelism. The TP communicator is built
    from `cudaGetDeviceCount()` inside one process — there is no `ncclGetUniqueId` broadcast and no rank/world
    bootstrap in ggml — so RCCL will drive the four **local** dies only. The single existing multi-machine path, the
    RPC backend, is TCP sockets with no RDMA verbs, so it gains nothing from PeerDirect/GPUDirect. Spanning nodes
    requires cross-process communicator setup added to ggml, plus the `amdgpu` peer-memory module for GPUDirect RDMA.
    **Out of scope for the current patchset optimisation campaign (lead, 2026-09-21); tracked as a future-state
    engineering task in `NEXT-STEPS.md` S7.** If no upstream multi-node path exists by then, it is ours to author.
- **R3.8 Fleet.** Sixteen nodes, 64 dies, hyperconverged (KVM, DB, Ceph beside the service); the configuration must be
  reproducible from the repository on every node (`settings/`, `scripts/gfx906/build.sh`).

## 4. Explicitly not required (and not to be worked on)

2K-context cells; more than 12 slots; Q4 weight files; any kernel or knob whose only measured gain is on an out-of-scope cell.

~~Single-stream decode speed; MTP for a single user~~ — **removed 2026-09-19**: these are the single-user profile's axis (R2.7),
not out of scope. They are still never gates or headline numbers for the *multi-user* build.

## 5. Acceptance: how a build or setting is judged

The acceptance chain runs on the production build candidate and on the current production build, interleaved, at the design point
(**TBC:** 8 slots × 128K per slot as the first design point; then the largest context R3.2 allows):

1. **Service run:** `llama-server -cb` with the slot count of R2.1, clients = 1.5 × slots, **sessions shaped per R2.4**
   (short user turn, 2K-8K generated, tool calls, context grown by generation) using `bench/chat-client.py`, with a
   **non-degenerate arrival model** — `--arrival-rate` (Poisson) or `--ramp`. Starting every client at t=0 produces an
   N-way simultaneous prefill no real server sees; measured 2026-09-19 that artifact alone cost **4 tok/s of median
   decode** (11.05 -> 15.15) at 4 x 64K q8_0, moving a configuration from under the R3.1 floor to over it.
   ~~sessions opening with 16K..(D-8K) documents and three turns of 1K user + 512 generated
   (`tools/service-client.py`)~~ — superseded 2026-09-19, see R2.4. Report: per-request decode rate median and p10 (R3.1), aggregate gen and total tok/s (R3.3), TTFT
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

R2.1 slot count inside 4..12 (lead points at 8 and 4) · R3.1 exact
per-request floor · R3.4 TTFT bound · R5 the first design point.

## Change log

- 2026-09-24 (lead): **R3.7 — all gfx906 code in `exabit-io/mx-llama.cpp`; `master` = substrate + both-profile patches** (mxxm-t's fork merged with llama.cpp v0.5.0, offered upstream as mxxm-t/mx-llama.cpp#17); the bin branches live there; `exabit-io/llama.cpp` retired. Same day, R2.2: the context floor is 64K.
- 2026-09-19 (lead, on the harness): **R2.4 rewritten — the session shape was wrong, and with it every
  service-axis measurement in the corpus.** The lead's point: a real chat session has the user saying little and
  the model generating a lot, with context accumulating from the model's own output turn after turn, plus tool
  calls and eventual context compression. The old harness injected a 16K-56K document and generated 512 tokens a
  turn — a ~33:1 prefill:decode ratio against reality's ~1:15. Section 5.1 updated to require
  `bench/chat-client.py` and an arrival model. Two harness defects found the same day and fixed: clients all
  started at t=0 (no arrival model), and an absolute `TAG` crashed the client AFTER the workload ran, silently
  losing every aggregate.
- 2026-09-19 (lead, on reviewing what Phase 4 actually ran): **R3.9 and R3.10 added.** MTP/speculative
  decoding and Flash-Next were tacitly implied from the start and my draft-1 reconstruction missed both —
  MTP appeared only as a single-user characteristic and in the §4 not-required list, Flash-Next only as a
  TBC. Neither had a §3 requirement, so nothing measured them. MTP is now in scope for both profiles and
  must be measured on/off per profile rather than assumed; the 2026-09-09 MTP-gate failure predates ROCm
  10.0 and has never been retested.
- 2026-09-19 (lead, later): R2.2 narrowed — design points are **8 × 192K q8_0 KV** and **4 × 256K q8_0 KV**.
  f16 at 8 × 128K was run and kept as a data point, but was Claude's choice for corpus comparability, not a
  requirement-derived one. Note Q8_0 is also the MODEL quant (R2.5) — the two must not be conflated.
- 2026-09-19 (lead): **R2.7 added — two build profiles, single-user and multi-user, maintained separately** because the patch
  sets are mutually exclusive between modes; each is gated and promoted on its own axis. R2.6 superseded and §4 amended
  accordingly. R3.3's >300 tok/s TBC resolved: not recoverable, never logged, discarded during the single-user work; the
  multi-user baseline is re-established by the current campaign. Removed from the §7 TBC list.
- 2026-09-09 draft 1 (Claude, after the lead's correction). Written after two days of work that optimised out-of-scope cells; see
  `reports/2026-09-09-night-report.md` for what that cost.
