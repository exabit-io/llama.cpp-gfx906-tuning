# gfx906 ISA re-read against the 2026-09-19/20 measurements

Internal analysis, 2026-09-20. Read-only review of

- `reference/amd-vega-7nm-isa-gfx906.txt` — "Vega" 7nm Instruction Set Architecture (296 PDF pages), cited below as **ISA §x.y (line N)**
- `reference/llvm-amdgpu-backend-user-guide.txt` — User Guide for AMDGPU Backend, LLVM 22.0.0git, cited as **LLVM (line N)**
- `reference/amd-infinity-fabric-link-user-guide-56978.txt` — cited as **IFL**
- `reference/amd-instinct-system-tuning-guide-57286.txt` — cited as **STG**

Companion to `ISA-NOTES.md` (which this does not repeat). No GPU workload was run; every number below is
either quoted from these documents, quoted from the measured facts supplied for this review, or arithmetic
over the two. Labels used throughout:

- **DOCUMENTED** — stated in one of the four documents, with the citation.
- **INFERENCE** — my arithmetic or reasoning over documented facts plus the measured facts.
- **NOT IN THESE DOCUMENTS** — the honest answer; the documents do not contain it.

Hardware under discussion: four Vega 20 / gfx906 dies, 64 CU per die at 1730 MHz, 32 GB HBM2 per die, one
XGMI ring (0b-0e-1e-1b), ROCm 10.0, llama.cpp b11067, Qwen3.8-27B-Q8_0, `-sm tensor` over four dies, flash
attention on, q8_0 KV.

---

## Executive summary — the five most actionable conclusions

**1. The depth term of the decode model is 9.6x off the memory roofline. It is the single biggest
opportunity on this machine, and the arithmetic points at byte amplification, not bandwidth.**
The measured `94.6 ms per million total KV tokens` corresponds to only **90 GB/s per die** of KV traffic
(8.5 KiB/token/die x 1e6 tokens = 8.70 GB per die, in 94.6 ms), against 880.6 GB/s measured streaming read —
**10.2% of roofline**, and 4.1x worse than this project's own single-stream FA-vec figure of 369 GB/s/die at
the same head size. The VALU roofline for the same work is 3.5-7.1 ms, so the kernel is not compute bound
either. The leading explanation is that the depth-dependent path moves ~4x the necessary bytes: the measured
compute buffer is **6.3-12.5 KiB per token per die**, the same order as the 8.5 KiB KV itself, and
8.5 + 2x12.5 = 33.5 KiB/token/die read back at 94.6 ms/Mtok works out to **354 GB/s/die** — within 4% of the
369 GB/s the FA-vec kernel was separately measured to sustain. If that is right, the kernel is *at* the
roofline while carrying 4x the payload, and the patch to look for is elimination of split-K / dequant scratch
traffic, not a faster inner loop. **Discriminating test:** vary the FA split-K / `GGML_CUDA_FA_ALL_QUANTS`
configuration and see whether the 94.6 coefficient tracks the compute-buffer-per-token figure. See §J.1.

**2. The 3.49 ms per-active-slot term is not launch overhead — it is unamortised weight/activation
re-loading, and it is therefore cap-insensitive work worth attacking at 125 W.** Kernel count per decode
step does not scale with slot count in llama.cpp (one graph, batch = S rows), so per-dispatch cost cannot
produce a linear slot term. A full weight pass is 6.834 GB/die / 880.6 GB/s = 7.76 ms; 3.49 ms is **45% of a
full weight re-read per additional row**. That matches the counter data already in `ISA-NOTES.md` §3
(memory-read instructions per wave 29 -> 96 for batch 1 -> 8, i.e. ~33% of an unamortised re-read per extra
column). The lever is more columns per loaded weight block and fewer instructions between load and
accumulate — the same lever that already returned +62% at 12 columns. See §A.

**3. The prefill -20.2% / decode -10.7% split under a power cap is exactly what a VALU-bound phase next to a
half-memory-bound phase predicts, and the ratio gives a number to design against: 0.107/0.202 = 53% of
decode time is frequency-scalable, 47% is not.** Prefill attention runs at roughly 36% of the packed-fp16
VALU peak (INFERENCE, §D) and its cap penalty is uniform to 0.5 percentage points across four very different
configurations — the signature of a single clock-limited resource. gfx906 has **no MFMA / matrix cores**
(DOCUMENTED, §E), so there is no instruction class that buys prefill out of the VALU roofline; only packed
fp16 (`V_PK_FMA_F16`, `V_DOT2_F32_F16`, 2x) and int8 (`V_DOT4_I32_I8`, 4x) do. **Prediction to test:** refit
the decode model at the 125 W cap. The 3.49 ms slot coefficient should rise ~20%, the 94.6 ms KV coefficient
should rise much less. If it does not, the model's term attribution is wrong.

**4. Under a 125 W cap, only ~10 W per die sits above the 115 W memory-streaming floor, so rank patches by
instructions issued per useful MAC, not by occupancy.** The ISA's own advice is the right heuristic: "It is
recommended to use the 32-bit encoding whenever possible" (ISA §6.1, line 1872). Ranked list in §J. The top
four: (i) delete redundant bytes (cap-invariant, §J.1); (ii) `V_DOT4_I32_I8` / `V_PK_*` everywhere the
quantisation allows, 4x/2x fewer lane-instructions per MAC; (iii) 32-bit encodings and `GLOBAL_*` with
`SADDR` + 13-bit immediate `OFFSET` instead of 64-bit VGPR addresses; (iv) push everything wave-uniform into
SALU/SMEM (`S_LOAD_DWORDX16`), which touches 1 value per wave instead of 64. **Do not raise occupancy** — the
project already measured -20 to -36% from doing so, and the ISA explains why (latency is hidden by
independent work *within* a wave as much as across waves; VGPRs allocate in blocks of 4, ISA §3.6.4 line 918).

**5. Two things in the reference set are simply not there, and one measured number may be
self-contradictory.** `amd-infinity-fabric-link-user-guide-56978` is an **MI100 bridge installation guide**
with no bandwidth, latency or topology data of any kind (§H) — it cannot support any XGMI claim about Vega 20.
The ISA contains no bandwidth figure, no clock-domain description, no power model and no instruction
throughput table (greps in §B, §C, §E) — every peak rate in this project's reports comes from outside these
documents. And **the 715 GB/s copy figure needs its convention pinned down**: if it counts only the bytes
copied in one direction, the implied DRAM traffic is 1430 GB/s, which exceeds the 1024 GB/s theoretical HBM2
peak and is impossible. See §B.

---

## A. What produces the 3.49 ms per-active-slot term

Model under review: `t = 17.6 ms + 3.49 ms x slots + 94.6 ms x KV_Mtok`, 0.5% mean error over 12 cells.

### A.0 The model validates against an independent trace

At S = 1 and negligible KV the model gives 17.6 + 3.49 = **21.1 ms per token**. The M1 kernel trace
(`reports/2026-09-08-m1-kernel-trace.md`, quoted in `ISA-NOTES.md` §3b) measured a single-stream four-die
token at **21.5 ms**, decomposed as 11.2 ms matrix-vector + 3.4 ms RCCL (128 kernels x 27 us) + 5.8 ms small
kernels at the ~4 us dispatch floor + 1.1 ms idle. That is a 2% agreement between a batched-server regression
and a single-stream kernel trace taken on a different build. **INFERENCE:** the constant term is real and is
roughly 9.2 ms of launch-and-allreduce overhead plus ~11 ms of matrix-vector work — which is why HIP graphs
are worth 7% and why kernel fusion is a first-class lever.

### A.1 Dispatch and wavefront-launch overhead is ruled out as the cause of the slot term

**DOCUMENTED.** Per-dispatch and per-wavefront setup costs on gfx906 are real and the documents enumerate them:

- Up to 16 User SGPRs are written by CP and apply to all wavefronts of the grid; System SGPRs (workgroup IDs,
  scratch wavefront offset) are written per wavefront by ADC/SPI (LLVM line 4014 ff., Table 78).
- gfx906 uses the **unpacked** work-item ID method — a separate VGPR per enabled dimension (LLVM Table 79;
  "Packed work-item IDs" appears in the Target Properties column only for gfx125x/gfx94x, LLVM lines 505-586).
- gfx906 has **no kernarg preload**: that feature is listed as a target feature for gfx90a/gfx942/gfx950 only
  (LLVM lines 538-582, and "Preloaded Kernel Arguments … On hardware that supports this feature", line 4130).
- "On dGPU over XGMI or PCIe the kernarg backing memory is allocated in host memory accessed as MTYPE UC
  (uncached) to avoid needing to invalidate the L2 cache" (LLVM line ~5170). So each wave's `s_load` of its
  kernargs is an uncached host-memory read that bypasses L2.
- "CP invalidates the L1 cache at the start of each kernel dispatch" (LLVM line ~5168).

**INFERENCE — but it does not explain the slot term.** In llama.cpp a decode step for S slots is one graph
with batch = S rows; the number of kernel dispatches is essentially independent of S. A cost that scales
linearly and steeply in S therefore cannot be per-dispatch. Quantitatively: 3.49 ms at a ~4 us dispatch floor
would require ~870 extra kernel launches per additional slot, which no part of the decode graph does. The
per-dispatch costs above belong to the **17.6 ms constant**, where the trace puts 5.8 ms of them, and they are
what HIP graphs and kernel fusion attack.

### A.2 What the arithmetic does support: unamortised loads per column

**INFERENCE.** Weights per die are 6517 MiB = 6.834e9 bytes. One full pass at the measured 880.6 GB/s is
**7.76 ms**. The measured slot term is 3.49 ms = **45% of a full weight pass per additional row.** Perfect
batching would make the marginal cost of a row almost free (weights read once, S columns of activations
against them); 45% says the weight or activation blocks are being re-fetched for roughly every second column.

This is independently corroborated by the counters already recorded in `ISA-NOTES.md` §3: going from batch 1
to batch 8 in MMVQ raised memory-read instructions per wave from 29 to 96. Fully amortised, batch 8 would
need ~29 + 8; measured is 96, i.e. ~9.6 extra load instructions per extra column against 29 for the first —
**33% of an unamortised re-read per column.** 45% (bytes, four dies, server) and 33% (instructions, one die,
microbenchmark) are the same phenomenon measured two ways.

The ISA-level reasons this costs what it does:

- **VALU issue is not the bottleneck.** ISA §2 (line 456) and the counters both say so: VALU busy was 20-30%
  at batch 1 and 8. What the SIMDs wait on is `VM_CNT` (ISA §4.4, line 1230: "VM_CNT … Incremented every
  time a vector-memory read or write … is issued. Decremented for reads when the data has been written back
  to the VGPRs"). Each redundant load lengthens the `S_WAITCNT vmcnt` chain.
- **Latency hiding is limited by VGPR budget, and the granularity is coarse.** "A wavefront can be allocated
  16 to 102 SGPRs, in units of 16 GPRs" (ISA §3.6.2, line 894); "VGPRs are allocated in groups of four
  Dwords" (ISA §3.6.4, line 918); LLVM confirms for GFX6-GFX9 `max(0, ceil(vgprs_used / 4) - 1)` (LLVM line
  3546). A 256-VGPR file per lane shared across resident waves plus a granularity of 4 means occupancy moves
  in coarse steps as the accumulator set grows with columns.
- **`S_WAITCNT 0` is mandatory after FLAT.** "Since the data for a FLAT load can come from either LDS or the
  texture cache … the only sensible S_WAITCNT value to use after FLAT instructions is zero" (ISA §9.2.2,
  line 3862), and a FLAT instruction "increments both VM_CNT and LGKM_CNT and [is] not considered done until
  both have been decremented" (ISA §9.2, line 3840). Any kernel using FLAT instead of GLOBAL for its weight
  loads cannot pipeline partial completions at all. **Survey item: check the emitted ISA for `flat_load_*`
  where `global_load_*` would do.**

### A.3 The other mechanisms the question asks about, with what the ISA says

- **Scalar vs vector setup per sequence.** "All kernel control flow is handled using scalar ALU instructions"
  and SALU "operates on one value per wavefront" (ISA §2, line 456). Constraint that shapes the code:
  "At most one SGPR can be read per instruction, but the value can be used for more than one operand"
  and "At most one literal constant can be used, and only when an SGPR or M0 is not used as a source"
  (ISA §6.2.1, line 1920). **INFERENCE:** per-sequence scalars (KV base pointer, sequence length, slot
  offsets) can be held in SGPRs and broadcast for free, but only one per instruction — so a kernel indexing
  by slot tends to spill slot metadata into VGPRs, which both costs registers and forces VALU address math.
- **LDS allocation and occupancy.** 64 kB per CU, "32 banks, each with 512 entries of 4 bytes" (ISA §2.2.1,
  line 503). "LDS space is allocated to a work-group or wavefront in contiguous blocks of 128 Dwords on
  128-Dword alignment" (ISA §3.6.5, line 924) — i.e. a **512-byte granule**, confirmed by LLVM's
  `roundup(lds-size / (128 * 4))` for GFX7-GFX12 (line 3834). **INFERENCE:** per-slot LDS staging rounds up
  to 512 B per workgroup; with many slots this is a real occupancy tax, and the project already measured that
  LDS staging of activations *lost* 16-52% because of the barrier it requires.
- **Barriers.** "Up to 16 wavefronts (1024 work-items) can be combined into a work-group … the `S_BARRIER`
  instruction can be used to force each wavefront to wait" (ISA §4.3, line 1180). LLVM adds that LDS has
  "multiple request queues shared by the SIMDs of a CU", so "a `s_waitcnt lgkmcnt(0)` is required to ensure
  synchronization between LDS operations and vector memory operations between wavefronts of a work-group"
  (LLVM line ~5104). Every barrier therefore costs a full LGKM drain, not just a rendezvous.
- **How concurrent sequences share CUs.** **NOT IN THESE DOCUMENTS** at the level asked. The ISA describes
  wavefronts, workgroups and the CU; it says nothing about how a runtime multiplexes independent request
  streams, and nothing about HSA queues or barrier packets. What it does say, and what matters for ragged
  batches, is: "This GPU does no optimization when EXEC = 0. The shader hardware executes every instruction,
  wasting instruction issue bandwidth. Use CBRANCH or VSKIP to rapidly skip over code when it is likely that
  the EXEC mask is zero" (ISA §3.3, note at line 678). **INFERENCE:** slots with unequal KV depth that are
  handled by a uniform padded loop burn issue slots — and therefore power — on dead lanes. At a 125 W cap
  that is pure waste. See §J.9.

---

## B. Vega 20 HBM2 peak vs the measured 880.6 / 715 GB/s

### B.1 The documents contain no bandwidth figure at all

**NOT IN THESE DOCUMENTS.** `grep -ci` over `amd-vega-7nm-isa-gfx906.txt`: `HBM` = 0 hits, `GB/s` = 0,
`TB/s` = 0, `memory clock` = 0. The eight hits for "bandwidth" are all qualitative (LDS "one order of
magnitude higher effective bandwidth than direct, uncached global memory", ISA §10, line 3978). The LLVM guide
has zero hits for `HBM` and `bandwidth`. The STG quotes PCIe numbers for an EPYC 7002 / MI50 board and its
three bandwidth figures are images that did not survive text extraction (STG §3, pages 9-11). **Any peak
HBM2 number used in this project comes from outside this reference set** and should be cited as such.

### B.2 The comparison, with the peak stated as external

**External fact (not from these documents):** Vega 20 carries a 4096-bit HBM2 interface at 2.0 Gbps/pin =
**1024 GB/s** nominal; this is the figure `ISA-NOTES.md` already uses.

| Quantity | Measured | Fraction of 1024 GB/s |
|---|---|---|
| Read | 880.6 GB/s | 86.0% |
| Copy | ~715 GB/s | 69.8% |
| Read / copy ratio | 1.23 | — |

**INFERENCE.** 86% of nominal on a pure read stream is the expected result for HBM2 and needs no exotic
explanation: refresh, `tRC`/`tFAW` bank-cycle limits and row-activate overhead account for roughly that much
on any DRAM. The read-vs-copy gap has a standard cause the documents do not describe but which is structural:
a copy interleaves reads and writes on the same DQ bus, paying read-to-write and write-to-read turnaround on
every direction change, and it must sustain two streams of row activations instead of one.

### B.3 Flag: the copy number may be self-contradictory

If 715 GB/s counts **only the bytes copied** (i.e. the destination bytes), the actual DRAM traffic is
2 x 715 = **1430 GB/s, which exceeds the 1024 GB/s theoretical peak and is therefore impossible.** The figure
is only consistent under the BabelStream-style convention where the reported rate already counts both the
read and the write. Before this number is used in any roofline argument it must be pinned to a convention,
because the two readings differ by 2x. **Action: check the probe source in `bench/`.**

### B.4 Channel interleaving, granularity and ECC

- **Channel interleaving — DOCUMENTED, qualitatively.** "The L2 cache has independent channels to service
  disjoint ranges of virtual addresses. Each CU has a separate request queue per channel. Therefore, the
  vector and scalar memory operations performed by wavefronts executing in different work-groups … can be
  reordered relative to each other" (LLVM, GFX6-GFX9 memory model, line ~5131). The number of channels and
  the interleaving stride are **NOT IN THESE DOCUMENTS**.
- **Access granularity — DOCUMENTED.** "For Dword or larger reads or writes, the two LSBs of the byte-address
  are ignored, thus forcing Dword alignment" (ISA §8.1.7, line 2996). Vector loads come in
  `DWORD/DWORDX2/DWORDX3/DWORDX4` widths (ISA §12.18.3, global opcode table from line 11251), and D16 /
  D16_HI variants move 16 bits per work-item (ISA §8.1.6, line 2985). The DRAM burst length and the L1/L2
  line size are **NOT IN THESE DOCUMENTS**.
- **ECC — partially DOCUMENTED, and it is the wrong ECC.** gfx906 lists `sramecc` as a target feature
  (LLVM line 505; code-object IDs `gfx906:sramecc-:xnack-`, line 2030). `sramecc` covers **on-chip SRAM**,
  not the HBM array; it is a code-object compatibility switch ("generate code that can only be loaded and
  executed in a process that has a matching setting for SRAMECC", LLVM line 946). Whether HBM ECC is enabled
  on these particular Radeon Pro Vega II dies, and what it would cost, is **NOT IN THESE DOCUMENTS**. Note
  the trap in LLVM Table 23 (line 803): under the `gfx9-generic` target, "sramecc is not available on
  gfx906" — a generic build silently loses it, along with `v_dot4_i32_i8`, `v_dot8_i32_i4`, `v_fmac_f32`
  and the rest of the dot family (lines 806-825).

---

## C. Why a power cap does not reduce HBM bandwidth while the die draws 115 W streaming

### C.1 What the documents actually say about clocks and power

**NOT IN THESE DOCUMENTS** — and this is worth stating plainly, because it is tempting to assume an ISA
manual describes the clock tree. It does not. Greps over the ISA: `sclk` = 0, `mclk` = 0, `DPM` = 0,
`voltage` = 0, `power` = 2 hits. The LLVM guide has no power or clock-domain content either.

The ISA's only statement about clock domains is a good one, however, and it establishes that **at least two
exist**:

> **7.2.4. S_MEMTIME** — This instruction reads a 64-bit clock counter into a pair of SGPRs …
> **7.2.5. S_MEMREALTIME** — This instruction reads a 64-bit "real time-counter" … The time value is from a
> clock for which the frequency is **constant (not affected by power modes or core clock frequency changes)**.
> — ISA §7.2.4-7.2.5, lines 2508-2515

So the ISA distinguishes a **core clock that power modes change** from a **constant-rate reference clock**.
That is the entire documented basis.

### C.2 Which domain the 999 MHz floor is, and why the memory system is immune

**INFERENCE, well supported.** The 999 MHz DPM floor is the **shader/core clock** — the domain `S_MEMTIME`
counts and the one ISA §7.2.5 says "power modes" change. Three lines of support:

1. The measured facts state it as "sclk pinned at the 999 MHz DPM floor", and sclk is the shader clock.
2. Fact 3 says bandwidth is *invariant* from 110 W down to 50 W while sclk is pinned at that floor. A single
   clock domain cannot both be pinned and govern a quantity that does not change; the memory rate must live
   elsewhere.
3. Fact 4 says prefill loses 20.2% and decode 10.7% from the same cap. A cap that touched the memory clock
   would have shown up in fact 3's streaming test.

**INFERENCE on the mechanism:** the per-die power cap is enforced by the SMU against the shader-core rail.
The HBM2 stack, its PHY, and the on-die fabric run on separate rails at frequencies the graphics DPM table
does not scale, so a streaming workload's DRAM and PHY power is a **fixed ~115 W floor** that the cap cannot
reach. That is exactly the shape of fact 3: bandwidth constant, sclk at its floor, die at 115 W even when the
cap is set to 50 W — i.e. **below ~115 W the cap is not binding at all; the DPM floor is.** The complete set
of Vega 20 clock domains (gfxclk, fclk, mclk, socclk, and their rails) is **NOT IN THESE DOCUMENTS**.

### C.3 A log-reading trap worth writing down

The shader-clock DPM floor is **999 MHz** and Vega 20's HBM2 memory clock is **~1000 MHz** (external fact).
These are different domains that happen to be numerically adjacent. Any log line reading "1000 MHz" must be
attributed to a domain before it is used, or a memory-clock reading will be mistaken for the sclk floor and
vice versa.

---

## D. Prefill -20.2% vs decode -10.7% under the cap

### D.1 The units each phase saturates

**DOCUMENTED — what the instructions are.**

| Phase | Shape | Dominant instruction classes (gfx906) |
|---|---|---|
| Prefill | large GEMM, thousands of rows | `V_DOT4_I32_I8` (quantised weights, ISA §12.10 op 40, line 7949) through MMQ tiles; `V_PK_FMA_F16` / `V_DOT2_F32_F16` in flash attention (ISA §6.7 line 2326, §12.10 op 35 line 7943); `DS_READ_B128` / `DS_WRITE_B128` for tile staging (ISA §12.13, lines 10128-10140); `BUFFER_LOAD_DWORDX4` |
| Decode | batched mat-vec, S rows | `GLOBAL_LOAD_DWORDX4` streaming weights; `V_DOT4_I32_I8` on far fewer MACs per loaded byte; `S_WAITCNT vmcnt` chains (ISA §4.4, line 1230) |

The distinction is arithmetic intensity, and the ISA does not need to say anything more than it already does:
prefill reuses each loaded weight across many rows and therefore runs out of **VALU issue** first; decode
reuses each loaded weight across S rows only and runs out of **VMEM** first.

### D.2 The asymmetry is quantitatively what a 53/47 split predicts

**INFERENCE.** Take the cap as reducing the effective shader clock by a factor `k`. A phase whose time is
entirely frequency-scalable loses `k`; a phase that is a fraction `f` frequency-scalable and `(1-f)` DRAM-bound
loses `f x k`, since fact 3 establishes the DRAM part is cap-invariant. Then:

    prefill:  k          = 20.2%
    decode:   f x k      = 10.7%   ->   f = 0.107 / 0.202 = 0.53

So **53% of a decode step is frequency-scalable VALU/issue work and 47% is cap-invariant memory work.** That
53% is where a power-capped decode optimisation has to land.

Two independent sanity checks on the "prefill is VALU-bound" half:

- **The uniformity is the proof.** Fact 4 says prefill's -20.2% holds to within 0.5 percentage points across
  four very different configurations. Only a single shared resource that scales with clock does that. A mixed
  compute/memory phase would have shown spread, because the mix differs by configuration.
- **Prefill attention is near its roofline.** The prefill fit `1/(0.000732 + 0.004856 x depth_Mtok)` gives a
  depth coefficient of **4.856 ms per token per Mtok of depth**. With head_dim 256 (§I), 24 query heads,
  16 KV layers, QK plus PV, and causal masking (average attended length ~ depth/2), the attention MACs per
  query token at depth `D` are `16 x 24 x 256 x 2 x (D/2) = 196608 x D/2`. At `D = 1e6`: 9.83e10 MACs =
  1.966e11 FLOP across the four dies, so **4.92e10 FLOP per die**, delivered in 4.856 ms =
  **10.1 TFLOP/s per die** sustained in prefill attention. Against a packed-fp16 peak of 28.3 TFLOP/s that is
  **36%**; against the fp32 FMA peak of 13.8 TFLOP/s it is **73%**. This agrees with the
  30%-of-packed-fp16 figure already in `ISA-NOTES.md` §2, within the factor-of-2 uncertainty of the
  causal-masking assumption. Either way prefill sits on the VALU roofline, which is why the cap bites it
  uniformly.

### D.3 The consequence for patch selection

**INFERENCE.** Because gfx906 has no MFMA (§E), the only way to move prefill is fewer lane-instructions per
MAC: `V_DOT4_I32_I8` (4x over scalar int8), `V_PK_FMA_F16` / `V_DOT2_F32_F16` (2x over unpacked fp16), and
tile shapes that keep the VALU fed without extra LDS traffic — which is precisely the per-target MMQ tile
table that returned +33%. A power-capped prefill gains from a patch in direct proportion to the
lane-instructions it removes, and gains nothing from bandwidth work.

For decode the reverse holds for 47% of the time, and for the other 53% the instruction-count argument
applies with full force.

---

## E. Quantised dot products, packed math, and the MFMA question

### E.1 The dot-product family — exact names and operand forms

**DOCUMENTED.** ISA preface "New Instructions" (page 9, lines 318-328) lists them as new in Vega 7nm, and
ISA §12.10 (VOP3P opcode table, lines 7943-7962) gives the semantics verbatim:

| Op | Name | Semantics as printed in the ISA |
|---|---|---|
| 35 | `V_DOT2_F32_F16` | `D.f32 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f32` |
| 38 | `V_DOT2_I32_I16` | `D.i32 = S0.i16[0] * S1.i16[0] + S0.i16[1] * S1.i16[1] + S2.i32` |
| 39 | `V_DOT2_U32_U16` | `D.u32 = S0.u16[0] * S1.u16[0] + S0.u16[1] * S1.u16[1] + S2.u32` |
| 40 | `V_DOT4_I32_I8` | `D.i32 = S0.i8[0]*S1.i8[0] + S0.i8[1]*S1.i8[1] + S0.i8[2]*S1.i8[2] + S0.i8[3]*S1.i8[3] + S2.i32` |
| 41 | `V_DOT4_U32_U8` | same, unsigned |
| 42 | `V_DOT8_I32_I4` | eight 4-bit products accumulated into `S2.i32` |
| 43 | `V_DOT8_U32_U4` | same, unsigned |

Note there is **no `V_DOT2_F16_F16`** and no bf16 form; the fp16 dot accumulates in fp32 only.

**Operand forms — DOCUMENTED.** All seven use the **VOP3P** microcode format: 64-bit encoding, fields
`VDST`, `SRC0/1/2`, `NEG_HI[10:8]`, `OPSEL[13:11]`, `OPSEL_HI2[14]`, `OPSEL_HI[60:59]`, `NEG[63:61]`, `CLMP[15]`
(ISA §13.3.6, Table 76, lines 14012-14060). `SRC0/1/2` may each be a VGPR (256-511), an SGPR (0-101), an
inline constant, or a literal — subject to ISA §6.2.1: **at most one SGPR per instruction** and no literal
when an SGPR is used (line 1920). `OPSEL`/`OPSEL_HI` select the low or high 16 bits of each source
independently, which is how a kernel picks halves of a packed register without a separate shift.

**Throughput — NOT IN THESE DOCUMENTS.** The ISA specifies semantics and encodings only; it contains no
cycle counts, no issue rates, and no throughput table (`grep -i throughput` returns only the LDS bank-conflict
discussion at line 4009). `ISA-NOTES.md`'s peak figures (int8 via `V_DOT4` = 56.7 TOPS, int4 via `V_DOT8` =
113 TOPS, packed fp16 28.3 TFLOP/s per die at 1730 MHz with 64 CU) are derived from the assumption of one
wave64 VOP3P per SIMD per 4 cycles — **an external assumption, not documented here.**

### E.2 Packed math

**DOCUMENTED.** ISA §6.7 (line 2326): "Vega adds support for packed math, which performs operations on two
16-bit values within a Dword as if they were separate threads… Packed math uses the instructions below and
the microcode format 'VOP3P'. This format adds op_sel and neg fields for both the low and high operands, and
removes ABS and OMOD."

Full list (ISA §6.7 line 2340, §12.10 opcodes 0-18 lines 7820-7918):
`V_PK_MAD_I16`, `V_PK_MUL_LO_U16`, `V_PK_ADD_I16`, `V_PK_SUB_I16`, `V_PK_LSHLREV_B16`, `V_PK_LSHRREV_B16`,
`V_PK_ASHRREV_I16`, `V_PK_MAX_I16`, `V_PK_MIN_I16`, `V_PK_MAD_U16`, `V_PK_ADD_U16`, `V_PK_SUB_U16`,
`V_PK_MAX_U16`, `V_PK_MIN_U16`, `V_PK_FMA_F16`, `V_PK_ADD_F16`, `V_PK_MUL_F16`, `V_PK_MIN_F16`, `V_PK_MAX_F16`.

Plus the mixed-precision VOP3P trio, which the ISA is careful to say are **not** packed math: `V_MAD_MIX_F32`
(op 32), `V_MAD_MIXLO_F16` (33), `V_MAD_MIXHI_F16` (34) — "perform a single MAD operation on a mixture of 16-
and 32-bit inputs… listed here because they use the VOP3P encoding" (ISA §6.7 note, line 2348). For these,
"the NEG_HI field acts instead as an absolute-value modifier" and OPSEL selects `src[31:0]`, `src[15:0]` or
`src[31:16]` per operand (ISA §12.10 ops 32-34).

Two gfx906-specific additions outside VOP3P that matter for instruction-count work (ISA §12.7, VOP2):
`V_FMAC_F32` (opcode 59, line 6338) — "VOP2 version of V_FMA_F32 with 3rd src VGPR address is the vDst" —
and `V_XNOR_B32` (opcode 61, line 6342). `V_FMA_F32` (VOP3A op 459, line 8117) and `V_MAD_F32` (VOP3A op 449,
line 8027) are 64-bit encodings; `V_FMAC_F32` is 32-bit. See §J.3.

There is also a Vega packing helper on the scalar side: `S_PACK_{LL,LH,HH}_B16_B32` (ISA preface, line 288).

### E.3 MFMA: confirmed absent

**DOCUMENTED — your belief is correct.**

- `grep -ci mfma amd-vega-7nm-isa-gfx906.txt` -> **0**. `grep -i matrix` -> 0 hits in any instruction context.
- There is no AGPR / accumulation register file: `compute_pgm_rsrc3.ACCUM_OFFSET` and the
  `amdgpu-agpr-alloc` attribute are documented as "only relevant on targets with AGPRs which support
  accum_offset (**gfx90a+**)" (LLVM lines 1453, 3871), and `compute_pgm_rsrc3` for GFX9 baseline has no such
  field.
- The VGPR-count encoding confirms the split: GFX6-GFX9 uses `vgprs_used 0..256`, while GFX90A/GFX942 use
  `0..512` with `vgprs_used = align(arch_vgprs, 4) + acc_vgprs` (LLVM lines 3546-3553). gfx906 has one
  256-entry architectural file and no accumulator file.
- The generation table files gfx906 under "GFX9" / Vega 7nm and gfx908 separately under **"CDNA 1"**
  (LLVM lines 12455-12458), with reference `[AMD-GCN-GFX908-CDNA1] AMD Instinct MI100 Instruction Set
  Architecture` (line 13118) — MI100 is where MFMA was introduced.

**Consequence:** every `mma`-family kernel in ggml-cuda is dead code on this machine, and the maximum
arithmetic density available is `V_DOT8_I32_I4` (8 MACs/lane/instr, unusable because llama.cpp quantises
activations to 8 bits), then `V_DOT4_I32_I8` (4), then packed fp16 (2), then fp32 FMA (1).

---

## F. What could make cost step at batch 8 / 16 / 24

### F.1 The honest answer first

**NOT IN THESE DOCUMENTS as a hardware boundary.** No documented gfx906 granularity is 8. The granularities
the documents do give are:

| Granularity | Value | Citation |
|---|---|---|
| Wavefront | 64 work-items | ISA §1.1 Table 1 (line 420), §2 (line 456) |
| VGPR allocation | groups of **4** Dwords | ISA §3.6.4 (line 918); LLVM `ceil(vgprs_used/4)-1` for GFX6-GFX9 (line 3548) |
| SGPR allocation | units of **16** | ISA §3.6.2 (line 894) |
| LDS allocation | **128 Dwords** = 512 B, 128-Dword aligned | ISA §3.6.5 (line 924); LLVM `roundup(lds/(128*4))` (line 3834) |
| LDS banks | **32** banks x 512 entries x 4 B | ISA §2.2.1 (line 503) |
| LDS read dispatch | "over **four** cycles in waterfall" | ISA §10.1 (line 3995) |
| Dot-product width | 2 / **4** / 8 sub-words | ISA §12.10 (lines 7943-7962) |
| Address alignment | Dword (2 LSBs ignored) | ISA §8.1.7 (line 2996) |

So a step at 8 is not a register-file or LDS boundary. It is 2 x the VGPR granule and 8 x the Dword.

### F.2 What the steps most plausibly are

**INFERENCE, in decreasing confidence.**

1. **The batch-8 step is software, not ISA.** llama.cpp selects MMVQ for batch <= 8 and MMQ above; the cost
   discontinuity at 9 is a kernel switch. Nothing in the ISA is involved. The 16 and 24 steps are then MMQ
   tile-width quantisation: J a multiple of 8, so batches of 9-16 pay for 16 columns and 17-24 pay for 24.
   This is the explanation the project's own data already supports (`ISA-NOTES.md` §2: batch 9 columns runs
   the MMQ 16-tile).
2. **Why 8 is the natural software constant, in ISA terms.** A Q8_0/Q8_1 block is 32 elements.
   `V_DOT4_I32_I8` consumes 4 bytes per instruction, so **one quantisation block = exactly 8 `V_DOT4`
   instructions**. A column group of 8 lets the compiler keep the 8 dot instructions of one loaded block
   resident with one accumulator per column, and the unroll falls out at 8. Multiples of 8 preserve that.
3. **VGPR granularity amplifies the step.** Accumulators grow one VGPR per (row x column); with the granule
   at 4 VGPRs (ISA §3.6.4), every 4 added accumulators can bump the allocation and drop a wave from the SIMD.
   That turns a smooth column sweep into a staircase, and it is why the project measured 16 columns needing
   113 VGPRs and losing on a single die.
4. **LDS tiles quantise in 512 B.** A tile whose column dimension is not a multiple of the granule wastes
   allocation and therefore occupancy (ISA §3.6.5). 512 B = 8 x 64 B, which again favours 8.
5. **Bank conflicts are a step function, not a gradient.** "If … more than one access attempt is made to the
   same bank at the same time, a bank conflict occurs… hardware prevents the attempted concurrent accesses
   … by turning them into serial accesses" and LDS ops "can complete in as little as two cycles, or take as
   many [as] 64 cycles" (ISA §10.1 line 4009, §10.3.3 line 4117). With 32 banks, a tile stride that is a
   multiple of 32 Dwords collapses to one bank — so tile widths flip between conflict-free and 32-way
   serialised at specific values rather than degrading smoothly.

**Not supported:** wave64 lane structure alone does not produce a step at 8; 64 is the only lane granularity
the ISA gives, and the 4-cycle cadence that produces 16-lane behaviour in practice is **NOT IN THESE
DOCUMENTS** (the closest statement is the LDS waterfall at ISA §10.1 line 3995).

---

## G. DPP, ds_swizzle, ds_permute and cross-lane reductions

### G.1 What each mechanism is

**DOCUMENTED.**

- **DPP** (ISA §13.3.9, Table 80, line 14240): "Data Parallel Primitives. This is a second dword which can
  follow **VOP1, VOP2 or VOPC** instructions (in place of a literal constant) to control selection of data
  from other lanes." Fields: `SRC0[39:32]`, `DPP_CTRL[48:40]`, `BC[51]` (bounds control), per-source
  `NEG`/`ABS`, `BANK_MASK[59:56]`, `ROW_MASK[63:60]`.
- **`DS_SWIZZLE_B32`** (ISA §12.13.1, line 10140): "Dword swizzle, **no data is written to LDS memory**.
  Swizzles input thread data based on offset mask and returns; note does not read or write the DS memory
  banks. Note that reading from an invalid thread results in 0x0." Modes: FFT (offset >= 0xe000), rotate
  (0xc000-0xdfff), full sharing within groups of 4 (offset[15]=1), and xor/or/and masking within groups of 32
  (offset[15]=0).
- **`DS_PERMUTE_B32` / `DS_BPERMUTE_B32`** (ISA §10.3.3, Table 45, line 4190): "Forward permute. Does not
  write any LDS memory. `LDS[dst] = src0; returnVal = LDS[thread_id]`" and the backward form
  "`LDS[thread_id] = src0 … returnVal = LDS[dst]`", "where thread_id is 0..63."

### G.2 The documented DPP_CTRL modes for a reduction

**DOCUMENTED** — ISA Table 81, lines 14300-14345:

| Mode | Hex | Function |
|---|---|---|
| `DPP_QUAD_PERM*` | 000-0FF | "Permute of four threads": `pix[n].srca = pix[(n&0x3c)+ dpp_cntl[n%4*2+1 : n%4*2]].srca` |
| `DPP_ROW_SL*` | 101-10F | Row shift left 1-15; out of range -> `bound_cntl` |
| `DPP_ROW_SR*` | 111-11F | Row shift right 1-15; out of range -> `bound_cntl` |
| `DPP_ROW_RR*` | 121-12F | Row rotate right 1-15 |
| `DPP_WF_SL1` / `RL1` / `SR1` / `RR1` | 130 / 134 / 138 / 13C | Whole-wavefront shift/rotate by 1 |
| `DPP_ROW_MIRROR` | 140 | `pix[n].srca = pix[15-(n&f)].srca` |
| `DPP_ROW_HALF_MIRROR` | 141 | `pix[n].srca = pix[7-(n&7)].srca` |
| `DPP_ROW_BCAST15` | 142 | "Broadcast 15th thread of each row to next row" (`if (n>15) pix[n].srca = pix[n & 0x30 - 1].srca`) |
| `DPP_ROW_BCAST31` | 143 | "Broadcast thread 31 to rows 2 and 3" (`if (n>31) …`) |

A "row" is 16 lanes (the mirror and mask definitions make this explicit). **INFERENCE:** the canonical wave64
sum is therefore six DPP steps — `row_shr:1, 2, 4, 8` to reduce within each 16-lane row, then
`row_bcast:15` and `row_bcast:31` to fold the four row results — each attached to a VOP2 `V_ADD_F32`, with
the result in lane 63 and `V_READLANE_B32` to extract it.

### G.3 Why DPP beats LDS, in the documents' own terms

**INFERENCE from DOCUMENTED properties.** A DPP reduction and an LDS/`ds_bpermute` reduction differ on five
counts, all documented:

1. **No LDS allocation.** LDS is allocated in 512-byte granules per workgroup (ISA §3.6.5) out of 64 kB per
   CU (ISA §2.2.1). A DPP reduction allocates nothing, so it does not trade occupancy for the reduction.
2. **No LGKM_CNT wait.** `LGKM_CNT` is "Incremented by 1 for every LDS or GDS instruction issued …
   Decremented by 1 for LDS/GDS reads … when the data has been returned to VGPRs" (ISA §4.4, line 1240), and
   LDS and SMEM share the counter, so an `s_waitcnt lgkmcnt(0)` after a shuffle also waits on unrelated
   scalar loads. DPP is a VALU operand modifier and touches no counter at all.
3. **No crossbar, no bank conflicts.** LDS indexed ops "can complete in as little as two cycles, or take as
   many 64 cycles, depending upon the number of bank conflicts" (ISA §10.3.3, line 4117); reads are
   "dispatched over four cycles in waterfall" (ISA §10.1, line 3995). `ds_swizzle`/`ds_permute` avoid the
   *memory* but still go through the LDS pipeline and its request queues, which LLVM notes are "shared by the
   SIMDs of a CU" and reorderable (LLVM line ~5104).
4. **No barrier.** A cross-wave LDS reduction needs `S_BARRIER` (ISA §4.3, line 1197) plus
   `s_waitcnt lgkmcnt(0)` (LLVM line ~5106). DPP is intra-wave and needs neither. This is the measured
   mechanism behind "staging one LDS copy of the activations per block lost 16-52%: the barrier"
   (`ISA-NOTES.md` §3).
5. **No extra instruction.** DPP is a second dword on an instruction the kernel was going to issue anyway, so
   a DPP add is one 64-bit instruction where the LDS path is `ds_bpermute` + `s_waitcnt` + `v_add`.

### G.4 Restrictions that will bite — read these before writing a DPP reduction

**DOCUMENTED.**

1. **DPP attaches only to VOP1, VOP2 and VOPC** (ISA §13.3.9 description, line 14240). It therefore **cannot**
   be applied to `V_DOT*`, `V_PK_*`, `V_MAD_MIX*` (all VOP3P) or `V_FMA_F32`/`V_MAD_F32` (VOP3A). A reduction
   tree must be built from VOP2 ops — `V_ADD_F32` (op 1), `V_MAX_F32` (op 11), `V_FMAC_F32` (op 59) — not
   from the dot instructions. **This is the single most common design error to check for in a candidate patch.**
2. **Explicit exclusion list** (ISA §12.19.1, line 11530): `V_MADMK_F32`, `V_MADAK_F32`, `V_MADMK_F16`,
   `V_MADAK_F16`, `V_READFIRSTLANE_B32`, `V_SWAP_B32`, `V_CLREXCP`, all F64 conversions and F64 math, and all
   64-bit compares. Notably `V_ADD_F32`, `V_MAX_F32` and `V_FMAC_F32` are **not** excluded.
3. **Wait states are mandatory and the hardware will not insert them** (ISA §4.5, Table 8, lines 1290-1320):
   - `VALU writes VGPR` -> `VALU DPP reads that VGPR`: **2 wait states**
   - `VALU writes EXEC` -> `VALU DPP op`: **5 wait states**, with the note "ALU does not forward EXEC to DPP."
   A hand-written DPP chain with back-to-back dependent steps is a correctness bug, not a slowdown.
4. **`BC` (bound_ctrl) as the ISA words it**: "Bounds Control: 0 = do not write when source is out of range,
   1 = write" (Table 80, line 14252). The `DPP_ROW_SL/SR/WF_*` entries say "else use bound_cntl" for
   out-of-range lanes. Note the assembler spelling `bound_ctrl:0` conventionally sets this field to 1
   (write 0 instead of skipping) — the mnemonic is inverted relative to the field, so read the field, not
   the mnemonic.
5. **`ROW_MASK` and `BANK_MASK` affect the destination write only**: "Applies to the VGPR destination write
   only, does not impact the thread mask when fetching source VGPR data" (Table 80, lines 14256-14268). They
   cannot be used to suppress a source fetch.
6. **`DS_SWIZZLE_B32` cannot cross the 32-lane halves of a wave64.** Every branch of the pseudocode ends with
   `j |= (i & 0x20)` (ISA §12.13.1, lines 10200-10240) — the group-of-32 bit is preserved unconditionally.
   So `ds_swizzle` alone can never complete a wave64 reduction; the last fold must be DPP `row_bcast31`,
   `ds_permute`/`ds_bpermute`, or `v_readlane`.
7. **`ds_swizzle` returns 0 for inactive lanes**: "reading from an invalid thread results in 0x0"
   (ISA §12.13.1, line 10143). Convenient for a sum, wrong for a max or a min.

---

## H. XGMI: what these documents support, and what they do not

### H.1 The Infinity Fabric Link guide contains no performance data whatsoever

**NOT IN THESE DOCUMENTS — flag this loudly.** `amd-infinity-fabric-link-user-guide-56978` is an
**MI100 bridge installation guide**, 18 pages, of which roughly half is regulatory compliance text (FCC,
Industry Canada, CE, VCCI, KC). Greps for `bandwidth`, `GB/s`, `Gbps`, `bidirectional`, `peer`, `link speed`,
`ring`, `topolog` return **zero substantive hits** (the only "bandwidth" hit is a liability disclaimer).
Its entire technical content is:

- "The MI100 accelerator supports the following Infinity Fabric™ link arrangement: 4 Piece
  (P/N: 102-D34601-00)" (IFL §1.1, line 133)
- "Install the four accelerator cards with an AMD Infinity Fabric™ Link ON as a 4-card assembly"
  (IFL §2.2, line 176)
- A screw torque pattern.

**It says nothing about Vega 20 and nothing about bandwidth.** No XGMI per-link or aggregate figure in this
project may be cited to it. The only transferable fact is that a four-card ring is the documented arrangement.

### H.2 The system tuning guide is the wrong platform, and its figures are images

**NOT IN THESE DOCUMENTS.** STG §3 measures "an A+A GPU Server with eight AMD Instinct™ MI50 GPUs and dual
AMD Epyc™ 7742" and is explicit that it is characterising **PCIe**, not XGMI: "Figure 3-1 PCIe Transfer Types
- **Without Instinct Infinity Fabric Installed**". The MI50 is Vega 20, so the platform is relevant, but all
three bandwidth figures (3-1, 3-2, 3-3) are images that did not survive text extraction — there are no
numbers to quote. The XGMI settings it does discuss ("4-Link xGMI Max Speed: 18Gbps … Up to 12.5% faster
GPU-to-Remote CPU DRAM and GPU-to-GPU & GPU-to-NIC transfers", STG page 8) are **EPYC socket-to-socket**
links, not GPU-to-GPU bridges, and do not exist on a Xeon W Mac Pro.

### H.3 What the measured topology implies for allreduce

**Project-measured, not from these documents** (`xgmi-session-HANDOVER.md`, artifact f811df1d): physical ring
0b-0e-1e-1b, HIP order 0-1-3-2; **33.5 GB/s per link per direction**, **256 GB/s with all eight directions
loaded**, **507 ns one-hop latency**; RCCL needs `/root/rccl_topo_fixed.xml` because the PSP firmware
mislabels the bridge pairs.

**INFERENCE.** A 4-GPU ring with bidirectional links means every GPU has exactly two neighbours and the
diameter is 2. The optimal allreduce is ring reduce-scatter + ring all-gather: 2 x (N-1)/N = 1.5x the message
size crossing each link, in 2 x (N-1) = 6 steps. A naive implementation that has one rank gather from all
others serialises two hops and loses the ring's parallelism entirely.

### H.4 Why RCCL beats llama.cpp's internal allreduce by 23% — documented mechanisms

**DOCUMENTED (LLVM GFX9 memory model, lines ~5125-5152).** Cross-device data movement on gfx906 is not free
at the cache level, and the guide is specific:

> To ensure coherence of local and remote memory writes of work-groups in different agents a **`buffer_wbl2`**
> is required. It will writeback dirty L2 cache lines of MTYPE RW (used for local coarse grain memory) and
> MTYPE NC (used for remote coarse grain memory)… To ensure coherence of local and remote memory reads of
> work-groups in different agents a **`buffer_invl2`** is required. It will invalidate L2 cache lines with
> MTYPE NC (used for remote coarse grain memory).

and

> MTYPE CC (used for local **fine grain** memory) causes write through to DRAM … MTYPE UC (used for remote
> **fine grain** memory) **bypasses the L2**.

and, for ordering:

> A `s_waitcnt vmcnt(0)` is required to ensure synchronization between vector memory operations of different
> CUs.

**INFERENCE — four candidate contributors to the 23%, all testable:**

1. **Memory-type choice.** Fine-grained (host-visible) allocations are MTYPE CC / UC: they write through or
   bypass L2 entirely. Coarse-grained device memory is RW / NC and can be cached, at the price of an explicit
   `buffer_wbl2` / `buffer_invl2` per fence. A naive allreduce that allocates its staging buffers
   fine-grained pays uncached traffic on every element; RCCL uses coarse-grained buffers plus explicit
   writeback/invalidate at the boundaries, i.e. **two L2 operations per phase instead of uncached traffic per
   line.** This is the mechanism most likely to be worth 23%.
2. **Fence count.** Each `s_waitcnt vmcnt(0)` drains outstanding VMEM for the whole wave (ISA §4.4, line
   1230). A per-element or per-chunk synchronised implementation pays many; a ring algorithm pays 6.
3. **Topology.** RCCL builds its rings from the topology file; without `rccl_topo_fixed.xml` the firmware's
   mislabelled hop table sends traffic the long way. The project already measured 24.8 -> 33.5 GB/s from
   fixing exactly this.
4. **Kernel count, not bandwidth.** The M1 trace puts the internal allreduce at **128 kernels x 27 us =
   3.4 ms per token**, i.e. 94 us per layer, while 507 ns of fabric latency and 33.5 GB/s per link mean the
   few kB per layer could cross in single-digit microseconds. **The allreduce is launch-bound, not
   bandwidth-bound**, which is why the custom peer-write allreduce gated to <= 4 decode rows returned +14.5%
   single-stream. Extending that gate is a high-value survey item, and the ISA supports it: `GLOBAL_STORE_*`
   with `SADDR` writes directly into peer VRAM, and `GLOBAL_ATOMIC_ADD` / `_INC` (ISA §12.18.3, opcodes at
   line 11251 ff.) provide the completion flag without a kernel boundary.

---

## I. Does 8.5 KiB/token/die check out for 4 KV heads at q8_0?

**Yes, exactly, and it pins head_dim = 256.**

Inputs: 16 of 65 blocks carry a KV cache; 4 KV heads; q8_0 = 32 elements in 34 bytes (32 quants + one fp16
scale) = **1.0625 bytes/element**; K and V both cached; four dies; measured 8.5 KiB/token/die and
34 KiB/token across four dies, constant to +/-0.3%.

Total bytes per token across all four dies:

    2 (K,V) x 16 layers x 4 KV heads x head_dim x 1.0625 B  =  136 x head_dim bytes

Set equal to the measured 34 KiB = 34816 B:

    head_dim = 34816 / 136 = 256          exactly

Per die: 136 x 256 / 4 = **8704 B = 8.5 KiB exactly** (8.5 x 1024 = 8704). The measured figure is not rounded;
it is the exact value.

**Independent confirmation from a different measurement.** `ISA-NOTES.md` §2 records the f16 KV path reading
369 GB/s/die and the project runs f16 KV at head size 256. For f16 (2 B/element) the same formula gives
2 x 4 KV-layers-per-die x 4 heads x 256 x 2 = **16384 B = 16 KiB/token/die**, and 16 / 8.5 = 1.88 = 2 / 1.0625
— the ratio the two quantisations require. Two independent measurements agree on head_dim = 256.

**Self-consistency with the rest of fact 6.** `n_embd` 5120 with 24 attention heads does **not** divide
(5120 / 24 = 213.3), so this architecture necessarily decouples `head_dim` from `n_embd / n_head` — head_dim
256 is not only consistent, it is required to be an independent parameter. The Q projection is then
24 x 256 = 6144 wide against an embedding of 5120, and the KV projection 4 x 256 = 1024, a GQA group of 6.
`rope.dimension_count = 64` means RoPE is applied to 64 of 256 dimensions per head — a **25% partial rotary**,
with 192 dimensions per head carrying no positional rotation.

**Why this matters for the survey.** head_dim 256 is the largest head size llama.cpp's HIP flash-attention
kernels support and the one with the worst register and LDS pressure. The `alex4300` **head-256 tile row**
noted in the fork survey is therefore not an optional extra — it is the tile row this model actually uses,
for all 16 of its KV layers. It should be at the top of the FA patch list.

**One arithmetic note to carry forward:** at head_dim 256, 24 q heads, 16 KV layers, the decode attention
arithmetic intensity is 393216 FLOP per 34816 bytes = **11.3 FLOP/byte**. The fp32-FMA ridge point at
1730 MHz is 13.8 TFLOP/s / 880.6 GB/s = 15.7 FLOP/byte, and at the 999 MHz DPM floor it is
8.0 TFLOP/s / 880.6 GB/s = 9.1 FLOP/byte. **Decode attention is memory-bound at full clock and crosses to
compute-bound at the DPM floor.** At a 125 W cap it sits near the crossover, which is precisely where
packed-fp16 arithmetic (`V_DOT2_F32_F16`, `V_PK_FMA_F16`) starts to matter for a phase that looked
memory-bound at 200 W.

---

## J. Ranked ISA-level optimisations for a 125 W cap

Framing: at 125 W only ~10 W per die sits above the 115 W memory-streaming floor, and fact 3 says the memory
subsystem is outside the cap's reach. Two consequences drive the whole ranking:

- **Bytes are cap-free; instructions are not.** Removing DRAM traffic shortens the step without competing for
  the capped budget at all. Removing lane-instructions frees budget that the cap is rationing.
- **There is no documented power model.** Neither document gives energy per instruction class
  (`grep -i power` on the ISA: 2 hits, neither relevant; the LLVM guide has none). The energy reasoning below
  is **INFERENCE** from the documented fact that SALU touches one value per wavefront where VALU touches 64
  (ISA §2, line 456), and from instruction counts. Treat the *ordering* as a hypothesis to test, not as
  documented fact.

### J.1 — RANK 1. Delete redundant DRAM traffic in the depth-dependent path

**Why first: it is the dominant term, it has ~9.6x of headroom, and it is entirely cap-invariant.**

Arithmetic. The KV coefficient is 94.6 ms per million total KV tokens. Per die that is
8.5 KiB x 1e6 = 8.704 GB of KV bytes, so the achieved rate on KV bytes is **90 GB/s per die = 10.2% of the
measured 880.6 GB/s**. The VALU roofline for the same work is 3.5 ms (packed fp16) to 7.1 ms (fp32 FMA), and
the memory roofline is 9.9 ms. At 94.6 ms the kernel is **9.6x off the binding roofline and 13-27x off the
other one** — it is bound by neither.

The measured compute buffer is **6.3-12.5 KiB per token per die**, reaching 9.7 GiB/die at 8 slots x 192K.
That is the same order as the 8.5 KiB KV cache itself and it scales per context token. If the depth path
writes and re-reads it:

    (8.5 KV + 2 x 12.5 scratch) KiB/token/die x 1e6 tokens / 94.6 ms  =  354 GB/s per die

which is **within 4% of the 369 GB/s/die the FA-vec kernel was separately measured to sustain**. Under that
reading the kernel is at its roofline while carrying ~4x the necessary bytes, and the patch is **traffic
elimination**, not a faster inner loop:

- Per-target FA tile shapes for **head_dim 256** (§I) sized to 64 kB LDS and wave64 — the same class of fix
  that gave MMQ +33%, and the `alex4300` head-256 tile row is the candidate.
- Re-price the **split-K / sequence-split** decision. More parallel blocks write more partial state; the
  combine then reads it back. On 64 CUs at 4-12 slots there is already enough parallelism from
  24 heads x KV blocks x 16 layers without splitting.
- Keep partials in VGPRs and fold them with **DPP** (§G) rather than through a scratch buffer.
- Consume q8_0 KV **in its quantised form** with `V_DOT4_I32_I8` instead of dequantising to fp16 in scratch.
  A dequant buffer is 2 bytes per element where the cache holds 1.0625 — an immediate 1.88x on that stream.
- ISA mechanics available: `GLOBAL_LOAD_*` with **`LDS = 1`** moves data between memory and LDS without
  passing through VGPRs (ISA §9.1 Table 41, `LDS` bit, line 3740; MUBUF equivalent in §8.1.9 line 3017), and
  `GLC = 0` keeps loads in L1 ("Typically, all loads … use GLC==0", ISA §8.1.10, line 3106) — check the
  emitted ISA for gratuitous `glc` on KV reads, which forces an L1 miss to L2 on every access.

**Test that discriminates the two hypotheses:** sweep a configuration that changes the compute buffer per
token (FA split count, `GGML_CUDA_FA_ALL_QUANTS`, KV type) and refit the 94.6 coefficient. If it tracks the
compute-buffer figure, it is byte amplification and the fix is traffic. If it does not, the kernel is
latency-bound and the fix is occupancy/MLP in the inner loop.

### J.2 — RANK 2. `V_DOT4_I32_I8` and `V_PK_*` everywhere the quantisation allows

**Why second: the largest reduction in lane-instructions per useful MAC available on this chip, and
instructions are what the cap rations.**

One `V_DOT4_I32_I8` performs 4 int8 MACs per lane in one issue slot (ISA §12.10 op 40, line 7949). The ggml
scalar fallback spends roughly 8 lane-instructions per 4 MACs, so the ratio is ~8x in instructions issued and
therefore ~8x in register-file reads and datapath toggles per MAC. `V_PK_FMA_F16` and `V_DOT2_F32_F16` give
2x on the fp16 paths. Against this, **`V_DOT8_I32_I4` stays unusable**: it multiplies 4-bit by 4-bit
(ISA op 42) and llama.cpp quantises activations to 8 bits.

Where to look, in order:
- **Flash attention QK and PV products.** Stock b10288 already defines `V_DOT2_F32_F16_AVAILABLE` for
  `__gfx906__`, so the work is not "add packed dots" but "confirm from the emitted ISA that the arithmetic is
  actually packed" — `ISA-NOTES.md` §3b flags unpacked arithmetic as the thing to check. Prefill attention at
  ~36% of the packed-fp16 peak (§D.2) is consistent with partially unpacked math.
- **K-quant scale unpacking.** The 4- and 6-bit scale shift-and-mask sequences are where `V_PK_LSHRREV_B16`,
  `V_PK_ASHRREV_I16`, `V_PK_MAD_U16` and **SDWA byte selects** replace multi-instruction sequences. Note the
  SDWA exclusion list (ISA §12.19.2, line 11570) rules out `V_MAC_F32`, `V_FMAC_F32`, `V_MADMK/MADAK`,
  `V_READFIRSTLANE_B32`, `V_CLREXCP`, `V_SWAP_B32` — but not the integer shifts and adds.
- **The Q8_1 activation scale.** Q8_1 carries a scale and a block sum; against Q8_0 weights only the scale is
  used, so quantising activations to Q8_0 saves a 4-byte load and a multiply per block per column.

**Build-level precondition, DOCUMENTED:** the `gfx9-generic` target **excludes** `v_dot4_i32_i8`,
`v_dot8_i32_i4`, `v_dot2_f32_f16`, `v_dot2_i32_i16`, `v_dot2_u32_u16`, `v_dot4_u32_u8`, `v_dot8_u32_u4` and
`v_fmac_f32` on gfx906 (LLVM Table 23, lines 806-825). Every build must target `gfx906` explicitly. Verify
this in CI, not by inspection.

### J.3 — RANK 3. 32-bit encodings and address-math elimination

**Why third: it removes instruction *bytes* and VGPRs at zero algorithmic risk, and the ISA tells you to.**

> "When an instruction is available in two microcode formats, it is up to the user to decide which to use.
> **It is recommended to use the 32-bit encoding whenever possible.**" — ISA §6.1, line 1872

Concrete substitutions, all DOCUMENTED:

- **`V_FMAC_F32` (VOP2, 32-bit, op 59, line 6338) for `V_FMA_F32`/`V_MAD_F32` (VOP3A, 64-bit, ops 459/449).**
  Half the instruction bytes per FMA, hence half the instruction-fetch and I-cache pressure. Cost: the third
  source must be the destination. Bonus: as a VOP2 it can also take a DPP modifier, which the VOP3 forms
  cannot (§G.4.1).
- **`GLOBAL_*` with `SADDR` instead of 64-bit VGPR addresses.** "Global: Use the SGPR to provide a base
  address; the VGPR provides a 32-bit offset" (ISA §9.1, Table 41, `SADDR` field, line 3755). That is **one
  VGPR per address instead of two**, plus a 13-bit signed immediate `OFFSET` in the same instruction
  (`OFFSET` field, line 3745) which removes an address `V_ADD` per access. New in Vega: "FLAT Microcode
  format: added an offset field" (ISA preface, line 309).
- **`GLOBAL_*` instead of `FLAT_*`.** A FLAT instruction "increment[s] both VM_CNT and LGKM_CNT and [is] not
  considered done until both have been decremented", and "the only sensible S_WAITCNT value to use after
  FLAT instructions is zero" (ISA §9.2, §9.2.2, lines 3840-3862). GLOBAL instructions "assume all workitem
  addresses fall in global memory space" (ISA §12.18.3, line 11295) and so avoid the dual-counter
  serialisation entirely. **Grep the disassembly for `flat_load`/`flat_store` — each one is a lost
  `s_waitcnt` granularity.**
- **Inline constants instead of literals.** VOP3P/VOP3 `SRC` encodings 128-248 cover integer -16..64 and the
  float constants 0.5, 1.0, 2.0, 4.0 and 1/(2*PI) with their negatives (ISA Table 76 source table, lines
  14063-14088), at zero instruction-stream cost. A literal costs a whole extra dword, and "Any of the 32-bit
  microcode formats may use a 32-bit literal constant, **but not VOP3**" (ISA §6.1, line 1894).

### J.4 — RANK 4. Scalar (SALU / SMEM) offload of everything wave-uniform

**Why fourth: the largest per-operation energy ratio on the chip, ~64:1, and it frees VGPRs at the same time.**

**DOCUMENTED.** "The scalar ALU operates on one value per wavefront and manages all control flow" vs "The
vector ALU maintains Vector GPRs that are unique for each work item" (ISA §1.1, Table 1, lines 420-440).
Scalar loads come in wide forms: `S_LOAD_DWORD{,X2,X4,X8,X16}` and `S_BUFFER_LOAD_DWORD{,X2,X4,X8,X16}` —
"Read 16 dwords from scalar data cache" (ISA §12.6, opcodes 0-12, lines 5591-5640 (X16 at 5606)). The scalar cache is
"shared by all wavefronts on a group of CUs" (LLVM line ~5121), so a uniform value fetched once is warm for
the neighbourhood.

**INFERENCE.** Anything uniform across a wavefront — quantisation block scales that are per-block not
per-lane, row/column strides, KV base pointers, sequence lengths, layer offsets, loop trip counts — belongs
in SGPRs. One `S_LOAD_DWORDX16` replaces sixteen `s_load_dword`s or, worse, a broadcast vector load. The
binding constraint is ISA §6.2.1: "At most one SGPR can be read per instruction, but the value can be used
for more than one operand" (line 1920) — so design the inner loop so that at most one scalar appears per
VALU instruction, and let it serve multiple operands.

Two mechanisms that compose with this:
- `SCRATCH_LOAD/STORE` to and from scalar memory is new in Vega ("Also added Scratch load/store to scalar
  memory", ISA preface line 306) — spill uniform state scalar-side.
- Control flow is already scalar: `S_CBRANCH_EXECZ`/`EXECNZ` and `S_SETVSKIP` (ISA §4.2, Table 7, line 1145)
  skip whole regions without issuing vector instructions. See J.9.

### J.5 — RANK 5. DPP reductions in place of LDS / `ds_bpermute`

**Why fifth: small per kernel, but it is in every kernel, and it costs nothing in registers or LDS.**

Full treatment in §G. The short version: `__shfl_xor` compiles to `ds_bpermute_b32` on GCN — a trip through
the LDS crossbar with an `LGKM_CNT` round trip — where a six-step DPP chain (`row_shr:1,2,4,8` then
`row_bcast:15`, `row_bcast:31`) does the same in-register with no LDS allocation, no counter and no barrier.
Every kernel's final per-row reduction pays this today. Watch the two wait-state rules (ISA §4.5: 2 states
after a VGPR write, 5 after an EXEC write) and remember DPP cannot ride on `V_DOT*` or `V_PK_*` (§G.4.1).

### J.6 — RANK 6. Wider loads, but measure before believing

`GLOBAL_LOAD_DWORDX4` moves 4 dwords per instruction and improves DRAM page locality.
**But the project already measured this losing:** "Aligned dword loads lost 9-17% with or without a branch"
and "Loading a whole 32-weight block per thread, a quarter of the load instructions, lost 16-32% at batch 8
because the dependent dot-product chain became four times longer (it wins only at batch 1, +5%)"
(`ISA-NOTES.md` §3). The ISA explains the mechanism: `VM_CNT` decrements only "when the data has been written
back to the VGPRs" (ISA §4.4, line 1234), so one wide load is one long wait where four narrow loads are four
overlappable ones. **Rank this low and gate it on batch size**: wide loads for batch 1, narrow for batched
decode.

### J.7 — RANK 7. Memory-to-LDS bypass, with the same caveat

`GLOBAL_*` with the `LDS` bit set, and MUBUF load-to-LDS (ISA §8.1.9, line 3017:
"allows reading data from a memory buffer directly into LDS without passing through VGPRs", for
`BUFFER_LOAD_{ubyte, sbyte, ushort, sshort, dword, format_x}`), remove a `ds_write` and the VGPRs that would
have staged the data. **But** using LDS at all reintroduces the barrier that cost 16-52% in the measured
sweep. Worth it only where the data is genuinely shared across a workgroup — FA tiles, MMQ tiles — never for
per-thread staging.

### J.8 — RANK 8. Do NOT raise occupancy; hold ~64 VGPRs and 4-8 independent chains

This is listed as an optimisation because the instinct to raise it is the trap. The project measured
rows-per-block 1 at any warp count — halving registers and doubling waves in flight — **losing 20-36%**,
because each loaded activation block then fed half as many rows; and rows 8 lost to register pressure
(`ISA-NOTES.md` §3).

**ISA grounding.** Latency is hidden two ways, and the second is the one that matters here: across resident
waves (governed by the 256-VGPR file and the 4-VGPR allocation granule, ISA §3.6.4 line 918) *and* by
independent instructions within one wave (governed by `VM_CNT`, ISA §4.4). More rows x columns per thread
means more independent dot-product chains behind the same loads. The measured optimum — **4-8 independent
chains inside a ~64-VGPR budget** — is where the two balance on this chip. At a power cap the argument gets
stronger, not weaker: more waves means more instructions issued for the same useful work.

### J.9 — RANK 9. EXEC-mask hygiene for ragged slot depths

**DOCUMENTED, and it is a direct power argument:**

> "This GPU does no optimization when EXEC = 0. The shader hardware executes every instruction, wasting
> instruction issue bandwidth. Use CBRANCH or VSKIP to rapidly skip over code when it is likely that the
> EXEC mask is zero." — ISA §3.3 note, line 678

Also: "VSKIP … 1 = skip (do not execute) any vector instructions: valu, vmem, export, lds, gds. 'Skipping'
instructions occurs at high-speed (10 wavefronts per clock cycle can skip one instruction). This is much
faster than issuing and discarding instructions" (ISA §3.5, MODE register, line 839).

**INFERENCE.** A batched server has slots at different KV depths by construction. Any kernel that loops to
the maximum depth with a mask issues instructions for lanes that contribute nothing — and under a power cap,
issued instructions are exactly what is being rationed. Two survey items follow: (i) check the FA and MMVQ
inner loops for `s_cbranch_execz` early-outs; (ii) at the scheduler level, grouping slots of similar depth
into the same batch reduces mask waste. Note also that fact 1's large per-slot term at *identical total KV*
(6x64K = 13.20 t/s/req vs 12x32K = 10.30) is the signature of per-sequence inefficiency, and mask waste is
one of its candidate causes.

### J.10 — RANK 10. Kernel count and the documented per-dispatch floor

Not an ISA optimisation, but it is 5.8 ms of the 17.6 ms constant (§A.0) and the documents explain why the
floor exists: gfx906 has **no kernarg preload** (LLVM line 4130 and the target-feature table), kernarg
backing memory on a dGPU is **MTYPE UC in host memory, bypassing L2** (LLVM line ~5170), and **"CP
invalidates the L1 cache at the start of each kernel dispatch"** (LLVM line ~5168). Each dispatch therefore
pays an uncached host round trip for its arguments plus a cold L1. That is a documented, structural argument
for HIP graphs (already +7%) and for fusing the linear-attention blocks' small kernels.

### J.11 What is NOT worth doing under the cap

- **`V_DOT8_I32_I4`** — 4-bit x 4-bit only; llama.cpp's activations are int8. Closed.
- **rocBLAS fp16 GEMM for prefill** — already retired: ties MMQ within the run-to-run band, and
  `V_DOT4_I32_I8` is the denser MAC.
- **Chasing HBM ECC overhead** — 880-892 GB/s of 1024 nominal is an ordinary HBM2 result (§B.2), and
  `sramecc` is on-chip SRAM, not the HBM array (§B.4).
- **Raising occupancy** — see J.8.

---

## Contradictions and open conflicts, flagged

1. **The KV coefficient is 4.1x worse than this project's own FA bandwidth figure.** 94.6 ms/Mtok implies
   90 GB/s/die on KV bytes; `ISA-NOTES.md` §2 records the FA-vec kernel at 369 GB/s/die of cache read at the
   same head size. Both cannot describe the same kernel doing the same bytes. Either the batched
   multi-sequence FA path is 4x less efficient than the single-stream vec path, or the depth path moves ~4x
   the KV bytes (the compute-buffer arithmetic in §J.1 fits the second to within 4%). **This must be resolved
   by a kernel trace before any FA patch is chosen, because the two readings point at different patches.**

2. **The 715 GB/s copy figure may be arithmetically impossible.** Under a one-direction-counted convention it
   implies 1430 GB/s of DRAM traffic against a 1024 GB/s theoretical peak. Pin the convention (§B.3).

3. **Prefill attention efficiency: 36% (this review) vs 30% (`ISA-NOTES.md` §2) vs "60% of fp32 FMA".**
   These are close but the derivations differ, and mine carries a factor-of-2 uncertainty from the
   causal-masking assumption (§D.2). The conclusion — prefill is VALU/clock-bound — is robust to the factor;
   the size of the remaining headroom is not. Do not size an FA prefill patch off either number without
   nailing the causal factor.

4. **"Nothing is clock-limited or thermally limited, everything is power limited" (fact 4) is true at 125 W
   and 200 W but false below ~115 W.** Fact 3 has the die drawing 115 W while streaming with the cap set as
   low as 50 W and sclk pinned at the 999 MHz DPM floor. Below ~115 W the **DPM floor**, not the cap, is
   binding: the die cannot go slower. The two facts are consistent only within the 125-200 W range, and 125 W
   is ~10 W from the crossover. Any extrapolation of the -20.2% / -10.7% slopes below 125 W will be wrong.

5. **Decode attention changes roofline side between 200 W and the DPM floor.** Arithmetic intensity is
   11.3 FLOP/byte (§I) against a ridge of 15.7 at 1730 MHz and 9.1 at 999 MHz. So a patch that looks
   pointless at 200 W (packed fp16 in FA) can matter at 125 W, and a bandwidth patch that looks decisive at
   200 W matters less at the floor. **Every FA patch must be benchmarked at both caps**; a single-cap result
   is not transferable.

6. **The reference set cannot support any XGMI claim.** `amd-infinity-fabric-link-user-guide-56978` is an
   MI100 installation guide with zero performance content (§H.1) and the system tuning guide is an
   EPYC/PCIe/MI50 document whose bandwidth figures are unextracted images (§H.2). Every XGMI number in this
   project's reports is project-measured and must be cited as such — never to these PDFs. The same applies to
   HBM peak bandwidth, all clock-domain descriptions, all power behaviour, and every instruction throughput
   figure: none of them are in these four documents.
