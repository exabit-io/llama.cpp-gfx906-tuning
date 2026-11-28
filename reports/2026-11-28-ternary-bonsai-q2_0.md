# 2026-11-28: Ternary-Bonsai-2.5-27B Q2_0 on gfx906 - benchmarked, not adopted

**Decision (lead, 2026-11-28): no switch.** The service stays on Qwen3.8-27B Q8_0 with tensor split. The ternary
model runs on the installed package, but on this box it is slower per request than what is served now, weaker on
plain text, and cannot use tensor split. The run was stopped after the first timed cell; what was not measured is
listed at the end.

Model: `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`, file `Ternary-Bonsai-2.5-27B-Q2_0.gguf` (revision `28a7093c`,
7,673,194,848 bytes, SHA256 checked against the hub's), a ternary build of Qwen3.8-27B (architecture `qwen35`, 64
blocks, width 5120, 262K context). In the file: 402 matrices in llama.cpp's `Q2_0` type (26.87 G weights, 7.04 GiB,
2.25 bits per weight with scales; token embedding and output head included) and 449 F32 norm vectors. The weights
are stored in a Hadamard-rotated basis (block 1024) and the runtime must transform the activations to match
(`prism.hadamard.*` metadata; upstream since b11556, so in our v1.1.0 = b12774).

All runs: macpro2019-01, 4x gfx906, installed package `mx-llama-cpp-gfx906 1.1.0+git20261127.c782c317d` (ROCm 10.1),
125 W per die, f16 KV, flash attention on. Scripts `tools/bonsai/`, raw files `data/raw/2026-11-28/bonsai/`; every
script has its reading rule in its header, written before its first run. Clock columns clean.

## 1. Does it run correctly (`bonsai-check.sh`; functional, no rate is a result)

| Mode | Greedy text (project prompt) | Perplexity, wikitext-2, 16K x 6 |
|---|---|---|
| one die | correct two-sentence answer | **9.3516 +/- 0.1270** |
| four dies, `-sm layer` | identical text | 9.3516 +/- 0.1270 |
| four dies, `-sm tensor` | **aborts** | aborts |
| for comparison: Qwen3.8-27B Q8_0, tensor split, same package | | 5.6088 +/- 0.0622 |

- The Hadamard runtime works on gfx906 (the card warns that builds without it produce garbage; 9.35 is not garbage).
  The GPU backend has MMQ and MMVQ kernels for `Q2_0` (gfx906 takes upstream's GCN tile table) and an FWHT kernel.
- **Tensor split cannot run this model on our tree**: `ggml-backend-meta.cpp:1610: GGML_ASSERT(split_state.ne[j] %
  div == 0)`, on a `RESHAPE` that turns a lane-split activation (6144 wide, 1536 per die) into the model's 1024-wide
  Hadamard blocks. The split-state propagation takes the per-die ratio over from the source and the blocks do not
  divide. It is in the shared tensor-split backend, not in an Exabit commit. Supporting it means block-aligned lane
  splits for Hadamard inputs (5, 6 and 17 blocks do not split evenly over four dies) or keeping those activations
  unsplit: development work, not a setting.
- Perplexity is 67 % above the Q8_0 model's on this test. The vendor's own scores are on agentic and tool-calling
  suites; none of those was run here.

## 2. One instance per die (`bonsai-perdie.sh`; lead: "just run a separate instance on each die")

**What one die holds** (model loaded, full context allocated, tiny run; functional):

| Slots x 64K on one die | 1 | 2 | 3 | 4 | 5 | 1 x 262144 |
|---|---|---|---|---|---|---|
| Peak VRAM, GiB | 13.7 | 18.0 | 22.2 | 26.4 | 30.7 | 27.4 |

So the box holds up to 20 slots of 64K (five per die) or four 262K sessions at once: about 9.5 GiB for the model and
its buffers and 4.25 GiB per 64K slot.

**Four instances at once, one slot x 65536 each (4 x 64K in total), 1,024 generated tokens, two runs:**

| Die | Decode tok/s per request (run 1 / run 2) | Prefill tok/s (run 1 / run 2) |
|---|---|---|
| 0 | 17.711 / 17.638 | 171.0 / 170.9 |
| 1 | 17.238 / 17.227 | 165.7 / 165.6 |
| 2 | 17.908 / 17.949 | 172.3 / 172.1 |
| 3 | 17.852 / 17.934 | 172.1 / 172.1 |

Readable by the rule (runs agree within 0.46 % in decode and 0.14 % in prefill on every die). Each run took 459 s.

| At 4 x 64K | Bonsai Q2_0, one instance per die | Qwen3.8-27B Q8_0, tensor split | Bonsai against Q8_0 |
|---|---|---|---|
| Decode per request, tok/s | 17.68 | 24.96 | **-29 %** |
| Aggregate decode, tok/s | 70.7 | 99.8 | -29 % |
| Aggregate prefill, tok/s | 680.9 | 818.6 | **-17 %** |
| Over R3.1's 12 tok/s per request | yes | yes | |

The Q8_0 column is the same package timed in the acceptance session of 2026-11-27/28 (`reports/2026-11-27-v110-timing.md`
s.7; prefill from the clean partial rerun), another session of the same boot: good for differences of this size, not
for small ones. DC power of the whole machine in the Bonsai decode phase: 768 W (0.092 decode tokens per joule).

Why smaller weights do not make it faster here (inference, not measured): each die does the whole 27B forward pass
alone, where tensor split shares every matrix product over four dies; and the `Q2_0` kernels are upstream's defaults,
without the tuned and fused paths the Q8_0 model has on this tree.

## 3. Not measured (stopped on the lead's decision)

Two and three slots per die (8 and 12 x 64K in total), one die alone against one die with busy neighbours, a timed
1 x 262144, layer split timing (one started run was stopped), the `-mtp` file, the vendor's `PTQ1_0`/`PQ2_0` files
(they need the vendor's fork), any quality suite beyond perplexity.

What would have to change for another look: tensor split support for the Hadamard blocks (the one with real upside),
tuned `Q2_0` kernels for gfx906, or a requirement that values context capacity over speed.

The file stays at `/root/models/Ternary-Bonsai-2.5-27B-Q2_0.gguf` (7.2 GiB).
