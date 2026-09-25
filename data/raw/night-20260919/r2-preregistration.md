# Round 2 — the substrate's patchsets (pre-registered 2026-09-25, before any round-2 data)

Base: master a23e12438 (substrate 528384980 + AR size gate + FA_QUANTS=all). Each arm REMOVES one substrate feature
(runtime switch where the fork has one, else a revert or a one-line switch-off patch), except one ADDITION arm.
Rule: binstats.py with binstats-r2.json; multi-user mean of n=5, single-user median of n=6 (lead 2026-09-25);
two-sided exact permutation; BH q<0.10 over 8 arms x 2 axes x 2 metrics = 32; effect floor 2%.

| arm | removes | how | reference |
|---|---|---|---|
| no-q8_1-cache | q8_1 activation cache (775a8051f, cfea4a1f6, 0a694d8ad) | `GGML_CUDA_Q8_1_CACHE=0` (covers the q8_repack reuse too) | base |
| no-token-graph | whole-token graph capture (751b6114c, a3ab67c89, 693375a1f, a355590d2) | `GGML_META_TOKEN_GRAPH=0` | base |
| no-parallel-dispatch | concurrent subgraph lanes (5d9efc8ca) | `GGML_META_PARALLEL_DISPATCH=0` (also turns the token graph off, which needs it) | **no-token-graph** (increment) |
| no-layout-cache | galloc layout cache (6d2012d8a, 42b3cfb63, 687ef0194, d5047d6aa) | `GGML_GALLOC_LAYOUT_CACHE=0` | base |
| no-gdn-chunked | chunked gated-delta-net prefill (55b275262, 1248e5e37, 4405b44cd, 0e3249e9d) | patch f62654251: `chunk_eligible = false` (revert conflicts; upstream v0.5.0 has no chunked kernel, so serial = upstream) | base |
| revert-input-staging | sched input staging (38e266b6a, c7069d868, 20af9a480) | `git revert` of the three (applies cleanly) | base |
| no-mmq-kernel-tweaks | gfx906 Q8_0 tile-load tpr=8 (5c4505b5d) + vec_dot unroll-2 (f1684f76c) | patch 2032a34e6: both guards forced false (upstream code) | base |
| mmq-config-reachable | ADDITION: fork's gfx906 Q8_0 MMQ config (2335544ab) made reachable again | patch a22eb39ef: VEGA20 dispatched before GCN (host and device); every other type stays on upstream's GCN table | base |

Reading the removal arms (pre-registered): binstats compares arm vs reference, so the FEATURE's verdict is the
mirror image. Arm regresses (q<0.10, <= -2%) -> the feature improves that axis -> keep. Arm improves (q<0.10, >= +2%)
-> the feature regresses that axis -> switched off (runtime env) or reverted for that profile. Neutral -> neutral.
Mixed -> the lead. The addition arm reads directly: improves -> the merge lost a real gain; restoring it changes the
substrate, so it goes to the lead (master + PR mxxm-t/mx-llama.cpp#17).

Found while preparing (0 GPU):
- **The v0.5.0 merge made the fork's gfx906 MMQ config unreachable.** Upstream v0.5.0 added mmq-config-gcn.cuh and
  dispatches GCN first; gfx906 is GCN5, so both the host (`GGML_CUDA_CC_IS_GCN`) and device (`#ifdef GCN`) paths
  return upstream's GCN table and the fork's `cc == VEGA20` / `__gfx906__` branches are dead. In mxxm-t's
  eefc4e732 they were live. Output is bit-exact either way (the correctness gate could not see it); only speed can.
  The fork's tpr=8 and unroll-2 kernel tweaks are unaffected (live, moved to mmq-load-tiles.cuh / mmq-vec-dot.cuh).
- The plan's switch for meta-token-graph (`GGML_META_TG_LIMIT=0`) is a debug bisect knob; the feature switch is
  `GGML_META_TOKEN_GRAPH` (default on). Corrected above.
- meta-xfer-rccl (04f89ab03 part) is NOT measured: its path needs `n_stages > 1` (`want_xfer_comm`), and this
  instrument runs one stage of four (tps = 4). Recorded by inspection.
- Later mxxm-t commits: a355590d2 goes with the token graph, 20af9a480 with input staging; de27e7509, 62d4be47d,
  be8ff98aa, cf6a98f73, 54702a718, 27f755681 are not round-2 arms and are classified by inspection in the records.

Run: chain-r2.sh <round-1b fncompat pid> -> installs the extended binrun.sh / fncompat.sh (revert + runtime-env arms)
after 1b exits -> ROUND=r2 binrun.sh -> fncompat.sh. Estimated 9 arms: 5.2 h multi + 10.7 h single + builds + compat.
