# tools/release/ - the release cycle's scripts (box-side copies, 2026-10-22)

Procedure: `RELEASE-RUNBOOK.md`. Working copies and logs on the box: `/root/rebase-v070/`, `/root/mx-llama-deb/`,
`/root/pytorch-gfx906-rocm10-2.14.1/`. All GPU scripts source `tools/gpu-test-env.sh` (macpro-fand pins 125 W) and run the
clamp watchdog by PID.

| Script | What it does |
|---|---|
| `build.sh SRC BUILD LOG` | gfx906 HIP build with tests, the tree's own defaults, ccache |
| `pr-test.sh BUILD OUT` | correctness suite: test-backend-ops x4 (`SKIP_TBO=1` skips), PPL 16K/6, `-sm layer -lm dio` greedy + fallback count, MTP via llama-server, Flash-Next |
| `fold-check.sh BUILD OUT` | decode-path folds on vs off (`UB=16`, `-v`), plus a repeat arm for determinism |
| `fold-isolate.sh BUILD OUT` | one fold off at a time |
| `kld-check.sh` | KL divergence of folds-on against all-folds-off logits, old vs new master |
| `dio-check.sh OUT NAME=BUILD...` | `-lm dio` buffered-fallback count across builds |
| `mtp-deb.sh` | MTP on the installed deb, idle host, 3 requests |
| `queue1.sh` / `queue2.sh` / `queue3.sh` | the v0.7.0 cycle's GPU queues (as run) |
| `build-mx-llama-deb.sh SRC` | the `mx-llama-cpp-gfx906` deb from a clean worktree of master |
| `build-torch-2.14.1.sh`, `build-torch-deb-2.14.1.sh`, `torch-stack-check.py` | PyTorch 2.14.1 wheels, the `python3-torch-gfx906` deb, and its GPU/CPU check |
| `publish.sh LLAMA_DEB TORCH_DEB` | sign both debs, tag resize-amdgpu-bars, one signed flat apt release per package |
| `round2-smoke.sh` | second-round smoke tests on the abliterated models |
| `q6k-campaign.sh` | Q6_K vs Q8_0 campaign (quality vs BF16, speed, MTP on/off, capacity) |
| `portnotes.py` | `git filter-branch --msg-filter` that appends port notes to ported commits |
| `port_norm_q8.py`, `port_norm_preadd.py` | the scripted ports of terms 02 and 03 onto the clamped-grid RMS norm kernel (record) |
| `ggufdiff.py`, `ggufcmp.py` | GGUF metadata/tensor-layout diff, and per-tensor byte comparison of two GGUFs |
| `mtp-modes.sh BUILD OUT [NP]` | v1.0.0: MTP target hidden rows on each path (`LLAMA_MTP_DEV`, `LLAMA_MTP_DEFERRED_PREFILL`): one long prompt then three identical greedy requests per arm, output hash compared |
| `ctx-check.sh BUILD OUT` | v1.0.0: `test-llama-archs`, `test-llama-context` (direct, one worker) and `test-fp8` over the dummy-model fixtures |
| `deb-test.sh OUT` | v1.0.0: reference on the previous package, install the new deb from the file, ELF/RPATH/native checks, suite and MTP on `/opt/mx-llama.cpp` |
| `publish-100.sh DEB` | v1.0.0: sign, tag, per-version release, then the rolling channel `mx-llama-cpp-gfx906` |
| `fork-delta.py OLD_UP OLD_FORK NEW_UP [NEW_FORK]` | after a merge: which files' fork delta changed (edited by the merge) and which did not |
| `queue-v100-1.sh` / `queue-v100-2.sh` | the v1.0.0 cycle's GPU queues (as run) |
| `dogfood-100.sh OUT` | v1.0.0: write the channel source line, reinstall from the channel, verify InRelease signature -> Packages SHA256 -> deb SHA256 -> origin signature in the deb |
| `drafters-100.sh OUT` | v1.0.0: DeepSeek-V4.1 + DSpark (`--fit-target 7168`; stock, merge build and its three feature paths) and Nemotron + LiLiCorr (stock, merge, master) |
| `np4-ab.sh OUT` | v1.0.0: single-stream MTP at `-np 4`, v0.9.0 master against v1.0.0 master, three alternating rounds (the 2026-11-14 run was disturbed, see the report) |
| `stall-diag.sh OUT [BUILD] [NREQ]` | v1.0.0: the same requests against one server with a verbose log; per request the speed and the largest gaps between log lines, plus SMI count, CE count, VRAM and CPU pressure before and after |
| `build-torch-2.15.0.sh`, `build-torch-deb-2.15.0.sh` | PyTorch 2.15.0 wheels (torch, vision, audio at PyTorch's CI pins) and the `python3-torch-gfx906` deb |
| `build-typing-extensions-deb.sh [VERSION]` | `python3-typing-extensions` backport for noble from the upstream wheel (hash checked against PyPI and the build venv) |
| `torch-deb-test.sh OUT` | reference on the installed PyTorch package, install the new debs from the files, library and dependency checks, `torch-stack-check.py` and `torch-ops-check.py` |
| `torch-ops-check.py` | GPU against CPU, one check per ROCm library, sort/topk, attention, a short training run, and the NCCL backends over the four dies; exit code = failed checks |
| `publish-torch.sh DEB`, `dogfood-torch.sh OUT` | PyTorch: sign, tag, per-version release, rolling channel `python3-torch-gfx906`; then switch the box to the channel and verify the chain |
| `raw-copy.sh`, `docs-update.py` | v1.0.0: copy the cycle's result lines into `data/raw/2026-11-14/`; move this repository's statements of the code state to the new cycle (run once) |
