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
