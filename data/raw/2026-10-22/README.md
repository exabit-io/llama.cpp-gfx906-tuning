# 2026-10-22 raw data: the v0.7.0 cycle

Cited by `reports/2026-10-22-v070-rebase.md`. Box-side originals in `/root/rebase-v070/` on macpro2019-01.

- `pr-test-*/progress`: correctness suite (merge, master source build, master deb).
- `fold-*/progress`, `isolate-*/progress`: decode-path fold perplexity (ubatch 16 except `fold-reconcile`, which is ubatch 32).
- `kld-*.txt`: KL divergence of the folds (llama-perplexity stdout).
- `dio/progress`: `-lm dio` fallback check on four builds.
- `test-backend-ops-summary.txt`: pass lines and failures of every test-backend-ops run.
- `mtp-deb/`: the three idle-host MTP runs on the installed deb (llama-server timings).
- `stack-check-*.txt`: PyTorch stack check, 2.13.0 and 2.14.1 debs; `torch-2.14.1-build.log`: the build driver log.
- `publish.log`, `pr-body.md`, `merge-v0.7.0-commit-message.txt`.
