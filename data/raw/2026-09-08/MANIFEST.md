# Raw evidence for the 2026-09-08 night report (copied from /root/rocm-tests/bench on 2026-09-08 21:10 UTC)

Every result table (`*.md`), client JSON and clock sample referenced by `reports/2026-09-08-night-report.md`,
`reports/2026-09-08-gfx906-branch-validation.md`, `reports/2026-09-08-tile-table-ablation.md`,
`reports/2026-09-08-fa-counters.md`, `reports/2026-09-08-warmup-samples.md` and `reports/2026-09-08-ppl-bisect.md`.
Server logs, `.kld` bases and perplexity logs stay on the box (3.8 GB); the runners that produced each file are in
`tools/` (same names as the `.md` prefixes: `gfx906-master-validate.sh`, `post22.sh` … `post27.sh`, `post25b.sh`,
`ci-paired.sh`, `server-final.sh`, `fa-counters-2.sh`, `ablation-bench.sh`, `flash-next-test.sh`, `bisect/`).
Every command line, model path, context, batch, `-npl` list and environment is in those scripts; the environment for
every measured cell is `settings/gfx906.env` (custom allreduce + RCCL ring) unless the row says `noar` / `X=1`.

## Binaries (sha256 of bin/llama-server, lib/libggml-hip.so; build time)
| prefix | role | llama-server | libggml-hip.so | built |
|---|---|---|---|---|
| /opt/llama.cpp | stock = upstream b10288 pristine | 1549870f04859035 | 2e47969d58e7ea9c | 2026-09-03 09:01 |
| /opt/llama.cpp-prod -> /opt/llama.cpp-mxxm-fh-nq | production = fork b10254 + patches/0001-0009 | dc260bf9b6e618f1 | 08d850a6bc378f6c | 2026-09-08 11:39 |
| /opt/llama.cpp-gfx906-master | branch before the MUL_MAT_ID fix (validation binary, sections D/E/F) | 9978100c555a3991 | 77cd88f3288c45c3 | 2026-09-08 13:30 |
| /opt/llama.cpp-gfx906-master-r2 | branch `gfx906` at 5d36d6fc8 (= 92a3ac9c2 minus docs), IEEE math | 139e7401bbe4a4e8 | 8138286342d30d8d | 2026-09-08 14:55 |
| /opt/llama.cpp-gfx906-master-r3 | r2 + `-funsafe-math-optimizations` (retired: no effect) | bd023e5b45c22c3f | e336ecd7a1d27a79 | 2026-09-08 18:30 |
| /opt/llama.cpp-b10912 | fork pristine at its tag b10912 | 02e29b19f44ff99f | 784b7ffada7215d8 | 2026-09-08 13:11 |
| /opt/llama.cpp-master | upstream pristine 5d806aa25 (2026-09-08) | d572434be1c399a2 | e5db4539337d8451 | 2026-09-08 13:29 |

Build recipe for every prefix: `scripts/gfx906/build.sh` in the repository (ROCm 7.14 TheRock ML-gfx906, hipcc
`-mllvm -amdgpu-sched-strategy=max-ilp`, `-march=native`, HIP graphs, RCCL, MMQ MFMA off); the ablation builds
(`/opt/llama.cpp-ablation-{a,b,c}`) are upstream b10288 + one fork commit each (ablation branch `ablation-tile`,
commits 39c917808 / f48d37902+6c2c03a95 / b29ea5262).

## Repack setting per row
The fork b10912 state (and the `gfx906` branch) repacks Q8_0 at upload by default. Rows labelled `nr1`, `--no-repack`
or `repack off` ran with `--no-repack` (llama-bench: `-nr 1`); everything else on those builds ran repacked. Production,
stock and pristine upstream have no repack path.

## Box state for every GPU row
Fans at full speed (t2fanrd `always_full_speed`), dies at `power_dpm_force_performance_level=high`, 200 W caps unless
the row is a cap sweep, host CPU RAPL 150 W, governor `performance`; the clamp watchdog (`tools/clamp-watchdog-v2.sh`)
kills the chain at PZ0G >= 1228 W or a die pinned at 1000 MHz. The `*-clocks.txt` files are the per-second sclk/power
sample for each queue (`tools/gpu-test-env.sh start_sampler`).
