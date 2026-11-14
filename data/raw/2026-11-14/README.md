# data/raw/2026-11-14 - llama.cpp v1.0.0 cycle

Result lines of the runs behind `reports/2026-11-14-v100-rebase.md`, copied from `/root/rebase-v100/` on macpro2019-01 by
`tools/release/raw-copy.sh`. One `<run>.progress.txt` per run directory (the scripts' own log lines: values, exit codes,
output hashes), the two queue logs, the publish log, and the texts sent to mxxm-t (`pr-body.md`, `pr-comment-models.md`,
`merge-msg.txt`, `pad-fix-msg.txt`). Server logs and model outputs stay on the box.

All 16K, 8K, 4K and 2K depths here are correctness smoke values (R2.2): comparable build to build, never serving results.
`np4-ab.progress.txt` is a disturbed run (report section 6): do not use its means.
