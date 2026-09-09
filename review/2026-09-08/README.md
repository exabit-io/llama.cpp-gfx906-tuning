# Review handoff for Claude

Open [the research report](vega20-review.html). It contains prioritized llama.cpp, LLVM/HIP, RCCL and packaging proposals with pinned source links. This folder is additive; original project files are unchanged.

## Tested optimizer patches

Apply in this order from the project root, preferably in a copy:

```sh
patch -p1 < review/2026-09-08/patches/0001-fix-q4v-lookup-and-command.patch
patch -p1 < review/2026-09-08/patches/0002-scope-modifiers-to-placement.patch
python3 review/2026-09-08/test_optimizer_patch.py optimize/optimize.py -v
```

The optimizer requires PuLP and a CBC solver. Validation here used an isolated virtual environment with PuLP 3.3.2; no system packages were changed. The output warns that PuLP 4 will change the solver/variable APIs, so pin or validate that dependency before an upgrade.

- Patch 0001 makes missing Q4V ladder keys fall through to recorded/model cells and prevents a shell comment from swallowing MTP flags.
- Patch 0002 makes the objective and per-stream constraint use the actual placement's topology/graph factors.
- Six regression tests pass on the patched candidate. Five fail on the original (two assertion failures and three errors); the existing f16/q8 ladder preservation test passes on both.
- Both delivered patches were independently applied in order to a second fresh copy; the resulting source matches the tested candidate.
- Candidate scenario runs completed for default production, `--build stock`, `--allow-quant --max-kl 0.04`, `--cap 125`, and `--kv-q4v`, each with `--alternatives 0`. Logs are in `evidence/`.

These patches do not fix the report's separate production-coefficient, provenance, long-context-prefill or cap-model findings. Do not treat the remaining estimates as certified observations.

## Next GPU work

Recover the actual production patch files and build manifest first. Then profile graph/allreduce eligibility and reproduce the reported production table. The report specifies MMVQ width/architecture guards, a full MMQ table/dispatch transplant, existing attention tuning, cache-safe activation-sum elimination, early MTP loop limits and conditional reduction work.

No proposed GPU kernel was compiled or benchmarked here. No remote machine was contacted and no GPU/power settings were changed. Existing raw reports were preserved.

## Report verification

The HTML is self-contained, with system fonts, semantic headings and a responsive comparison table. Structural and browser verification details are recorded in `evidence/report-qa.json`. The internal source ledger, search/gap record and input hashes preserve provenance for follow-up work.
