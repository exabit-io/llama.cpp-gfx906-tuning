# reference/ — the four documents the ISA notes and the kernel work cite

Each document was extracted from its PDF to a plain-text file with `=== [PDF PAGE n] ===` markers, plus a
`.toc.md` section map giving line numbers into the text (the line numbers quoted in `CLAUDE.md` and
`ISA-NOTES.md` refer to these text files).

| Text file (local) | Section map (in git) | Source |
|---|---|---|
| `amd-vega-7nm-isa-gfx906.txt` | `amd-vega-7nm-isa-gfx906.toc.md` | AMD, "Vega" 7nm Instruction Set Architecture Reference Guide (2020) — https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-7nm-shader-instruction-set-architecture.pdf |
| `amd-infinity-fabric-link-user-guide-56978.txt` | `amd-infinity-fabric-link-user-guide-56978.toc.md` | AMD Infinity Fabric Link User Guide, part 56978 |
| `amd-instinct-system-tuning-guide-57286.txt` | `amd-instinct-system-tuning-guide-57286.toc.md` | System Tuning Guide for AMD Instinct GPU Servers with EPYC 7002 CPUs, part 57286 |
| `llvm-amdgpu-backend-user-guide.txt` | `llvm-amdgpu-backend-user-guide.toc.md` | User Guide for AMDGPU Backend, LLVM (Apache-2.0 with LLVM exception) — https://llvm.org/docs/AMDGPUUsage.html |

The three AMD texts are AMD's copyright ("all rights reserved") and are **not** in the repository (`.gitignore`);
download the PDFs from AMD and re-extract (`pdftotext -layout`, then insert the page markers) to reproduce the
line numbers, or use the section maps to find the same passages in the PDFs. The LLVM guide is redistributable
and is included.
