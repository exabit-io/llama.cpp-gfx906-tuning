# S1b candidate 1: gfx906 code-object metadata for mul_mat_vec_repacked_nc<2,1,NCOLS,2,64,Q8_0>
Extraction: llvm-objcopy --dump-section .hip_fatbin on build/.../q8_repack/mul-mat.cu.o; clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx906 --unbundle; llvm-readelf --notes (ROCm 7.14, /opt/rocm/llvm/bin).

| NCOLS | a: VGPR / spills / scratch B | a2 (bare bound): VGPR / spills | a3 (NCOLS>10 ? 2 : 4 waves/EU): VGPR / spills |
|---:|---|---|---|
| 2 | 40 / 0 / 0 | 40 / 0 | 40 / 0 |
| 3 | 58 / 0 / 0 | 66 / 0 | 58 / 0 |
| 4 | 51 / 0 / 0 | 78 / 0 | 51 / 0 |
| 5 | 53 / 0 / 0 | 90 / 0 | 53 / 0 |
| 6 | 55 / 0 / 0 | 103 / 0 | 55 / 0 |
| 7 | 57 / 0 / 0 | 116 / 0 | 57 / 0 |
| 8 | 59 / 0 / 0 | 129 / 0 | 59 / 0 |
| 9 | 61 / 0 / 0 | 142 / 0 | 61 / 0 |
| 10 | 63 / 0 / 0 | 155 / 0 | 63 / 0 |
| 11 | 64 / 1 / 8 | 168 / 0 | 128 / 2 |
| 12 | 64 / 4 / 20 | 181 / 0 | 67 / 0 |
| 13 | 64 / 6 / 28 | 194 / 0 | 69 / 0 |
| 14 | 64 / 10 / 36 | 207 / 0 | 71 / 0 |
| 15 | 64 / 13 / 44 | 220 / 0 | 73 / 0 |
| 16 | 64 / 18 / 52 | 228 / 0 | 75 / 0 |
Widths 2 (16 lanes) and 3 (32 lanes) narrow-lane variants: 39-40 / 50-52 VGPRs, no spills, in every build.
