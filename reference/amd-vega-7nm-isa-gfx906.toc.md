# Section map: amd-vega-7nm-isa-gfx906.txt

Source: `amd-vega-7nm-isa-gfx906.pdf` (296 PDF pages). Line numbers refer to `amd-vega-7nm-isa-gfx906.txt`.
Use: grep the txt for keywords, or jump straight to a section with Read offset=<line>.

| Section | PDF page | Line |
|---|---|---|
| "Vega" 7nm Instruction Set Architecture: Reference Guide | 1 | 6 |
| Untitled | 1 | 6 |
| Contents | 2 | 12 |
| Preface | 7 | 203 |
| &nbsp;&nbsp;About This Document | 7 | 203 |
| &nbsp;&nbsp;Audience | 7 | 203 |
| &nbsp;&nbsp;Organization | 7 | 203 |
| &nbsp;&nbsp;Conventions | 8 | 247 |
| &nbsp;&nbsp;Related Documents | 8 | 247 |
| &nbsp;&nbsp;New Features of "Vega" 7nm Devices | 8 | 247 |
| &nbsp;&nbsp;New Instructions | 9 | 293 |
| &nbsp;&nbsp;Contact Information | 9 | 293 |
| Chapter 1. Introduction | 11 | 347 |
| &nbsp;&nbsp;1.1. Terminology | 12 | 378 |
| Chapter 2. Program Organization | 14 | 456 |
| &nbsp;&nbsp;2.1. Compute Shaders | 14 | 456 |
| &nbsp;&nbsp;2.2. Data Sharing | 15 | 503 |
| &nbsp;&nbsp;&nbsp;&nbsp;2.2.1. Local Data Share (LDS) | 15 | 503 |
| &nbsp;&nbsp;&nbsp;&nbsp;2.2.2. Global Data Share (GDS) | 16 | 527 |
| &nbsp;&nbsp;2.3. Device Memory | 16 | 527 |
| Chapter 3. Kernel State | 17 | 573 |
| &nbsp;&nbsp;3.1. State Overview | 17 | 573 |
| &nbsp;&nbsp;3.2. Program Counter (PC) | 18 | 630 |
| &nbsp;&nbsp;3.3. EXECute Mask | 18 | 630 |
| &nbsp;&nbsp;3.4. Status registers | 19 | 678 |
| &nbsp;&nbsp;3.5. Mode register | 20 | 736 |
| &nbsp;&nbsp;3.6. GPRs and LDS | 21 | 793 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.6.1. Out-of-Range behavior | 22 | 849 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.6.2. SGPR Allocation and storage | 23 | 894 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.6.3. SGPR Alignment | 23 | 894 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.6.4. VGPR Allocation and Alignment | 23 | 894 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.6.5. LDS Allocation and Clamping | 23 | 894 |
| &nbsp;&nbsp;3.7. M# Memory Descriptor | 24 | 936 |
| &nbsp;&nbsp;3.8. SCC: Scalar Condition code | 24 | 936 |
| &nbsp;&nbsp;3.9. Vector Compares: VCC and VCCZ | 24 | 936 |
| &nbsp;&nbsp;3.10. Trap and Exception registers | 25 | 979 |
| &nbsp;&nbsp;&nbsp;&nbsp;3.10.1. Trap Status register | 26 | 1025 |
| &nbsp;&nbsp;3.11. Memory Violations | 27 | 1082 |
| Chapter 4. Program Flow Control | 28 | 1126 |
| &nbsp;&nbsp;4.1. Program Control | 28 | 1126 |
| &nbsp;&nbsp;4.2. Branching | 28 | 1126 |
| &nbsp;&nbsp;4.3. Workgroups | 29 | 1180 |
| &nbsp;&nbsp;4.4. Data Dependency Resolution | 29 | 1180 |
| &nbsp;&nbsp;4.5. Manually Inserted Wait States (NOPs) | 30 | 1230 |
| &nbsp;&nbsp;4.6. Arbitrary Divergent Control Flow | 32 | 1340 |
| Chapter 5. Scalar ALU Operations | 34 | 1415 |
| &nbsp;&nbsp;5.1. SALU Instruction Formats | 34 | 1415 |
| &nbsp;&nbsp;5.2. Scalar ALU Operands | 34 | 1415 |
| &nbsp;&nbsp;5.3. Scalar Condition Code (SCC) | 37 | 1570 |
| &nbsp;&nbsp;5.4. Integer Arithmetic Instructions | 37 | 1570 |
| &nbsp;&nbsp;5.5. Conditional Instructions | 38 | 1626 |
| &nbsp;&nbsp;5.6. Comparison Instructions | 38 | 1626 |
| &nbsp;&nbsp;5.7. Bit-Wise Instructions | 38 | 1626 |
| &nbsp;&nbsp;5.8. Access Instructions | 40 | 1741 |
| Chapter 6. Vector ALU Operations | 42 | 1855 |
| &nbsp;&nbsp;6.1. Microcode Encodings | 42 | 1855 |
| &nbsp;&nbsp;6.2. Operands | 43 | 1891 |
| &nbsp;&nbsp;&nbsp;&nbsp;6.2.1. Instruction Inputs | 43 | 1891 |
| &nbsp;&nbsp;&nbsp;&nbsp;6.2.2. Instruction Outputs | 44 | 1927 |
| &nbsp;&nbsp;&nbsp;&nbsp;6.2.3. Out-of-Range GPRs | 46 | 2044 |
| &nbsp;&nbsp;6.3. Instructions | 46 | 2044 |
| &nbsp;&nbsp;6.4. Denormalized and Rounding Modes | 48 | 2167 |
| &nbsp;&nbsp;6.5. ALU Clamp Bit Usage | 49 | 2221 |
| &nbsp;&nbsp;6.6. VGPR Indexing | 49 | 2221 |
| &nbsp;&nbsp;&nbsp;&nbsp;6.6.1. Indexing Instructions | 49 | 2221 |
| &nbsp;&nbsp;&nbsp;&nbsp;6.6.2. Specific Cases | 50 | 2272 |
| &nbsp;&nbsp;6.7. Packed Math | 51 | 2326 |
| Chapter 7. Scalar Memory Operations | 52 | 2355 |
| &nbsp;&nbsp;7.1. Microcode Encoding | 52 | 2355 |
| &nbsp;&nbsp;7.2. Operations | 53 | 2404 |
| &nbsp;&nbsp;&nbsp;&nbsp;7.2.1. S_LOAD_DWORD, S_STORE_DWORD | 53 | 2404 |
| &nbsp;&nbsp;&nbsp;&nbsp;7.2.2. Scalar Atomic Operations | 54 | 2451 |
| &nbsp;&nbsp;&nbsp;&nbsp;7.2.3. S_DCACHE_INV, S_DCACHE_WB | 55 | 2499 |
| &nbsp;&nbsp;&nbsp;&nbsp;7.2.4. S_MEMTIME | 55 | 2499 |
| &nbsp;&nbsp;&nbsp;&nbsp;7.2.5. S_MEMREALTIME | 55 | 2499 |
| &nbsp;&nbsp;7.3. Dependency Checking | 55 | 2499 |
| &nbsp;&nbsp;7.4. Alignment and Bounds Checking | 55 | 2499 |
| Chapter 8. Vector Memory Operations | 57 | 2552 |
| &nbsp;&nbsp;8.1. Vector Memory Buffer Instructions | 57 | 2552 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.1. Simplified Buffer Addressing | 58 | 2599 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.2. Buffer Instructions | 58 | 2599 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.3. VGPR Usage | 60 | 2698 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.4. Buffer Data | 61 | 2754 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.5. Buffer Addressing | 62 | 2807 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.6. 16-bit Memory Operations | 67 | 2966 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.7. Alignment | 67 | 2966 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.8. Buffer Resource | 67 | 2966 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.9. Memory Buffer Load to LDS | 68 | 3017 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.1.10. GLC Bit Explained | 69 | 3076 |
| &nbsp;&nbsp;8.2. Vector Memory (VM) Image Instructions | 70 | 3112 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.2.1. Image Instructions | 71 | 3161 |
| &nbsp;&nbsp;8.3. Image Opcodes with No Sampler | 72 | 3219 |
| &nbsp;&nbsp;8.4. Image Opcodes with a Sampler | 73 | 3272 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.4.1. VGPR Usage | 75 | 3398 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.4.2. Image Resource | 76 | 3452 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.4.3. Image Sampler | 78 | 3572 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.4.4. Data Formats | 79 | 3632 |
| &nbsp;&nbsp;&nbsp;&nbsp;8.4.5. Vector Memory Instruction Data Dependencies | 80 | 3673 |
| Chapter 9. Flat Memory Instructions | 82 | 3693 |
| &nbsp;&nbsp;9.1. Flat Memory Instruction | 82 | 3693 |
| &nbsp;&nbsp;9.2. Instructions | 84 | 3812 |
| &nbsp;&nbsp;&nbsp;&nbsp;9.2.1. Ordering | 84 | 3812 |
| &nbsp;&nbsp;&nbsp;&nbsp;9.2.2. Important Timing Consideration | 84 | 3812 |
| &nbsp;&nbsp;9.3. Addressing | 85 | 3865 |
| &nbsp;&nbsp;9.4. Global | 85 | 3865 |
| &nbsp;&nbsp;9.5. Scratch | 85 | 3865 |
| &nbsp;&nbsp;9.6. Memory Error Checking | 86 | 3910 |
| &nbsp;&nbsp;9.7. Data | 86 | 3910 |
| &nbsp;&nbsp;9.8. Scratch Space (Private) | 87 | 3956 |
| Chapter 10. Data Share Operations | 88 | 3973 |
| &nbsp;&nbsp;10.1. Overview | 88 | 3973 |
| &nbsp;&nbsp;10.2. Dataflow in Memory Hierarchy | 89 | 4001 |
| &nbsp;&nbsp;10.3. LDS Access | 89 | 4001 |
| &nbsp;&nbsp;&nbsp;&nbsp;10.3.1. LDS Direct Reads | 90 | 4032 |
| &nbsp;&nbsp;&nbsp;&nbsp;10.3.2. LDS Parameter Reads | 90 | 4032 |
| &nbsp;&nbsp;&nbsp;&nbsp;10.3.3. Data Share Indexed and Atomic Access | 91 | 4078 |
| Chapter 11. Exporting Pixel and Vertex Data | 95 | 4250 |
| &nbsp;&nbsp;11.1. Microcode Encoding | 95 | 4250 |
| &nbsp;&nbsp;11.2. Operations | 96 | 4293 |
| &nbsp;&nbsp;&nbsp;&nbsp;11.2.1. Pixel Shader Exports | 96 | 4293 |
| &nbsp;&nbsp;&nbsp;&nbsp;11.2.2. Vertex Shader Exports | 96 | 4293 |
| &nbsp;&nbsp;11.3. Dependency Checking | 96 | 4293 |
| Chapter 12. Instructions | 98 | 4359 |
| &nbsp;&nbsp;12.1. SOP2 Instructions | 98 | 4359 |
| &nbsp;&nbsp;12.2. SOPK Instructions | 103 | 4631 |
| &nbsp;&nbsp;12.3. SOP1 Instructions | 105 | 4745 |
| &nbsp;&nbsp;12.4. SOPC Instructions | 115 | 5266 |
| &nbsp;&nbsp;12.5. SOPP Instructions | 116 | 5321 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.5.1. Send Message | 120 | 5555 |
| &nbsp;&nbsp;12.6. SMEM Instructions | 120 | 5555 |
| &nbsp;&nbsp;12.7. VOP2 Instructions | 128 | 6029 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.7.1. VOP2 using VOP3 encoding | 133 | 6317 |
| &nbsp;&nbsp;12.8. VOP1 Instructions | 133 | 6317 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.8.1. VOP1 using VOP3 encoding | 147 | 7047 |
| &nbsp;&nbsp;12.9. VOPC Instructions | 148 | 7097 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.9.1. VOPC using VOP3A encoding | 160 | 7814 |
| &nbsp;&nbsp;12.10. VOP3P Instructions | 160 | 7814 |
| &nbsp;&nbsp;12.11. VINTERP Instructions | 162 | 7920 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.11.1. VINTERP using VOP3 encoding | 163 | 7966 |
| &nbsp;&nbsp;12.12. VOP3A & VOP3B Instructions | 163 | 7966 |
| &nbsp;&nbsp;12.13. LDS & GDS Instructions | 182 | 8945 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.13.1. DS_SWIZZLE_B32 Details | 203 | 10128 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.13.2. LDS Instruction Limitations | 205 | 10212 |
| &nbsp;&nbsp;12.14. MUBUF Instructions | 206 | 10266 |
| &nbsp;&nbsp;12.15. MTBUF Instructions | 211 | 10557 |
| &nbsp;&nbsp;12.16. MIMG Instructions | 212 | 10607 |
| &nbsp;&nbsp;12.17. EXPORT Instructions | 217 | 10905 |
| &nbsp;&nbsp;12.18. FLAT, Scratch and Global Instructions | 218 | 10968 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.18.1. Flat Instructions | 218 | 10968 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.18.2. Scratch Instructions | 222 | 11192 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.18.3. Global Instructions | 223 | 11251 |
| &nbsp;&nbsp;12.19. Instruction Limitations | 227 | 11494 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.19.1. DPP | 227 | 11494 |
| &nbsp;&nbsp;&nbsp;&nbsp;12.19.2. SDWA | 228 | 11547 |
| Chapter 13. Microcode Formats | 229 | 11589 |
| &nbsp;&nbsp;13.1. Scalar ALU and Control Formats | 230 | 11651 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.1.1. SOP2 | 231 | 11705 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.1.2. SOPK | 234 | 11850 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.1.3. SOP1 | 236 | 11974 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.1.4. SOPC | 239 | 12123 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.1.5. SOPP | 241 | 12234 |
| &nbsp;&nbsp;13.2. Scalar Memory Format | 243 | 12360 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.2.1. SMEM | 243 | 12360 |
| &nbsp;&nbsp;13.3. Vector ALU Formats | 246 | 12560 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.1. VOP2 | 246 | 12560 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.2. VOP1 | 249 | 12729 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.3. VOPC | 253 | 12992 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.4. VOP3A | 262 | 13590 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.5. VOP3B | 267 | 13891 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.6. VOP3P | 269 | 13996 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.7. SDWA | 271 | 14091 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.8. SDWAB | 273 | 14212 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.3.9. DPP | 273 | 14212 |
| &nbsp;&nbsp;13.4. Vector Parameter Interpolation Format | 275 | 14320 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.4.1. VINTRP | 275 | 14320 |
| &nbsp;&nbsp;13.5. LDS and GDS format | 276 | 14374 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.5.1. DS | 276 | 14374 |
| &nbsp;&nbsp;13.6. Vector Memory Buffer Formats | 281 | 14714 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.6.1. MTBUF | 281 | 14714 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.6.2. MUBUF | 283 | 14829 |
| &nbsp;&nbsp;13.7. Vector Memory Image Format | 286 | 15024 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.7.1. MIMG | 286 | 15024 |
| &nbsp;&nbsp;13.8. Flat Formats | 290 | 15288 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.8.1. FLAT | 291 | 15352 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.8.2. GLOBAL | 293 | 15477 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.8.3. SCRATCH | 295 | 15615 |
| &nbsp;&nbsp;13.9. Export Format | 296 | 15675 |
| &nbsp;&nbsp;&nbsp;&nbsp;13.9.1. EXP | 296 | 15675 |
