# Section map: llvm-amdgpu-backend-user-guide.txt

Source: `llvm-amdgpu-backend-user-guide.pdf` (187 PDF pages; print capture of the LLVM 22.0.0git AMDGPU backend user guide, 2026-08-10).
Line numbers refer to `llvm-amdgpu-backend-user-guide.txt`. TOC synthesized from the document's own contents listing; blank line = title wrapped in source TOC, grep for it instead.

| Section | PDF page | Line |
|---|---|---|
| Introduction | 3 | 157 |
| LLVM | 3 | 161 |
| &nbsp;&nbsp;&nbsp;&nbsp;Target Triples | 3 | 162 |
| &nbsp;&nbsp;&nbsp;&nbsp;Processors | 3 | 191 |
| &nbsp;&nbsp;&nbsp;&nbsp;Generic Processor Versioning |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Target Features | 13 | 914 |
| &nbsp;&nbsp;&nbsp;&nbsp;Target ID | 14 | 985 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 to V3 Target ID | 14 | 998 |
| &nbsp;&nbsp;&nbsp;&nbsp;Embedding Bundled Code Objects | 14 | 1008 |
| &nbsp;&nbsp;&nbsp;&nbsp;Address Spaces | 14 | 1014 |
| &nbsp;&nbsp;&nbsp;&nbsp;Memory Scopes | 16 | 1159 |
| &nbsp;&nbsp;&nbsp;&nbsp;LLVM IR Intrinsics | 17 | 1228 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;‘llvm.amdgcn.cooperative.atomic’ Intrinsics | 17 | 1232 |
| LLVM IR Metadata | 18 | 1274 |
| &nbsp;&nbsp;&nbsp;&nbsp;‘amdgpu.last.use’ Metadata | 18 | 1277 |
| &nbsp;&nbsp;&nbsp;&nbsp;‘amdgpu.no.remote.memory’ Metadata | 18 | 1282 |
| &nbsp;&nbsp;&nbsp;&nbsp;‘amdgpu.no.fine.grained.memory’ Metadata | 18 | 1298 |
| &nbsp;&nbsp;&nbsp;&nbsp;‘amdgpu.ignore.denormal.mode’ Metadata | 19 | 1318 |
| LLVM IR Attributes | 19 | 1328 |
| Calling Conventions | 21 | 1492 |
| &nbsp;&nbsp;&nbsp;&nbsp;AMDGPU MCExpr | 22 | 1559 |
| &nbsp;&nbsp;&nbsp;&nbsp;Function Resource Usage | 22 | 1566 |
| ELF Code Object | 23 | 1607 |
| &nbsp;&nbsp;&nbsp;&nbsp;Header | 23 | 1611 |
| &nbsp;&nbsp;&nbsp;&nbsp;Sections | 27 | 1867 |
| &nbsp;&nbsp;&nbsp;&nbsp;Note Records | 27 | 1908 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 Note Records |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 and Above Note Records | 29 | 2034 |
| &nbsp;&nbsp;&nbsp;&nbsp;Symbols | 30 | 2064 |
| &nbsp;&nbsp;&nbsp;&nbsp;Relocation Records | 30 | 2090 |
| &nbsp;&nbsp;&nbsp;&nbsp;Loaded Code Object Path Uniform Resource Identifier (URI) | 31 | 2148 |
| DWARF Debug Information | 32 | 2189 |
| &nbsp;&nbsp;&nbsp;&nbsp;Register Identifier | 32 | 2201 |
| &nbsp;&nbsp;&nbsp;&nbsp;Memory Space Identifier | 33 | 2269 |
| &nbsp;&nbsp;&nbsp;&nbsp;Address Space Identifier | 33 | 2291 |
| &nbsp;&nbsp;&nbsp;&nbsp;Lane identifier | 34 | 2354 |
| &nbsp;&nbsp;&nbsp;&nbsp;Operation Expressions | 34 | 2362 |
| &nbsp;&nbsp;&nbsp;&nbsp;Base Type Conversions |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Debugger Information Entry Attributes | 35 | 2393 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DW_AT_LLVM_lane_pc | 35 | 2398 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DW_AT_LLVM_active_lane | 38 | 2636 |
| &nbsp;&nbsp;Call Frame Information | 38 | 2648 |
| &nbsp;&nbsp;Accelerated Access | 38 | 2668 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lookup By Address Section Header | 38 | 2671 |
| &nbsp;&nbsp;Line Number Information | 38 | 2682 |
| &nbsp;&nbsp;32-Bit and 64-Bit DWARF Formats | 39 | 2715 |
| &nbsp;&nbsp;Unit Headers | 39 | 2721 |
| Code Conventions | 39 | 2727 |
| &nbsp;&nbsp;AMDHSA | 39 | 2730 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object Metadata | 39 | 2733 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 Metadata | 39 | 2742 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 Metadata |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V4 Metadata | 46 | 3197 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V5 Metadata | 46 | 3213 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V6 Metadata | 48 | 3322 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kernel Dispatch | 48 | 3329 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Spaces |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Image and Samplers                                                                         latest |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;HSA Signals | 49 | 3416 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;HSA AQL Queue              Claude is active in this tab group |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kernel Descriptor | 49 | 3427 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 Kernel Descriptor |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initial Kernel Execution State | 58 | 4014 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Preloaded Kernel Arguments | 60 | 4130 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kernel Prolog | 60 | 4146 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CFI | 60 | 4151 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;M0 | 60 | 4156 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stack Pointer | 60 | 4164 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Frame Pointer | 60 | 4168 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Flat Scratch | 60 | 4175 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Private Segment Buffer | 61 | 4229 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model | 62 | 4263 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fence and Address Spaces | 63 | 4342 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX6-GFX9 | 63 | 4359 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX90A |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX942 | 92 | 6427 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX10-GFX11 | 111 | 7774 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX12 | 128 | 8947 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Memory Model GFX125x | 147 | 10352 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;‘llvm.amdgcn.cooperative.atomic’ Intrinsics | 166 | 11672 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trap Handler ABI | 166 | 11690 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Call Convention | 168 | 11863 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kernel Functions | 168 | 11869 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Non-Kernel Functions | 168 | 11878 |
| &nbsp;&nbsp;&nbsp;&nbsp;AMDPAL |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object Metadata | 172 | 12064 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;User Data | 174 | 12247 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Per-Shader Table | 175 | 12310 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Spill Table | 175 | 12319 |
| &nbsp;&nbsp;&nbsp;&nbsp;Unspecified OS | 175 | 12327 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trap Handler ABI | 175 | 12330 |
| Core file format |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;Core file header | 176 | 12355 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Split files | 176 | 12358 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Unified file | 176 | 12369 |
| &nbsp;&nbsp;&nbsp;&nbsp;Core file notes | 176 | 12372 |
| &nbsp;&nbsp;&nbsp;&nbsp;Memory segments | 176 | 12404 |
| Source Languages | 177 | 12416 |
| &nbsp;&nbsp;&nbsp;&nbsp;OpenCL | 177 | 12417 |
| &nbsp;&nbsp;&nbsp;&nbsp;HCC | 177 | 12433 |
| &nbsp;&nbsp;&nbsp;&nbsp;Assembler | 177 | 12437 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Instructions | 177 | 12441 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Operands | 178 | 12490 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Modifiers | 178 | 12493 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Instruction Examples | 178 | 12496 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DS | 178 | 12497 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FLAT | 178 | 12505 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MUBUF | 178 | 12514 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SMRD/SMEM | 178 | 12523 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SOP1 | 178 | 12532 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SOP2 | 179 | 12548 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SOPC | 179 | 12561 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SOPP | 179 | 12569 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VALU | 179 | 12586 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 Predefined Symbols | 180 | 12638 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.option.machine_version_major | 180 | 12644 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.option.machine_version_minor | 180 | 12648 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.option.machine_version_stepping | 180 | 12652 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.kernel.vgpr_count | 180 | 12656 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.kernel.sgpr_count | 180 | 12661 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 Directives | 180 | 12666 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.hsa_code_object_version major, minor | 180 | 12672 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.hsa_code_object_isa [major, minor, stepping, vendor, arch] | 180 | 12675 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgpu_hsa_kernel (name) | 181 | 12688 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amd_kernel_code_t | 181 | 12692 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V2 Example Source Code | 181 | 12714 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 and Above Predefined Symbols                                              latest |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn.gfx_generation_number | 181 | 12753 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Claude is active in this tab group |  |  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn.gfx_generation_minor | 182 | 12762 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn.gfx_generation_stepping | 182 | 12766 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn.next_free_vgpr | 182 | 12770 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn.next_free_sgpr | 182 | 12776 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 and Above Directives | 182 | 12782 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgcn_target <target-triple> “-” <target-id> | 182 | 12786 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdhsa_code_object_version <version> | 182 | 12794 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdhsa_kernel <name> | 182 | 12798 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.amdgpu_metadata | 185 | 12976 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Code Object V3 and Above Example Source Code | 185 | 12983 |
| Additional Documentation | 186 | 13112 |
