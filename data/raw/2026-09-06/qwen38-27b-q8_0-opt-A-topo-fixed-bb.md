macpro2019-01:254561:254561 [3] NCCL INFO ncclOsDlopen(libnccl-env.so) failed: libnccl-env.so: cannot open shared object file: No such file or directory
macpro2019-01:254561:254561 [3] NCCL INFO ENV/Plugin: Could not find: libnccl-env.so
macpro2019-01:254561:254561 [3] NCCL INFO ROCr version 1.21
macpro2019-01:254561:254561 [3] NCCL INFO Kernel version 7.0
macpro2019-01:254561:254561 [3] NCCL INFO Skipping /proc/config.gz (zcat available, file not found)
macpro2019-01:254561:254561 [3] NCCL INFO CONFIG_PCI_P2PDMA=y in /boot/config-7.0.0-30-generic
macpro2019-01:254561:254561 [3] NCCL INFO CONFIG_DMABUF_MOVE_NOTIFY=y in /boot/config-7.0.0-30-generic
macpro2019-01:254561:254561 [3] NCCL INFO DMA_BUF Support Enabled
macpro2019-01:254561:254561 [3] NCCL INFO Kernel version: 7.0.0-30-generic

[2026-09-06 18:39:07] macpro2019-01:254561:254561 [3] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/init.cc:303 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
macpro2019-01:254561:254561 [3] NCCL INFO Hipruntime version: 71460850, firmware version: 0
macpro2019-01:254561:254561 [0] NCCL INFO Kernel version: 7.0.0-30-generic

[2026-09-06 18:39:07] macpro2019-01:254561:254561 [0] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/init.cc:303 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
macpro2019-01:254561:254561 [0] NCCL INFO RCCL version : 2.30.4-HEAD:2b22ab0
HIP version  : 7.14.60850-0000000
ROCm version : 7.14.0.7-9999-2b22ab0195
Hostname     : macpro2019-01
Librccl path : /opt/rocm/lib/librccl.so.1
macpro2019-01:254561:254561 [1] NCCL INFO Kernel version: 7.0.0-30-generic

[2026-09-06 18:39:09] macpro2019-01:254561:254561 [1] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/init.cc:303 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
macpro2019-01:254561:254561 [2] NCCL INFO Kernel version: 7.0.0-30-generic

[2026-09-06 18:39:09] macpro2019-01:254561:254561 [2] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/init.cc:303 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
macpro2019-01:254561:254561 [3] NCCL INFO Kernel version: 7.0.0-30-generic

[2026-09-06 18:39:09] macpro2019-01:254561:254561 [3] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/init.cc:303 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
macpro2019-01:254561:254699 [0] NCCL INFO RCCL: Detected 0 IB devices, primary type: 0 (count: 0)
macpro2019-01:254561:254699 [0] NCCL INFO ncclOsDlopen(librccl-net.so) failed: librccl-net.so: cannot open shared object file: No such file or directory
macpro2019-01:254561:254699 [0] NCCL INFO NET/Plugin: Could not find: librccl-net.so
macpro2019-01:254561:254699 [0] NCCL INFO Could not open libibverbs.so
macpro2019-01:254561:254699 [0] NCCL INFO NET/IB : No device found.
macpro2019-01:254561:254699 [0] NCCL INFO NET/IB : Using [RO]; OOB enp3s0:192.168.1.2<0>
macpro2019-01:254561:254699 [0] NCCL INFO Failed to initialize NET plugin IB
macpro2019-01:254561:254699 [0] NCCL INFO NET/Socket : Using [0]enp3s0:192.168.1.2<0>
macpro2019-01:254561:254699 [0] NCCL INFO Initialized NET plugin Socket
macpro2019-01:254561:254699 [0] NCCL INFO Assigned NET plugin Socket to comm
macpro2019-01:254561:254699 [0] NCCL INFO ncclOsDlopen(libnccl-gin.so) failed: libnccl-gin.so: cannot open shared object file: No such file or directory
macpro2019-01:254561:254699 [0] NCCL INFO GIN/Plugin: Could not find: libnccl-gin.so
macpro2019-01:254561:254699 [0] NCCL INFO Failed to initialize any GIN plugin
macpro2019-01:254561:254699 [0] NCCL INFO Using network Socket

[2026-09-06 18:39:11] macpro2019-01:254561:254699 [0] /root/rocm/code/TheRock/build/comm-libs/rccl/build/hipify/src/misc/amdsmi_wrap.cc:214 NCCL WARN RCCL_USE_AMD_SMI_LIB not set, but HIP_FABRIC_API is defined. Fabric support is only available through AMD SMI. Rerun with RCCL_USE_AMD_SMI_LIB=1 to enable AMD SMI and UALoE fabric support.
macpro2019-01:254561:254699 [0] NCCL INFO [node_id = 4; gpu_id = 15848; unique_id = 928446178975424807; location_id = 7680; bdf = 7680; domain = 0; partition = 0], 
macpro2019-01:254561:254699 [0] NCCL INFO [node_id = 3; gpu_id = 28203; unique_id = 14672297270832902477; location_id = 6912; bdf = 6912; domain = 0; partition = 0], 
macpro2019-01:254561:254699 [0] NCCL INFO [node_id = 1; gpu_id = 41510; unique_id = 14946454722782306556; location_id = 2816; bdf = 2816; domain = 0; partition = 0], 
macpro2019-01:254561:254699 [0] NCCL INFO [node_id = 2; gpu_id = 61925; unique_id = 14947571689157173500; location_id = 3584; bdf = 3584; domain = 0; partition = 0], 
macpro2019-01:254561:254699 [0] NCCL INFO initialized internal alternative rsmi functionality
macpro2019-01:254561:254699 [0] NCCL INFO [Rank 0] ncclCommInitAll_impl comm 0x56af439d46e0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId b000 commId 0x1b36da65599e1fd7 - Init START
macpro2019-01:254561:254701 [2] NCCL INFO Initialized NET plugin Socket
macpro2019-01:254561:254701 [2] NCCL INFO Assigned NET plugin Socket to comm
macpro2019-01:254561:254701 [2] NCCL INFO Failed to initialize any GIN plugin
macpro2019-01:254561:254701 [2] NCCL INFO Using network Socket
macpro2019-01:254561:254701 [2] NCCL INFO [node_id = 4; gpu_id = 15848; unique_id = 928446178975424807; location_id = 7680; bdf = 7680; domain = 0; partition = 0], 
macpro2019-01:254561:254701 [2] NCCL INFO [node_id = 3; gpu_id = 28203; unique_id = 14672297270832902477; location_id = 6912; bdf = 6912; domain = 0; partition = 0], 
macpro2019-01:254561:254701 [2] NCCL INFO [node_id = 1; gpu_id = 41510; unique_id = 14946454722782306556; location_id = 2816; bdf = 2816; domain = 0; partition = 0], 
macpro2019-01:254561:254701 [2] NCCL INFO [node_id = 2; gpu_id = 61925; unique_id = 14947571689157173500; location_id = 3584; bdf = 3584; domain = 0; partition = 0], 
macpro2019-01:254561:254701 [2] NCCL INFO [Rank 2] ncclCommInitAll_impl comm 0x56af43c170c0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 1b000 commId 0x1b36da65599e1fd7 - Init START
macpro2019-01:254561:254702 [3] NCCL INFO Initialized NET plugin Socket
macpro2019-01:254561:254702 [3] NCCL INFO Assigned NET plugin Socket to comm
macpro2019-01:254561:254702 [3] NCCL INFO Failed to initialize any GIN plugin
macpro2019-01:254561:254702 [3] NCCL INFO Using network Socket
macpro2019-01:254561:254702 [3] NCCL INFO [node_id = 4; gpu_id = 15848; unique_id = 928446178975424807; location_id = 7680; bdf = 7680; domain = 0; partition = 0], 
macpro2019-01:254561:254702 [3] NCCL INFO [node_id = 3; gpu_id = 28203; unique_id = 14672297270832902477; location_id = 6912; bdf = 6912; domain = 0; partition = 0], 
macpro2019-01:254561:254702 [3] NCCL INFO [node_id = 1; gpu_id = 41510; unique_id = 14946454722782306556; location_id = 2816; bdf = 2816; domain = 0; partition = 0], 
macpro2019-01:254561:254702 [3] NCCL INFO [node_id = 2; gpu_id = 61925; unique_id = 14947571689157173500; location_id = 3584; bdf = 3584; domain = 0; partition = 0], 
macpro2019-01:254561:254702 [3] NCCL INFO [Rank 3] ncclCommInitAll_impl comm 0x56af44d4d6d0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 1e000 commId 0x1b36da65599e1fd7 - Init START
macpro2019-01:254561:254702 [3] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
macpro2019-01:254561:254700 [1] NCCL INFO Initialized NET plugin Socket
macpro2019-01:254561:254700 [1] NCCL INFO Assigned NET plugin Socket to comm
macpro2019-01:254561:254700 [1] NCCL INFO Failed to initialize any GIN plugin
macpro2019-01:254561:254700 [1] NCCL INFO Using network Socket
macpro2019-01:254561:254700 [1] NCCL INFO [node_id = 4; gpu_id = 15848; unique_id = 928446178975424807; location_id = 7680; bdf = 7680; domain = 0; partition = 0], 
macpro2019-01:254561:254700 [1] NCCL INFO [node_id = 3; gpu_id = 28203; unique_id = 14672297270832902477; location_id = 6912; bdf = 6912; domain = 0; partition = 0], 
macpro2019-01:254561:254700 [1] NCCL INFO [node_id = 1; gpu_id = 41510; unique_id = 14946454722782306556; location_id = 2816; bdf = 2816; domain = 0; partition = 0], 
macpro2019-01:254561:254700 [1] NCCL INFO [node_id = 2; gpu_id = 61925; unique_id = 14947571689157173500; location_id = 3584; bdf = 3584; domain = 0; partition = 0], 
macpro2019-01:254561:254700 [1] NCCL INFO [Rank 1] ncclCommInitAll_impl comm 0x56af43b3a4e0 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId e000 commId 0x1b36da65599e1fd7 - Init START
macpro2019-01:254561:254699 [0] NCCL INFO Loading topology file /root/rccl_topo_fixed.xml
macpro2019-01:254561:254701 [2] NCCL INFO Loading topology file /root/rccl_topo_fixed.xml
macpro2019-01:254561:254700 [1] NCCL INFO Loading topology file /root/rccl_topo_fixed.xml
macpro2019-01:254561:254699 [0] NCCL INFO Loading unnamed topology
macpro2019-01:254561:254701 [2] NCCL INFO Loading unnamed topology
macpro2019-01:254561:254702 [3] NCCL INFO Loading topology file /root/rccl_topo_fixed.xml
macpro2019-01:254561:254700 [1] NCCL INFO Loading unnamed topology
macpro2019-01:254561:254702 [3] NCCL INFO Loading unnamed topology
macpro2019-01:254561:254699 [0] NCCL INFO TOPO/NET : Importing network plugins to topology
macpro2019-01:254561:254699 [0] NCCL INFO ncclTopoPopulateNics : Filled enp3s0 in topo with pciPath=/sys/devices/pci0000:00/0000:00:1d.0/0000:03:00.0 net=1 gin=(null) keep=1 coll=(null)
macpro2019-01:254561:254700 [1] NCCL INFO TOPO/NET : Importing network plugins to topology
macpro2019-01:254561:254700 [1] NCCL INFO ncclTopoPopulateNics : Filled enp3s0 in topo with pciPath=/sys/devices/pci0000:00/0000:00:1d.0/0000:03:00.0 net=1 gin=(null) keep=1 coll=(null)
macpro2019-01:254561:254701 [2] NCCL INFO TOPO/NET : Importing network plugins to topology
macpro2019-01:254561:254701 [2] NCCL INFO ncclTopoPopulateNics : Filled enp3s0 in topo with pciPath=/sys/devices/pci0000:00/0000:00:1d.0/0000:03:00.0 net=1 gin=(null) keep=1 coll=(null)
macpro2019-01:254561:254702 [3] NCCL INFO TOPO/NET : Importing network plugins to topology
macpro2019-01:254561:254702 [3] NCCL INFO ncclTopoPopulateNics : Filled enp3s0 in topo with pciPath=/sys/devices/pci0000:00/0000:00:1d.0/0000:03:00.0 net=1 gin=(null) keep=1 coll=(null)
macpro2019-01:254561:254702 [3] NCCL INFO Tuning index set to: 0
macpro2019-01:254561:254702 [3] NCCL INFO === System : maxBw 24.0 totalBw 48.0 ===
macpro2019-01:254561:254702 [3] NCCL INFO CPU/0-0 (1/1/2)
macpro2019-01:254561:254702 [3] NCCL INFO + PCI[12.0] - PCI/0-7000 (10b5874710b58747)
macpro2019-01:254561:254702 [3] NCCL INFO               + PCI[12.0] - PCI/0-9000 (100214a000000000)
macpro2019-01:254561:254702 [3] NCCL INFO                             + PCI[12.0] - DEV/0-b000 (100266a3106b0203)
macpro2019-01:254561:254702 [3] NCCL INFO                                           + LOC[5000.0] - GPU/0-b000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254702 [3] NCCL INFO               + PCI[12.0] - PCI/0-c000 (100214a000000000)
macpro2019-01:254561:254702 [3] NCCL INFO                             + PCI[12.0] - DEV/0-e000 (100266a3106b0203)
macpro2019-01:254561:254702 [3] NCCL INFO                                           + LOC[5000.0] - GPU/0-e000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254702 [3] NCCL INFO + PCI[12.0] - PCI/0-17000 (10b5874710b58747)
macpro2019-01:254561:254702 [3] NCCL INFO               + PCI[12.0] - PCI/0-19000 (100214a000000000)
macpro2019-01:254561:254702 [3] NCCL INFO                             + PCI[12.0] - DEV/0-1b000 (100266a3106b0203)
macpro2019-01:254561:254702 [3] NCCL INFO                                           + LOC[5000.0] - GPU/0-1b000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254702 [3] NCCL INFO               + PCI[12.0] - PCI/0-1c000 (100214a000000000)
macpro2019-01:254561:254702 [3] NCCL INFO                             + PCI[12.0] - DEV/0-1e000 (100266a3106b0203)
macpro2019-01:254561:254702 [3] NCCL INFO                                           + LOC[5000.0] - GPU/0-1e000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254702 [3] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254702 [3] NCCL INFO + PCI[1.5] - NIC/0-3000
macpro2019-01:254561:254702 [3] NCCL INFO ==========================================
macpro2019-01:254561:254702 [3] NCCL INFO GPU/0-b000 :GPU/0-b000 (0/5000.0/LOC) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (4/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (1/5000.0/LOC) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (3/24.0/XGMI) 
macpro2019-01:254561:254702 [3] NCCL INFO GPU/0-e000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (0/5000.0/LOC) GPU/0-1b000 (4/24.0/XGMI) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (1/5000.0/LOC) DEV/0-1b000 (3/24.0/XGMI) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254702 [3] NCCL INFO GPU/0-1b000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (4/24.0/XGMI) GPU/0-1b000 (0/5000.0/LOC) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (3/24.0/XGMI) DEV/0-1b000 (1/5000.0/LOC) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254702 [3] NCCL INFO GPU/0-1e000 :GPU/0-b000 (4/24.0/XGMI) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (0/5000.0/LOC) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (3/24.0/XGMI) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (1/5000.0/LOC) 
macpro2019-01:254561:254702 [3] NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 3 is 0-31. (GPU affinity = 0-31 ; CPU affinity = 0-55).
macpro2019-01:254561:254702 [3] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254702 [3] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1e000 GPU/0-1b000
macpro2019-01:254561:254702 [3] NCCL INFO  1 : GPU/0-b000 GPU/0-1b000 GPU/0-1e000 GPU/0-e000
macpro2019-01:254561:254702 [3] NCCL INFO ringGraph->nChannels = 2 
macpro2019-01:254561:254702 [3] NCCL INFO Pattern 1, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254702 [3] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1b000 GPU/0-1e000
macpro2019-01:254561:254702 [3] NCCL INFO  1 : GPU/0-1e000 GPU/0-e000 GPU/0-b000 GPU/0-1b000
macpro2019-01:254561:254702 [3] NCCL INFO GFX9 cheap fence is OFF
macpro2019-01:254561:254702 [3] NCCL INFO Rank 3: 1 Net devices
macpro2019-01:254561:254702 [3] NCCL INFO Rank 3: 0 CollNet devices
macpro2019-01:254561:254699 [0] NCCL INFO Tuning index set to: 0
macpro2019-01:254561:254699 [0] NCCL INFO === System : maxBw 24.0 totalBw 48.0 ===
macpro2019-01:254561:254699 [0] NCCL INFO CPU/0-0 (1/1/2)
macpro2019-01:254561:254699 [0] NCCL INFO + PCI[12.0] - PCI/0-7000 (10b5874710b58747)
macpro2019-01:254561:254699 [0] NCCL INFO               + PCI[12.0] - PCI/0-9000 (100214a000000000)
macpro2019-01:254561:254699 [0] NCCL INFO                             + PCI[12.0] - DEV/0-b000 (100266a3106b0203)
macpro2019-01:254561:254699 [0] NCCL INFO                                           + LOC[5000.0] - GPU/0-b000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254699 [0] NCCL INFO               + PCI[12.0] - PCI/0-c000 (100214a000000000)
macpro2019-01:254561:254699 [0] NCCL INFO                             + PCI[12.0] - DEV/0-e000 (100266a3106b0203)
macpro2019-01:254561:254699 [0] NCCL INFO                                           + LOC[5000.0] - GPU/0-e000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254699 [0] NCCL INFO + PCI[12.0] - PCI/0-17000 (10b5874710b58747)
macpro2019-01:254561:254699 [0] NCCL INFO               + PCI[12.0] - PCI/0-19000 (100214a000000000)
macpro2019-01:254561:254699 [0] NCCL INFO                             + PCI[12.0] - DEV/0-1b000 (100266a3106b0203)
macpro2019-01:254561:254699 [0] NCCL INFO                                           + LOC[5000.0] - GPU/0-1b000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254699 [0] NCCL INFO               + PCI[12.0] - PCI/0-1c000 (100214a000000000)
macpro2019-01:254561:254699 [0] NCCL INFO                             + PCI[12.0] - DEV/0-1e000 (100266a3106b0203)
macpro2019-01:254561:254699 [0] NCCL INFO                                           + LOC[5000.0] - GPU/0-1e000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254699 [0] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254699 [0] NCCL INFO + PCI[1.5] - NIC/0-3000
macpro2019-01:254561:254699 [0] NCCL INFO ==========================================
macpro2019-01:254561:254699 [0] NCCL INFO GPU/0-b000 :GPU/0-b000 (0/5000.0/LOC) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (4/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (1/5000.0/LOC) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (3/24.0/XGMI) 
macpro2019-01:254561:254699 [0] NCCL INFO GPU/0-e000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (0/5000.0/LOC) GPU/0-1b000 (4/24.0/XGMI) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (1/5000.0/LOC) DEV/0-1b000 (3/24.0/XGMI) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254699 [0] NCCL INFO GPU/0-1b000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (4/24.0/XGMI) GPU/0-1b000 (0/5000.0/LOC) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (3/24.0/XGMI) DEV/0-1b000 (1/5000.0/LOC) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254699 [0] NCCL INFO GPU/0-1e000 :GPU/0-b000 (4/24.0/XGMI) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (0/5000.0/LOC) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (3/24.0/XGMI) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (1/5000.0/LOC) 
macpro2019-01:254561:254699 [0] NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is 0-31. (GPU affinity = 0-31 ; CPU affinity = 0-55).
macpro2019-01:254561:254699 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254699 [0] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1e000 GPU/0-1b000
macpro2019-01:254561:254699 [0] NCCL INFO  1 : GPU/0-b000 GPU/0-1b000 GPU/0-1e000 GPU/0-e000
macpro2019-01:254561:254699 [0] NCCL INFO ringGraph->nChannels = 2 
macpro2019-01:254561:254699 [0] NCCL INFO Pattern 1, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254699 [0] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1b000 GPU/0-1e000
macpro2019-01:254561:254699 [0] NCCL INFO  1 : GPU/0-1e000 GPU/0-e000 GPU/0-b000 GPU/0-1b000
macpro2019-01:254561:254699 [0] NCCL INFO GFX9 cheap fence is OFF
macpro2019-01:254561:254699 [0] NCCL INFO Rank 0: 1 Net devices
macpro2019-01:254561:254699 [0] NCCL INFO Rank 0: 0 CollNet devices
macpro2019-01:254561:254700 [1] NCCL INFO Tuning index set to: 0
macpro2019-01:254561:254700 [1] NCCL INFO === System : maxBw 24.0 totalBw 48.0 ===
macpro2019-01:254561:254700 [1] NCCL INFO CPU/0-0 (1/1/2)
macpro2019-01:254561:254700 [1] NCCL INFO + PCI[12.0] - PCI/0-7000 (10b5874710b58747)
macpro2019-01:254561:254700 [1] NCCL INFO               + PCI[12.0] - PCI/0-9000 (100214a000000000)
macpro2019-01:254561:254700 [1] NCCL INFO                             + PCI[12.0] - DEV/0-b000 (100266a3106b0203)
macpro2019-01:254561:254700 [1] NCCL INFO                                           + LOC[5000.0] - GPU/0-b000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254700 [1] NCCL INFO               + PCI[12.0] - PCI/0-c000 (100214a000000000)
macpro2019-01:254561:254700 [1] NCCL INFO                             + PCI[12.0] - DEV/0-e000 (100266a3106b0203)
macpro2019-01:254561:254700 [1] NCCL INFO                                           + LOC[5000.0] - GPU/0-e000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254700 [1] NCCL INFO + PCI[12.0] - PCI/0-17000 (10b5874710b58747)
macpro2019-01:254561:254700 [1] NCCL INFO               + PCI[12.0] - PCI/0-19000 (100214a000000000)
macpro2019-01:254561:254700 [1] NCCL INFO                             + PCI[12.0] - DEV/0-1b000 (100266a3106b0203)
macpro2019-01:254561:254700 [1] NCCL INFO                                           + LOC[5000.0] - GPU/0-1b000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254700 [1] NCCL INFO               + PCI[12.0] - PCI/0-1c000 (100214a000000000)
macpro2019-01:254561:254700 [1] NCCL INFO                             + PCI[12.0] - DEV/0-1e000 (100266a3106b0203)
macpro2019-01:254561:254700 [1] NCCL INFO                                           + LOC[5000.0] - GPU/0-1e000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254700 [1] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254700 [1] NCCL INFO + PCI[1.5] - NIC/0-3000
macpro2019-01:254561:254700 [1] NCCL INFO ==========================================
macpro2019-01:254561:254700 [1] NCCL INFO GPU/0-b000 :GPU/0-b000 (0/5000.0/LOC) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (4/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (1/5000.0/LOC) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (3/24.0/XGMI) 
macpro2019-01:254561:254700 [1] NCCL INFO GPU/0-e000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (0/5000.0/LOC) GPU/0-1b000 (4/24.0/XGMI) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (1/5000.0/LOC) DEV/0-1b000 (3/24.0/XGMI) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254700 [1] NCCL INFO GPU/0-1b000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (4/24.0/XGMI) GPU/0-1b000 (0/5000.0/LOC) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (3/24.0/XGMI) DEV/0-1b000 (1/5000.0/LOC) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254700 [1] NCCL INFO GPU/0-1e000 :GPU/0-b000 (4/24.0/XGMI) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (0/5000.0/LOC) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (3/24.0/XGMI) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (1/5000.0/LOC) 
macpro2019-01:254561:254700 [1] NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 1 is 0-31. (GPU affinity = 0-31 ; CPU affinity = 0-55).
macpro2019-01:254561:254700 [1] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254700 [1] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1e000 GPU/0-1b000
macpro2019-01:254561:254700 [1] NCCL INFO  1 : GPU/0-b000 GPU/0-1b000 GPU/0-1e000 GPU/0-e000
macpro2019-01:254561:254700 [1] NCCL INFO ringGraph->nChannels = 2 
macpro2019-01:254561:254700 [1] NCCL INFO Pattern 1, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254700 [1] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1b000 GPU/0-1e000
macpro2019-01:254561:254700 [1] NCCL INFO  1 : GPU/0-1e000 GPU/0-e000 GPU/0-b000 GPU/0-1b000
macpro2019-01:254561:254700 [1] NCCL INFO GFX9 cheap fence is OFF
macpro2019-01:254561:254700 [1] NCCL INFO Rank 1: 1 Net devices
macpro2019-01:254561:254700 [1] NCCL INFO Rank 1: 0 CollNet devices
macpro2019-01:254561:254701 [2] NCCL INFO Tuning index set to: 0
macpro2019-01:254561:254701 [2] NCCL INFO === System : maxBw 24.0 totalBw 48.0 ===
macpro2019-01:254561:254701 [2] NCCL INFO CPU/0-0 (1/1/2)
macpro2019-01:254561:254701 [2] NCCL INFO + PCI[12.0] - PCI/0-7000 (10b5874710b58747)
macpro2019-01:254561:254701 [2] NCCL INFO               + PCI[12.0] - PCI/0-9000 (100214a000000000)
macpro2019-01:254561:254701 [2] NCCL INFO                             + PCI[12.0] - DEV/0-b000 (100266a3106b0203)
macpro2019-01:254561:254701 [2] NCCL INFO                                           + LOC[5000.0] - GPU/0-b000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254701 [2] NCCL INFO               + PCI[12.0] - PCI/0-c000 (100214a000000000)
macpro2019-01:254561:254701 [2] NCCL INFO                             + PCI[12.0] - DEV/0-e000 (100266a3106b0203)
macpro2019-01:254561:254701 [2] NCCL INFO                                           + LOC[5000.0] - GPU/0-e000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254701 [2] NCCL INFO + PCI[12.0] - PCI/0-17000 (10b5874710b58747)
macpro2019-01:254561:254701 [2] NCCL INFO               + PCI[12.0] - PCI/0-19000 (100214a000000000)
macpro2019-01:254561:254701 [2] NCCL INFO                             + PCI[12.0] - DEV/0-1b000 (100266a3106b0203)
macpro2019-01:254561:254701 [2] NCCL INFO                                           + LOC[5000.0] - GPU/0-1b000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-1e000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-b000
macpro2019-01:254561:254701 [2] NCCL INFO               + PCI[12.0] - PCI/0-1c000 (100214a000000000)
macpro2019-01:254561:254701 [2] NCCL INFO                             + PCI[12.0] - DEV/0-1e000 (100266a3106b0203)
macpro2019-01:254561:254701 [2] NCCL INFO                                           + LOC[5000.0] - GPU/0-1e000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-1b000
macpro2019-01:254561:254701 [2] NCCL INFO                                           + XGMI[24.0] - DEV/0-e000
macpro2019-01:254561:254701 [2] NCCL INFO + PCI[1.5] - NIC/0-3000
macpro2019-01:254561:254701 [2] NCCL INFO ==========================================
macpro2019-01:254561:254701 [2] NCCL INFO GPU/0-b000 :GPU/0-b000 (0/5000.0/LOC) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (4/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (1/5000.0/LOC) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (3/24.0/XGMI) 
macpro2019-01:254561:254701 [2] NCCL INFO GPU/0-e000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (0/5000.0/LOC) GPU/0-1b000 (4/24.0/XGMI) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (1/5000.0/LOC) DEV/0-1b000 (3/24.0/XGMI) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254701 [2] NCCL INFO GPU/0-1b000 :GPU/0-b000 (3/24.0/XGMI) GPU/0-e000 (4/24.0/XGMI) GPU/0-1b000 (0/5000.0/LOC) GPU/0-1e000 (3/24.0/XGMI) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (2/24.0/XGMI) DEV/0-e000 (3/24.0/XGMI) DEV/0-1b000 (1/5000.0/LOC) DEV/0-1e000 (2/24.0/XGMI) 
macpro2019-01:254561:254701 [2] NCCL INFO GPU/0-1e000 :GPU/0-b000 (4/24.0/XGMI) GPU/0-e000 (3/24.0/XGMI) GPU/0-1b000 (3/24.0/XGMI) GPU/0-1e000 (0/5000.0/LOC) CPU/0-0 (4/12.0/PHB) DEV/0-b000 (3/24.0/XGMI) DEV/0-e000 (2/24.0/XGMI) DEV/0-1b000 (2/24.0/XGMI) DEV/0-1e000 (1/5000.0/LOC) 
macpro2019-01:254561:254701 [2] NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 2 is 0-31. (GPU affinity = 0-31 ; CPU affinity = 0-55).
macpro2019-01:254561:254701 [2] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254701 [2] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1e000 GPU/0-1b000
macpro2019-01:254561:254701 [2] NCCL INFO  1 : GPU/0-b000 GPU/0-1b000 GPU/0-1e000 GPU/0-e000
macpro2019-01:254561:254701 [2] NCCL INFO ringGraph->nChannels = 2 
macpro2019-01:254561:254701 [2] NCCL INFO Pattern 1, crossNic 0, nChannels 2, bw 24.000000/24.000000, type XGMI/PIX, sameChannels 0
macpro2019-01:254561:254701 [2] NCCL INFO  0 : GPU/0-b000 GPU/0-e000 GPU/0-1b000 GPU/0-1e000
macpro2019-01:254561:254701 [2] NCCL INFO  1 : GPU/0-1e000 GPU/0-e000 GPU/0-b000 GPU/0-1b000
macpro2019-01:254561:254701 [2] NCCL INFO GFX9 cheap fence is OFF
macpro2019-01:254561:254701 [2] NCCL INFO Rank 2: 1 Net devices
macpro2019-01:254561:254701 [2] NCCL INFO Rank 2: 0 CollNet devices
macpro2019-01:254561:254702 [3] NCCL INFO comm 0x56af44d4d6d0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
macpro2019-01:254561:254702 [3] NCCL INFO Tree 1 : -1 -> 3 -> 1/-1/-1
macpro2019-01:254561:254702 [3] NCCL INFO Tree 3 : -1 -> 3 -> 1/-1/-1
macpro2019-01:254561:254702 [3] NCCL INFO Ring 0 : 1 -> 3 -> 2 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 1 : 2 -> 3 -> 1 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 2 : 1 -> 3 -> 2 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 3 : 2 -> 3 -> 1 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 4 : 1 -> 3 -> 2 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 5 : 2 -> 3 -> 1 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 6 : 1 -> 3 -> 2 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Ring 7 : 2 -> 3 -> 1 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 1/-1/-1->3->-1 [2] -1/-1/-1->3->2 [3] 1/-1/-1->3->-1 [4] -1/-1/-1->3->2 [5] 1/-1/-1->3->-1 [6] -1/-1/-1->3->2 [7] 1/-1/-1->3->-1 comm 0x56af44d4d6d0 nRanks 04 busId 1e000
macpro2019-01:254561:254702 [3] NCCL INFO P2P Chunksize set to 524288
macpro2019-01:254561:254702 [3] NCCL INFO ncclOsDlopen(libnccl-profiler.so) failed: libnccl-profiler.so: cannot open shared object file: No such file or directory
macpro2019-01:254561:254702 [3] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so
macpro2019-01:254561:254702 [3] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
macpro2019-01:254561:254746 [0] NCCL INFO [Proxy Service] Device 3 CPU core 28
macpro2019-01:254561:254746 [3] NCCL INFO proxy listening socket at 192.168.1.2<60055>
macpro2019-01:254561:254702 [3] NCCL INFO ncclP2pSchedule: group size used is 4
macpro2019-01:254561:254747 [0] NCCL INFO [Proxy Service UDS] Device 3 CPU core 30
macpro2019-01:254561:254699 [0] NCCL INFO Local Net device counts across ranks: min 1 max 1
macpro2019-01:254561:254699 [0] NCCL INFO Local CollNet device counts across ranks: min 0 max 0
macpro2019-01:254561:254699 [0] NCCL INFO comm 0x56af439d46e0 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
macpro2019-01:254561:254700 [1] NCCL INFO comm 0x56af43b3a4e0 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
macpro2019-01:254561:254699 [0] NCCL INFO [RINGS]      00     01
macpro2019-01:254561:254699 [0] NCCL INFO [RINGS]  00->02 00->01
macpro2019-01:254561:254699 [0] NCCL INFO Tree 0 : -1 -> 0 -> 1/-1/-1
macpro2019-01:254561:254700 [1] NCCL INFO Tree 0 : 0 -> 1 -> 2/-1/-1
macpro2019-01:254561:254700 [1] NCCL INFO Tree 2 : 0 -> 1 -> 2/-1/-1
macpro2019-01:254561:254700 [1] NCCL INFO Tree 1 : 3 -> 1 -> 0/-1/-1
macpro2019-01:254561:254700 [1] NCCL INFO Tree 3 : 3 -> 1 -> 0/-1/-1
macpro2019-01:254561:254700 [1] NCCL INFO Ring 0 : 0 -> 1 -> 3 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 1 : 3 -> 1 -> 0 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 2 : 0 -> 1 -> 3 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 3 : 3 -> 1 -> 0 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 4 : 0 -> 1 -> 3 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 5 : 3 -> 1 -> 0 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 6 : 0 -> 1 -> 3 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Ring 7 : 3 -> 1 -> 0 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254700 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 0/-1/-1->1->3 [2] 2/-1/-1->1->0 [3] 0/-1/-1->1->3 [4] 2/-1/-1->1->0 [5] 0/-1/-1->1->3 [6] 2/-1/-1->1->0 [7] 0/-1/-1->1->3 comm 0x56af43b3a4e0 nRanks 04 busId e000
macpro2019-01:254561:254701 [2] NCCL INFO comm 0x56af43c170c0 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
macpro2019-01:254561:254700 [1] NCCL INFO P2P Chunksize set to 524288
macpro2019-01:254561:254700 [1] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
macpro2019-01:254561:254701 [2] NCCL INFO Ring 0 : 3 -> 2 -> 0 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 1 : 0 -> 2 -> 3 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 2 : 3 -> 2 -> 0 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 3 : 0 -> 2 -> 3 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 4 : 3 -> 2 -> 0 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 5 : 0 -> 2 -> 3 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 6 : 3 -> 2 -> 0 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Ring 7 : 0 -> 2 -> 3 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] -1/-1/-1->2->0 [2] 3/-1/-1->2->1 [3] -1/-1/-1->2->0 [4] 3/-1/-1->2->1 [5] -1/-1/-1->2->0 [6] 3/-1/-1->2->1 [7] -1/-1/-1->2->0 comm 0x56af43c170c0 nRanks 04 busId 1b000
macpro2019-01:254561:254701 [2] NCCL INFO P2P Chunksize set to 524288
macpro2019-01:254561:254701 [2] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
macpro2019-01:254561:254699 [0] NCCL INFO Tree 2 : -1 -> 0 -> 1/-1/-1
macpro2019-01:254561:254699 [0] NCCL INFO Channel 00/08 : 0 1 3 2
macpro2019-01:254561:254699 [0] NCCL INFO Channel 01/08 : 0 2 3 1
macpro2019-01:254561:254699 [0] NCCL INFO Channel 02/08 : 0 1 3 2
macpro2019-01:254561:254699 [0] NCCL INFO Channel 03/08 : 0 2 3 1
macpro2019-01:254561:254699 [0] NCCL INFO Channel 04/08 : 0 1 3 2
macpro2019-01:254561:254699 [0] NCCL INFO Channel 05/08 : 0 2 3 1
macpro2019-01:254561:254749 [0] NCCL INFO [Proxy Service] Device 1 CPU core 30
macpro2019-01:254561:254699 [0] NCCL INFO Channel 06/08 : 0 1 3 2
macpro2019-01:254561:254699 [0] NCCL INFO Channel 07/08 : 0 2 3 1
macpro2019-01:254561:254701 [2] NCCL INFO ncclP2pSchedule: group size used is 4
macpro2019-01:254561:254700 [1] NCCL INFO ncclP2pSchedule: group size used is 4
macpro2019-01:254561:254749 [1] NCCL INFO proxy listening socket at 192.168.1.2<42541>
macpro2019-01:254561:254751 [0] NCCL INFO [Proxy Service UDS] Device 1 CPU core 30
macpro2019-01:254561:254748 [0] NCCL INFO [Proxy Service] Device 2 CPU core 31
macpro2019-01:254561:254748 [2] NCCL INFO proxy listening socket at 192.168.1.2<48259>
macpro2019-01:254561:254750 [0] NCCL INFO [Proxy Service UDS] Device 2 CPU core 19
macpro2019-01:254561:254699 [0] NCCL INFO Ring 0 : 2 -> 0 -> 1 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 1 : 1 -> 0 -> 2 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 2 : 2 -> 0 -> 1 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 3 : 1 -> 0 -> 2 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 4 : 2 -> 0 -> 1 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 5 : 1 -> 0 -> 2 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 6 : 2 -> 0 -> 1 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Ring 7 : 1 -> 0 -> 2 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 2/-1/-1->0->1 [2] 1/-1/-1->0->-1 [3] 2/-1/-1->0->1 [4] 1/-1/-1->0->-1 [5] 2/-1/-1->0->1 [6] 1/-1/-1->0->-1 [7] 2/-1/-1->0->1 comm 0x56af439d46e0 nRanks 04 busId b000
macpro2019-01:254561:254699 [0] NCCL INFO P2P Chunksize set to 524288
macpro2019-01:254561:254699 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
macpro2019-01:254561:254752 [0] NCCL INFO [Proxy Service] Device 0 CPU core 28
macpro2019-01:254561:254752 [0] NCCL INFO proxy listening socket at 192.168.1.2<54495>
macpro2019-01:254561:254699 [0] NCCL INFO ncclP2pSchedule: group size used is 4
macpro2019-01:254561:254753 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 28
macpro2019-01:254561:254701 [2] NCCL INFO Channel 01/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 00/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 03/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 02/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 05/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 04/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 07/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 06/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 00/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 01/0 : 3[1e000] -> 1[e000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 02/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 01/0 : 0[b000] -> 2[1b000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 03/0 : 3[1e000] -> 1[e000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 04/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 03/0 : 0[b000] -> 2[1b000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 05/0 : 3[1e000] -> 1[e000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 06/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 05/0 : 0[b000] -> 2[1b000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 07/0 : 3[1e000] -> 1[e000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 00/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 07/0 : 0[b000] -> 2[1b000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 02/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 04/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 06/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 01/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 00/0 : 3[1e000] -> 2[1b000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 03/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 02/0 : 3[1e000] -> 2[1b000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 05/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 04/0 : 3[1e000] -> 2[1b000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 07/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Channel 06/0 : 3[1e000] -> 2[1b000] via P2P/direct pointer comm 0x56af44d4d6d0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
macpro2019-01:254561:254699 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
macpro2019-01:254561:254702 [3] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
macpro2019-01:254561:254699 [0] NCCL INFO Channel 01/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
macpro2019-01:254561:254699 [0] NCCL INFO Channel 03/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 05/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254699 [0] NCCL INFO Channel 07/0 : 0[b000] -> 1[e000] via P2P/direct pointer comm 0x56af439d46e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 00/0 : 1[e000] -> 2[1b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 00/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 02/0 : 1[e000] -> 2[1b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 02/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 04/0 : 1[e000] -> 2[1b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 04/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 06/0 : 1[e000] -> 2[1b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 06/0 : 2[1b000] -> 3[1e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 01/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 01/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 03/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 03/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 05/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 05/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 07/0 : 1[e000] -> 3[1e000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 07/0 : 2[1b000] -> 0[b000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 00/0 : 2[1b000] -> 1[e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 02/0 : 2[1b000] -> 1[e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 04/0 : 2[1b000] -> 1[e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254701 [2] NCCL INFO Channel 06/0 : 2[1b000] -> 1[e000] via P2P/direct pointer comm 0x56af43c170c0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 00/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 02/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 04/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254700 [1] NCCL INFO Channel 06/0 : 1[e000] -> 0[b000] via P2P/direct pointer comm 0x56af43b3a4e0 nRanks 04
macpro2019-01:254561:254702 [3] NCCL INFO Connected all trees
macpro2019-01:254561:254699 [0] NCCL INFO Connected all trees
macpro2019-01:254561:254701 [2] NCCL INFO Connected all trees
macpro2019-01:254561:254700 [1] NCCL INFO Connected all trees
macpro2019-01:254561:254755 [0] NCCL INFO [Proxy Progress] Device 3 CPU core 30
macpro2019-01:254561:254756 [0] NCCL INFO [Proxy Progress] Device 2 CPU core 3
macpro2019-01:254561:254757 [0] NCCL INFO [Proxy Progress] Device 0 CPU core 0
macpro2019-01:254561:254758 [0] NCCL INFO [Proxy Progress] Device 1 CPU core 30
macpro2019-01:254561:254702 [3] NCCL INFO ncclOsDlopen(libnccl-tuner.so) failed: libnccl-tuner.so: cannot open shared object file: No such file or directory
macpro2019-01:254561:254702 [3] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so
macpro2019-01:254561:254702 [3] NCCL INFO RCCL Tuning index:0
macpro2019-01:254561:254702 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
macpro2019-01:254561:254702 [3] NCCL INFO comm:0x56af44d4d6d0, nRanks:4, nNodes:1, coll channels:8 collnet channels:8, nvls channels:0, p2p channels:8, p2p channels per peer:2, shiftSize:-1
macpro2019-01:254561:254702 [3] NCCL INFO Symmetric memory is not supported. cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1
macpro2019-01:254561:254699 [0] NCCL INFO RCCL Tuning index:0
macpro2019-01:254561:254699 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
macpro2019-01:254561:254699 [0] NCCL INFO comm:0x56af439d46e0, nRanks:4, nNodes:1, coll channels:8 collnet channels:8, nvls channels:0, p2p channels:8, p2p channels per peer:2, shiftSize:-1
macpro2019-01:254561:254699 [0] NCCL INFO Symmetric memory is not supported. cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1
macpro2019-01:254561:254701 [2] NCCL INFO RCCL Tuning index:0
macpro2019-01:254561:254701 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
macpro2019-01:254561:254701 [2] NCCL INFO comm:0x56af43c170c0, nRanks:4, nNodes:1, coll channels:8 collnet channels:8, nvls channels:0, p2p channels:8, p2p channels per peer:2, shiftSize:-1
macpro2019-01:254561:254701 [2] NCCL INFO Symmetric memory is not supported. cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1
macpro2019-01:254561:254699 [0] NCCL INFO CC Off, workFifoBytes 4194304
macpro2019-01:254561:254700 [1] NCCL INFO RCCL Tuning index:0
macpro2019-01:254561:254700 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
macpro2019-01:254561:254700 [1] NCCL INFO comm:0x56af43b3a4e0, nRanks:4, nNodes:1, coll channels:8 collnet channels:8, nvls channels:0, p2p channels:8, p2p channels per peer:2, shiftSize:-1
macpro2019-01:254561:254700 [1] NCCL INFO Symmetric memory is not supported. cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1
macpro2019-01:254561:254701 [2] NCCL INFO RCCL Unroll Factor (pre-set): 4
macpro2019-01:254561:254701 [2] NCCL INFO ncclCommInitAll_impl comm 0x56af43c170c0 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 1b000 commId 0x1b36da65599e1fd7 - Init COMPLETE
macpro2019-01:254561:254701 [2] NCCL INFO Init timings - ncclCommInitAll_impl: rank 2 nranks 4 total 3.39 (kernels 2.92, alloc 0.02, bootstrap 0.06, allgathers 0.00, topo 0.35, graphs 0.00, connections 0.04, rest 0.00)
macpro2019-01:254561:254700 [1] NCCL INFO RCCL Unroll Factor (pre-set): 4
macpro2019-01:254561:254699 [0] NCCL INFO RCCL Unroll Factor (pre-set): 4
macpro2019-01:254561:254702 [3] NCCL INFO RCCL Unroll Factor (pre-set): 4
macpro2019-01:254561:254700 [1] NCCL INFO ncclCommInitAll_impl comm 0x56af43b3a4e0 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId e000 commId 0x1b36da65599e1fd7 - Init COMPLETE
macpro2019-01:254561:254700 [1] NCCL INFO Init timings - ncclCommInitAll_impl: rank 1 nranks 4 total 3.39 (kernels 2.98, alloc 0.01, bootstrap 0.00, allgathers 0.02, topo 0.33, graphs 0.00, connections 0.04, rest 0.00)
macpro2019-01:254561:254702 [3] NCCL INFO ncclCommInitAll_impl comm 0x56af44d4d6d0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 1e000 commId 0x1b36da65599e1fd7 - Init COMPLETE
macpro2019-01:254561:254702 [3] NCCL INFO Init timings - ncclCommInitAll_impl: rank 3 nranks 4 total 3.39 (kernels 2.95, alloc 0.01, bootstrap 0.03, allgathers 0.05, topo 0.30, graphs 0.00, connections 0.04, rest 0.00)
macpro2019-01:254561:254699 [0] NCCL INFO ncclCommInitAll_impl comm 0x56af439d46e0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId b000 commId 0x1b36da65599e1fd7 - Init COMPLETE
macpro2019-01:254561:254699 [0] NCCL INFO Init timings - ncclCommInitAll_impl: rank 0 nranks 4 total 3.39 (kernels 1.81, alloc 0.01, bootstrap 1.17, allgathers 0.04, topo 0.31, graphs 0.00, connections 0.04, rest 0.00)

llama_batched_bench: n_kv_max = 18432, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   19.266 |   850.39 |    6.604 |   155.06 |   25.870 |   672.90 |

macpro2019-01:254561:254561 [3] NCCL INFO Memory used = 193020224
macpro2019-01:254561:254561 [3] NCCL INFO Memory used = 209797440
macpro2019-01:254561:254561 [3] NCCL INFO Memory used = 184631616
macpro2019-01:254561:254561 [3] NCCL INFO Memory used = 184631616
