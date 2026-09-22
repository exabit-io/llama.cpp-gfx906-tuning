# probe-alloc — per-die memory itemisation for the capacity ladder (R3.2)   2026-09-19T20:32:01+00:00

Buffer sizes depend on `-c`, `-npl` and the KV type, not on prompt length, so these are the
same allocations the ladder made. Budget is **31 GiB per die** (R3.2); weights ~6.3 GiB/die.

| slots | depth | offload | model MiB/die | KV MiB/die | compute MiB/die | sum GiB/die | headroom GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 64K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 8 | 192K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 4 | 256K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 8 | 64K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 4 | 128K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 8 | 96K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 8 | 128K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |
| 4 | 192K | offloaded 66/66 layers to GPU | ? | ? | ? | 0.00 | 31.00 |

# done 2026-09-19T20:34:08+00:00
