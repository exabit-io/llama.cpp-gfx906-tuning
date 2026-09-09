# qwen38-27b-mxxmfh-powercap-floor  2026-09-08T02:03:12  adaptive power-cap search, production build, llama-server tp4 -np 16; waves of 32 x (1300 in / 256 out) at 16 clients; energy from the 1 s sampler (dies: hwmon power1_input; bays: SMC PZ3G+PZ4G)

## Stage A: coarse ladder [200, 65, 40, 20, 10], 1 down/up pass pairs, 16 clients

| cap W | samples | gen tok/kJ bays (geomean) | +/- (log-sd) | vs best | agg gen tok/s | vs 200 W | TTFT s | bays W | die temp C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2 | 105.0 | 1.0% | -28.5% | 87.3 | +0.0% | 5.23 | 843 | 70 |
| 65 | 2 | 146.8 | 0.5% | -0.2% | 60.0 | -31.3% | 7.93 | 413 | 49 |
| 40 | 2 | 147.0 | 0.2% | -0.0% | 60.0 | -31.2% | 7.96 | 411 | 47 |
| 20 | 2 | 147.0 | 0.0% | -0.0% | 59.9 | -31.4% | 7.94 | 410 | 47 |
| 10 | 2 | 147.0 | 0.0% | +0.0% | 60.0 | -31.2% | 7.95 | 410 | 47 |

Pooled between-sample spread (log-sd): 0.7%

Coarse winner: 10 W
