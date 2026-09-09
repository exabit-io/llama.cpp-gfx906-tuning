# qwen38-27b-mxxmfh-powercap-adaptive  2026-09-07T22:54:04  adaptive power-cap search, production build, llama-server tp4 -np 16; waves of 32 x (1300 in / 256 out) at 16 clients; energy from the 1 s sampler (dies: hwmon power1_input; bays: SMC PZ3G+PZ4G)

## Stage A: coarse ladder [200, 185, 170, 155, 140, 125, 110, 95, 80], 1 down/up pass pairs, 16 clients

| cap W | samples | gen tok/kJ bays (geomean) | +/- (log-sd) | vs best | agg gen tok/s | vs 200 W | TTFT s | bays W | die temp C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2 | 105.0 | 0.0% | -28.4% | 87.1 | +0.0% | 5.24 | 843 | 73 |
| 185 | 2 | 108.0 | 0.1% | -26.4% | 86.6 | -0.6% | 5.32 | 811 | 71 |
| 170 | 2 | 113.0 | 0.0% | -23.1% | 84.7 | -2.7% | 5.46 | 759 | 69 |
| 155 | 2 | 117.9 | 0.5% | -19.7% | 82.5 | -5.3% | 5.61 | 705 | 65 |
| 140 | 2 | 124.1 | 0.3% | -15.4% | 79.4 | -8.9% | 5.83 | 644 | 61 |
| 125 | 2 | 130.9 | 0.1% | -10.8% | 75.3 | -13.5% | 6.13 | 579 | 57 |
| 110 | 2 | 137.0 | 0.2% | -6.7% | 70.0 | -19.6% | 6.58 | 515 | 53 |
| 95 | 2 | 141.9 | 0.1% | -3.3% | 63.3 | -27.4% | 7.30 | 451 | 50 |
| 80 | 2 | 146.8 | 0.1% | +0.0% | 60.3 | -30.8% | 7.88 | 413 | 47 |

Pooled between-sample spread (log-sd): 0.3%

Coarse winner: 80 W

## Stage B: fine ladder [65, 70, 75, 80, 85, 90, 95] (5 W steps), down/up pass pairs, 16 clients

| cap W | samples | gen tok/kJ bays (geomean) | +/- (log-sd) | vs best | agg gen tok/s | vs 200 W | TTFT s | bays W | die temp C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 95 | 8 | 141.1 | 0.4% | -4.3% | 63.3 | +nan% | 7.30 | 450 | 49 |
| 90 | 8 | 143.7 | 0.2% | -2.6% | 61.5 | +nan% | 7.60 | 431 | 48 |
| 85 | 8 | 146.0 | 0.2% | -1.0% | 60.6 | +nan% | 7.80 | 419 | 47 |
| 80 | 8 | 147.3 | 0.3% | -0.1% | 60.3 | +nan% | 7.86 | 414 | 47 |
| 75 | 8 | 147.2 | 0.3% | -0.1% | 60.2 | +nan% | 7.91 | 412 | 47 |
| 70 | 8 | 147.4 | 0.3% | +0.0% | 60.2 | +nan% | 7.92 | 411 | 47 |
| 65 | 8 | 147.3 | 0.3% | -0.1% | 60.1 | +nan% | 7.93 | 411 | 47 |

Pooled between-sample spread (log-sd): 0.3%

stopped at the pair cap (4): lead 0.1% vs pooled spread 0.3%. Fine winner: 70 W

## Validation: 70 W vs 200 W at 8 and 16 clients, two samples each

| cap W | clients | samples | gen tok/kJ bays | gen tok/kJ dies | agg gen tok/s | total tok/s | TTFT s | bays W | dies W | die temp C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 8 | 2 | 110.0 | 127.5 | 40.8 | 248 | 7.18 | 372 | 321 | 48 |
| 70 | 16 | 2 | 147.1 | 181.2 | 60.1 | 365 | 7.92 | 411 | 334 | 47 |
| 200 | 8 | 2 | 83.5 | 102.0 | 55.5 | 337 | 4.77 | 672 | 550 | 58 |
| 200 | 16 | 2 | 105.0 | 126.6 | 88.5 | 538 | 5.24 | 848 | 704 | 70 |

## Trade-off over every stage A/B sample: cap, gen tok/kJ (bays), throughput vs 200 W

| cap W | samples | gen tok/kJ bays | vs 200 W per-kJ | agg gen tok/s | vs 200 W |
|---:|---:|---:|---:|---:|---:|
| 200 | 2 | 105.0 | +0.0% | 87.1 | +0.0% |
| 185 | 2 | 108.0 | +2.8% | 86.6 | -0.6% |
| 170 | 2 | 113.0 | +7.5% | 84.7 | -2.7% |
| 155 | 2 | 117.9 | +12.2% | 82.5 | -5.3% |
| 140 | 2 | 124.1 | +18.2% | 79.4 | -8.9% |
| 125 | 2 | 130.9 | +24.6% | 75.3 | -13.5% |
| 110 | 2 | 137.0 | +30.4% | 70.0 | -19.6% |
| 95 | 10 | 141.3 | +34.5% | 63.3 | -27.4% |
| 90 | 8 | 143.7 | +36.8% | 61.5 | -29.4% |
| 85 | 8 | 146.0 | +39.0% | 60.6 | -30.4% |
| 80 | 10 | 147.2 | +40.1% | 60.3 | -30.8% |
| 75 | 8 | 147.2 | +40.1% | 60.2 | -30.9% |
| 70 | 8 | 147.4 | +40.4% | 60.2 | -30.9% |
| 65 | 8 | 147.3 | +40.2% | 60.1 | -31.0% |

Verdict: best serving performance per watt at 70 W per die (16 clients, production build), throughput -30.9% vs 200 W. Best cap within a throughput floor: -2%: 185 W; -5%: 170 W; -10%: 140 W.

## Reading the result (added 2026-09-08 02:05 after the run)

The fine stage found a floor, not a peak. From 85 W down to 65 W every sample is the same operating point: 60.0-60.6 tok/s, 411-419 W at the bays,
331-343 W at the dies (83-86 W each), 146-148 gen tok/kJ. The clock sampler shows the dies at sclk level 0/1 (999-1000 MHz, the firmware's
minimum GFX clock) for ~80% of samples at the 65 W cap; mclk (1000 MHz), socclk (971 MHz) and fclk (1166 MHz) have no DPM on this firmware.
The SMU has no lower state, so caps below about 85 W are not honoured: the die draws ~83 W under this load whatever the cap says (idle is ~34 W).
The cap starts to bind at about 85-90 W (dies mix 1000/1204 MHz at 90 W, 89-93 W each at 95 W). The "70 W winner" is the plateau picking a
label by noise (four pairs, lead 0.1% vs 0.3% spread); read it as "cap <= 85 W".

Validation at 8 clients: the floor point gives 110 gen tok/kJ at 40.8 tok/s against 83.5 at 55.5 tok/s for 200 W (+32% per kJ, -26% throughput).
Concurrency moves per-kJ more than the cap does: 200 W at 16 clients (105) is about the same per kJ as the floor at 8 clients (110).

Recommendation table (16 clients, production build, bays energy):

| goal | cap per die | tok/s | gen tok/kJ | note |
|---|---:|---:|---:|---|
| max throughput | 200 | 87.1 | 105 | reference |
| within 2% | 185 | 86.6 | 108 | +3% per kJ |
| within 5% | 170 | 84.7 | 113 | +8% per kJ |
| within 10% | 140 | 79.4 | 124 | +18% per kJ |
| max per-watt | 85 (or anything lower) | 60.6 | 146 | DPM floor; -30% throughput, +39% per kJ |
