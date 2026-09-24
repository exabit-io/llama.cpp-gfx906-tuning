# Fan power at idle — what max-RPM testing has been spending on cooling   2026-09-20T00:51:02+00:00

GPUs idle, host idle, same state otherwise. Median SMC DC total (PZ0G) over 90 s.

| fan mode | DC total W | fan RPM (1/2/3/4) |
|---|---:|---|
| max (all corpus testing) | 255.1 | 1200/2509/2510/2499 |
| production PWM curve | 233.9 | 496/486/486/493 |

**Fan power delta at idle: 21.2 W** of the DC total.

Note: at idle the PWM curve sits near its floor, so this is close to the MAXIMUM
delta. Under serving load the production curve ramps and the gap narrows.
