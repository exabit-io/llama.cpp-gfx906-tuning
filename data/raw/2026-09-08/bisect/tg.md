| build | tg128 mean ± sd | samples |
|---|---:|---|

## per-sample tp4 tg128 x5 (post22c)
| r2-repack-on | 48.3 ± 12.1 | 41.5 30.0 56.6 56.6 56.6 |
| r2-repack-off | 55.3 ± 1.8 | 52.8 53.8 56.7 56.5 56.5 |
| production | 56.8 ± 1.6 | 53.9 57.5 57.7 57.6 57.5 |
| fork-b10912-pristine | 49.2 ± 8.3 | 42.5 38.2 55.3 55.2 55.0 |
| upstream-master-pristine | 45.5 ± 2.3 | 41.4 46.6 46.6 46.6 46.5 |
| production-0907 | 54.3 ± 1.9 | 50.9 54.3 55.0 55.6 55.5 |

## fork-history bisect round 1 (first-parent b10254..b10912, positions 16/32/48/64/80/96/112)
| fork-115a32c1e | 41.6 ± 16.7 | 30.4 17.6 53.5 53.5 53.1 |
| fork-16628b027 | 40.1 ± 19.6 | 23.1 14.7 54.1 54.4 54.4 |
| fork-803f00bb7 | 53.6 ± 0.9 | 52.2 53.2 54.3 54.3 54.1 |
| fork-8253e3fbf | 53.0 ± 1.7 | 50.7 51.8 54.3 54.3 54.0 |
| fork-9435cfcc4 | 47.0 ± 12.3 | 27.8 41.8 55.4 55.3 55.0 |
| fork-a65e71eb2 | 41.1 ± 19.3 | 23.3 16.9 55.4 55.4 54.5 |
| fork-def64b335 | 42.0 ± 19.2 | 25.4 17.0 55.9 55.9 55.6 |

## fork-history bisect round 2 (positions 33..47, oldest first)
| fork-e21ccb704 | 43.5 ± 14.5 | 31.8 24.0 54.1 54.0 53.7 |
| fork-97e14020c | 51.7 ± 3.0 | 49.5 47.5 54.1 54.0 53.7 |

## repack switch vs code (post22f): -nr 1 = --no-repack
| fork-32-8253e3fbf-nr1 | fail | Expecting value: line 1 column 1 (char 0) |
| fork-32-8253e3fbf-nr0 | fail | Expecting value: line 1 column 1 (char 0) |
| fork-33-e21ccb704-nr1 | fail | Expecting value: line 1 column 1 (char 0) |
| fork-33-e21ccb704-nr0 | fail | Expecting value: line 1 column 1 (char 0) |
| fork-b10912-nr1 | 54.3 ± 1.4 | 52.3 53.1 55.4 55.3 55.1 |
| r2-nr1-again | 57.4 ± 1.4 | 55.4 56.7 58.5 58.3 58.2 |
| r2-nr0-again | 43.2 ± 19.1 | 27.2 17.9 57.1 56.9 56.8 |
| production-again | 58.4 ± 1.4 | 55.9 59.1 59.1 59.0 58.9 |

## per-sample tg128 x5
| r3-nr1 | 57.8 ± 1.7 | 55.4 56.7 59.2 59.0 58.9 |
| r3-nr0 | 56.6 ± 1.8 | 54.5 54.8 58.0 57.9 57.8 |
