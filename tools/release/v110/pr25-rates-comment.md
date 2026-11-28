The rates the description promised, at full clock (same box: four gfx906 dies, Qwen3.8-27B Q8_0, `-sm tensor`, MTP `--spec-draft-n-max 3`, one session, one request at a time). "Long" is a 2,089-token prompt and about 137 generated tokens, once per arm; "short" is an 11-token prompt and 256 generated tokens, three times per arm.

| Arm | long: prefill tok/s | long: decode tok/s | short: decode tok/s | short: prompt tok/s |
|---|---|---|---|---|
| ring, `-np 1` | 556.9 | 62.9 | 73.1 - 73.9 | 34 - 39 |
| ring with `LLAMA_RS_REPLAY=1` (must be the ring again) | 549.6 | 62.6 | 73.1 - 73.8 | 34 - 39 |
| snapshots, `-np 4` (the default) | 534.6 | 74.1 | 78.8 - 79.3 | 38 - 43 |
| replay, `-np 4`, `LLAMA_RS_REPLAY=1` | 449.6 | 70.0 | 75.3 - 75.9 | 17 - 18 |
| snapshots on the recurrent kernel, `-np 4` | 527.2 | 70.3 | 76.3 - 77.1 | 35 - 40 |

Arms that write different text do not do equal work in decode (the accepted drafts differ), so the comparison to read is replay against snapshots on the recurrent kernel, which write the same text on all four requests: prefill -15 %, decode -0.4 % (long) and -1.5 % (short), and an 11-token prompt at less than half the rate. Against the default snapshots, which also run the faster GDN kernels, replay is -16 % in prefill and about -4 to -5 % in decode. Replay needs 653 MiB of recurrent state where the snapshots need 2,394 MiB. That trade is why this branch keeps the snapshots as the default and leaves replay behind `LLAMA_RS_REPLAY=1`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
