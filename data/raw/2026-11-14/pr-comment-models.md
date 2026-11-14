Follow-up, as promised above: DeepSeek-V4.1 with DSpark, Nemotron 3.5 Bolt with LiLiCorr, and a correction to the `test-llama-context` count. Same box (4x gfx906, ROCm 10.0); stock is llama.cpp v1.0.0 (`b1b770d`), "this branch" is `a73c601e6`.

**DeepSeek-V4.1-Flash-1017 UD-IQ1_S** (auto-fit offload, greedy)

| | stock v1.0.0 | this branch |
|---|---|---|
| `-sm layer`, 64 tokens | coherent, 23.5 tok/s | coherent, 23.5 tok/s, same text as on #23 |
| `-sm tensor`, 64 tokens | not run | coherent, 30.9 tok/s, same text as on #23 |
| perplexity 2K x 4, `-sm layer` | 2.2264 +/- 0.0599 | 2.2164 +/- 0.0595 (#23: 2.2164) |
| server without a drafter, 192 tokens, two prompts | 15.1 / 15.6 tok/s | 18.2 / 18.9 tok/s |
| server with DSpark, 192 tokens, `ignore_eos` | 124/199 accepted at 20.9; 129/185 at 23.0 tok/s | 115/227 at 20.1; 129/185 at 25.3 tok/s |

On this branch DSpark gives the same text and acceptance on all three feature paths, and the same as on #23: deferred prompt encode (your default) 20.1 / 25.3 tok/s, `LLAMA_DFLASH_DEFER=0` (upstream's device buffer) 21.1 / 25.9, and with `LLAMA_DFLASH_DEV=0` as well (host copies) 20.5 / 25.9. The servers ran with `--fit-target 7168`: without it the fit fills the dies with the target and the drafter cannot allocate, on stock too.

**Nemotron 3.5 Bolt 74B-A9B** (`nemotron_h_moe`, one die)

- Perplexity 4K x 4: 4.8620 +/- 0.1243 on stock and on this branch. Greedy decode 38.3 tok/s on stock, 40.6 here.
- LiLiCorr (`--spec-type draft-dflash --spec-draft-n-max 6`), four prompts, 192 tokens: 506/968 drafts accepted on stock, 481/1076 here. Both are identical, prompt by prompt, to the v0.9.0 builds of #23.
- `-sm tensor` is refused for this architecture, as before.

**`test-llama-context` on four dies: the count is 142 or 143 of 144, not always 142.** `test-llama-archs` writes the dummy models with new random weights on every run, and the result moves with them. In three four-die runs (this branch twice, our downstream branch once) `gemma4-dense` failed `cp_h` / `cp_d` every time and `mistral-large4-moe` in two of the three. The cause and the workaround (`LLAMA_KV_REPLICA=0`, or one visible die) are as described in the PR text.
