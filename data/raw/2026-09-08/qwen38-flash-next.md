# qwen38-flash-next  2026-09-08T15:38:45+00:00  /opt/llama.cpp-gfx906-master, four dies -sm tensor, model 108G (--load-mode dio)

## llama-bench -p 2048 -n 128 -r 2, tp4
| PLE shard | pp2048 | tg128 |
|---|---:|---:|
| 1 | 365.24 ± 0.52 | 20.64 ± 8.00 |  |
| 0 | 357.87 ± 0.41 | 17.74 ± 4.13 |  |

## perplexity -c 2048 --chunks 4 (PLE shard on): Final estimate: PPL = 1.9264 +/- 0.04984

## greedy sample (120 tokens)
```
Explain in three sentences why a tensor split over four GPUs needs an allreduce after every layer.

<think>
The user is asking about distributed training with tensors split across four GPUs and why an allreduce is needed after every layer. Let me think carefully about this.

Actually, I need to be precise here. The question says "a tensor split over four GPUs." This could refer to different parallelism strategies:

1. **Data parallelism**: Each GPU has a full copy of the model, but processes different data batches. After the forward pass, gradients are computed locally, and an allreduce is needed to average gradients across GPUs before the optimizer step. But this is after the backward


```
# done 2026-09-08T15:49:55+00:00
