patch:            substrate-de27e7509 | https://github.com/mxxm-t/mx-llama.cpp | de27e7509 | kv-cache: keep a reused layer's KV cache on the device that reads it (replica)
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: inactive here: its own message — "Replicas exist only under -sm layer on more than one device; a single device, -sm tensor and models without KV reuse build the same graph as before" (this instrument: -sm tensor, Qwen3.8-27B has no KV reuse)
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  the service runs -sm layer across devices with a KV-reuse model (gemma-4, DeepSeek-V4.1): its own measurements +21% / +128% tg there
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
