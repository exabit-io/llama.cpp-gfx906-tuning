patch:            v-cache-type-sweep | GGML_CUDA_FA_QUANTS + runtime -ctv | q4_1 / q5_0 / q5_1 as the V cache | does the q4_0-V decode win continue at other widths?
axis:             multi-user
zero point:       build-faq with -ctv q8_0
recipe:           would be 4x64K and 1x254K, plus 8 slots (untested for q4_0 as well) | -ctk q8_0 | 125 W/die
metric:           decode tok/s/slot; improves requires q<0.10 and effect >= +2%
result:           NOT MEASURED. q5_1 was run once as a diagnostic on a build without its kernel and is not a result.
effect:           unknown
stats:            n/a
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              technique-requires-implementation
would change if:  a sweep is run with each combination actually compiled into GGML_CUDA_FA_QUANTS
notes:            WHAT HAPPENED TO q5_1, recorded because it is instructive rather than useful.
                  I ran `-ctv q5_1` once on build-faq, whose FA_QUANTS does not include q8_0-q5_1, and
                  read the resulting 8% decode drop as proof that uncompiled combinations "fall back
                  silently". Two errors:
                  1. CONFOUNDED. q5_1 carries 6.0 bits per value against q4_0's 4.5 -- 33% more V bytes.
                     On a bandwidth-bound decode some of that 8% is simply the wider cache. The test
                     varied quantisation width AND compiled-vs-uncompiled at once.
                  2. NOT SILENT. The log said exactly what happened: "no FlashAttention vector kernel
                     compiled for K/V types q8_0-q5_1, converting K and V to f16 instead (slow). Add
                     q8_0-q5_1 to GGML_CUDA_FA_QUANTS to compile it." I asserted the opposite from
                     inference while the answer sat in a log I had already captured.
                  The real mechanism is an f16 CONVERSION, not a generic slow path, and it is detectable
                  in any run log -- now enforced by tools/assert-no-fa-fallback.sh.
                  WORTH DOING: q4_0-V is +4.4%/+8.3% decode at 4 and 1 slots, so a proper sweep of
                  q4_1 (5.0 bits), q5_0 (5.5) and q5_1 (6.0) would show whether the gain is monotonic in
                  bits or whether q4_0 is a sweet spot. It must include 8 SLOTS, where q4_0-V itself is
                  unestablished and the 2026-09-07 data says it loses 3%.
