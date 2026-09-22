#!/usr/bin/env python3
"""
optimize.py - choose the llama.cpp launch configuration for the four-die gfx906 box
by mixed-integer linear programming over the measured data in data/benchmarks.json.

Why an LP:  every knob is a one-hot choice and the knobs multiply (decode = base * f_topo *
f_graphs * f_draft ...).  In log space the product is a sum, so log(throughput) is LINEAR in the
binary choice variables, and the memory budget is linear in slots x context.  That makes the whole
thing a MILP that CBC solves in milliseconds, and it lets a workload be expressed as constraints
(context >= X, per-stream >= Y tok/s, streams >= Z) instead of by hand-picking rows from tables.

    maximise   w_dec * log(decode_tps) + w_pre * log(prefill_tps)
    subject to one-hot per factor, memory per die <= 32 GiB - margin, workload constraints

Decision factors
    placement  tp4 | tp2 | dp4 (four single-die instances) | single (one die) | layer4
    slots      1..8, 16, 24, 32 per instance (-np)
    ctx        context per slot (-c / -np), 2K .. 256K
    kv         f16 | q8_0            (-ctk/-ctv)
    ubatch     512 | 1024 | 2048 | 4096   (-ub)
    draft      0 | 1 | 2 | 3 | 4 | 6   (--spec-type draft-mtp --spec-draft-n-max)   tp4 only
    topo       default | fixed16      (NCCL_TOPO_FILE + NCCL_MIN_NCHANNELS=16)
    graphs     on | off               (GGML_CUDA_DISABLE_GRAPHS)
    quant      Q8_0 | Q4_0 | Q4_1 | Q4_K_M | Q6_K   (model file; fixed to Q8_0 unless --allow-quant; --max-kl bounds quality)
    build      --build stock | mmvq16 | fastpath | prod   (which llama.cpp binary; default prod = ML-gfx906 fork + fast-path kernel)
    kv q8_0/q4_0   4-bit value cache, only with --kv-q4v (needs a GGML_CUDA_FA_ALL_QUANTS build)
    cap        --cap W  per-die power cap (scales decode/prefill by the measured cap curves; reports gen tok/kJ)

Every coefficient comes from a measured cell when one exists; otherwise from the reports' own
fitted linear models, and the output says which.  Run  python3 optimize.py --help.
"""
import argparse, json, math, os, sys
from collections import OrderedDict

try:
    import pulp
except ImportError:
    sys.exit("pip install pulp   (CBC solver ships with it)")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "data", "benchmarks.json")))

K = 1024
DIE_GIB = 32.0

# ----------------------------------------------------------------------------------------------
# 1. Core decode / prefill tables  D(placement, slots, depth, kv), P(placement, depth)
#    Measured cells first; the reports' fitted step models fill the rest (flagged "model").
# ----------------------------------------------------------------------------------------------
PLACEMENTS = ["tp4", "tp2", "dp4", "single", "layer4"]
INSTANCES = {"tp4": 1, "tp2": 1, "dp4": 4, "single": 1, "layer4": 1}
SLOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32]
DEPTHS = [2 * K, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 128 * K, 160 * K, 192 * K, 256 * K]
KVS = ["f16", "q8_0", "q8_0/q4_0"]
UBATCH = [512, 1024, 2048, 4096]
DRAFTS = [0, 1, 2, 3, 4, 6]
TOPOS = ["default", "fixed16"]
GRAPHS = ["on", "off"]
QUANTS = ["Q8_0", "Q4_0", "Q4_1", "Q4_K_M", "Q6_K"]
BUILDS = ["stock", "mmvq16", "fastpath", "prod"]

bb = DATA["batched_bench_tp4_2k"]["rows"]          # batch, prompt, gen512, per, ms, gen2048, per
MS_STEP_2K = {r[0]: r[4] for r in bb}              # ms per decode step at ~2K depth, tp4
GEN_2K = {r[0]: r[5] for r in bb}                  # aggregate t/s at 2K (2048-token prompts)
lad = DATA["context_ladder_tp4"]
LAD_DEPTHS = lad["depths"]                         # 2048 .. 261888

F16_SLOPE = 0.045     # ms per sequence per 1K of depth (tp4, f16)   report 2 s.3
Q8_SLOPE = 0.086      # ms per sequence per 1K of depth (tp4, q8_0)
Q8_BASE_OFFSET = 2.2  # ms: q8_0 step at 2K is ~2 ms above f16 at n=4 and 8 (calibrated)
SINGLE_SLOPE = 0.237  # ms per sequence per 1K on ONE die (20.25 -> 17.69 t/s over 2K..32K)
# CORRECTED 2026-09-22. The old value, 0.092, was measured on a build whose GGML_CUDA_FA_QUANTS did
# not include q8_0-q4_0. An uncompiled FA combination does not fail -- it silently runs on a slower
# generic path (verified: q5_1, uncompiled, ran 8% slow at identical prefill). So the old number timed
# the FALLBACK, not the kernel, and had the SIGN WRONG: q8_0-K/q4_0-V is FASTER than q8_0/q8_0, not
# slower. Measured on v0.4.1 + RCCL + the corrected FA_QUANTS, n=4 per arm:
#   4x64K   decode 15.883 -> 16.587  = +4.44%   (q=0.0762)
#   1x254K  decode 18.588 -> 20.132  = +8.30%   (q=0.0762)
# Scaling Q8_SLOPE by the measured 254K gain gives the value below. It is a SCALED ESTIMATE, not a
# directly fitted slope: a proper refit needs the campaign cells wired into data/benchmarks.json,
# which is still owed. Flagged so nobody reads it as a measured fit.
Q4V_SLOPE = 0.0794    # = Q8_SLOPE / 1.083, from the +8.30% decode gain at 1x254K (scaled estimate)
Q4V_BASE_OFFSET = 2.3


def _nearest_ladder_idx(depth):
    for i, d in enumerate(LAD_DEPTHS):
        if abs(d - depth) / depth < 0.05:
            return i
    return None


def _apply_base_factor(tps, n, slope, dk, bd, streams=None):
    """a kernel gain bd (>1) shortens the depth-independent part of the step; the cache-read part is untouched"""
    if bd == 1.0:
        return tps
    streams = streams or n
    step = streams / tps * 1000                      # ms per step for `streams` sequences
    base = step - slope * n * dk
    return streams / (base / bd + slope * n * dk) * 1000


def decode_tps(p, n, depth, kv, bd=1.0):
    """aggregate decode tokens/s for the whole box on the stock build (bd=1) or with a kernel gain bd applied
    to the depth-independent part of the step; returns (value, provenance)"""
    dk = depth / K
    slope_kv = {"f16": F16_SLOPE, "q8_0": Q8_SLOPE, "q8_0/q4_0": Q4V_SLOPE}[kv]
    v, prov = _decode_tps_stock(p, n, depth, kv)
    if v is None:
        return v, prov
    if p in ("tp4", "tp2", "layer4"):
        return _apply_base_factor(v, n, slope_kv, dk, bd), prov
    s1 = SINGLE_SLOPE * (1.9 if kv != "f16" else 1.0)
    return _apply_base_factor(v, n, s1, dk, bd, streams=n * INSTANCES[p]), prov


def _decode_tps_stock(p, n, depth, kv):
    dk = depth / K
    if p == "tp4":
        i = _nearest_ladder_idx(depth)
        ladder_values = lad["decode_tps"].get(f"n{n}_{kv}")
        if n in (4, 8) and i is not None and ladder_values is not None:
            v = ladder_values[i]
            if v is not None:
                return v, "measured"
        if n == 8:
            for ckv, cctx, cpre, cwall, cdec, cper in lad["ceiling_n8"]["rows"]:   # 8 x 160K f16, 8 x 192K q8_0
                if ckv == kv and abs(cctx - depth) / depth < 0.05:
                    return cdec, "measured"
        if n == 1 and kv == "f16":
            ss = lad["single_stream"]
            for d, v in zip(ss["depths"], ss["decode_tps"]):
                if abs(d - depth) / depth < 0.05:
                    return v, "measured"
        if depth == 2 * K and kv == "f16" and n in GEN_2K:
            return GEN_2K[n], "measured"
        if n not in MS_STEP_2K:
            return None, "unmeasured"
        if kv == "q8_0/q4_0" and n == 8:          # FA_ALL_QUANTS build, run-through s.8
            for kk, p2, d2, p32, d32, fits in DATA["fa_all_quants_build"]["rows"]:
                if kk == "q8_0/q4_0":
                    if abs(depth - 2 * K) / depth < 0.05:
                        return d2, "measured"
                    if abs(depth - 32 * K) / depth < 0.05:
                        return d32, "measured"
        base = MS_STEP_2K[n] - F16_SLOPE * n * 2      # strip the 2K of depth the bench carried
        if kv == "f16":
            step = base + F16_SLOPE * n * dk
        elif kv == "q8_0":
            step = base + Q8_BASE_OFFSET + Q8_SLOPE * n * dk
        else:                                       # q8_0 keys / q4_0 values: 152.8 @2K, 107.4 @32K for n=8
            step = base + Q4V_BASE_OFFSET + Q4V_SLOPE * n * dk
        return n / step * 1000, "model"
    if p == "single":
        if n != 1:
            return None, "unmeasured"
        sc = DATA["scaling_llama_bench"]["rows"][0]
        meas = {2 * K: sc[2], 16 * K: sc[4], 32 * K: sc[6]}
        if depth in meas and kv == "f16":
            return meas[depth], "measured"
        step = 49.4 - SINGLE_SLOPE * 2 + SINGLE_SLOPE * dk * (1.9 if kv != "f16" else 1.0)
        return 1000 / step, "model"
    if p == "dp4":
        # four independent single-die instances; n = streams per die
        rows = DATA["four_instances_dp4"]["rows"]
        for prompt, k, spd, pre, dec, per in rows:
            if spd == n and k == kv and abs(prompt - depth) / depth < 0.05:
                return dec, "measured"
        if n not in (1, 8):
            return None, "unmeasured"
        # per-die step at 2K: n=1 -> 1000/19.3 = 51.8 ms; n=8 -> 8 tokens / (206/4 t/s) = 155.3 ms
        base1 = {1: 51.8, 8: 155.3}[n] - SINGLE_SLOPE * n * 2
        slope = SINGLE_SLOPE * (1.9 if kv != "f16" else 1.0)
        step = base1 + (1.5 if kv != "f16" else 0) + slope * n * dk
        return 4 * n / step * 1000, "model"
    if p == "tp2":
        if n != 1:
            return None, "unmeasured"
        sc = DATA["scaling_llama_bench"]["rows"][1]
        meas = {2 * K: sc[2], 16 * K: sc[4], 32 * K: sc[6]}
        if depth in meas and kv == "f16":
            return meas[depth], "measured"
        step = 1000 / 31.23 - 0.09 * 2 + 0.09 * dk * (1.9 if kv != "f16" else 1.0)
        return 1000 / step, "model"
    if p == "layer4":
        if n != 1 or depth > 4 * K or kv != "f16":
            return None, "unmeasured"
        return 20.1, "measured(server, per-request)"
    return None, "unmeasured"


def prefill_tps(p, depth):
    """average prompt-processing rate over a prompt of length `depth`; (value, provenance)"""
    dk = depth / K
    if p == "tp4":
        i = _nearest_ladder_idx(depth)
        if i is not None and lad["prefill_tps"]["n4_f16"][i] is not None:
            return lad["prefill_tps"]["n4_f16"][i], "measured"
        for ckv, cctx, cpre, cwall, cdec, cper in lad["ceiling_n8"]["rows"]:
            if abs(cctx - depth) / depth < 0.05:
                return cpre, "measured"
        # average of 1/(1.18 + 0.0121*d) over d in [0, dk]
        return 1000 * dk / (1.18 * dk + 0.0121 * dk * dk / 2), "model"
    if p == "single":
        sc = DATA["scaling_llama_bench"]["rows"][0]
        meas = {2 * K: sc[1], 16 * K: sc[3], 32 * K: sc[5]}
        if depth in meas:
            return meas[depth], "measured"
        return 1000 * dk / (4.26 * dk + 0.0484 * dk * dk / 2), "model"   # 4x tp4 per-token costs
    if p == "dp4":
        if depth <= 2 * K:
            return 929, "measured"
        if depth == 8 * K:
            return 902, "measured"
        return 4 * prefill_tps("single", depth)[0], "model"
    if p == "tp2":
        sc = DATA["scaling_llama_bench"]["rows"][1]
        meas = {2 * K: sc[1], 16 * K: sc[3], 32 * K: sc[5]}
        if depth in meas:
            return meas[depth], "measured"
        return 1000 * dk / (2.27 * dk + 0.0242 * dk * dk / 2), "model"
    if p == "layer4":
        return 220, "measured(server: 1300 tokens in 5.87 s)"
    return None, "unmeasured"


# ----------------------------------------------------------------------------------------------
# 2. Multiplicative modifiers (all measured, report 2 s.7 / s.8; report 1 s.2)
# ----------------------------------------------------------------------------------------------
def f_topo(topo, n, p):
    if p != "tp4" or topo == "default":
        return 1.0, 1.0
    return (47.30 / 45.60 if n == 1 else 155.4 / 154.6), 848 / 834          # decode, prefill


def f_graphs(g, n, p):
    if g == "on" or p == "layer4":
        return 1.0
    return 42.31 / 45.60 if n == 1 else 150.2 / 154.6


UB_PREFILL = {512: 689 / 810, 1024: 774 / 810, 2048: 1.0, 4096: 817 / 810}
UB_COMPUTE_F16 = {int(k): v for k, v in DATA["memory_per_die_tp4"]["compute_buffer_f16_GiB_by_ubatch"].items()}

QUANT = {r[0]: r for r in DATA["quant_comparison_llama_bench"]["rows"]}
# Q4_1 (run-through s.6): Q4_0's speed within 3% -> +13% decode, +33% prefill on four dies; one-die tg scaled from Q4_0
QUANT["Q4_1"] = ["Q4_1", 16.33, None, None, QUANT["Q8_0"][4] * 1.33, QUANT["Q4_0"][5] * 1.13 / 1.11, QUANT["Q8_0"][6] * 1.13]
QUANT_KL = {r[0]: r[4] for r in DATA["quant_quality"]["rows"]}          # mean KL divergence vs Q8_0, nats


def f_quant(q, p):
    """(decode factor, prefill factor, weights GiB on disk) relative to Q8_0"""
    r, r8 = QUANT[q], QUANT["Q8_0"]
    if p in ("tp4", "tp2", "layer4"):
        return r[6] / r8[6], r[4] / r8[4], r[1]
    return r[5] / r8[5], (r[4] / r8[4]), r[1]        # single die / dp4: tg from one-die column


# ---- builds (run-through s.9, s.10c, s.11) -------------------------------------------------------
_MR = {r[0]: r for r in DATA["mmvq_rewrite_builds"]["rows"]}
_PB = {r[0]: r for r in DATA["production_build"]["bench"]["rows"]}
_FT = {(r[0], r[1]): r for r in DATA["fork_tile_table"]["rows"]}
_M16 = {r[0]: r for r in DATA["mmvq16_patch"]["rows"]}
_STOCK_TP4 = {8: 160.9, 12: 122.3, 16: 152.3}
_STOCK_DIE = {4: 48.3, 8: 52.7}


def _interp(x, pts):
    """linear interpolation over sorted (x, y) points; clamps outside"""
    pts = sorted(pts)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def f_build(build, p, n, q, depth=None):
    """(decode factor, prefill factor, provenance) of a build relative to stock b10288 for this placement/slots/quant.
    Measured: prod tp4 n=1 (fork), 8/12/16, 24/32 (M2); one die n=1, 4, 8; prefill by quant on the fork. Others interpolated."""
    if build == "stock":
        return 1.0, 1.0, "measured"
    split = p in ("tp4", "tp2", "tp3", "layer4")
    if build == "mmvq16":
        pts = [(8, 158.2 / 159.5), (9, 162.6 / 96.5), (12, 174.6 / 122.5), (16, 170.7 / 152.5)] if split else [(8, 1.0), (12, 51.7 / 54.7), (16, 46.6 / 63.0)]
        if n < 8 or n > 16:
            return 1.0, 1.0, "measured" if n <= 8 else "model"
        return _interp(n, pts), 1.0, ("measured" if n in (8, 9, 12, 16) else "model")
    # fastpath = the kernel on upstream b10288 (key mmvq_rewrite_builds); prod = the production build's own cells
    # (key production_build.bench, 2026-09-07 s.11; 24/32 slots from M2, key m2_slots_prod). Review 2026-09-08 P2: the
    # production factors used the upstream-fastpath table (one die at 8 slots 72.2 instead of the recorded 69.1).
    # The factor is applied to the depth-independent part of the step (_apply_base_factor), so it is derived as the
    # base-term ratio at the 2K anchor, not the raw throughput ratio: (step_stock - cache) / (step_prod - cache).
    hyb, stock = _MR["fast path, hybrid rows 4/2 (serving build)"], _MR["stock b10288"]
    pb, ps = _PB["prod (fork + kernel)"], _PB["stock b10288"]
    def base_ratio(nn, t_stock, t_prod, slope, dk):
        cache = slope * nn * dk
        return (1000.0 * nn / t_stock - cache) / (1000.0 * nn / t_prod - cache)
    if split:
        if build == "prod":
            # columns: build, die_pp2048, tp4_pp2048, tp4_dec_b8, tp4_dec_b12, tp4_dec_b16, die_dec_b4, die_dec_b8, ...
            pts = [(1, base_ratio(1, 47.3, 48.8, F16_SLOPE, 2.0)),
                   (8, base_ratio(8, ps[3], pb[3], F16_SLOPE, 2.0)), (12, base_ratio(12, ps[4], pb[4], F16_SLOPE, 2.0)),
                   (16, base_ratio(16, ps[5], pb[5], F16_SLOPE, 2.0)),
                   (24, base_ratio(24, 161.7, 192.64, F16_SLOPE, 2.0)), (32, base_ratio(32, 179.2, 212.68, F16_SLOPE, 2.0))]
            meas = {1, 8, 12, 16, 24, 32}
        else:
            pts = [(1, 1.0), (8, hyb[1] / stock[1]), (12, hyb[2] / stock[2]), (16, hyb[3] / stock[3])]
            meas = {1, 8, 12, 16}
        if n > 16 and build != "prod":
            dec, dprov = 1.0, "model"       # batches 24/32 run the MMQ tile kernel: unmeasured on the fast-path-only build
        else:
            dec, dprov = _interp(n, pts), ("measured" if n in meas else "model")
    else:
        if build == "prod":
            # one die: 512-token prompts in the production table; single stream from s.11 (20.52 vs 20.26)
            pts = [(1, base_ratio(1, 20.26, 20.52, SINGLE_SLOPE, 0.5)), (4, base_ratio(4, ps[6], pb[6], SINGLE_SLOPE, 0.5)),
                   (8, base_ratio(8, ps[7], pb[7], SINGLE_SLOPE, 0.5))]
        else:
            pts = [(1, 1.0), (4, hyb[4] / stock[4]), (8, hyb[5] / stock[5])]
        if n > 8:
            return None, None, "unmeasured"   # 16-column kernel loses on one die with the plain patch; fast path at 9-16 unmeasured
        dec, dprov = _interp(n, pts), ("measured" if n in (1, 4, 8) else "model")
    if build == "fastpath":
        return dec, 1.0, dprov
    # prod adds the fork's gfx906 MMQ tile table: prefill by quant (2K anchors)
    pre_by_q = {"Q8_0": (1130 / 848 if split else 315.0 / 233.6), "Q6_K": (827 / 658 if split else 228.7 / 175.0), "Q4_K_M": (796 / 746 if split else 216.7 / 201.3)}
    pre = pre_by_q.get(q, 1.0)
    pprov = "measured" if q in pre_by_q else "model"
    if split and depth is not None:
        # Review 2026-09-08 P2: the tile table speeds the matmul, not attention, so the gain shrinks with depth. Measured
        # on the split at Q8_0 (M2, key m2_slots_prod, against the stock context ladder): 1.333 at 2K, 1.357 at 8K, 1.319
        # at 32K; beyond 32K the report-2 component model (1.18 ms + 0.0121 ms per 1K of depth per token, matmul part
        # only sped up) scaled to the 32K anchor. Other quants scale by the same shape.
        dk = depth / K
        anchors = [(2.0, 1130 / 848), (8.0, 1100.53 / 811.43), (32.0, 959.10 / 727.07)]
        if dk <= 32.0:
            shape = _interp(dk, anchors) / anchors[0][1]
        else:
            g = 1130 / 848
            model = lambda x: (1.18 + 0.0121 * x) / (1.18 / g + 0.0121 * x)
            shape = (anchors[-1][1] / anchors[0][1]) * model(dk) / model(32.0)
            pprov = "model"
        pre *= shape
    if q not in ("Q8_0",):
        dprov = "model"                       # decode factor measured at Q8_0 only
    return dec, pre, (dprov if pprov == "measured" else "model")


# ---- power cap (run-through: production build, tp4 -np 16, 16 clients; morning sweep for single stream) ----
CAP_16 = {r[0]: r[2] for r in DATA["power_cap_study"]["rows"]}                       # agg gen tok/s at 16 clients
CAP_KJ = {r[0]: r[4] for r in DATA["power_cap_study"]["rows"]}                       # gen tok per kJ at the bays
_MS = {r[0]: r for r in DATA["power_cap_study"]["morning_sweep_stock_8slots"]["rows"][:5]}
# TODO 13 (2026-09-08): the phases separated on the production build, tp4 — decode-only at 16 slots and prefill-only pp2048
# per cap (200/170/140/125/85 W). Used for the 8+-stream decode factor and for every prefill factor; the 16-client server
# aggregate (CAP_16) keeps the tok/kJ column and the single-stream cap curve stays the morning sweep above 150 W.
_CP = {r[0]: r for r in DATA["power_cap_phases"]["rows"][:5]}
CAP_DEC16 = sorted((w, r[1] / _CP[200][1]) for w, r in _CP.items())                 # decode-only, 16 slots
CAP_PRE = sorted((w, r[3] / _CP[200][3]) for w, r in _CP.items())                   # prefill-only, tp4 pp2048


def f_cap(cap, n):
    """(decode factor, prefill factor, provenance) for a per-die power cap in W, relative to 200 W"""
    if cap is None or cap >= 200:
        return 1.0, 1.0, "measured"
    if n >= 8:                                  # decode-only curve at 16 slots (TODO 13), not the server aggregate
        dec, prov = _interp(cap, CAP_DEC16), ("measured" if cap in _CP else "model")
    else:
        pts1 = sorted((w, r[2] / _MS[200][2]) for w, r in _MS.items())              # tg256 at 200..150 W
        if cap >= 150:
            dec, prov = _interp(cap, pts1), ("measured" if cap in _MS else "model")
        else:                                   # below the morning sweep: follow the decode-only curve's shape
            dec, prov = _interp(cap, CAP_DEC16) * (pts1[0][1] / _interp(150, CAP_DEC16)), "model"
    pre = _interp(cap, CAP_PRE)                 # prefill-only curve, measured at 200/170/140/125/85 W (TODO 13)
    if cap not in _CP:
        prov = "model"
    return dec, pre, prov


# speculative decoding with the model's own MTP head (tp4 only). factor on decode t/s.
# keyed (streams, draft) -> factor at short depth; realistic-prompt table (conservative "free prose")
SPEC = DATA["speculative_mtp_tp4"]
_c = {}
for var, conc, agg, per, ttft, acc in SPEC["concurrency_np8"]["rows"]:
    _c[(var.split(" ")[0], conc)] = agg
DRAFT_FACTOR = {}
for conc in (1, 2, 4, 8):
    base = _c[("none", conc)]
    for d in (1, 2, 3):
        DRAFT_FACTOR[(conc, d)] = _c[(f"mtp-n{d}", conc)] / base
DRAFT_FACTOR[(16, 1)] = _c[("mtp-n1", 16)] / _c[("none", 16)]
DRAFT_FACTOR[(16, 3)] = _c[("mtp-n3", 16)] / _c[("none", 16)]
_m1 = SPEC["draft_length_single_stream"]
for d, free, edit, af, ae in _m1["rows"]:
    if d in (4, 6):
        DRAFT_FACTOR[(1, d)] = free / _m1["baseline"]["free_prose_tps"]
DRAFT_EDIT_FACTOR = {d: edit / _m1["baseline"]["code_edit_tps"] for d, free, edit, af, ae in _m1["rows"]}
DRAFT_DEPTH_FACTOR = {32 * K: 61.40 / 40.45, 128 * K: 60.20 / 35.24}   # single stream, default draft
# run-through s.5, decode-only steady state at 2 and 4 slots (server timings)
_DD = SPEC["depth_steady_state_factors"]["rows"]
DRAFT_DEPTH_N = {(4, 1): _DD["n4_d1"], (4, 2): _DD["n4_d2"], (2, 2): _DD["n2_d2"], (2, 3): _DD["n2_d3"]}


# M3 (2026-09-08, key mtp_prod_build): MTP on the production build, wave-2 decode ratios vs no draft at 2K / 32K.
# The verify batch may reach 16 rows there (16-column MMVQ), so 4 slots x draft 3 and 8 slots x draft 1 are allowed and measured.
_MP = {(r[1], r[2]): {} for r in DATA["mtp_prod_build"]["rows"]}
for _r in DATA["mtp_prod_build"]["rows"]:
    _MP[(_r[1], _r[2])]["2K" if _r[0] <= 2048 else "32K"] = 1.0 + _r[4]
DRAFT_PROD = {k: v for k, v in _MP.items() if k[1] > 0}
_BUILD = "prod"   # set by solve(); f_draft has no build argument


def f_draft(d, n, depth, p):
    """returns (factor, provenance) or (None, reason) when the combination is not allowed"""
    if d == 0:
        return 1.0, "-"
    if p != "tp4":
        return None, "MTP measured on tp4 only"
    if _BUILD == "prod" and (n, d) in DRAFT_PROD:
        return DRAFT_PROD[(n, d)]["32K" if depth >= 32 * K else "2K"], "measured(prod)"
    if n == 1 and depth >= 32 * K and d == 3:
        key = 128 * K if depth >= 128 * K else 32 * K
        return DRAFT_DEPTH_FACTOR[key], "measured(depth)"
    if (n, d) in DRAFT_DEPTH_N and depth >= 32 * K:
        return DRAFT_DEPTH_N[(n, d)]["128K" if depth >= 128 * K else "32K"], "measured(depth)"
    if (n, d) in DRAFT_FACTOR:
        return DRAFT_FACTOR[(n, d)], "measured"
    if n * (d + 1) > (16 if _BUILD == "prod" else 8):
        return None, "slots x (draft+1) beyond the MMVQ kernel width (8 stock, 16 production): unmeasured, expected loss"
    return None, "unmeasured"


# ----------------------------------------------------------------------------------------------
# 3. Memory per die (GiB)
# ----------------------------------------------------------------------------------------------
M4 = DATA["memory_per_die_tp4"]
MD = DATA["memory_per_die_dp4"]
# KiB of KV cache per token per die in tensor split (f16 16 = 8 K + 8 V; q8_0 34/64 of that; q4_0 V 18/64)
KV_KIB = {"f16": M4["kv_KiB_per_token_per_die"]["f16"], "q8_0": M4["kv_KiB_per_token_per_die"]["q8_0"], "q8_0/q4_0": 4.25 + 2.25}


def memory_gib(p, n, ctx, kv, ub, q):
    cells = n * ctx
    wq = f_quant(q, p)[2] - 1.3        # token embedding stays in host RAM
    if p == "tp4":
        w = wq / 4
        kvg = KV_KIB[kv] * cells / (K * K)
        st = M4["recurrent_state_GiB_per_slot"] * n
        # q8_0: the FA path keeps an f16 scratch copy that grows with the cache; the reported compute
        # buffer is 4.9 / 7.1 / 9.3 GiB at 1.0 / 1.5 / 2.0 M cells (ub 2048), i.e. 0.5 + 4.4 per M cells,
        # and it REPLACES the f16 figure rather than adding to it.  q8_0/q4_0 on the FA_ALL_QUANTS build
        # fits 8 x 256K (run-through s.8), so it is modelled with the f16 compute buffer.
        comp = UB_COMPUTE_F16[ub] if kv != "q8_0" else (0.5 + 4.4 * cells / 1e6) + (UB_COMPUTE_F16[ub] - UB_COMPUTE_F16[2048])
        return w + kvg + st + comp
    if p == "tp2":
        w = wq / 2
        kvg = 2 * KV_KIB[kv] * cells / (K * K)
        st = M4["recurrent_state_GiB_per_slot"] * 2 * n
        comp = UB_COMPUTE_F16[ub] * 1.4 if kv != "q8_0" else (0.5 + 8.8 * cells / 1e6) + (UB_COMPUTE_F16[ub] - UB_COMPUTE_F16[2048])
        return w + kvg + st + comp
    if p in ("dp4", "single"):
        w = wq
        kvg = 4 * KV_KIB[kv] * cells / (K * K)
        st = MD["recurrent_state_GiB_per_slot"] * n
        comp = MD["compute_buffer_GiB"] * UB_COMPUTE_F16[ub] / UB_COMPUTE_F16[2048]
        if kv == "q8_0":
            comp = (0.5 + 17.6 * cells / 1e6) + (UB_COMPUTE_F16[ub] - UB_COMPUTE_F16[2048])
        return w + kvg + st + comp
    if p == "layer4":
        # 16 blocks per die, weights /4; cache for the 4 attention blocks on that die -> ~1/4 per die
        return wq / 4 + KV_KIB[kv] * cells / (K * K) + M4["recurrent_state_GiB_per_slot"] * n + UB_COMPUTE_F16[ub]
    return 1e9


# ----------------------------------------------------------------------------------------------
# 4. The MILP
# ----------------------------------------------------------------------------------------------
def solve(scn, allow_quant=False, margin_gib=1.0, verbose=False, exclude_model=False,
          build="prod", cap=None, allow_q4v=False, max_kl=None):
    global _BUILD
    _BUILD = build
    """scn: dict with keys
         name, w_dec, w_pre, depth (working depth, tokens), ctx_min (capacity per slot),
         streams_min, streams_max, per_stream_min (tok/s), placements (optional list), draft_ok (bool)
    """
    prob = pulp.LpProblem("gfx906_llama_cpp", pulp.LpMaximize)
    depth = scn["depth"]
    quants = QUANTS if allow_quant else ["Q8_0"]
    if max_kl is not None:
        quants = [q for q in quants if QUANT_KL[q] <= max_kl]
    kvs = KVS if allow_q4v else [k for k in KVS if k != "q8_0/q4_0"]
    plist = scn.get("placements", PLACEMENTS)

    # core joint variable z[p,n,ctx,kv,q]  (ctx only matters for memory; depth is the scenario's)
    Z = OrderedDict()
    for p in plist:
        for n in SLOTS:
            for ctx in DEPTHS:
                if ctx < scn["ctx_min"] or ctx < depth:
                    continue
                for kv in kvs:
                    for q in quants:
                        if 9 <= n <= 15 and build == "stock":
                            continue            # measured, but the stock stair makes them dominated; keep the search small
                        bd, bp, bprov = f_build(build, p, n, q, depth)
                        if bd is None:
                            continue
                        dec, prov = decode_tps(p, n, depth, kv, bd)
                        if dec is None or (exclude_model and prov == "model"):
                            continue
                        pre, pprov = prefill_tps(p, depth)
                        fd, fp, _ = f_quant(q, p)
                        if q != "Q8_0" and (n > 1 or depth > 32 * K):
                            prov = "model"      # quant factors were measured at batch 1, depth <= 32K
                        if build != "stock" and depth > 8 * K:
                            prov = "model"      # kernel gains measured at 2K (and 512); applied to the depth-independent part at depth
                        cd, cp, cprov = f_cap(cap, n)
                        if "model" in (bprov, cprov):
                            prov = "model"
                        if bprov == "model" or cprov == "model":
                            pprov = "model"
                        if exclude_model and (prov == "model" or pprov == "model"):
                            continue            # review 2026-09-08 P2: filter after the build/quant/cap composition, not before
                        Z[(p, n, ctx, kv, q)] = dict(dec=dec * fd * cd, pre=pre * fp * bp * cp, prov=prov, pprov=pprov,
                                                     interp=(bprov == "model"), capm=(cprov == "model"),
                                                     var=pulp.LpVariable(f"z_{p}_{n}_{ctx // K}k_{kv.replace('/', '_')}_{q}", cat="Binary"))
    if not Z:
        return None, "no feasible core configuration for this workload"
    prob += pulp.lpSum(v["var"] for v in Z.values()) == 1

    # one-hot factors
    ub_v = {u: pulp.LpVariable(f"ub_{u}", cat="Binary") for u in UBATCH}
    tp_v = {t: pulp.LpVariable(f"topo_{t}", cat="Binary") for t in TOPOS}
    gr_v = {g: pulp.LpVariable(f"graphs_{g}", cat="Binary") for g in GRAPHS}
    prob += pulp.lpSum(ub_v.values()) == 1
    prob += pulp.lpSum(tp_v.values()) == 1
    prob += pulp.lpSum(gr_v.values()) == 1

    # joint (slots, draft) and (slots, topo/graphs) variables, linked to the core choice
    nd_v, nt_v, ng_v = {}, {}, {}
    for n in SLOTS:
        core_n = [v["var"] for k, v in Z.items() if k[1] == n]
        if not core_n:
            continue
        for d in DRAFTS:
            nd_v[(n, d)] = pulp.LpVariable(f"nd_{n}_{d}", cat="Binary")
        prob += pulp.lpSum(nd_v[(n, d)] for d in DRAFTS) == pulp.lpSum(core_n)
    # Topology and graph effects depend on both placement and slot count.
    # Use the same placement in the objective, latency constraint and final output.
    for p in plist:
        for n in SLOTS:
            core_pn = [v["var"] for k, v in Z.items() if k[0] == p and k[1] == n]
            if not core_pn:
                continue
            for t in TOPOS:
                nt_v[(p, n, t)] = pulp.LpVariable(f"nt_{p}_{n}_{t}", cat="Binary")
                prob += nt_v[(p, n, t)] <= tp_v[t]
            prob += pulp.lpSum(nt_v[(p, n, t)] for t in TOPOS) == pulp.lpSum(core_pn)
            for g in GRAPHS:
                ng_v[(p, n, g)] = pulp.LpVariable(f"ng_{p}_{n}_{g}", cat="Binary")
                prob += ng_v[(p, n, g)] <= gr_v[g]
            prob += pulp.lpSum(ng_v[(p, n, g)] for g in GRAPHS) == pulp.lpSum(core_pn)

    # draft legality depends on placement too: forbid draft>0 with non-tp4 core or unmeasured combos
    for (n, d), var in nd_v.items():
        if d == 0:
            continue
        if not scn.get("draft_ok", True):
            prob += var == 0
            continue
        for k, v in Z.items():
            if k[1] == n:
                f, why = f_draft(d, n, depth, k[0])
                if f is None:
                    prob += var + v["var"] <= 1

    # ---- objective (log-linear) ----
    def L(x):
        return math.log(x)

    # Tie-breaks, so that gains inside the 2% run-to-run noise do not buy complexity:
    #   LAMBDA_MEM  : 0.2% of objective per GiB of VRAM used (prefers the leaner config, minimal ctx)
    #   EPS         : 0.01% for every non-default knob (draft, topo file, graphs off, ub != 2048, q8_0 kv)
    #   MODEL_DISCOUNT: cells that come from the fitted step models (not a measured point) are
    #                   discounted 3%, so an extrapolated cell only wins over a measured one when
    #                   its predicted gain is bigger than the run-to-run spread
    LAMBDA_MEM, EPS, MODEL_DISCOUNT = 0.002, 1e-4, math.log(0.97)
    obj = []
    for k, v in Z.items():
        p, n, ctx, kv, q = k
        val = scn["w_dec"] * L(v["dec"]) + scn["w_pre"] * L(v["pre"])
        val -= LAMBDA_MEM * memory_gib(p, n, ctx, kv, 2048, q)
        val -= EPS * (kv != "f16")
        val += MODEL_DISCOUNT * (v["prov"] == "model")
        val += MODEL_DISCOUNT * v["interp"]          # build factor interpolated between measured slot counts: a second 3%
        val += math.log(0.99) * v["capm"]            # cap curve interpolated: 1%
        obj.append(val * v["var"])
    for (n, d), var in nd_v.items():
        if d:
            f, _ = f_draft(d, n, depth, "tp4")
            if f:
                obj.append((scn["w_dec"] * L(f) - EPS) * var)
    for (p, n, t), var in nt_v.items():
        fd, fp = f_topo(t, n, p)
        obj.append((scn["w_dec"] * L(fd) + scn["w_pre"] * L(fp) - EPS * (t != "default")) * var)
    for (p, n, g), var in ng_v.items():
        obj.append((scn["w_dec"] * L(f_graphs(g, n, p)) - EPS * (g != "on")) * var)
    for u, var in ub_v.items():
        obj.append((scn["w_pre"] * L(UB_PREFILL[u]) - LAMBDA_MEM * (UB_COMPUTE_F16[u] - UB_COMPUTE_F16[2048]) - EPS * (u != 2048)) * var)
    prob += pulp.lpSum(obj)
    # excluded core cells (used for the runner-up list)
    for k in scn.get("_exclude", []):
        if k in Z:
            prob += Z[k]["var"] == 0

    # ---- memory: core part + ubatch part <= budget (both linear) ----
    budget = DIE_GIB - margin_gib
    for k, v in Z.items():
        p, n, ctx, kv, q = k
        for u, uv in ub_v.items():
            m = memory_gib(p, n, ctx, kv, u, q)
            if m > budget:
                prob += v["var"] + uv <= 1        # this (core, ubatch) pair does not fit

    # ---- workload constraints ----
    for k, v in Z.items():
        p, n = k[0], k[1]
        streams = n * INSTANCES[p]
        if streams < scn.get("streams_min", 1) or streams > scn.get("streams_max", 10 ** 9):
            prob += v["var"] == 0
        if scn.get("per_stream_min"):
            # cheap pre-prune: no modifier combination reaches 2.2x, so these cells can never meet the floor
            if v["dec"] / streams * 2.2 < scn["per_stream_min"]:
                prob += v["var"] == 0
    # per-stream floor exactly, in log space: log dec + log f_draft + log f_topo + log f_graphs - log streams >= log floor
    if scn.get("per_stream_min"):
        lhs = []
        for k, v in Z.items():
            lhs.append((L(v["dec"]) - L(k[1] * INSTANCES[k[0]])) * v["var"])
        for (n, d), var in nd_v.items():
            if d:
                f, _ = f_draft(d, n, depth, "tp4")
                if f:
                    lhs.append(L(f) * var)
        for (p, n, t), var in nt_v.items():
            lhs.append(L(f_topo(t, n, p)[0]) * var)
        for (p, n, g), var in ng_v.items():
            lhs.append(L(f_graphs(g, n, p)) * var)
        prob += pulp.lpSum(lhs) >= L(scn["per_stream_min"])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
    if pulp.LpStatus[status] != "Optimal":
        return None, pulp.LpStatus[status]

    sel = [k for k, v in Z.items() if v["var"].value() > 0.5][0]
    p, n, ctx, kv, q = sel
    d = [d for (nn, d), var in nd_v.items() if nn == n and var.value() > 0.5][0]
    t = [t for t, var in tp_v.items() if var.value() > 0.5][0]
    g = [g for g, var in gr_v.items() if var.value() > 0.5][0]
    u = [u for u, var in ub_v.items() if var.value() > 0.5][0]
    core = Z[sel]
    fdr = f_draft(d, n, depth, p)[0] if d else 1.0
    ftd, ftp = f_topo(t, n, p)
    fg = f_graphs(g, n, p)
    dec = core["dec"] * fdr * ftd * fg
    pre = core["pre"] * ftp * UB_PREFILL[u]
    streams = n * INSTANCES[p]
    # largest context per slot that still fits with these settings (same slots/kv/ub/quant)
    ctx_max = ctx
    step = 4096
    while memory_gib(p, n, ctx_max + step, kv, u, q) <= DIE_GIB - margin_gib and ctx_max + step <= 262144:
        ctx_max += step
    # request-level figure for a (prompt, gen) shape: prompts are read one slot at a time, then all decode
    P_, G_ = scn.get("shape", (1300, 256))
    t_req = streams * P_ / pre + G_ / (dec / streams)
    res = dict(placement=p, slots=n, ctx=ctx, ctx_max=ctx_max, kv=kv, quant=q, ubatch=u, draft=d, topo=t, graphs=g,
               build=build, cap=cap, kj=(_interp(cap, sorted(CAP_KJ.items())) if cap else CAP_KJ[200]),
               streams=streams, decode_tps=dec, per_stream_tps=dec / streams, prefill_tps=pre,
               mem_gib=memory_gib(p, n, ctx, kv, u, q), provenance=core["prov"], prefill_prov=core["pprov"],
               draft_edit_upper=(DRAFT_EDIT_FACTOR.get(d) if (d and n == 1) else None),
               shape=(P_, G_), req_tps=streams * G_ / t_req, req_per_min=streams / t_req * 60,
               objective=pulp.value(prob.objective), core=sel)
    return res, "Optimal"


# ----------------------------------------------------------------------------------------------
# 5. Workloads
# ----------------------------------------------------------------------------------------------
SCENARIOS = [
    dict(name="A. one user, short context (chat / code, <= 8K)",
         w_dec=1.0, w_pre=0.15, depth=2 * K, ctx_min=8 * K, streams_min=1, streams_max=1),
    dict(name="B. one user, 32K working context",
         w_dec=1.0, w_pre=0.25, depth=32 * K, ctx_min=32 * K, streams_min=1, streams_max=1),
    dict(name="C. one user, 128K working context",
         w_dec=1.0, w_pre=0.25, depth=128 * K, ctx_min=128 * K, streams_min=1, streams_max=1),
    dict(name="D. small team chat server: max aggregate, every stream >= 12 tok/s, 16K per slot",
         w_dec=1.0, w_pre=0.15, depth=4 * K, ctx_min=16 * K, streams_min=2, per_stream_min=12.0),
    dict(name="E. busy server: max aggregate, every stream >= 6 tok/s (reading speed), 32K per slot",
         w_dec=1.0, w_pre=0.15, depth=8 * K, ctx_min=32 * K, streams_min=4, per_stream_min=6.0),
    dict(name="F. offline batch generation, short prompts, no latency floor",
         w_dec=1.0, w_pre=0.05, depth=2 * K, ctx_min=4 * K, streams_min=8),
    dict(name="G. long-context server: 128K per slot, max aggregate, >= 4 streams",
         w_dec=1.0, w_pre=0.3, depth=128 * K, ctx_min=128 * K, streams_min=4),
    dict(name="H. full 256K context, as many slots as fit",
         w_dec=1.0, w_pre=0.3, depth=256 * K, ctx_min=256 * K, streams_min=1),
    dict(name="I. prompt ingestion (RAG indexing): maximise prefill, 32K documents",
         w_dec=0.1, w_pre=1.0, depth=32 * K, ctx_min=32 * K, streams_min=1),
    dict(name="J. eight slots at the memory ceiling: 8 streams, 160K each",
         w_dec=1.0, w_pre=0.3, depth=160 * K, ctx_min=160 * K, streams_min=8, streams_max=8),
]


def fmt(res):
    env = []
    if res["topo"] == "fixed16":
        env.append("NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16")
    if res["graphs"] == "off":
        env.append("GGML_CUDA_DISABLE_GRAPHS=1")
    p = res["placement"]
    dev = {"tp4": "--device rocm0,rocm1,rocm2,rocm3 -sm tensor", "tp2": "--device rocm0,rocm1 -sm tensor",
           "single": "--device rocm0", "dp4": "--device rocmN   (one server per die, N=0..3)",
           "layer4": "--device rocm0,rocm1,rocm2,rocm3 -sm layer"}[p]
    n, ctx = res["slots"], res["ctx"]
    binp = {"stock": "$LLAMA_STOCK/", "mmvq16": "$LLAMA_MMVQ16/", "fastpath": "$LLAMA_FASTPATH/", "prod": "$LLAMA_PROD/"}[res["build"]]
    cmd = f"{binp}llama-server -m Qwen3.8-27B-{res['quant']}.gguf {dev} -fa on -np {n} -c {n * ctx} -b 2048 -ub {res['ubatch']} -cb"
    if res["kv"] == "q8_0":
        cmd += " -ctk q8_0 -ctv q8_0"
    elif res["kv"] == "q8_0/q4_0":
        cmd += " -ctk q8_0 -ctv q4_0"
    if res["draft"]:
        cmd += f" --spec-type draft-mtp --spec-draft-n-max {res['draft']}"
    if p in ("dp4", "single"):
        env = [e for e in env if not e.startswith("NCCL")]
    lines = [
        f"  placement {p:7s} slots/instance {n:<3d} streams {res['streams']:<3d} ctx/slot {ctx // K}K (up to {res['ctx_max'] // K}K fits)  kv {res['kv']:9s} ub {res['ubatch']}  draft {res['draft']}  topo {res['topo']}  graphs {res['graphs']}  quant {res['quant']}  build {res['build']}"
        + (f"  cap {res['cap']} W (~{res['kj']:.0f} gen tok/kJ at 16 clients)" if res['cap'] else ""),
        f"  decode {res['decode_tps']:7.1f} tok/s aggregate  ({res['per_stream_tps']:.1f} per stream)   prefill {res['prefill_tps']:6.0f} tok/s   memory/die {res['mem_gib']:.1f} GiB   [decode {res['provenance']}, prefill {res['prefill_prov']}]",
        f"  request-level for {res['shape'][0]}-token prompts / {res['shape'][1]} generated: ~{res['req_tps']:.0f} tok/s, ~{res['req_per_min']:.1f} requests/min (simple prefill-then-decode model, optimistic by 15-25% vs report 1)",
    ]
    if res["draft_edit_upper"]:
        lines.append(f"  (draft factor is the conservative free-prose figure; on code-edit prompts the same draft gave {res['draft_edit_upper']:.2f}x)")
    if res["cap"] and res["cap"] < 200:
        lines.append(f"  cap {res['cap']} W per die: set with settings/powercap.sh {res['cap']}; caps below ~85 W are accepted by the driver and not honoured")
    if res["kv"] == "q8_0/q4_0":
        lines.append("  # needs a GGML_CUDA_FA_ALL_QUANTS=ON build")
    lines.append("  " + (" ".join(env) + " " if env else "") + cmd)
    return "\n".join(lines)


def _core_keys(scn, a):
    quants = QUANTS if a.allow_quant else ["Q8_0"]
    return [(p, n, ctx, kv, q) for p in scn.get("placements", PLACEMENTS) for n in SLOTS for ctx in DEPTHS
            for kv in KVS for q in quants]


def _solve_args(a):
    return dict(allow_quant=a.allow_quant, margin_gib=a.margin, exclude_model=a.measured_only,
                build=a.build, cap=a.cap, allow_q4v=a.kv_q4v, max_kl=a.max_kl)


def fmt_alt(res, best):
    d = (res["decode_tps"] / best["decode_tps"] - 1) * 100
    pr = (res["prefill_tps"] / best["prefill_tps"] - 1) * 100
    return (f"    runner-up: {res['placement']} np {res['slots']} ({res['streams']} streams) ctx {res['ctx'] // K}K kv {res['kv']} ub {res['ubatch']} draft {res['draft']} {res['quant']}"
            f" -> decode {res['decode_tps']:.1f} ({d:+.0f}%), prefill {res['prefill_tps']:.0f} ({pr:+.0f}%), {res['mem_gib']:.1f} GiB/die [{res['provenance']}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-quant", action="store_true", help="let the model file (Q8_0/Q4_0/Q4_1/Q4_K_M/Q6_K) be a decision variable")
    ap.add_argument("--max-kl", type=float, help="with --allow-quant: only quants whose mean KL vs Q8_0 is <= this (Q6_K 0.019, Q4_K_M 0.036, Q4_1 0.056, Q4_0 0.091 nats)")
    ap.add_argument("--build", choices=BUILDS, default="prod", help="llama.cpp binary: stock b10288 | mmvq16 | fastpath | prod (fork tile table + fast path; default)")
    ap.add_argument("--cap", type=int, help="per-die power cap in W (200 = uncapped); scales by the measured cap curves and reports gen tok/kJ")
    ap.add_argument("--kv-q4v", action="store_true", help="allow the q8_0-keys / q4_0-values cache (needs a GGML_CUDA_FA_ALL_QUANTS build)")
    ap.add_argument("--margin", type=float, default=1.0, help="GiB of VRAM per die to leave free (default 1.0)")
    ap.add_argument("--measured-only", action="store_true", help="exclude cells that come from the fitted models")
    ap.add_argument("--depth", type=int, help="custom workload: working depth in tokens")
    ap.add_argument("--ctx", type=int, help="custom workload: required context per slot in tokens")
    ap.add_argument("--streams", type=int, help="custom workload: minimum concurrent streams")
    ap.add_argument("--per-stream", type=float, help="custom workload: minimum tok/s per stream")
    ap.add_argument("--w-pre", type=float, default=0.15, help="custom workload: weight of log(prefill) vs log(decode)=1")
    ap.add_argument("--max-streams", type=int, help="custom workload: maximum concurrent streams")
    ap.add_argument("--shape", default="1300,256", help="prompt,gen token counts for the request-level figure (default 1300,256)")
    ap.add_argument("--alternatives", type=int, default=2, help="how many runner-up core configurations to list (default 2)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    shape = tuple(int(x) for x in a.shape.split(","))
    scns = SCENARIOS
    if a.depth or a.ctx or a.streams or a.per_stream or a.max_streams:
        depth = a.depth or 2 * K
        scns = [dict(name="custom", w_dec=1.0, w_pre=a.w_pre, depth=depth, ctx_min=a.ctx or depth,
                     streams_min=a.streams or 1, streams_max=a.max_streams or 10 ** 9, per_stream_min=a.per_stream)]
    print(f"gfx906 x4 / Qwen3.8-27B: MILP over {len(SLOTS)}x{len(DEPTHS)}x{len(KVS)} core cells x ubatch x draft x topo x graphs"
          f"{' x quant' if a.allow_quant else ''}; build {a.build}; cap {a.cap or 200} W; VRAM budget {DIE_GIB - a.margin:.0f} GiB/die\n")
    for s in scns:
        s = dict(s, shape=shape)
        res, st = solve(s, verbose=a.verbose, **_solve_args(a))
        print(s["name"])
        print(fmt(res) if res else f"  infeasible: {st}")
        if res:
            # runner-ups: exclude every core cell with the same (placement, slots, kv, quant) and re-solve
            seen = {(res["placement"], res["slots"], res["kv"], res["quant"])}
            excl = [k for k in _core_keys(s, a) if (k[0], k[1], k[3], k[4]) in seen]
            for _ in range(a.alternatives):
                alt, st2 = solve(dict(s, _exclude=list(excl)), **_solve_args(a))
                if not alt:
                    break
                print(fmt_alt(alt, res))
                seen.add((alt["placement"], alt["slots"], alt["kv"], alt["quant"]))
                excl = [k for k in _core_keys(s, a) if (k[0], k[1], k[3], k[4]) in seen]
        print()


if __name__ == "__main__":
    main()
