#!/usr/bin/env python3
"""Regression checks for the isolated Q4V patch. Requires the optimizer's PuLP dependency.

Usage: python test_optimizer_patch.py /path/to/patched/optimize/optimize.py
The production project is never edited by this script.
"""
import importlib.util
import math
from pathlib import Path
import shlex
import sys
import unittest

optimizer_path = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("review_optimizer", optimizer_path)
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


class Q4VRegression(unittest.TestCase):
    def test_recorded_q4v_cells_are_reachable(self):
        self.assertEqual(opt._decode_tps_stock("tp4", 8, 2048, "q8_0/q4_0"), (152.8, "measured"))
        self.assertEqual(opt._decode_tps_stock("tp4", 8, 32768, "q8_0/q4_0"), (107.4, "measured"))

    def test_unrecorded_q4v_cell_is_a_model(self):
        for n in (4, 8):
            value, provenance = opt._decode_tps_stock("tp4", n, 131072, "q8_0/q4_0")
            self.assertTrue(math.isfinite(value) and value > 0)
            self.assertEqual(provenance, "model")

    def test_existing_f16_and_q8_ladders_are_preserved(self):
        for n in (4, 8):
            for kv in ("f16", "q8_0"):
                for depth, value in zip(opt.LAD_DEPTHS, opt.lad["decode_tps"][f"n{n}_{kv}"]):
                    if value is not None:
                        self.assertEqual(opt._decode_tps_stock("tp4", n, depth, kv), (value, "measured"))

    def test_q4v_command_keeps_speculative_flags(self):
        # A formatting fixture, not an assertion that this configuration is measured.
        result = dict(placement="tp4", slots=1, streams=1, ctx=32768, ctx_max=32768,
                      kv="q8_0/q4_0", quant="Q8_0", ubatch=2048, draft=3,
                      topo="default", graphs="on", build="stock", cap=None,
                      decode_tps=1.0, per_stream_tps=1.0, prefill_tps=1.0, mem_gib=1.0,
                      provenance="model", prefill_prov="model", shape=(1300,256),
                      req_tps=1.0, req_per_min=1.0, draft_edit_upper=None)
        rendered = opt.fmt(result)
        tokens = shlex.split(rendered.splitlines()[-1], comments=True)
        self.assertEqual(tokens[tokens.index("-ctv") + 1], "q4_0")
        self.assertEqual(tokens[tokens.index("--spec-draft-n-max") + 1], "3")
        self.assertIn("GGML_CUDA_FA_ALL_QUANTS=ON", rendered)


class PlacementRegression(unittest.TestCase):
    def solve_floor(self, floor):
        return opt.solve(dict(name="floor regression", w_dec=1, w_pre=.15,
                              depth=2048, ctx_min=2048, placements=["dp4"],
                              streams_min=4, streams_max=4, per_stream_min=floor),
                         build="stock")

    def test_fictitious_tp4_gain_cannot_satisfy_dp4_floor(self):
        result, status = self.solve_floor(20)
        self.assertIsNone(result)
        self.assertEqual(status, "Infeasible")

    def test_attainable_dp4_floor_still_succeeds(self):
        result, status = self.solve_floor(19)
        self.assertEqual(status, "Optimal")
        self.assertGreaterEqual(result["per_stream_tps"], 19)
        self.assertEqual(result["topo"], "default")


if __name__ == "__main__":
    unittest.main()
