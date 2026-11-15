"""pydeps_check.py -- the parts of PyTorch that use sympy and setuptools at run time, on the interpreter that runs it.
    /usr/bin/python3 -I pydeps_check.py
One line per check: "ok/FAIL <name>: detail". Exit code = number of failed checks. CPU only (no GPU is touched).
"""
import importlib.metadata as md
import os
import sys
import tempfile
import time

import torch

results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        results.append(True)
        print(f"ok   {name}: {detail} ({time.time() - t0:.1f} s)", flush=True)
    except Exception as e:  # noqa: BLE001
        results.append(False)
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)


def c_export_dynamic():
    from torch.export import Dim, export

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 8)

        def forward(self, x, y):
            z = torch.cat([x, y], dim=0)               # batch = bx + by
            return self.lin(z).relu().sum(dim=0) / z.shape[0], z[: z.shape[0] // 2]

    bx, by = Dim("bx", min=2, max=64), Dim("by", min=2, max=64)
    ep = export(M(), (torch.randn(4, 16), torch.randn(6, 16)), dynamic_shapes={"x": {0: bx}, "y": {0: by}})
    out = ep.module()(torch.randn(9, 16), torch.randn(11, 16))
    assert out[0].shape == (8,) and out[1].shape == (10, 16), [o.shape for o in out]
    return f"dynamic batch export runs at new sizes; range constraints {len(ep.range_constraints)}"


def c_symbolic_fx():
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(a, b):
        n = a.shape[0] * 2 + b.shape[1] % 3
        return (a.reshape(-1)[: a.numel() // 2].sum() + b.sum()) * n, torch.arange(n)

    g = make_fx(f, tracing_mode="symbolic")(torch.randn(6, 4), torch.randn(3, 7))
    r = g(torch.randn(10, 4), torch.randn(3, 8))
    assert r[1].numel() == 10 * 2 + 8 % 3, r[1].numel()
    return f"symbolic trace with {len(list(g.graph.nodes))} nodes generalises to new shapes"


def c_shape_env():
    import sympy
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.utils._sympy.functions import FloorDiv, Mod
    from torch.utils._sympy.value_ranges import ValueRanges, bound_sympy

    s0, s1 = sympy.symbols("s0 s1", positive=True, integer=True)
    env = ShapeEnv()
    e = FloorDiv(s0 * 4 + 2, 2) + Mod(s1 * 6, 3)
    simp = env.simplify(e)
    vr = bound_sympy(s0 * 2 + 1, {s0: ValueRanges(2, 10)})
    assert vr.lower == 5 and vr.upper == 21, vr
    return f"simplify({e}) = {simp}; bound_sympy gives [{vr.lower}, {vr.upper}]"


def c_compile_cpu():
    @torch.compile(dynamic=True)
    def f(x):
        return torch.nn.functional.gelu(x @ x.T).sum(dim=-1) / x.shape[0]

    a = f(torch.randn(8, 32))
    b = f(torch.randn(13, 32))                           # second shape: guards, no hard recompile error
    assert a.shape == (8,) and b.shape == (13,)
    return "torch.compile(dynamic=True) on the CPU compiles and reruns at a second shape"


SRC = r"""
#include <torch/extension.h>
torch::Tensor scaled_add(torch::Tensor a, torch::Tensor b, double s) { return a + s * b; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scaled_add", &scaled_add); }
"""


def c_cpp_extension(use_ninja):
    def run():
        from torch.utils import cpp_extension
        d = tempfile.mkdtemp(prefix="pydeps-ext-")
        src = os.path.join(d, "ext.cpp")
        open(src, "w").write(SRC)
        name = "pydeps_ext_ninja" if use_ninja else "pydeps_ext_setuptools"
        if use_ninja:
            mod = cpp_extension.load(name=name, sources=[src], build_directory=d, verbose=False, with_cuda=False)
        else:
            # the setuptools build_ext path (what `python setup.py build_ext` of an extension package uses)
            import subprocess
            setup = os.path.join(d, "setup.py")
            open(setup, "w").write(
                "from setuptools import setup\nfrom torch.utils.cpp_extension import BuildExtension, CppExtension\n"
                f"setup(name='{name}', ext_modules=[CppExtension('{name}', ['ext.cpp'])], cmdclass={{'build_ext': BuildExtension.with_options(use_ninja=False)}})\n")
            p = subprocess.run([sys.executable, "-I", setup, "build_ext", "--inplace"], cwd=d, capture_output=True, text=True)
            assert p.returncode == 0, (p.stderr or p.stdout)[-600:]
            sys.path.insert(0, d)
            mod = __import__(name)
        a, b = torch.ones(3), torch.arange(3.0)
        assert torch.equal(mod.scaled_add(a, b, 2.0), a + 2 * b)
        return "C++ extension built and called"
    return run


def main():
    print(f"torch {torch.__version__}  sympy {md.version('sympy')}  setuptools {md.version('setuptools')}  "
          f"typing_extensions {md.version('typing_extensions')}  python {sys.version.split()[0]}", flush=True)
    check("export_dynamic_shapes", c_export_dynamic)
    check("symbolic_fx_trace", c_symbolic_fx)
    check("shape_env_sympy", c_shape_env)
    check("compile_cpu_dynamic", c_compile_cpu)
    check("cpp_extension_ninja", c_cpp_extension(True))
    check("cpp_extension_setuptools", c_cpp_extension(False))
    failed = results.count(False)
    print(f"pydeps_check: {results.count(True)} passed, {failed} failed", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
