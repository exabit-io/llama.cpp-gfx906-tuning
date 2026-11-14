"""ops_check.py -- GPU-against-CPU smoke checks for python3-torch-gfx906, one per ROCm library the wheel links, plus the
operators PyTorch 2.15 fixed on ROCm (topk order, argmax on large reductions, layer_norm, COO coalesce, cdist) and the
torch.distributed backends (2.15 makes a rewritten NCCL backend the default; RCCL stands in for NCCL here).

    /usr/bin/python3 -I ops_check.py            # system package
    VENV/bin/python -I ops_check.py             # a build venv

Every check prints one line: "ok   <name>: <detail>" or "FAIL <name>: <error>". Exit code = number of failed checks.
Works on 2.14 and 2.15 (backends a version does not have are reported as "skip"). Timings are informational.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

DEV = "cuda:0"
results = []


def check(name):
    def deco(fn):
        def run():
            try:
                torch.manual_seed(0)
                detail = fn()
                results.append((name, True))
                print(f"ok   {name}: {detail}", flush=True)
            except Skip as e:
                print(f"skip {name}: {e}", flush=True)
            except Exception as e:  # noqa: BLE001 - every failure is reported, none stops the run
                results.append((name, False))
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        run.__name__ = name
        return run
    return deco


class Skip(Exception):
    pass


def rel(a, b):
    return ((a.double().cpu() - b.double().cpu()).abs().max() / b.double().cpu().abs().max().clamp_min(1e-30)).item()


@check("blas")
def c_blas():
    a = torch.randn(512, 384)
    b = torch.randn(384, 256)
    v = torch.randn(384)
    e1 = rel(a.to(DEV) @ b.to(DEV), a @ b)
    e2 = rel(a.to(DEV) @ v.to(DEV), a @ v)
    e3 = rel(torch.bmm(a.to(DEV).expand(8, -1, -1), b.to(DEV).expand(8, -1, -1)), torch.bmm(a.expand(8, -1, -1), b.expand(8, -1, -1)))
    ad = a.double()
    e4 = rel(ad.to(DEV) @ b.double().to(DEV), ad @ b.double())
    assert max(e1, e2, e3) < 1e-4 and e4 < 1e-12, (e1, e2, e3, e4)
    return f"gemm/gemv/bmm fp32 rel err {max(e1, e2, e3):.1e}, fp64 gemm {e4:.1e}"


@check("linalg")
def c_linalg():
    a = torch.randn(200, 200, dtype=torch.float64)
    spd = a @ a.T + 200 * torch.eye(200, dtype=torch.float64)
    b = torch.randn(200, 8, dtype=torch.float64)
    g = spd.to(DEV)
    e_solve = rel(torch.linalg.solve(g, b.to(DEV)), torch.linalg.solve(spd, b))
    e_chol = rel(torch.linalg.cholesky(g), torch.linalg.cholesky(spd))
    q, r = torch.linalg.qr(a.to(DEV))
    e_qr = rel(q @ r, a)
    e_svd = rel(torch.linalg.svdvals(a.to(DEV)), torch.linalg.svdvals(a))
    e_eigh = rel(torch.linalg.eigvalsh(g), torch.linalg.eigvalsh(spd))
    e_inv = rel(torch.linalg.inv(g) @ g, torch.eye(200, dtype=torch.float64))
    worst = max(e_solve, e_chol, e_qr, e_svd, e_eigh, e_inv)
    assert worst < 1e-8, (e_solve, e_chol, e_qr, e_svd, e_eigh, e_inv)
    return f"solve/cholesky/qr/svdvals/eigvalsh/inv fp64 worst rel err {worst:.1e}"


@check("fft")
def c_fft():
    x = torch.randn(4, 256, 300)
    e1 = rel(torch.view_as_real(torch.fft.fft2(x.to(DEV))), torch.view_as_real(torch.fft.fft2(x)))
    e2 = rel(torch.view_as_real(torch.fft.rfft(x.to(DEV), dim=-1)), torch.view_as_real(torch.fft.rfft(x, dim=-1)))
    e3 = rel(torch.fft.irfft(torch.fft.rfft(x.to(DEV), dim=-1), n=300, dim=-1), x)
    assert max(e1, e2, e3) < 1e-4, (e1, e2, e3)
    return f"fft2/rfft/irfft round trip rel err {max(e1, e2, e3):.1e}"


@check("sparse")
def c_sparse():
    n = 2000
    idx = torch.randint(0, n, (2, 30000))
    val = torch.randn(30000)
    coo = torch.sparse_coo_tensor(idx, val, (n, n))
    dense = coo.to_dense()
    x = torch.randn(n, 16)
    g = coo.to(DEV)
    e_coalesce = rel(g.coalesce().to_dense(), dense)          # 2.15 fixed COO coalesce on ROCm
    e_spmm = rel(torch.sparse.mm(g, x.to(DEV)), dense @ x)
    csr = dense.to_sparse_csr()
    e_csr = rel(csr.to(DEV) @ x.to(DEV), dense @ x)
    assert max(e_coalesce, e_spmm, e_csr) < 1e-4, (e_coalesce, e_spmm, e_csr)
    return f"coalesce {e_coalesce:.1e}, COO spmm {e_spmm:.1e}, CSR spmm {e_csr:.1e}"


@check("random")
def c_random():
    g = torch.Generator(device=DEV)
    g.manual_seed(1234)
    a = torch.randn(1 << 22, device=DEV, generator=g)
    g.manual_seed(1234)
    b = torch.randn(1 << 22, device=DEV, generator=g)
    u = torch.rand(1 << 22, device=DEV)
    p = torch.randperm(100000, device=DEV)
    assert torch.equal(a, b), "same seed, different stream"
    assert abs(a.mean().item()) < 5e-3 and abs(a.std().item() - 1) < 5e-3, (a.mean().item(), a.std().item())
    assert abs(u.mean().item() - 0.5) < 5e-3 and 0 <= u.min().item() and u.max().item() < 1
    assert torch.equal(p.sort().values.cpu(), torch.arange(100000))
    m = torch.multinomial(torch.tensor([0.1, 0.2, 0.7], device=DEV), 200000, replacement=True).float().mean().item()
    assert abs(m - 1.6) < 0.02, m
    return f"randn mean {a.mean().item():+.1e} std {a.std().item():.4f}, seeded stream repeats, randperm is a permutation"


@check("sort_topk_argmax")
def c_sort():
    x = torch.argsort(torch.rand(64, 5000), dim=-1).float()     # every row a permutation of 0..4999: no ties, so indices are unique
    xi = torch.randint(0, 50, (64, 5000)).float()               # many ties
    big = torch.randn(1 << 25)                                  # large reduction (2.15 fixed rare wrong indices on ROCm)
    bad = []

    def want(cond, label):
        if not cond:
            bad.append(label)

    s_g = torch.sort(x.to(DEV), dim=-1)
    s_c = torch.sort(x, dim=-1)
    want(torch.equal(s_g.values.cpu(), s_c.values), "sort values")
    want(torch.equal(s_g.indices.cpu(), s_c.indices), "sort indices")
    st_g = torch.sort(xi.to(DEV), dim=-1, stable=True)
    st_c = torch.sort(xi, dim=-1, stable=True)
    want(torch.equal(st_g.indices.cpu(), st_c.indices), "stable sort indices with ties")
    for k in (8, 256, 300, 2000):                               # 2.15 fixed nondeterministic order beyond 256 elements
        t_g = torch.topk(x.to(DEV), k, dim=-1)
        t_c = torch.topk(x, k, dim=-1)
        want(torch.equal(t_g.values.cpu(), t_c.values), f"topk k={k} values")
        want(torch.equal(t_g.indices.cpu(), t_c.indices), f"topk k={k} indices (no ties)")
    t1 = torch.topk(xi.to(DEV), 300, dim=-1)
    t2 = torch.topk(xi.to(DEV), 300, dim=-1)
    want(torch.equal(t1.values.cpu(), torch.topk(xi, 300, dim=-1).values), "topk with ties: values")
    want(torch.equal(xi.to(DEV).gather(-1, t1.indices), t1.values), "topk with ties: indices point at the values")
    want(torch.equal(t1.indices, t2.indices), "topk with ties: same indices on a second call")
    want(big.to(DEV).argmax().item() == big.argmax().item(), "argmax over 2^25")
    want(big.to(DEV).argmin().item() == big.argmin().item(), "argmin over 2^25")
    m_g = x.to(DEV).max(dim=-1)
    want(torch.equal(m_g.indices.cpu(), x.max(dim=-1).indices), "max(dim) indices")
    want(torch.equal(torch.unique(xi.to(DEV)).cpu(), torch.unique(xi)), "unique")
    v, i = torch.kthvalue(x.to(DEV), 1234, dim=-1)
    want(torch.equal(v.cpu(), torch.kthvalue(x, 1234, dim=-1).values), "kthvalue")
    assert not bad, "; ".join(bad)
    return "sort, stable sort, topk (k = 8..2000, with and without ties), argmax/argmin over 2^25, max(dim), unique, kthvalue equal the CPU"


@check("conv_norm_pool")
def c_conv():
    x = torch.randn(4, 16, 48, 48)
    conv = torch.nn.Conv2d(16, 32, 3, padding=1)
    ct = torch.nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
    bn = torch.nn.BatchNorm2d(32).eval()
    y_c = ct(F.max_pool2d(bn(conv(x)), 2))
    conv_g, ct_g, bn_g = conv.to(DEV), ct.to(DEV), bn.to(DEV)
    y_g = ct_g(F.max_pool2d(bn_g(conv_g(x.to(DEV))), 2))
    e1 = rel(y_g, y_c)
    xg = x.to(DEV).requires_grad_(True)
    xc = x.clone().requires_grad_(True)
    conv_g(xg).square().mean().backward()
    conv.cpu()(xc).square().mean().backward()
    e2 = rel(xg.grad, xc.grad)
    h = torch.randn(8, 100, 384)
    ln = torch.nn.LayerNorm(384)
    e3 = rel(ln.to(DEV)(h.to(DEV)), ln.cpu()(h))                # 2.15 fixed layer_norm forward on ROCm
    gn = torch.nn.GroupNorm(8, 32)
    z = torch.randn(4, 32, 20, 20)
    e4 = rel(gn.to(DEV)(z.to(DEV)), gn.cpu()(z))
    assert max(e1, e2, e3, e4) < 1e-4, (e1, e2, e3, e4)
    return f"conv+bn+pool+convT {e1:.1e}, conv input grad {e2:.1e}, layer_norm {e3:.1e}, group_norm {e4:.1e}"


@check("rnn")
def c_rnn():
    lstm = torch.nn.LSTM(64, 96, num_layers=2, batch_first=True)
    gru = torch.nn.GRU(64, 96, batch_first=True)
    x = torch.randn(4, 50, 64)
    o_c, _ = lstm(x)
    g_c, _ = gru(x)
    o_g, _ = lstm.to(DEV)(x.to(DEV))
    g_g, _ = gru.to(DEV)(x.to(DEV))
    e1, e2 = rel(o_g, o_c), rel(g_g, g_c)
    xg = x.to(DEV).requires_grad_(True)
    lstm(xg)[0].sum().backward()
    assert max(e1, e2) < 1e-4 and torch.isfinite(xg.grad).all(), (e1, e2)
    return f"2-layer LSTM {e1:.1e}, GRU {e2:.1e}, LSTM backward finite"


@check("attention_amp")
def c_attn():
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = F.scaled_dot_product_attention(q.to(DEV), k.to(DEV), v.to(DEV), is_causal=True)
    e1 = rel(out, ref)
    mask = torch.rand(2, 1, 128, 128) > 0.2
    mask[..., 0] = True
    e2 = rel(F.scaled_dot_product_attention(q.to(DEV), k.to(DEV), v.to(DEV), attn_mask=mask.to(DEV)),
             F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    errs = {}
    for dt in (torch.float16, torch.bfloat16):
        with torch.autocast("cuda", dtype=dt):
            o = F.scaled_dot_product_attention(q.to(DEV), k.to(DEV), v.to(DEV), is_causal=True)
        assert o.dtype == dt and torch.isfinite(o).all()
        errs[str(dt).split(".")[-1]] = rel(o.float(), ref)
    assert max(e1, e2) < 1e-4 and errs["float16"] < 2e-2 and errs["bfloat16"] < 1e-1, (e1, e2, errs)
    return f"SDPA causal {e1:.1e}, masked {e2:.1e}, autocast fp16 {errs['float16']:.1e}, bf16 {errs['bfloat16']:.1e}"


@check("transformer_train")
def c_train():
    def make():
        torch.manual_seed(7)
        return torch.nn.Sequential(torch.nn.Embedding(1000, 128),
                                   torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True), 2),
                                   torch.nn.Linear(128, 1000))
    tok = torch.randint(0, 1000, (16, 64))
    tgt = torch.roll(tok, -1, dims=1)
    losses = {}
    for dev in ("cpu", DEV):
        m = make().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        ls = []
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(tok.to(dev)).flatten(0, 1), tgt.to(dev).flatten())
            loss.backward()
            opt.step()
            ls.append(loss.item())
        losses[dev] = ls
    c, g = losses["cpu"], losses[DEV]
    assert abs(c[0] - g[0]) < 1e-3, (c[0], g[0])
    assert g[-1] < 0.7 * g[0] and abs(c[-1] - g[-1]) < 0.15 * c[-1], (c[-1], g[-1])
    return f"first loss cpu {c[0]:.4f} gpu {g[0]:.4f}; after 30 AdamW steps cpu {c[-1]:.4f} gpu {g[-1]:.4f}"


@check("indexing")
def c_index():
    n = 100000
    idx = torch.randint(0, 1000, (n,))
    src = torch.randn(n)
    e1 = rel(torch.zeros(1000, device=DEV).scatter_add_(0, idx.to(DEV), src.to(DEV)), torch.zeros(1000).scatter_add_(0, idx, src))
    e2 = rel(torch.zeros(1000, device=DEV).index_put_((idx.to(DEV),), src.to(DEV), accumulate=True), torch.zeros(1000).index_put_((idx,), src, accumulate=True))
    bag = torch.nn.EmbeddingBag(1000, 32, mode="sum")
    off = torch.arange(0, n, 50)
    e3 = rel(bag.to(DEV)(idx.to(DEV), off.to(DEV)), bag.cpu()(idx, off))
    t = torch.randn(300, 400)
    m = t > 0.3
    assert torch.equal(t.to(DEV)[m.to(DEV)].cpu(), t[m]) and torch.equal(torch.nonzero(m.to(DEV)).cpu(), torch.nonzero(m))
    e4 = rel(torch.cumsum(t.to(DEV).double(), 1), torch.cumsum(t.double(), 1))
    assert max(e1, e2, e3) < 1e-3 and e4 < 1e-12, (e1, e2, e3, e4)
    return f"scatter_add {e1:.1e}, index_put accumulate {e2:.1e}, embedding_bag {e3:.1e}, masked select/nonzero equal, cumsum {e4:.1e}"


@check("cdist_pdist")
def c_cdist():
    a = torch.randn(3000, 64)
    b = torch.randn(2500, 64)
    e1 = rel(torch.cdist(a.to(DEV), b.to(DEV)), torch.cdist(a, b))
    e2 = rel(F.pdist(a[:1500].to(DEV)), F.pdist(a[:1500]))
    assert max(e1, e2) < 1e-3, (e1, e2)
    return f"cdist 3000x2500 {e1:.1e}, pdist 1500 {e2:.1e}"


@check("memory_serialize")
def c_mem():
    import io
    free0, total = torch.cuda.mem_get_info(0)
    x = torch.empty(1 << 30, dtype=torch.uint8, device=DEV)
    used = torch.cuda.memory_allocated(0)
    del x
    torch.cuda.empty_cache()
    free1, _ = torch.cuda.mem_get_info(0)
    assert used >= 1 << 30 and free1 >= free0 - (64 << 20), (used, free0, free1)
    buf = io.BytesIO()
    sd = {"w": torch.randn(64, 64, device=DEV), "b": torch.arange(10, device="cuda:1")}
    torch.save(sd, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=True)
    assert back["w"].device == sd["w"].device and torch.equal(back["w"], sd["w"]) and back["b"].device.index == 1
    return f"{total / 2**30:.1f} GiB die, 1 GiB alloc/free returns, save/load weights_only keeps devices"


def _dist_worker(rank, world, backend, port, q):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group(backend, rank=rank, world_size=world, device_id=torch.device("cuda", rank))
        t = torch.full((1 << 20,), float(rank + 1), device=f"cuda:{rank}")
        dist.all_reduce(t)
        b = torch.full((1024,), float(rank), device=f"cuda:{rank}")
        dist.broadcast(b, src=0)
        gathered = [torch.empty(4, device=f"cuda:{rank}") for _ in range(world)]
        dist.all_gather(gathered, torch.full((4,), float(rank), device=f"cuda:{rank}"))
        big = torch.ones(256 << 20, device=f"cuda:{rank}")
        dist.all_reduce(big)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(5):
            dist.all_reduce(big)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        ok = (t[0].item() == sum(range(1, world + 1)) and b[0].item() == 0.0 and [g[0].item() for g in gathered] == [float(r) for r in range(world)])
        impl = type(dist.distributed_c10d._get_default_group()._get_backend(torch.device("cuda", rank))).__name__
        q.put((rank, ok, dt, impl, ""))
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        q.put((rank, False, 0.0, "", f"{type(e).__name__}: {str(e)[:200]}"))


def dist_check(backend, port):
    @check(f"distributed[{backend}]")
    def run():
        import torch.multiprocessing as mp
        import torch.distributed as dist
        if backend != "nccl":
            # 2.15 registers its extra built-in backends through the wheel's torch.distributed.backends entry points
            from importlib.metadata import entry_points
            if not any(ep.name.lower() == backend for ep in entry_points(group="torch.distributed.backends")):
                raise Skip(f"this torch has no '{backend}' backend")
        world = torch.cuda.device_count()
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        ps = [ctx.Process(target=_dist_worker, args=(r, world, backend, port, q)) for r in range(world)]
        for p in ps:
            p.start()
        out = []
        deadline = time.time() + 300
        while len(out) < world and time.time() < deadline:
            try:
                out.append(q.get(timeout=5))
            except Exception:  # noqa: BLE001 - queue.Empty: keep waiting while workers live
                if not any(p.is_alive() for p in ps):
                    break
        for p in ps:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        errs = [o[4] for o in out if o[4]]
        assert len(out) == world, f"{len(out)} of {world} ranks reported; exit codes {[p.exitcode for p in ps]}"
        assert not errs, errs[0]
        assert all(o[1] for o in out), "wrong collective result"
        dt = max(o[2] for o in out)
        return f"{world} ranks, all_reduce/broadcast/all_gather correct, 1 GiB all_reduce {dt * 1e3:.0f} ms ({1.0 / dt:.2f} GiB/s), implementation {out[0][3]}"
    return run


def main():
    print(f"torch {torch.__version__}  hip {torch.version.hip}  devices {torch.cuda.device_count()}  file {torch.__file__}", flush=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
    for fn in (c_blas, c_linalg, c_fft, c_sparse, c_random, c_sort, c_conv, c_rnn, c_attn, c_train, c_index, c_cdist, c_mem):
        fn()
    for i, backend in enumerate(("nccl", "nccl-lazy", "nccl-legacy")):
        dist_check(backend, 29600 + i)()
    failed = [n for n, ok in results if not ok]
    print(f"ops_check: {len(results) - len(failed)} passed, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}", flush=True)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
