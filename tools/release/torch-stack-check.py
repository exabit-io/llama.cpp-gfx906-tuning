"""Stack check for python3-torch-gfx906: torch_check.py (matmul per die, MIOpen conv, P2P, RCCL all-reduce) plus the
CPU backends (MKL, oneDNN) and the torchvision / torchaudio GPU paths. Run with the system python, no venv:
    /usr/bin/python3 -I stack_check.py
"""
import sys

sys.path.insert(0, "/root/pytorch-gfx906")  # spawned RCCL workers import torch_check by name; spawn copies sys.path

import torch
import torch_check as tc


def cpu_checks():
    print("torch file:", torch.__file__)
    print("  MKL", torch.backends.mkl.is_available(), " MKLDNN", torch.backends.mkldnn.is_available(),
          " OpenMP", torch.backends.openmp.is_available(), " CPU capability", torch.backends.cpu.get_cpu_capability())
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    err = ((a @ b) - (a.double() @ b.double()).float()).abs().max().item()
    print(f"  CPU fp32 matmul 1024^3 max abs err vs fp64 {err:.2e}")
    assert err < 1e-2


def vision_audio_checks():
    import torchvision
    import torchaudio
    from torchvision.ops import nms, roi_align
    print("torchvision", torchvision.__version__, " torchaudio", torchaudio.__version__)
    torch.manual_seed(0)
    boxes = torch.rand(2000, 4) * 500
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.rand(2000)
    keep_gpu = nms(boxes.cuda(), scores.cuda(), 0.5).cpu()
    keep_cpu = nms(boxes, scores, 0.5)
    print(f"  torchvision nms on GPU: {keep_gpu.numel()} kept, equal to CPU: {torch.equal(keep_gpu, keep_cpu)}")
    assert torch.equal(keep_gpu, keep_cpu)
    feat = torch.randn(2, 16, 64, 64)
    rois = torch.tensor([[0, 4.0, 4.0, 40.0, 40.0], [1, 10.0, 8.0, 50.0, 60.0]])
    rg = roi_align(feat.cuda(), rois.cuda(), (7, 7), 0.5, 2).cpu()
    rc = roi_align(feat, rois, (7, 7), 0.5, 2)
    print(f"  torchvision roi_align GPU vs CPU max abs diff {(rg - rc).abs().max().item():.2e}")
    assert (rg - rc).abs().max().item() < 1e-4
    wave = torch.randn(1, 16000, device="cuda")
    spec_ = torchaudio.transforms.MelSpectrogram(sample_rate=16000).cuda()(wave)
    print("  torchaudio MelSpectrogram on GPU:", tuple(spec_.shape))


if __name__ == "__main__":
    cpu_checks()
    vision_audio_checks()
    tc.main()
    sys.exit(0)
