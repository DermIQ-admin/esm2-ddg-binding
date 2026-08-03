"""Shared utilities: device reporting and reproducibility.

Run the environment check directly with:

    .venv\\Scripts\\python.exe -m src.utils
"""

from __future__ import annotations

import random


def describe_device() -> dict:
    """Report what PyTorch can actually see, and prove the GPU really works.

    `torch.cuda.is_available()` returning True is necessary but NOT sufficient:
    it can be True on a mismatched driver/runtime pairing where kernel launches
    then fail at the first real operation. So this runs an actual matmul on the
    GPU and checks the result against the CPU, which is the honest test.
    """
    import torch

    info = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if not info["cuda_available"]:
        info["error"] = "CUDA not available — training would silently fall back to CPU."
        return info

    props = torch.cuda.get_device_properties(0)
    info.update(
        {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            # (8, 9) == Ada Lovelace, which is what an RTX 4090 should report.
            "capability": torch.cuda.get_device_capability(0),
            "total_vram_gb": round(props.total_memory / 1024**3, 2),
            "multiprocessors": props.multi_processor_count,
            # bf16 matters for training: same dynamic range as fp32, so mixed
            # precision needs no GradScaler. See the note in src/train.py.
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )

    # The real test: run something on the GPU and check it against the CPU.
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    gpu_result = (a.cuda() @ b.cuda()).cpu()
    info["matmul_matches_cpu"] = bool(
        torch.allclose(gpu_result, a @ b, atol=1e-3, rtol=1e-3)
    )
    info["max_abs_error"] = float((gpu_result - a @ b).abs().max())

    return info


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Note this does not by itself make CUDA fully deterministic — some cuDNN
    kernels are nondeterministic regardless. Good enough for reproducible
    splits and comparable runs, which is what we need.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    width = max(len(k) for k in describe_device())
    for key, value in describe_device().items():
        print(f"{key:<{width}} : {value}")
