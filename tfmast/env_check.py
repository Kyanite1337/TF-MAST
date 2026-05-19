from __future__ import annotations

import json
import platform

import torch


def collect_env() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [],
        "mamba_ssm": False,
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append({"index": idx, "name": props.name, "total_memory_gb": round(props.total_memory / (1024**3), 2)})
        x = torch.randn(8, 8, device="cuda")
        info["cuda_tensor_check"] = float((x @ x).mean().detach().cpu())
    try:
        import mamba_ssm  # noqa: F401
        info["mamba_ssm"] = True
    except Exception as exc:
        info["mamba_ssm_error"] = str(exc)
    return info


def main() -> None:
    info = collect_env()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info["cuda_available"]:
        raise SystemExit("CUDA is not available")


if __name__ == "__main__":
    main()
