from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class MaskBatch:
    encoder_mask: torch.Tensor
    decoder_mask: torch.Tensor


class MaskBank:
    def __init__(
        self,
        mask_ratio: float = 0.5,
        strategies: Sequence[str] | None = None,
        patch_size: tuple[int, int] = (1, 4),
        decoder_mask_ratio: float | None = None,
    ):
        self.mask_ratio = float(mask_ratio)
        self.decoder_mask_ratio = float(decoder_mask_ratio if decoder_mask_ratio is not None else mask_ratio)
        self.strategies = list(strategies or ["block", "temporal", "sensor", "multi_scale"])
        self.patch_size = tuple(patch_size)

    def __call__(self, x: torch.Tensor) -> MaskBatch:
        shape = self._grid_shape(x)
        return MaskBatch(
            encoder_mask=self._sample_mask(shape, x.device, self.mask_ratio),
            decoder_mask=self._sample_mask(shape, x.device, self.decoder_mask_ratio),
        )

    def _grid_shape(self, x: torch.Tensor) -> tuple[int, int, int]:
        batch, channels, width = x.shape
        patch_width = self.patch_size[1]
        return batch, channels // self.patch_size[0], width // patch_width

    def _sample_mask(self, shape: tuple[int, int, int], device: torch.device, ratio: float) -> torch.Tensor:
        batch, channels, steps = shape
        out = torch.zeros(shape, dtype=torch.bool, device=device)
        for b in range(batch):
            strategy = self.strategies[torch.randint(0, len(self.strategies), (1,), device=device).item()]
            out[b] = self._strategy_mask(channels, steps, ratio, strategy, device)
        return out

    def _strategy_mask(self, channels: int, steps: int, ratio: float, strategy: str, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(channels, steps, dtype=torch.bool, device=device)
        total = max(1, int(round(channels * steps * ratio)))
        if strategy in {"sensor", "sensor-wise", "sensor_wise"}:
            n_channels = max(1, int(round(channels * ratio)))
            idx = torch.randperm(channels, device=device)[:n_channels]
            mask[idx, :] = True
        elif strategy == "temporal":
            n_steps = max(1, int(round(steps * ratio)))
            idx = torch.randperm(steps, device=device)[:n_steps]
            mask[:, idx] = True
        elif strategy in {"multi_scale", "multi-scale"}:
            remaining = total
            while remaining > 0:
                h = int(torch.randint(1, min(channels, 4) + 1, (1,), device=device).item())
                w = int(torch.randint(1, min(steps, 4) + 1, (1,), device=device).item())
                c0 = int(torch.randint(0, max(1, channels - h + 1), (1,), device=device).item())
                t0 = int(torch.randint(0, max(1, steps - w + 1), (1,), device=device).item())
                before = mask.sum()
                mask[c0 : c0 + h, t0 : t0 + w] = True
                remaining -= int(mask.sum().item() - before.item())
                if mask.sum().item() >= total:
                    break
        else:
            flat = mask.flatten()
            start = int(torch.randint(0, flat.numel(), (1,), device=device).item())
            length = max(1, min(total, flat.numel()))
            idx = (torch.arange(length, device=device) + start) % flat.numel()
            flat[idx] = True
        if mask.sum().item() < total and strategy not in {"sensor", "sensor-wise", "sensor_wise", "temporal"}:
            missing = total - int(mask.sum().item())
            candidates = (~mask).flatten().nonzero(as_tuple=False).flatten()
            if candidates.numel() > 0:
                extra = candidates[torch.randperm(candidates.numel(), device=device)[:missing]]
                mask.flatten()[extra] = True
        return mask


def expand_patch_mask(mask: torch.Tensor, width: int, patch_size: tuple[int, int] = (1, 4)) -> torch.Tensor:
    patch_width = patch_size[1]
    expanded = mask.repeat_interleave(patch_width, dim=2)
    return expanded[:, :, :width]
