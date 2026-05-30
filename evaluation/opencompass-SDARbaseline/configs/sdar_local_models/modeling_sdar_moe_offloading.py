"""Local SDAR modeling entry that builds the MoE-Offloading runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch


HERE = Path(__file__).resolve()
OPENCOMPASS_ROOT = HERE.parents[2]
MOE_OFFLOADING_ROOT = HERE.parents[3] / "MoE-Offloading"

for path in (OPENCOMPASS_ROOT, MOE_OFFLOADING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.sdar_local_models.modeling_sdar_moe import (  # noqa: E402,F401
    SDARMoeAttention,
    SDARMoeConfig,
    SDARMoeDecoderLayer,
    SDARMoeMLP,
    SDARMoeModel,
    SDARMoePreTrainedModel,
    SDARMoeRMSNorm,
    SDARMoeRotaryEmbedding,
    SDARMoeSparseMoeBlock,
)
from baseline.sdar_builder import sdar_build_model  # noqa: E402


def _resolve_device(device=None, device_map=None) -> torch.device:
    if device is not None:
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return torch.device(device)

    if isinstance(device_map, str):
        if device_map in {"cuda", "gpu"}:
            return torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        if device_map.startswith("cuda:"):
            return torch.device(device_map)

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class SDARMoeForCausalLM(SDARMoePreTrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        device = _resolve_device(
            device=kwargs.pop("device", None),
            device_map=kwargs.pop("device_map", None),
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        enable_gpu_cache = kwargs.pop("enable_gpu_cache", True)
        cache_policy = kwargs.pop("cache_policy", "topk_lru")
        topk_lru_logit_percentile = kwargs.pop(
            "topk_lru_logit_percentile", 90.0
        )
        cache_slots_per_layer = kwargs.pop("cache_slots_per_layer", 16)

        if kwargs:
            print(
                "Unused SDAR offloading from_pretrained kwargs:",
                sorted(kwargs.keys()),
            )

        return sdar_build_model(
            device=device,
            state_path=str(pretrained_model_name_or_path),
            enable_gpu_cache=enable_gpu_cache,
            cache_policy=cache_policy,
            topk_lru_logit_percentile=topk_lru_logit_percentile,
            cache_slots_per_layer=cache_slots_per_layer,
            torch_dtype=torch_dtype,
        )
