"""
SDAR MoE builder with CPU offloading and optional GPU expert cache.

This keeps the SDAR block-diffusion decoding path intact and only swaps the
runtime MoE implementation from in-memory experts to the MoE-Offloading
runtime used by this project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import GenerationConfig

from baseline.expert_cache import ExpertCache
from baseline.nvtx_utils import nvtx_range
from baseline.sdar_layers import GateRegistry, SDARSparseMoeWrapper
from baseline.utils import with_default_dtype


OPENCOMPASS_ROOT = Path(__file__).resolve().parents[2] / "opencompass"
if str(OPENCOMPASS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOMPASS_ROOT))

from configs.sdar_local_models.modeling_sdar_moe import (  # noqa: E402
    SDARMoeAttention,
    SDARMoeConfig,
    SDARMoeDecoderLayer,
    SDARMoeForCausalLM,
    SDARMoeMLP,
    SDARMoeRMSNorm,
    SDARMoeSparseMoeBlock,
)


_ORIGINAL_ATTENTION_FORWARD = SDARMoeAttention.forward
_ORIGINAL_DECODER_LAYER_INIT = SDARMoeDecoderLayer.__init__
_ORIGINAL_SPARSE_MOE_INIT = SDARMoeSparseMoeBlock.__init__


def _apply_monkey_patches() -> None:
    if not getattr(SDARMoeAttention, "_moe_offloading_nvtx_patch", False):

        def patched_attention_forward(self, *args, **kwargs):
            with nvtx_range(f"Attention_Layer{self.layer_idx}"):
                return _ORIGINAL_ATTENTION_FORWARD(self, *args, **kwargs)

        SDARMoeAttention.forward = patched_attention_forward
        SDARMoeAttention._moe_offloading_nvtx_patch = True

    if not getattr(SDARMoeDecoderLayer, "_moe_offloading_patch", False):

        def patched_decoder_layer_init(self, config: SDARMoeConfig, layer_idx: int):
            nn.Module.__init__(self)
            self.hidden_size = config.hidden_size
            self.self_attn = SDARMoeAttention(config, layer_idx)

            use_moe = (
                layer_idx not in getattr(config, "mlp_only_layers", [])
                and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
            )
            if use_moe:
                self.mlp = SDARMoeSparseMoeBlock(config)
            else:
                self.mlp = SDARMoeMLP(
                    config, intermediate_size=config.intermediate_size
                )

            self.input_layernorm = SDARMoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = SDARMoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        SDARMoeDecoderLayer.__init__ = patched_decoder_layer_init
        SDARMoeDecoderLayer._moe_offloading_patch = True

    if not getattr(SDARMoeSparseMoeBlock, "_moe_offloading_patch", False):

        def patched_sparse_moe_init(self, config: SDARMoeConfig):
            nn.Module.__init__(self)
            target_experts = getattr(config, "_target_experts", config.num_experts)
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.norm_topk_prob = config.norm_topk_prob
            self.gate = nn.Linear(config.hidden_size, target_experts, bias=False)
            self.experts = nn.ModuleList([])

        SDARMoeSparseMoeBlock.__init__ = patched_sparse_moe_init
        SDARMoeSparseMoeBlock._moe_offloading_patch = True


def _normalize_device(device_or_map) -> torch.device:
    if isinstance(device_or_map, torch.device):
        return device_or_map
    if isinstance(device_or_map, str):
        if device_or_map in {"cuda", "gpu"}:
            return torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        return torch.device(device_or_map)
    if isinstance(device_or_map, int):
        return torch.device(f"cuda:{device_or_map}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _resolve_build_dtype(torch_dtype) -> torch.dtype:
    if torch_dtype is None:
        return torch.bfloat16
    if torch_dtype != torch.bfloat16:
        print(
            f"⚠️ SDAR offloading builder currently uses torch.bfloat16. "
            f"Requested dtype {torch_dtype} will be ignored."
        )
    return torch.bfloat16


def _group_weight_files(index_path: Path) -> Dict[str, list[str]]:
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    file_weights: Dict[str, list[str]] = {}
    for weight_name, file_name in weight_map.items():
        file_weights.setdefault(file_name, []).append(weight_name)
    return file_weights


def _set_tensor_by_path(model: nn.Module, weight_name: str, tensor: torch.Tensor) -> None:
    parts = weight_name.split(".")
    current = model
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)

    attr_name = parts[-1]
    if hasattr(current, attr_name):
        existing = getattr(current, attr_name)
        if isinstance(existing, nn.Parameter):
            existing.data = tensor
            return

    setattr(current, attr_name, nn.Parameter(tensor))


def _load_all_weights_unified(
    model: SDARMoeForCausalLM,
    state_path: Path,
    device: torch.device,
    config: SDARMoeConfig,
    expert_cache: ExpertCache,
    build_dtype: torch.dtype,
) -> None:
    index_path = state_path / "model.safetensors.index.json"
    file_weights = _group_weight_files(index_path)
    total_files = len(file_weights)

    for loaded_files, (file_name, weight_names) in enumerate(file_weights.items(), start=1):
        file_path = state_path / file_name
        weights = load_file(str(file_path), device="cpu")

        expert_weights = {
            name: weights[name]
            for name in weight_names
            if name in weights and "experts." in name
        }
        if expert_weights:
            expert_cache._process_weights_batch(expert_weights, config)

        for weight_name in weight_names:
            if weight_name not in weights or "experts." in weight_name:
                continue

            tensor = weights[weight_name].to(build_dtype).to(device)
            _set_tensor_by_path(model, weight_name, tensor)

        del weights
        if loaded_files % 8 == 0 or loaded_files == total_files:
            print(f"Processed {loaded_files}/{total_files} weight files...")
            torch.cuda.empty_cache()


def _iter_sparse_layers(model: SDARMoeForCausalLM) -> Iterable[tuple[int, nn.Module]]:
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            yield layer_idx, layer


def sdar_build_model(
    device,
    state_path: str,
    enable_gpu_cache: bool = True,
    cache_policy: str = "topk_lru",
    topk_lru_logit_percentile: float = 90.0,
    cache_slots_per_layer: int = 16,
    torch_dtype: Optional[torch.dtype] = None,
):
    _apply_monkey_patches()
    GateRegistry.clear()

    device = _normalize_device(device)
    build_dtype = _resolve_build_dtype(torch_dtype)
    state_path = Path(state_path)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = SDARMoeConfig.from_pretrained(str(state_path))
    original_num_experts = config.num_experts
    config.num_experts = 0
    config._target_experts = original_num_experts
    config._original_num_experts = original_num_experts

    with device, with_default_dtype(build_dtype):
        model = SDARMoeForCausalLM(config)

    expert_cache = ExpertCache(
        str(state_path),
        device,
        original_num_experts,
        enable_gpu_cache=enable_gpu_cache,
        cache_policy=cache_policy,
        topk_lru_logit_percentile=topk_lru_logit_percentile,
        cache_slots_per_layer=cache_slots_per_layer,
    )
    expert_cache.num_layers = config.num_hidden_layers
    expert_cache.num_experts = original_num_experts

    _load_all_weights_unified(
        model=model,
        state_path=state_path,
        device=device,
        config=config,
        expert_cache=expert_cache,
        build_dtype=build_dtype,
    )

    if expert_cache.enable_gpu_cache:
        expert_cache.init_gpu_cache()

    for layer_idx, layer in _iter_sparse_layers(model):
        gate = layer.mlp.gate
        layer.mlp = SDARSparseMoeWrapper(config, layer_idx, gate, expert_cache)
        if (layer_idx + 1) % 8 == 0:
            torch.cuda.empty_cache()

    try:
        model.generation_config = GenerationConfig.from_pretrained(str(state_path))
    except Exception:
        model.generation_config = GenerationConfig()

    torch.cuda.empty_cache()
    return model
