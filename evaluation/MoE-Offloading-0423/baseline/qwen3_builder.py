"""
Simple Qwen3 MoE builder - Direct copy of Mixtral approach but for Qwen3
"""

import json
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock, Qwen3MoeRMSNorm, Qwen3MoeAttention
from safetensors.torch import load_file
from baseline.nvtx_utils import nvtx_range

# Monkey patch 0: 给Attention添加NVTX标记
original_attention_forward = Qwen3MoeAttention.forward

def patched_attention_forward(self, *args, **kwargs):
    with nvtx_range(f"Attention_Layer{self.layer_idx}"):
        return original_attention_forward(self, *args, **kwargs)

Qwen3MoeAttention.forward = patched_attention_forward

# Monkey patch 1: 强制所有层都使用Qwen3MoeSparseMoeBlock
original_decoder_layer_init = Qwen3MoeDecoderLayer.__init__

def patched_decoder_layer_init(self, config: Qwen3MoeConfig, layer_idx: int):
    """修复的__init__方法，强制所有层都使用MoE结构"""

    # 使用显式的super调用，避免monkey patch中的super()问题
    from transformers.modeling_layers import GradientCheckpointingLayer
    GradientCheckpointingLayer.__init__(self)

    self.hidden_size = config.hidden_size

    self.self_attn = Qwen3MoeAttention(config, layer_idx)

    # 强制所有层都使用MoE结构，不管原来的判断逻辑
    self.mlp = Qwen3MoeSparseMoeBlock(config)

    self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# Monkey patch 2: 修复Qwen3MoeSparseMoeBlock的gate维度
original_sparse_moe_init = Qwen3MoeSparseMoeBlock.__init__

def patched_sparse_moe_init(self, config):
    """修复的Qwen3MoeSparseMoeBlock，使用目标专家数量作为gate维度"""

    # 使用显式的super调用，避免monkey patch中的super()问题
    nn.Module.__init__(self)

    # 使用目标专家数量（用于gate维度），而不是0
    target_experts = getattr(config, '_target_experts', config.num_experts)

    self.num_experts = config.num_experts  # 实际experts数量（为0）
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob

    # gate使用目标专家数量，这样有正确的维度
    self.gate = nn.Linear(config.hidden_size, target_experts, bias=False)

    # 创建空的experts列表（节省显存），由我们的offload系统处理
    self.experts = nn.ModuleList([])

# 应用monkey patches
Qwen3MoeDecoderLayer.__init__ = patched_decoder_layer_init
Qwen3MoeSparseMoeBlock.__init__ = patched_sparse_moe_init

from baseline.expert_cache import Qwen3ExpertCache
from baseline.qwen3_layers import Qwen3SparseMoeWrapper, Qwen3ExpertMLP
from baseline.utils import with_default_dtype


def qwen3_build_model(
    device, 
    state_path,
    enable_gpu_cache: bool = True,
    cache_policy: str = "topk_lru",
    topk_lru_logit_percentile: float = 90.0,
    cache_slots_per_layer: int = 16,
):
    """Simple Qwen3 build - Mixtral style"""
    # 1. Load config
    config = AutoConfig.from_pretrained(state_path)

    # 临时修改num_experts为0，并强制保持MoE结构
    original_num_experts = config.num_experts
    config.num_experts = 0  # 不创建任何experts，节省显存
    config._force_moe_structure = True  # 强制保持MoE结构，不退化到MLP
    config._target_experts = 128  # 恢复为正常的128个expert，用于gate维度
    config._original_num_experts = original_num_experts

    # 保存原始num_experts用于权重加载判断
    # 2. Create base model with reduced number of experts
    with device, with_default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM(config)

    # 3. Setup expert cache (like Mixtral)
    # Buffer size = 128 experts (shared by all layers, each layer overwrites from idx=0)
    buffer_size = original_num_experts  # 128
    expert_cache = Qwen3ExpertCache(
        state_path, device, buffer_size=buffer_size,
        enable_gpu_cache=enable_gpu_cache,
        cache_policy=cache_policy,
        topk_lru_logit_percentile=topk_lru_logit_percentile,
        cache_slots_per_layer=cache_slots_per_layer,
    )

    # 🚀 优化：使用简单字典缓存，无需复杂的untyped storage初始化
    # 简单缓存会自动按需初始化，无需预先分配存储
    num_layers = config.num_hidden_layers
    num_experts = getattr(config, '_target_experts', config.num_experts)
    expert_cache.num_layers = num_layers  # 设置基本信息用于边界检查
    expert_cache.num_experts = num_experts

    # 4. Load all weights in one pass: non-expert weights to GPU, expert weights to CPU cache
    _load_all_weights_unified(model, state_path, device, config, expert_cache)

    # 4.5 初始化 GPU Cache（在 CPU cache 准备好后）
    if expert_cache.enable_gpu_cache:
        expert_cache.init_gpu_cache()

    # 5. Replace MoE layers with wrappers (like Mixtral's _setup_experts)
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        # Qwen3: All layers should be MoE (decoder_sparse_step=1, mlp_only_layers=[])
        if hasattr(layer.mlp, 'gate'):
            # Fix gate dimension to match our target_experts=128
            original_gate = layer.mlp.gate
            fixed_gate = nn.Linear(config.hidden_size, config._target_experts, bias=False)
            # Load correct gate weights from the model files
            gate_weight_name = f"model.layers.{layer_idx}.mlp.gate.weight"
            index_path = f"{state_path}/model.safetensors.index.json"

            # Find which file contains the gate weight
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]

            if gate_weight_name in weight_map:
                file_name = weight_map[gate_weight_name]
                file_path = f"{state_path}/{file_name}"
                weights = load_file(file_path, device='cpu')

                if gate_weight_name in weights:
                    gate_weight_data = weights[gate_weight_name].to(torch.bfloat16)
                    # IMPORTANT: Only take first _target_experts weights from the 128-dim gate
                    gate_weight_data = gate_weight_data[:config._target_experts, :]
                    # Move gate weight to the correct device
                    gate_weight_data = gate_weight_data.to(device)
                    # Gate weight loaded successfully
                    fixed_gate.weight.data = gate_weight_data
                else:
                    # Fallback: repeat the single expert weight
                    fixed_gate.weight.data = original_gate.weight.data.repeat(config._target_experts, 1)[:config._target_experts]
            else:
                # Fallback: repeat the single expert weight
                fixed_gate.weight.data = original_gate.weight.data.repeat(config._target_experts, 1)[:config._target_experts]

            # Replace with our wrapper, using the fixed gate
            layer.mlp = Qwen3SparseMoeWrapper(
                config, layer_idx, fixed_gate, expert_cache, model.model.layers
            )
        else:
            # Non-MoE layer, skip replacement
            pass

        # Clear cache every few layers to prevent memory buildup during construction
        if (layer_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Final cache cleanup to minimize reserved memory
    torch.cuda.empty_cache()

    return model


def _load_all_weights_unified(model, state_path, device, config, expert_cache):
    """统一的权重加载：非expert权重到GPU，expert权重到CPU cache"""
    index_path = f"{state_path}/model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Group weights by file
    file_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_weights:
            file_weights[file_name] = []
        file_weights[file_name].append(weight_name)

    # Check if we modified num_experts from original
    num_experts_modified = hasattr(config, '_original_num_experts') and config.num_experts < config._original_num_experts
    loaded_files = 0

    # Load weights file by file (to CPU first, then distribute)
    for file_name, weight_names in file_weights.items():
        file_path = f"{state_path}/{file_name}"
        # Load to CPU first to avoid temporary GPU memory spikes
        weights = load_file(file_path, device='cpu')
        loaded_files += 1

        # 🚀 优化：批量处理专家权重，提高效率
        expert_weights_dict = {name: weights[name] for name in weight_names
                             if name in weights and 'experts.' in name}
        if expert_weights_dict:
            expert_cache._process_weights_batch(expert_weights_dict, config)

        # Process non-expert weights individually
        for weight_name in weight_names:
            if weight_name in weights:
                if 'experts.' in weight_name:
                    # Already processed in batch above
                    continue
                elif num_experts_modified and 'mlp.gate.weight' in weight_name:
                    # Skip gate weights when we've reduced num_experts
                    continue
                else:
                    # Non-expert weights -> GPU
                    parts = weight_name.split('.')
                    current = model
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], nn.Parameter(weights[weight_name].to(torch.bfloat16).to(device)))

        # Clear CPU memory for this file
        del weights
        import gc
        gc.collect()

        if loaded_files % 4 == 0:
            print(f"Processed {loaded_files}/{len(file_weights)} files...")
