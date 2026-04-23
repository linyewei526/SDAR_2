"""
GPT-OSS Model Builder - Transformers-based with Monkey Patching
类似Qwen3的实现风格：使用transformers的官方模型结构，通过monkey patching替换MoE层

GPT-OSS模型特点：
- 24层 (num_hidden_layers)
- 32个专家 (num_local_experts)
- 每token选4个专家 (num_experts_per_tok)
- hidden_size: 2880
- intermediate_size: 2880
- 使用合并的 gate_up_proj
- router: mlp.router (带bias)
- sliding_attention / full_attention 交替
"""

import json
import torch
from torch import nn
from transformers import AutoConfig
from safetensors.torch import load_file
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssDecoderLayer,
    GptOssMLP,
    GptOssTopKRouter,
    GptOssExperts,
    GptOssRMSNorm,
    GptOssAttention,
)
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from baseline.nvtx_utils import nvtx_range

# ===== Monkey Patch 0: 给Attention添加NVTX标记 =====
original_attention_forward = GptOssAttention.forward

def patched_attention_forward(self, *args, **kwargs):
    with nvtx_range(f"Attention_Layer{self.layer_idx}"):
        return original_attention_forward(self, *args, **kwargs)

GptOssAttention.forward = patched_attention_forward


# ===== Monkey Patch 1: 修改DecoderLayer的__init__用于offloading =====
original_decoder_layer_init = GptOssDecoderLayer.__init__

def patched_decoder_layer_init(self, config: GptOssConfig, layer_idx: int):
    """修复的__init__方法，用于offloading场景（不创建experts节省显存）"""
    from transformers.modeling_layers import GradientCheckpointingLayer
    GradientCheckpointingLayer.__init__(self)
    
    self.hidden_size = config.hidden_size
    self.layer_idx = layer_idx  # 保存layer_idx供后续使用
    self.self_attn = GptOssAttention(config=config, layer_idx=layer_idx)
    
    # 总是创建MLP（router会创建，但experts可能被跳过）
    self.mlp = GptOssMLP(config)
    
    self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.attention_type = config.layer_types[layer_idx]

GptOssDecoderLayer.__init__ = patched_decoder_layer_init


# ===== Monkey Patch 2: 修改GptOssMLP的__init__为空experts =====
original_mlp_init = GptOssMLP.__init__

def patched_mlp_init(self, config):
    """修复的MLP，如果设置了_skip_experts则不创建experts节省显存"""
    super(GptOssMLP, self).__init__()
    
    # 使用目标专家数量（用于router维度）
    target_experts = getattr(config, '_target_experts', config.num_local_experts)
    
    # 临时修改config的num_local_experts用于router创建
    original_num_experts = config.num_local_experts
    config.num_local_experts = target_experts
    
    # 创建router（维度为target_experts）
    self.router = GptOssTopKRouter(config)
    
    # 恢复原始值
    config.num_local_experts = original_num_experts
    
    # 只有当不跳过experts时才创建
    if not getattr(config, '_skip_experts', False):
        self.experts = GptOssExperts(config)
    else:
        self.experts = None

GptOssMLP.__init__ = patched_mlp_init


# 导入offloading组件
from baseline.expert_cache import GptOssExpertCache
from baseline.gptoss_layers import GptOssSparseMoeWrapper, GateRegistry
from baseline.utils import with_default_dtype


def gptoss_build_model(
    device, 
    state_path: str,
    enable_gpu_cache: bool = True,
    cache_policy: str = "topk_lru",
    topk_lru_logit_percentile: float = 90.0,
    cache_slots_per_layer: int = 8,
):
    """
    构建GPT-OSS MoE模型（带offloading）- 使用transformers官方模型 + monkey patching

    Args:
        device: 目标设备
        state_path: 模型权重路径

    Returns:
        GptOssForCausalLM: 构建好的模型
    """
    # 0. 清空GateRegistry（防止多次构建时残留）
    GateRegistry.clear()
    
    # 1. 加载配置（使用官方AutoConfig）
    config = AutoConfig.from_pretrained(state_path, trust_remote_code=True)

    # 将config转为dict供后续使用
    config_dict = config.to_dict()

    # 2. 设置配置标记，让monkey patch跳过experts创建
    config._skip_experts = True
    config._target_experts = config.num_local_experts
    
    # 3. 创建基础模型（experts不会被创建，节省显存）
    with device, with_default_dtype(torch.bfloat16):
        model = GptOssForCausalLM(config)
    
    # 4. 设置expert cache
    buffer_size = config.num_local_experts  # 32个buffer对应32个专家
    expert_cache = GptOssExpertCache(
        state_path, device, buffer_size=buffer_size,
        enable_gpu_cache=enable_gpu_cache,
        cache_policy=cache_policy,
        topk_lru_logit_percentile=topk_lru_logit_percentile,
        cache_slots_per_layer=cache_slots_per_layer,
    )
    
    # 5. 加载权重（非expert权重到GPU，expert权重到CPU cache）
    _load_all_weights_unified(model, state_path, device, config, expert_cache)

    # 5.5 初始化 GPU Cache（在 CPU cache 准备好后）
    if expert_cache.enable_gpu_cache:
        expert_cache.init_gpu_cache()
    
    # 6. 先加载所有router权重，注册到GateRegistry
    print("   Loading routers and registering to GateRegistry...")
    index_path = f"{state_path}/model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        router = layer.mlp.router
        
        router_weight_name = f"model.layers.{layer_idx}.mlp.router.weight"
        router_bias_name = f"model.layers.{layer_idx}.mlp.router.bias"
        
        if router_weight_name in weight_map:
            file_name = weight_map[router_weight_name]
            file_path = f"{state_path}/{file_name}"
            weights = load_file(file_path, device='cpu')
            
            if router_weight_name in weights:
                router.weight.data = weights[router_weight_name].to(torch.bfloat16).to(device)
            if router_bias_name in weights:
                router.bias.data = weights[router_bias_name].to(torch.bfloat16).to(device)
            
            del weights
        
        # 注册到GateRegistry
        GateRegistry.register_gate(layer_idx, router)
    
    # 7. 替换MoE层为offloading wrapper
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        router = layer.mlp.router
        
        # 替换为MoE wrapper
        layer.mlp = GptOssSparseMoeWrapper(
            config_dict, layer_idx, router, expert_cache
        )
        
        if (layer_idx + 1) % 8 == 0:
            print(f"   Processed {layer_idx + 1}/{config.num_hidden_layers} layers...")
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    print(f"✅ GPT-OSS model built successfully with prefetch optimization!")
    
    return model


def _load_all_weights_unified(model, state_path, device, config, expert_cache):
    """
    统一的权重加载（transformers兼容）：
    - 非expert权重 → GPU模型参数
    - expert权重 → CPU cache供offloading使用
    """
    index_path = f"{state_path}/model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    
    # 按文件分组权重
    file_weights = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in file_weights:
            file_weights[file_name] = []
        file_weights[file_name].append(weight_name)
    
    loaded_files = 0
    
    for file_name, weight_names in file_weights.items():
        file_path = f"{state_path}/{file_name}"
        weights = load_file(file_path, device='cpu')
        loaded_files += 1
        
        # 批量处理专家权重 → CPU cache
        expert_weights_dict = {name: weights[name] for name in weight_names
                              if name in weights and 'experts' in name}
        if expert_weights_dict:
            expert_cache._process_weights_batch(expert_weights_dict, config)
        
        # 处理非专家权重 → GPU模型参数
        for weight_name in weight_names:
            if weight_name in weights:
                # 跳过专家权重和router权重（router在build_model中单独加载）
                if 'experts' in weight_name:
                    continue
                elif 'router' in weight_name:
                    continue
                else:
                    # 使用 transformers 标准的参数设置方式
                    parts = weight_name.split('.')
                    try:
                        current = model
                        for part in parts[:-1]:
                            if part.isdigit():
                                current = current[int(part)]
                            else:
                                current = getattr(current, part)
                        
                        param_name = parts[-1]
                        param_value = weights[weight_name].to(torch.bfloat16).to(device)
                        
                        # 检查是否是Parameter还是普通Tensor
                        if hasattr(current, param_name):
                            existing = getattr(current, param_name)
                            if isinstance(existing, nn.Parameter):
                                existing.data = param_value
                            else:
                                setattr(current, param_name, nn.Parameter(param_value))
                        else:
                            setattr(current, param_name, nn.Parameter(param_value))
                    except AttributeError:
                        # 某些权重可能因为monkey patch而找不到目标
                        pass
        
        del weights
        import gc
        gc.collect()
        
        if loaded_files % 3 == 0:
            print(f"   Loaded {loaded_files}/{len(file_weights)} weight files...")
    
    print(f"✅ All weights loaded!")
