"""
GPT-OSS MoE Layers - 优化版
实现prefetch和异步流水线优化

完全参考官方transformers实现
兼容官方GptOssTopKRouter（返回router_scores, router_indices）
"""

import time
import torch
from torch import nn
from torch.nn import functional as F

from .debug_config import (
    PRINT_EXPERT_DETAILS,
    PRINT_PREFETCH_DEBUG,
    PREFETCH_ENABLED
)
from .nvtx_utils import nvtx_range

# ===== Prefetch MACRO =====
# Prefetch top-k experts based on predicted weights (0 = disable prefetch)
PREFETCH_TOPK = 2
# ===== END PREFETCH MACRO =====


class GateRegistry:
    """全局router注册表，用于安全的跨层访问"""
    _instance = None
    _gates = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_gate(cls, layer_idx: int, gate: nn.Module):
        """注册router"""
        cls._gates[layer_idx] = gate

    @classmethod
    def get_gate(cls, layer_idx: int):
        """获取router"""
        return cls._gates.get(layer_idx, None)

    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._gates.clear()


class GptOssSimpleMoE(nn.Module):
    """
    GPT-OSS Optimized MoE with prefetch and async pipeline
    - 计算router得到active experts
    - 使用下一层gate预测下一层experts进行prefetch
    - 批量加载和计算expert outputs
    - 异步prefetch与compute overlap
    
    兼容两种router类型：
    1. nn.Linear: 返回logits，需要自己做topk和softmax
    2. GptOssTopKRouter: 返回(router_scores, router_indices)
    """

    # Class-level statistics collection
    layer_expert_counts = []
    layer_times = []
    forward_session_id = 0
    decode_token_counter = 0

    # Router prefetch data collection
    router_inputs_for_prefetch = {}
    next_layer_predictions = {}

    def __init__(self, config, layer_idx: int, router_layer, expert_cache):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expert_cache = expert_cache
        self.router = router_layer

        # Expert参数
        self.hidden_size = config['hidden_size']
        self.num_experts = config.get('num_local_experts', 32)
        self.top_k = config.get('num_experts_per_tok', 4)
        self.intermediate_size = config['intermediate_size']
        self.num_hidden_layers = config.get('num_hidden_layers', 24)

        # GPT-OSS特有参数
        self.limit = config.get('swiglu_limit', 7.0)
        self.alpha = 1.702

        # 注册router到全局注册表
        GateRegistry.register_gate(layer_idx, router_layer)

    def forward(self, hidden_states):
        """
        优化的forward pass，支持prefetch和异步流水线
        """
        layer_start = time.time()

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        src_dtype = hidden_states.dtype

        # ===== Router Input Collection for Prefetch =====
        if PREFETCH_ENABLED:
            router_input_flat = hidden_states_flat.clone().detach()

        # ===== Router计算 =====
        with nvtx_range(f"MoE_Routing_Layer{self.layer_idx}"):
            # 兼容官方GptOssTopKRouter（返回router_scores, router_indices）
            router_output = self.router(hidden_states_flat)
            
            if isinstance(router_output, tuple):
                # 官方GptOssTopKRouter: 返回 (router_scores, router_indices)
                router_scores, selected_experts = router_output
            else:
                # nn.Linear: 返回logits，需要自己做topk和softmax
                router_logits = router_output
                routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
                routing_weights = F.softmax(routing_weights, dim=1, dtype=routing_weights.dtype)
                router_scores = torch.zeros_like(router_logits)
                router_scores.scatter_(1, selected_experts, routing_weights)

            # 🚀 Next layer router prediction for prefetch
            next_layer_selected_experts = None
            next_layer_selected_experts_cpu = None
            if PREFETCH_ENABLED and self.layer_idx < self.num_hidden_layers - 1:
                next_layer_router = GateRegistry.get_gate(self.layer_idx + 1)
                if next_layer_router is not None:
                    with torch.no_grad():
                        # Use next layer router with current hidden states to predict next layer experts
                        next_router_output = next_layer_router(hidden_states_flat)
                        
                        if isinstance(next_router_output, tuple):
                            next_layer_weights, next_layer_selected_experts = next_router_output
                            # 取top-K
                            if next_layer_selected_experts.shape[-1] > PREFETCH_TOPK:
                                next_layer_selected_experts = next_layer_selected_experts[:, :PREFETCH_TOPK]
                        else:
                            next_layer_logits = next_router_output
                            next_layer_weights = F.softmax(next_layer_logits, dim=1, dtype=torch.float)
                            sorted_indices = torch.argsort(next_layer_weights, dim=-1, descending=True)
                            next_layer_selected_experts = sorted_indices[:, :PREFETCH_TOPK]

                        if PRINT_PREFETCH_DEBUG:
                            top_experts = next_layer_selected_experts[0].cpu().tolist()
                            print(f"Layer {self.layer_idx:2d} -> Next layer {self.layer_idx+1} prediction: {top_experts}")

                        # DtoH获取expert list用于prefetch
                        next_layer_selected_experts_cpu = next_layer_selected_experts.flatten().cpu()

        # Print Expert Selection (for decode phase)
        if PRINT_EXPERT_DETAILS and hidden_states_flat.shape[0] == 1:
            expert_ids = selected_experts[0].cpu().tolist()
            print(f"🔍 Layer {self.layer_idx:2d} | Expert selection: {expert_ids}")

        # Decode token counter
        if hidden_states_flat.shape[0] == 1:
            if self.layer_idx == 0:
                pass  # Decode token header
            if self.layer_idx == self.num_hidden_layers - 1:
                GptOssSimpleMoE.decode_token_counter += 1

        # ===== 找出unique expert IDs并加载experts =====
        with nvtx_range("MoE_Expert_Load_Prep"):
            flat_experts = selected_experts.reshape(-1)
            flat_experts_cpu = flat_experts.cpu()
            expert_counts = torch.bincount(flat_experts_cpu, minlength=self.num_experts)
            expert_offsets_cpu = torch.cumsum(expert_counts, dim=0) - expert_counts

            active_expert_ids = torch.where(expert_counts > 0)[0].tolist()
            expert_counts_list = expert_counts.tolist()
            expert_offsets_list = expert_offsets_cpu.tolist()

            # 为动态 cache 策略计算每个 expert 的 logit score
            router_logits = None
            if self.expert_cache.enable_gpu_cache and self.expert_cache.gpu_cache_manager is not None:
                policy_name = self.expert_cache.gpu_cache_manager.cache_policy_name
                if policy_name in ("lru", "lfu", "topk_lru", "tinylfu"):
                    max_scores_per_expert, _ = router_scores.max(dim=0)
                    router_logits = {eid: max_scores_per_expert[eid].item() for eid in active_expert_ids}

            # ===== 批量加载experts到GPU =====
            expert_indices = [(eid, eid) for eid in active_expert_ids]
            expert_to_buffer_mapping = self.expert_cache.batch_load_experts_continuous(
                self.layer_idx, expert_indices, router_logits
            )

            # 缓存统计信息
            num_tokens_cached = selected_experts.shape[0]
            expert_count_cached = len(active_expert_ids)

            # 🚀 启动Prefetch（与compute并行）
            if PREFETCH_ENABLED and next_layer_selected_experts_cpu is not None and self.layer_idx < self.num_hidden_layers - 1:
                with nvtx_range(f"Prefetch_Start_Layer{self.layer_idx}"):
                    self._parallel_prefetch(next_layer_selected_experts_cpu)

        # 🚀 GPU排序（延迟到prefetch之后，与prefetch HtoD overlap）
        with nvtx_range("Global_Expert_Routing_Prep_GPU_Sort"):
            num_tokens = selected_experts.shape[0]

            token_indices = torch.arange(num_tokens, device=selected_experts.device).repeat_interleave(self.top_k)
            k_ranks = torch.tile(torch.arange(self.top_k, device=selected_experts.device), (num_tokens,))

            sorted_experts, perm = torch.sort(flat_experts)
            sorted_tokens = token_indices[perm]
            sorted_ranks = k_ranks[perm]

        # ===== 批量专家计算 =====

        # NVTX: Gather输入和准备阶段（不包括Batched_Expert_Compute）
        with nvtx_range("Expert_Input_Gather_Prep"):
            # Gather输入
            all_input_states = hidden_states_flat[sorted_tokens]
            # GPT-OSS使用router_scores而不是routing_weights
            all_routing_weights = router_scores[sorted_tokens, sorted_experts].unsqueeze(1)

            # 批量expert计算
            expert_outputs = torch.empty_like(all_input_states)
            pool = self.expert_cache.buffer_manager.gpu_memory_pool

            num_active = len(active_expert_ids)
            if num_active > 0:
                max_tok = max(expert_counts_list[eid] for eid in active_expert_ids)

                # 按专家分组构造批输入
                batched_inputs = all_input_states.new_zeros(num_active, max_tok, hidden_dim)
                batched_rweights = all_routing_weights.new_zeros(num_active, max_tok, 1)

                gate_up_views, down_views = [], []
                gate_up_bias_views, down_bias_views = [], []
                expert_offsets_local, expert_counts_local = [], []

                for row_idx, expert_id in enumerate(active_expert_ids):
                    count = expert_counts_list[expert_id]
                    offset = expert_offsets_list[expert_id]
                    expert_offsets_local.append(offset)
                    expert_counts_local.append(count)

                    if count > 0:
                        batched_inputs[row_idx, :count] = all_input_states[offset : offset + count]
                        batched_rweights[row_idx, :count] = all_routing_weights[offset : offset + count]

                    virtual_idx = expert_to_buffer_mapping[expert_id]
                    gpu_buffer = self.expert_cache.buffer_manager.get_expert_view_for_computation(virtual_idx)
                    pool = gpu_buffer['memory_pool']

                    # Gate-Up projection
                    gu_off = gpu_buffer['gate_up_proj']['offset']
                    gu_shape = gpu_buffer['gate_up_proj']['shape']
                    gate_up_views.append(pool[gu_off : gu_off + gu_shape[0] * gu_shape[1]].view(gu_shape))

                    # Gate-Up bias
                    if 'gate_up_proj_bias' in gpu_buffer and gpu_buffer['gate_up_proj_bias']['size'] > 0:
                        gub_off = gpu_buffer['gate_up_proj_bias']['offset']
                        gub_shape = gpu_buffer['gate_up_proj_bias']['shape']
                        gate_up_bias_views.append(pool[gub_off : gub_off + gub_shape[0]].view(gub_shape))
                    else:
                        gate_up_bias_views.append(None)

                    # Down projection
                    d_off = gpu_buffer['down_proj']['offset']
                    d_shape = gpu_buffer['down_proj']['shape']
                    down_views.append(pool[d_off : d_off + d_shape[0] * d_shape[1]].view(d_shape))

                    # Down bias
                    if 'down_proj_bias' in gpu_buffer and gpu_buffer['down_proj_bias']['size'] > 0:
                        db_off = gpu_buffer['down_proj_bias']['offset']
                        db_shape = gpu_buffer['down_proj_bias']['shape']
                        down_bias_views.append(pool[db_off : db_off + db_shape[0]].view(db_shape))
                    else:
                        down_bias_views.append(None)

        # 批量计算
        with nvtx_range("Batched_Expert_Compute"):
                gate_up_w = torch.stack(gate_up_views, dim=0).contiguous()
                down_w = torch.stack(down_views, dim=0).contiguous()

                # Gate-Up: [batch, max_tok, hidden] @ [batch, hidden, inter*2] = [batch, max_tok, inter*2]
                gate_up_out = torch.bmm(batched_inputs, gate_up_w)

                # Add bias if present
                if gate_up_bias_views[0] is not None:
                    gate_up_bias_stacked = torch.stack([b for b in gate_up_bias_views if b is not None], dim=0)
                    gate_up_out = gate_up_out + gate_up_bias_stacked.unsqueeze(1)

                # GPT-OSS SwiGLU: gate[::2], up[1::2], gate * sigmoid(gate * alpha) * (up + 1)
                gate = gate_up_out[..., ::2]
                up = gate_up_out[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu

                # Down: [batch, max_tok, inter] @ [batch, inter, hidden] = [batch, max_tok, hidden]
                exp_out = torch.bmm(gated_output, down_w)

                # Add down bias if present
                if down_bias_views[0] is not None:
                    down_bias_stacked = torch.stack([b for b in down_bias_views if b is not None], dim=0)
                    exp_out = exp_out + down_bias_stacked.unsqueeze(1)

                exp_out = exp_out * batched_rweights

                # 写回
                for row_idx, count in enumerate(expert_counts_local):
                    if count == 0:
                        continue
                    offset = expert_offsets_local[row_idx]
                    expert_outputs[offset : offset + count] = exp_out[row_idx, :count]

        # ===== 初始化并Scatter输出 =====
        next_states = torch.zeros_like(hidden_states_flat)
        with nvtx_range("Batch_Result_Scatter"):
            next_states.scatter_add_(
                0,
                sorted_tokens.unsqueeze(1).expand(-1, hidden_dim),
                expert_outputs.to(src_dtype)
            )

        layer_time = time.time() - layer_start

        # Statistics collection
        if self.layer_idx == 0:
            GptOssSimpleMoE.layer_expert_counts = []
            GptOssSimpleMoE.layer_times = []

        GptOssSimpleMoE.layer_expert_counts.append(expert_count_cached)
        GptOssSimpleMoE.layer_times.append(layer_time * 1000)

        if self.layer_idx == self.num_hidden_layers - 1:

            if PREFETCH_ENABLED:
                GptOssSimpleMoE.router_inputs_for_prefetch = {}

        output_tensor = next_states.reshape(batch_size, sequence_length, hidden_dim)
        return output_tensor, router_scores

    def _parallel_prefetch(self, next_layer_selected_experts_cpu):
        """
        并行prefetch：使用预测的下一层专家信息

        Args:
            next_layer_selected_experts_cpu: 下一层预测的专家tensor（CPU）
        """
        # 等待上一个prefetch完成
        self.expert_cache.buffer_manager.get_prefetch_stream().synchronize()

        # 获取下一层需要prefetch的专家ID列表（去重）
        prefetch_expert_ids = next_layer_selected_experts_cpu.unique().tolist()

        if not prefetch_expert_ids:
            return

        # 获取下一层的专家权重（打包的tensor）
        next_layer_idx = self.layer_idx + 1
        prefetch_tensors_dict = {}
        missing_experts = []

        for expert_id in prefetch_expert_ids:
            packed_tensor = self.expert_cache.simple_expert_cache.get((next_layer_idx, expert_id))
            if packed_tensor is not None:
                prefetch_tensors_dict[expert_id] = packed_tensor
            else:
                missing_experts.append(expert_id)

        if PRINT_PREFETCH_DEBUG and missing_experts:
            print(f"Layer {self.layer_idx:2d} -> Missing {len(missing_experts)} experts: {missing_experts}")

        # 在prefetch_stream上执行prefetch操作
        with torch.cuda.stream(self.expert_cache.buffer_manager.get_prefetch_stream()):
            prefetched_count = 0
            for expert_id, packed_tensor in prefetch_tensors_dict.items():
                success = self.expert_cache.buffer_manager.prefetch_expert(next_layer_idx, expert_id, packed_tensor)
                if success:
                    prefetched_count += 1


class GptOssSparseMoeWrapper(GptOssSimpleMoE):
    """Alias for compatibility"""
    pass
