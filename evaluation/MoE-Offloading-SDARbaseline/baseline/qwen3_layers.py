import time
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.activations import ACT2FN

from .debug_config import (
    PRINT_EXPERT_DETAILS,
    PRINT_PREFETCH_DEBUG,
    MOE_DEBUG_ENABLED,
    PREFETCH_ENABLED,
    BMM_ENABLED,
    PRELAUNCH_ENABLED
)
from .nvtx_utils import nvtx_range

# ===== Prefetch MACRO =====
# Prefetch top-k experts based on predicted weights (0 = disable prefetch)
PREFETCH_TOPK = 4
# ===== END PREFETCH MACRO =====

# 🚀 Global Gate Registry for cross-layer access
class GateRegistry:
    """全局gate注册表，用于安全的跨层访问"""
    _instance = None
    _gates = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_gate(cls, layer_idx: int, gate: nn.Module):
        """注册gate"""
        cls._gates[layer_idx] = gate

    @classmethod
    def get_gate(cls, layer_idx: int):
        """获取gate"""
        return cls._gates.get(layer_idx, None)

    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._gates.clear()


class Qwen3ExpertMLP(nn.Module):
    """Simple Qwen3 Expert MLP - NO MEMORY WRAPPER"""
    def __init__(self, config: Qwen3MoeConfig, quant_config=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # SwiGLU activation
        current_hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        current_hidden_states = self.down_proj(current_hidden_states)
        return current_hidden_states


class Qwen3SimpleMoE(nn.Module):
    """Ultra-simple MoE layer - NO COMPLEX INHERITANCE"""

    # Class-level statistics collection
    layer_expert_counts = []
    layer_times = []
    forward_session_id = 0
    decode_token_counter = 0  # Track decode tokens

    # MoE activation data collection (only used when MOE_DEBUG_ENABLED=True)
    moe_activation_log = []  # List of dicts containing layer data

    # Router prefetch data collection
    router_inputs_for_prefetch = {}  # {layer_idx: router_input}
    next_layer_predictions = {}  # {layer_idx: predictions}

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, gate_layer, expert_cache, layers):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expert_cache = expert_cache

        # Use the existing gate
        self.gate = gate_layer

        # 🚀 注册gate到全局注册表，用于安全的跨层访问
        GateRegistry.register_gate(layer_idx, gate_layer)

        # Expert parameters
        self.hidden_size = config.hidden_size
        # 使用目标expert数量而不是当前的config.num_experts=1
        self.num_experts = getattr(config, '_target_experts', config.num_experts)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Store router_logits for model output
        self.last_router_logits = None

    def forward(self, hidden_states):
        layer_start = time.time()

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # ===== Router Input Collection for Prefetch =====
        if PREFETCH_ENABLED:
            router_input_flat = hidden_states_flat.clone().detach()
        # ===== END PREFETCH =====

        # Route to experts (following official Qwen3MoeSparseMoeBlock logic)
        router_logits = self.gate(hidden_states_flat)

        # Check if this is decode phase (only one token)
        if router_logits.shape[0] == 1:

            # Increment decode counter when at last layer
            if self.layer_idx == 47:  # Assuming 48 layers (0-47)
                Qwen3SimpleMoE.decode_token_counter += 1

        # ===== Router Calculation =====
        with nvtx_range(f"Routing_Layer{self.layer_idx}"):
            # Current layer router calculation
            full_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(full_routing_weights, self.top_k, dim=-1)

            # 🚀 Next layer router prediction for prefetch (same input, next layer gate)
            next_layer_selected_experts = None
            next_layer_selected_experts_cpu = None  # 用于prefetch的CPU数据
            if PREFETCH_ENABLED and self.layer_idx < 47:
                next_layer_gate = GateRegistry.get_gate(self.layer_idx + 1)
                if next_layer_gate is not None:
                    with torch.no_grad():
                        # Use next layer gate with current hidden states to predict next layer experts
                        next_layer_logits = next_layer_gate(hidden_states_flat)
                        next_layer_weights = F.softmax(next_layer_logits, dim=1, dtype=torch.float)

                        # 🚀 根据PREFETCH_TOPK选择topk专家（支持topk=0自然工作）
                        # 使用argsort而不是topk，因为topk不支持k=0
                        sorted_indices = torch.argsort(next_layer_weights, dim=-1, descending=True)
                        next_layer_selected_experts = sorted_indices[:, :PREFETCH_TOPK]

                        # 打印预测结果（token 0的top-K expert IDs + weights）
                        if PRINT_PREFETCH_DEBUG:
                            weights_cpu = next_layer_weights[0].cpu().tolist()
                            top_experts = sorted_indices[0, :PREFETCH_TOPK].cpu().tolist()
                            top_weights = [weights_cpu[e] for e in top_experts]
                            print(f"Layer {self.layer_idx:2d} -> Next layer {self.layer_idx+1} prediction (token 0): {[(e, f'{w:.4f}') for e, w in zip(top_experts, top_weights)]}")

                        # 🚀 在default stream上做DtoH，获取expert list用于prefetch
                        next_layer_selected_experts_cpu = next_layer_selected_experts.flatten().cpu()
                else:
                    raise RuntimeError(f"Layer {self.layer_idx:2d}: Next layer gate not found for prefetch prediction!")

            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # ===== Print Expert Selection (for each token) =====
        if PRINT_EXPERT_DETAILS and router_logits.shape[0] == 1:  # Only print for single token (decode phase)
            expert_ids = selected_experts[0].cpu().tolist()
            print(f"🔍 Layer {self.layer_idx:2d} | Expert selection: {expert_ids}")

        # ===== MoE DEBUG: Collect expert distribution =====
        if MOE_DEBUG_ENABLED:
            # Create expert token distribution: [num_experts] array showing how many tokens each expert gets
            expert_token_distribution = torch.zeros(self.num_experts, dtype=torch.int32)
            for token_idx in range(selected_experts.shape[0]):
                for expert_idx in selected_experts[token_idx]:
                    expert_token_distribution[expert_idx] += 1

            # Store layer data
            layer_data = {
                'layer_idx': self.layer_idx,
                'router_input_shape': router_input_flat.shape,
                'router_input': router_input_flat.float().cpu().numpy().tolist(),
                'router_logits': router_logits.float().cpu().numpy().tolist(),
                'selected_experts': selected_experts.cpu().numpy().tolist(),
                'expert_token_distribution': expert_token_distribution.cpu().numpy().tolist()
            }
            Qwen3SimpleMoE.moe_activation_log.append(layer_data)

        
            # Print summary for this layer
            active_experts = (expert_token_distribution > 0).sum().item()
            print(f"Layer {self.layer_idx:2d}: ✅ Actual active_experts={active_experts}, expert_token_dist={expert_token_distribution.cpu().numpy().tolist()}")
        # ===== END DEBUG =====

        # ===== Router Prediction for Prefetch =====
        if PREFETCH_ENABLED:
            # Store current router input for next layer prediction
            Qwen3SimpleMoE.router_inputs_for_prefetch[self.layer_idx] = router_input_flat.clone().detach()
        # ===== END PREFETCH =====

        # Ensure hidden_states_flat has the right dtype
        # The input might be float32, but our weights are bfloat16
        if hidden_states_flat.dtype != self.gate.weight.dtype:
            hidden_states_flat = hidden_states_flat.to(self.gate.weight.dtype)

        # Final output
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_flat.dtype,
            device=hidden_states_flat.device
        )

        # 🚀 优化：先做CPU同步，获取active_expert_ids
        # 直接 DtoH 然后 CPU bincount，避免 GPU kernel overhead
        flat_experts = selected_experts.reshape(-1)  # [N*top_k]
        # 直接传到 CPU，避免 GPU bincount 的 kernel launch overhead
        flat_experts_cpu = flat_experts.cpu()
        expert_counts = torch.bincount(flat_experts_cpu, minlength=self.num_experts)
        expert_offsets_cpu = torch.cumsum(expert_counts, dim=0) - expert_counts

        # 获取active_expert_ids用于加载experts
        active_expert_ids = torch.where(expert_counts > 0)[0].tolist()
        expert_counts_list = expert_counts.tolist()
        expert_offsets_list = expert_offsets_cpu.tolist()

        # 🚀 一次性加载所有experts并获取映射表
        router_logits = None
        if self.expert_cache.enable_gpu_cache and self.expert_cache.gpu_cache_manager is not None:
            policy_name = self.expert_cache.gpu_cache_manager.cache_policy_name
            if policy_name in ("lru", "lfu", "topk_lru", "tinylfu"):
                max_logits_per_expert, _ = full_routing_weights.max(dim=0)
                max_logits_cpu = max_logits_per_expert.cpu()
                router_logits = {eid: max_logits_cpu[eid].item() for eid in active_expert_ids}

        expert_indices = [(eid, eid) for eid in active_expert_ids]
        expert_to_buffer_mapping = self.expert_cache.batch_load_experts_continuous(
            self.layer_idx, expert_indices, router_logits
        )

        # 🚀 PreLaunch Control: 如果不启用PreLaunch，则等待load完成
        if not PRELAUNCH_ENABLED:
            torch.cuda.synchronize()

        # 🚀 提前缓存shape信息和selected_experts，避免expert compute结束后的多次cuda同步
        num_tokens_cached = selected_experts.shape[0]
        expert_count_cached = len(active_expert_ids)

        # 🚀 启动Prefetch（DtoH已在default stream上完成）
        if PREFETCH_ENABLED and next_layer_selected_experts_cpu is not None and self.layer_idx < 47:
            with nvtx_range(f"Next_Layer_Prefetch_Layer{self.layer_idx}"):
                self._parallel_prefetch(next_layer_selected_experts_cpu)

        # 🚀 延迟GPU排序到prefetch之后，与prefetch HtoD overlap
        # 这样default stream在等待prefetch HtoD期间可以做GPU计算，避免bubble
        with nvtx_range(f"Reorder_Layer{self.layer_idx}"):
            num_tokens = selected_experts.shape[0]

            # 对应的token索引：[0,0, 1,1, 2,2,...]
            token_indices = torch.arange(num_tokens, device=selected_experts.device).repeat_interleave(self.top_k)
            # 对应的top_k排名：[0,1, 0,1, 0,1,...]
            k_ranks = torch.tile(torch.arange(self.top_k, device=selected_experts.device), (num_tokens,))

            # 一次性全局排序（纯GPU异步操作，无CPU同步）
            sorted_experts, perm = torch.sort(flat_experts)
            sorted_tokens = token_indices[perm]
            sorted_ranks = k_ranks[perm]

        # 阶段2: 统一批量专家计算 - Gather -> Compute -> Scatter

        with nvtx_range(f"Gather_Layer{self.layer_idx}"):
            # 优化1: 统一Gather输入（一次性索引操作）
            all_input_states = hidden_states_flat[sorted_tokens]  # [total_tokens, hidden_dim]
            all_routing_weights = routing_weights[sorted_tokens, sorted_ranks].unsqueeze(1)  # [total_tokens, 1]
            expert_outputs = torch.empty_like(all_input_states)

            num_active = len(active_expert_ids)
            batched_inputs = None
            batched_rweights = None
            gate_views, up_views, down_views = [], [], []
            expert_offsets_local, expert_counts_local = [], []
            per_expert_compute_args = []

            if num_active > 0:
                if BMM_ENABLED:
                    max_tok = max(expert_counts_list[eid] for eid in active_expert_ids)
                    batched_inputs = all_input_states.new_zeros(num_active, max_tok, hidden_dim)
                    batched_rweights = all_routing_weights.new_zeros(num_active, max_tok, 1)

                    for row_idx, expert_id in enumerate(active_expert_ids):
                        count = expert_counts_list[expert_id]
                        offset = expert_offsets_list[expert_id]
                        expert_offsets_local.append(offset)
                        expert_counts_local.append(count)

                        if count > 0:
                            batched_inputs[row_idx, :count] = all_input_states[offset : offset + count]
                            batched_rweights[row_idx, :count] = all_routing_weights[offset : offset + count]

                        gpu_buffer = self.expert_cache.buffer_manager.get_expert_view_for_computation(
                            expert_to_buffer_mapping[expert_id]
                        )
                        pool = gpu_buffer['memory_pool']

                        g_off = gpu_buffer['gate_proj']['offset']
                        g_shape = gpu_buffer['gate_proj']['shape']
                        gate_views.append(pool[g_off : g_off + g_shape[0] * g_shape[1]].view(g_shape))

                        u_off = gpu_buffer['up_proj']['offset']
                        u_shape = gpu_buffer['up_proj']['shape']
                        up_views.append(pool[u_off : u_off + u_shape[0] * u_shape[1]].view(u_shape))

                        d_off = gpu_buffer['down_proj']['offset']
                        d_shape = gpu_buffer['down_proj']['shape']
                        down_views.append(pool[d_off : d_off + d_shape[0] * d_shape[1]].view(d_shape))
                else:
                    for expert_id in active_expert_ids:
                        count = expert_counts_list[expert_id]
                        offset = expert_offsets_list[expert_id]

                        if count == 0:
                            continue

                        expert_input_states = all_input_states[offset : offset + count]
                        expert_routing_weights = all_routing_weights[offset : offset + count]

                        gpu_buffer = self.expert_cache.buffer_manager.get_expert_view_for_computation(
                            expert_to_buffer_mapping[expert_id]
                        )
                        pool = gpu_buffer['memory_pool']

                        g_off = gpu_buffer['gate_proj']['offset']
                        g_shape = gpu_buffer['gate_proj']['shape']
                        gate_w = pool[g_off : g_off + g_shape[0] * g_shape[1]].view(g_shape)

                        u_off = gpu_buffer['up_proj']['offset']
                        u_shape = gpu_buffer['up_proj']['shape']
                        up_w = pool[u_off : u_off + u_shape[0] * u_shape[1]].view(u_shape)

                        d_off = gpu_buffer['down_proj']['offset']
                        d_shape = gpu_buffer['down_proj']['shape']
                        down_w = pool[d_off : d_off + d_shape[0] * d_shape[1]].view(d_shape)

                        per_expert_compute_args.append(
                            (
                                offset,
                                count,
                                expert_input_states,
                                expert_routing_weights,
                                gate_w,
                                up_w,
                                down_w,
                            )
                        )

        with nvtx_range(f"Expert_Compute_Layer{self.layer_idx}"):
            if num_active > 0:
                if BMM_ENABLED:
                    gate_w = torch.stack(gate_views, dim=0).contiguous()
                    up_w = torch.stack(up_views, dim=0).contiguous()
                    down_w = torch.stack(down_views, dim=0).contiguous()

                    gate_out = torch.nn.functional.silu(torch.bmm(batched_inputs, gate_w.transpose(1, 2)))
                    up_out = torch.bmm(batched_inputs, up_w.transpose(1, 2))
                    exp_out = torch.bmm(gate_out * up_out, down_w.transpose(1, 2))
                    exp_out = exp_out * batched_rweights

                    for row_idx, count in enumerate(expert_counts_local):
                        if count == 0:
                            continue
                        offset = expert_offsets_local[row_idx]
                        expert_outputs[offset : offset + count] = exp_out[row_idx, :count]
                else:
                    for (
                        offset,
                        count,
                        expert_input_states,
                        expert_routing_weights,
                        gate_w,
                        up_w,
                        down_w,
                    ) in per_expert_compute_args:
                        expert_output = F.linear(
                            torch.nn.functional.silu(F.linear(expert_input_states, gate_w)) *
                            F.linear(expert_input_states, up_w),
                            down_w,
                        )
                        expert_outputs[offset : offset + count] = expert_output * expert_routing_weights

        # 优化3: 统一Scatter输出（一次性scatter_add）
        with nvtx_range(f"Scatter_Layer{self.layer_idx}"):
            final_hidden_states.scatter_add_(
                0,
                sorted_tokens.unsqueeze(1).expand(-1, hidden_dim),
                expert_outputs
            )

        # 🚀 prefetch已经在CUDA stream中并行运行，无需等待GIL

        layer_time = time.time() - layer_start

        # Reset statistics at the beginning of each forward sequence
        if self.layer_idx == 0:
            Qwen3SimpleMoE.layer_expert_counts = []
            Qwen3SimpleMoE.layer_times = []

        # Collect statistics（使用缓存的值，避免访问GPU tensor）
        Qwen3SimpleMoE.layer_expert_counts.append(expert_count_cached)
        Qwen3SimpleMoE.layer_times.append(layer_time * 1000)  # Convert to ms

        if self.layer_idx == 47:  # Assuming 48 layers (0-47)

            # ===== MoE DEBUG: Clear data for next pass =====
            if MOE_DEBUG_ENABLED:
                Qwen3SimpleMoE.moe_activation_log = []
            # ===== END DEBUG =====

            # ===== Prefetch: Clear data for next pass =====
            if PREFETCH_ENABLED:
                Qwen3SimpleMoE.router_inputs_for_prefetch = {}
            # ===== END PREFETCH =====

        # No need to clear GPU cache - each layer overwrites the shared buffer pool

        output_tensor = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # Save router_logits for model output
        self.last_router_logits = router_logits

        return output_tensor, router_logits

    def _parallel_prefetch(self, next_layer_selected_experts_cpu):
        """
        并行prefetch：使用预测的下一层专家信息（CPU数据）

        Args:
            next_layer_selected_experts_cpu: 下一层预测的专家tensor（CPU，已在default stream上DtoH完成）
        """
        # 🚀 先等待上一个 prefetch 完成（CPU阻塞）
        # 确保不会有多个 prefetch 同时运行
        self.expert_cache.buffer_manager.get_prefetch_stream().synchronize()

        # 获取下一层需要prefetch的专家ID列表（去重）- 此时数据已在CPU上，无同步
        prefetch_expert_ids = next_layer_selected_experts_cpu.unique().tolist()

        if not prefetch_expert_ids:
            return

        # 获取下一层的专家权重
        next_layer_idx = self.layer_idx + 1
        prefetch_weights_dict = {}
        missing_experts = []

        for expert_id in prefetch_expert_ids:
            gate_tensor = self.expert_cache.simple_expert_cache.get((next_layer_idx, expert_id, 'gate'))
            up_tensor = self.expert_cache.simple_expert_cache.get((next_layer_idx, expert_id, 'up'))
            down_tensor = self.expert_cache.simple_expert_cache.get((next_layer_idx, expert_id, 'down'))

            if gate_tensor is not None and up_tensor is not None and down_tensor is not None:
                prefetch_weights_dict[expert_id] = {
                    'gate': gate_tensor,
                    'up': up_tensor,
                    'down': down_tensor
                }
            else:
                missing_experts.append(expert_id)

        # 打印missing信息（如果有）
        if PRINT_PREFETCH_DEBUG and missing_experts:
            print(f"Layer {self.layer_idx:2d} -> Missing {len(missing_experts)} experts (not in cache): {missing_experts}")

        # 在prefetch_stream上执行prefetch操作
        with torch.cuda.stream(self.expert_cache.buffer_manager.get_prefetch_stream()):
            # 🚀 连续的prefetch操作，与当前层计算并行
            prefetched_count = 0
            for expert_id, weights in prefetch_weights_dict.items():
                success = self.expert_cache.buffer_manager.prefetch_expert(next_layer_idx, expert_id, weights)
                if success:
                    prefetched_count += 1

class Qwen3ExpertWrapper:
    """Simple wrapper - just pass through"""
    def __init__(self, expert_module, device):
        self.expert_module = expert_module.to(device)

    def to(self, device):
        return self.expert_module.to(device)

    def __call__(self, x):
        return self.expert_module(x)

class Qwen3SparseMoeWrapper(Qwen3SimpleMoE):
    """Alias for compatibility"""
    pass
