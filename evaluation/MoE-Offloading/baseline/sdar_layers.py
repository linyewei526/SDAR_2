import time
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .debug_config import (
    BMM_ENABLED,
    MOE_DEBUG_ENABLED,
    PREFETCH_ENABLED,
    PRELAUNCH_ENABLED,
    PRINT_EXPERT_DETAILS,
    PRINT_PREFETCH_DEBUG,
)
from .nvtx_utils import nvtx_range
from .sdar_runtime_trace import is_expert_recording_enabled, record_layer_activity


PREFETCH_TOPK = 4


class GateRegistry:
    """Global gate registry for next-layer prefetch prediction."""

    _instance = None
    _gates = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_gate(cls, layer_idx: int, gate: nn.Module):
        cls._gates[layer_idx] = gate

    @classmethod
    def get_gate(cls, layer_idx: int):
        return cls._gates.get(layer_idx, None)

    @classmethod
    def clear(cls):
        cls._gates.clear()


class SDARSimpleMoE(nn.Module):
    """Offloading MoE wrapper for SDAR block-diffusion decoding."""

    layer_expert_counts = []
    layer_times = []
    forward_session_id = 0
    decode_token_counter = 0
    moe_activation_log = []
    router_inputs_for_prefetch = {}
    next_layer_predictions = {}

    def __init__(self, config, layer_idx: int, gate_layer, expert_cache):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.expert_cache = expert_cache
        self.gate = gate_layer

        GateRegistry.register_gate(layer_idx, gate_layer)

        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "_target_experts", config.num_experts)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.last_router_logits = None

    def forward(self, hidden_states):
        layer_start = time.time()

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        if PREFETCH_ENABLED:
            router_input_flat = hidden_states_flat.clone().detach()

        router_logits_tensor = self.gate(hidden_states_flat)

        with nvtx_range(f"Routing_Layer{self.layer_idx}"):
            full_routing_weights = F.softmax(
                router_logits_tensor, dim=1, dtype=torch.float
            )
            routing_weights, selected_experts = torch.topk(
                full_routing_weights, self.top_k, dim=-1
            )

            next_layer_selected_experts_cpu = None
            if PREFETCH_ENABLED and self.layer_idx < self.num_hidden_layers - 1:
                next_layer_gate = GateRegistry.get_gate(self.layer_idx + 1)
                if next_layer_gate is None:
                    raise RuntimeError(
                        f"Layer {self.layer_idx}: next layer gate not found for prefetch"
                    )

                with torch.no_grad():
                    next_layer_logits = next_layer_gate(hidden_states_flat)
                    next_layer_weights = F.softmax(
                        next_layer_logits, dim=1, dtype=torch.float
                    )
                    sorted_indices = torch.argsort(
                        next_layer_weights, dim=-1, descending=True
                    )
                    next_layer_selected_experts = sorted_indices[:, :PREFETCH_TOPK]

                    if PRINT_PREFETCH_DEBUG and next_layer_selected_experts.numel() > 0:
                        weights_cpu = next_layer_weights[0].cpu().tolist()
                        top_experts = next_layer_selected_experts[0].cpu().tolist()
                        top_weights = [weights_cpu[e] for e in top_experts]
                        print(
                            f"Layer {self.layer_idx:2d} -> Next layer {self.layer_idx + 1} "
                            f"prediction (token 0): {[(e, f'{w:.4f}') for e, w in zip(top_experts, top_weights)]}"
                        )

                    next_layer_selected_experts_cpu = (
                        next_layer_selected_experts.flatten().cpu()
                    )

            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            routing_weights = routing_weights.to(hidden_states_flat.dtype)

        if PRINT_EXPERT_DETAILS and router_logits_tensor.shape[0] == 1:
            expert_ids = selected_experts[0].cpu().tolist()
            print(f"Layer {self.layer_idx:2d} | Expert selection: {expert_ids}")

        if MOE_DEBUG_ENABLED:
            expert_token_distribution = torch.zeros(self.num_experts, dtype=torch.int32)
            for token_idx in range(selected_experts.shape[0]):
                for expert_idx in selected_experts[token_idx]:
                    expert_token_distribution[expert_idx] += 1

            layer_data = {
                "layer_idx": self.layer_idx,
                "router_input_shape": tuple(router_input_flat.shape),
                "router_logits": router_logits_tensor.float().cpu().numpy().tolist(),
                "selected_experts": selected_experts.cpu().numpy().tolist(),
                "expert_token_distribution": expert_token_distribution.cpu()
                .numpy()
                .tolist(),
            }
            SDARSimpleMoE.moe_activation_log.append(layer_data)

        if PREFETCH_ENABLED:
            SDARSimpleMoE.router_inputs_for_prefetch[self.layer_idx] = (
                router_input_flat.clone().detach()
            )

        if hidden_states_flat.dtype != self.gate.weight.dtype:
            hidden_states_flat = hidden_states_flat.to(self.gate.weight.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states_flat.dtype,
            device=hidden_states_flat.device,
        )

        flat_experts = selected_experts.reshape(-1)
        flat_experts_cpu = flat_experts.cpu()
        expert_counts = torch.bincount(flat_experts_cpu, minlength=self.num_experts)
        expert_offsets_cpu = torch.cumsum(expert_counts, dim=0) - expert_counts

        active_expert_ids = torch.where(expert_counts > 0)[0].tolist()
        expert_counts_list = expert_counts.tolist()
        expert_offsets_list = expert_offsets_cpu.tolist()

        router_logits = None
        if (
            self.expert_cache.enable_gpu_cache
            and self.expert_cache.gpu_cache_manager is not None
        ):
            policy_name = self.expert_cache.gpu_cache_manager.cache_policy_name
            if policy_name in ("lru", "lfu", "topk_lru", "tinylfu"):
                max_logits_per_expert, _ = full_routing_weights.max(dim=0)
                max_logits_cpu = max_logits_per_expert.cpu()
                router_logits = {
                    eid: max_logits_cpu[eid].item() for eid in active_expert_ids
                }

        expert_indices = [(eid, eid) for eid in active_expert_ids]
        expert_to_buffer_mapping = self.expert_cache.batch_load_experts_continuous(
            self.layer_idx, expert_indices, router_logits
        )
        if is_expert_recording_enabled():
            record_layer_activity(
                self.layer_idx,
                active_expert_count=len(active_expert_ids),
                load_trace=getattr(
                    self.expert_cache.buffer_manager, "last_load_trace", {}
                ),
            )

        if not PRELAUNCH_ENABLED:
            torch.cuda.synchronize()

        expert_count_cached = len(active_expert_ids)

        if (
            PREFETCH_ENABLED
            and next_layer_selected_experts_cpu is not None
            and self.layer_idx < self.num_hidden_layers - 1
        ):
            with nvtx_range(
                f"Next_Layer_Prefetch_Layer{self.layer_idx}",
                stream=self.expert_cache.buffer_manager.get_prefetch_stream(),
            ):
                self._parallel_prefetch(next_layer_selected_experts_cpu)

        with nvtx_range(f"Reorder_Layer{self.layer_idx}"):
            num_tokens = selected_experts.shape[0]
            token_indices = torch.arange(
                num_tokens, device=selected_experts.device
            ).repeat_interleave(self.top_k)
            k_ranks = torch.tile(
                torch.arange(self.top_k, device=selected_experts.device),
                (num_tokens,),
            )
            sorted_experts, perm = torch.sort(flat_experts)
            sorted_tokens = token_indices[perm]
            sorted_ranks = k_ranks[perm]

        with nvtx_range(f"Gather_Layer{self.layer_idx}"):
            all_input_states = hidden_states_flat[sorted_tokens]
            all_routing_weights = routing_weights[sorted_tokens, sorted_ranks].unsqueeze(1)
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
                    batched_inputs = all_input_states.new_zeros(
                        num_active, max_tok, hidden_dim
                    )
                    batched_rweights = all_routing_weights.new_zeros(
                        num_active, max_tok, 1
                    )

                    for row_idx, expert_id in enumerate(active_expert_ids):
                        count = expert_counts_list[expert_id]
                        offset = expert_offsets_list[expert_id]
                        expert_offsets_local.append(offset)
                        expert_counts_local.append(count)

                        if count > 0:
                            batched_inputs[row_idx, :count] = all_input_states[
                                offset : offset + count
                            ]
                            batched_rweights[row_idx, :count] = all_routing_weights[
                                offset : offset + count
                            ]

                        gpu_buffer = (
                            self.expert_cache.buffer_manager.get_expert_view_for_computation(
                                expert_to_buffer_mapping[expert_id]
                            )
                        )
                        pool = gpu_buffer["memory_pool"]

                        g_off = gpu_buffer["gate_proj"]["offset"]
                        g_shape = gpu_buffer["gate_proj"]["shape"]
                        gate_views.append(
                            pool[g_off : g_off + g_shape[0] * g_shape[1]].view(g_shape)
                        )

                        u_off = gpu_buffer["up_proj"]["offset"]
                        u_shape = gpu_buffer["up_proj"]["shape"]
                        up_views.append(
                            pool[u_off : u_off + u_shape[0] * u_shape[1]].view(u_shape)
                        )

                        d_off = gpu_buffer["down_proj"]["offset"]
                        d_shape = gpu_buffer["down_proj"]["shape"]
                        down_views.append(
                            pool[d_off : d_off + d_shape[0] * d_shape[1]].view(d_shape)
                        )
                else:
                    for expert_id in active_expert_ids:
                        count = expert_counts_list[expert_id]
                        offset = expert_offsets_list[expert_id]
                        if count == 0:
                            continue

                        expert_input_states = all_input_states[offset : offset + count]
                        expert_routing_weights = all_routing_weights[offset : offset + count]

                        gpu_buffer = (
                            self.expert_cache.buffer_manager.get_expert_view_for_computation(
                                expert_to_buffer_mapping[expert_id]
                            )
                        )
                        pool = gpu_buffer["memory_pool"]

                        g_off = gpu_buffer["gate_proj"]["offset"]
                        g_shape = gpu_buffer["gate_proj"]["shape"]
                        gate_w = pool[g_off : g_off + g_shape[0] * g_shape[1]].view(g_shape)

                        u_off = gpu_buffer["up_proj"]["offset"]
                        u_shape = gpu_buffer["up_proj"]["shape"]
                        up_w = pool[u_off : u_off + u_shape[0] * u_shape[1]].view(u_shape)

                        d_off = gpu_buffer["down_proj"]["offset"]
                        d_shape = gpu_buffer["down_proj"]["shape"]
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

                    gate_out = torch.nn.functional.silu(
                        torch.bmm(batched_inputs, gate_w.transpose(1, 2))
                    )
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
                            torch.nn.functional.silu(F.linear(expert_input_states, gate_w))
                            * F.linear(expert_input_states, up_w),
                            down_w,
                        )
                        expert_outputs[offset : offset + count] = (
                            expert_output * expert_routing_weights
                        )

        with nvtx_range(f"Scatter_Layer{self.layer_idx}"):
            final_hidden_states.scatter_add_(
                0,
                sorted_tokens.unsqueeze(1).expand(-1, hidden_dim),
                expert_outputs,
            )

        layer_time = time.time() - layer_start

        if self.layer_idx == 0:
            SDARSimpleMoE.layer_expert_counts = []
            SDARSimpleMoE.layer_times = []

        SDARSimpleMoE.layer_expert_counts.append(expert_count_cached)
        SDARSimpleMoE.layer_times.append(layer_time * 1000)

        if self.layer_idx == self.num_hidden_layers - 1:
            if MOE_DEBUG_ENABLED:
                SDARSimpleMoE.moe_activation_log = []
            if PREFETCH_ENABLED:
                SDARSimpleMoE.router_inputs_for_prefetch = {}

        output_tensor = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        self.last_router_logits = router_logits
        return output_tensor, router_logits

    def _parallel_prefetch(self, next_layer_selected_experts_cpu):
        self.expert_cache.buffer_manager.get_prefetch_stream().synchronize()

        prefetch_expert_ids = next_layer_selected_experts_cpu.unique().tolist()
        if not prefetch_expert_ids:
            return

        next_layer_idx = self.layer_idx + 1
        prefetch_weights_dict = {}
        missing_experts = []

        for expert_id in prefetch_expert_ids:
            gate_tensor = self.expert_cache.simple_expert_cache.get(
                (next_layer_idx, expert_id, "gate")
            )
            up_tensor = self.expert_cache.simple_expert_cache.get(
                (next_layer_idx, expert_id, "up")
            )
            down_tensor = self.expert_cache.simple_expert_cache.get(
                (next_layer_idx, expert_id, "down")
            )

            if gate_tensor is not None and up_tensor is not None and down_tensor is not None:
                prefetch_weights_dict[expert_id] = {
                    "gate": gate_tensor,
                    "up": up_tensor,
                    "down": down_tensor,
                }
            else:
                missing_experts.append(expert_id)

        if PRINT_PREFETCH_DEBUG and missing_experts:
            print(
                f"Layer {self.layer_idx:2d} -> Missing {len(missing_experts)} experts "
                f"(not in cache): {missing_experts}"
            )

        with torch.cuda.stream(self.expert_cache.buffer_manager.get_prefetch_stream()):
            for expert_id, weights in prefetch_weights_dict.items():
                self.expert_cache.buffer_manager.prefetch_expert(
                    next_layer_idx, expert_id, weights
                )


class SDARSparseMoeWrapper(SDARSimpleMoE):
    """Compatibility alias."""

    pass
