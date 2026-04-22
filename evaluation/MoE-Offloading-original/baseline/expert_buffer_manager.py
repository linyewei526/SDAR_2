"""
Unified Expert Buffer Manager - 统一的专家 Buffer 管理器 (Baseline 整合版)

支持 GPT-OSS 和 Qwen3 两种模型架构，通过 expert_config 配置区分差异。

架构特点:
- 物理层: 连续的 GPU 内存池
- 映射层: expert_id -> buffer_idx 双向映射
- 零拷贝重定位: 仅修改映射表
- 动态分配: 根据需要分配buffer
- GPU Cache 集成: 支持虚拟索引 (预留 cache slot 空间)
"""

import torch
from typing import Dict, Optional, List, Any, Tuple

from .nvtx_utils import nvtx_range


class ExpertWrapper:
    """
    Expert 计算包装器
    根据 expert_config 中的结构配置执行相应的前向传播
    """

    def __init__(self, gpu_buffer: Dict, gpu_memory_pool: torch.Tensor, expert_config: Dict):
        self.gpu_buffer = gpu_buffer
        self.gpu_memory_pool = gpu_memory_pool
        self.expert_config = expert_config

        # 解析配置
        self.structure = expert_config.get('structure', 'separate')  # 'merged' or 'separate'
        self.has_bias = expert_config.get('has_bias', False)
        self.activation = expert_config.get('activation', 'swiglu')
        self.swiglu_limit = expert_config.get('swiglu_limit', 7.0)
        self.alpha = expert_config.get('alpha', 1.702)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Expert 前向传播

        Args:
            hidden_states: [batch, hidden_size]

        Returns:
            output: [batch, hidden_size]
        """
        if self.structure == 'merged':
            # GPT-OSS: gate_up_proj (merged)
            return self._forward_merged(hidden_states)
        else:
            # Qwen3: separate gate, up, down
            return self._forward_separate(hidden_states)

    def _forward_merged(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """GPT-OSS 风格: merged gate_up_proj + bias"""
        # Gate-Up projection
        gate_up_offset = self.gpu_buffer['gate_up_proj']['offset']
        gate_up_shape = self.gpu_buffer['gate_up_proj']['shape']
        gate_up_weight = self.gpu_memory_pool[
            gate_up_offset:gate_up_offset + gate_up_shape[0] * gate_up_shape[1]
        ].view(gate_up_shape)

        gate_up_out = torch.nn.functional.linear(hidden_states, gate_up_weight)

        # Add bias
        if self.has_bias and 'gate_up_proj_bias' in self.gpu_buffer:
            bias_offset = self.gpu_buffer['gate_up_proj_bias']['offset']
            bias_size = self.gpu_buffer['gate_up_proj_bias']['size']
            bias_view = self.gpu_memory_pool[bias_offset:bias_offset + bias_size]
            gate_up_out = gate_up_out + bias_view

        # Split gate and up
        intermediate_size = gate_up_shape[0] // 2
        gate_out = gate_up_out[..., :intermediate_size]
        up_out = gate_up_out[..., intermediate_size:]

        # SwiGLU activation with clamp (GPT-OSS specific)
        gate_out = torch.clamp(gate_out, min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = torch.nn.functional.silu(gate_out) * up_out

        # Down projection
        down_offset = self.gpu_buffer['down_proj']['offset']
        down_shape = self.gpu_buffer['down_proj']['shape']
        down_weight = self.gpu_memory_pool[
            down_offset:down_offset + down_shape[0] * down_shape[1]
        ].view(down_shape)

        output = torch.nn.functional.linear(hidden, down_weight)

        # Add down bias
        if self.has_bias and 'down_proj_bias' in self.gpu_buffer:
            down_bias_offset = self.gpu_buffer['down_proj_bias']['offset']
            down_bias_size = self.gpu_buffer['down_proj_bias']['size']
            down_bias_view = self.gpu_memory_pool[down_bias_offset:down_bias_offset + down_bias_size]
            output = output + down_bias_view

        return output

    def _forward_separate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Qwen3 风格: separate gate, up, down (no bias)"""
        # Gate projection
        gate_offset = self.gpu_buffer['gate_proj']['offset']
        gate_shape = self.gpu_buffer['gate_proj']['shape']
        gate_weight = self.gpu_memory_pool[
            gate_offset:gate_offset + gate_shape[0] * gate_shape[1]
        ].view(gate_shape)
        gate_out = torch.nn.functional.silu(torch.nn.functional.linear(hidden_states, gate_weight))

        # Up projection
        up_offset = self.gpu_buffer['up_proj']['offset']
        up_shape = self.gpu_buffer['up_proj']['shape']
        up_weight = self.gpu_memory_pool[
            up_offset:up_offset + up_shape[0] * up_shape[1]
        ].view(up_shape)
        up_out = torch.nn.functional.linear(hidden_states, up_weight)

        # Down projection
        down_offset = self.gpu_buffer['down_proj']['offset']
        down_shape = self.gpu_buffer['down_proj']['shape']
        down_weight = self.gpu_memory_pool[
            down_offset:down_offset + down_shape[0] * down_shape[1]
        ].view(down_shape)

        # SwiGLU: silu(gate) * up -> down
        hidden = gate_out * up_out
        output = torch.nn.functional.linear(hidden, down_weight)

        return output


class ExpertBufferManager:
    """
    统一的 Expert Buffer Manager

    通过 expert_config 区分不同模型架构:
    - GPT-OSS: structure='merged', has_bias=True
    - Qwen3: structure='separate', has_bias=False

    支持两种初始化方式:
    1. ExpertBufferManager(state_path, device, buffer_size) - 自动检测模型类型
    2. ExpertBufferManager(device, buffer_size, expert_config) - 直接传入配置
    """

    def __init__(
        self,
        state_path_or_device,
        device_or_buffer_size,
        buffer_size_or_config=None,
        gpu_cache_manager: Optional[Any] = None
    ):
        """
        初始化 Buffer Manager

        方式1: ExpertBufferManager(state_path, device, buffer_size, gpu_cache_manager)
            state_path: 模型路径，自动从 config.json 加载配置
            device: GPU 设备
            buffer_size: 临时 buffer 数量

        方式2: ExpertBufferManager(device, buffer_size, expert_config, gpu_cache_manager)
            device: GPU 设备
            buffer_size: 临时 buffer 数量
            expert_config: 专家结构配置 dict
        """
        # 判断调用方式
        if isinstance(state_path_or_device, torch.device):
            # 方式2: 直接传入配置
            device = state_path_or_device
            buffer_size = device_or_buffer_size
            expert_config = buffer_size_or_config or {}
            self.state_path = None
        elif isinstance(state_path_or_device, str):
            # 方式1: 从 state_path 自动加载配置
            state_path = state_path_or_device
            device = device_or_buffer_size
            buffer_size = buffer_size_or_config
            self.state_path = state_path
            expert_config = self._load_expert_config(state_path)
        else:
            raise TypeError(
                f"First argument must be torch.device or str (state_path), "
                f"got {type(state_path_or_device).__name__}"
            )

        self.device = device
        self.buffer_size = buffer_size
        self.expert_config = expert_config
        self.gpu_cache_manager = gpu_cache_manager

        # 验证必需字段
        required_fields = ['hidden_size', 'intermediate_size']
        missing_fields = [f for f in required_fields if f not in expert_config]
        if missing_fields:
            raise ValueError(
                f"expert_config missing required fields: {missing_fields}. "
                f"Required: {required_fields}"
            )

        # 解析维度
        self.hidden_size = expert_config['hidden_size']
        self.intermediate_size = expert_config['intermediate_size']

        # 1. 分配物理 GPU buffer 池
        self._allocate_physical_buffers()

        # 2. Buffer 状态管理
        self.buffer_status: List[bool] = [False] * buffer_size

        # 3. 映射表
        self.current_layer_mapping: Dict[int, int] = {}  # expert_id -> buffer_idx
        self.prefetch_mapping: Dict[int, int] = {}
        self.prefetch_in_progress: Dict[int, int] = {}
        self.prefetch_terminated = False

        # 4. CUDA Stream
        self.prefetch_stream = torch.cuda.Stream()

        # 5. Wrapper 缓存
        self.wrapper_cache: Dict[int, ExpertWrapper] = {}

        # 6. 统计信息
        self.total_experts_loaded = 0
        self.prefetch_hits = 0
        self.compute_loads = 0

        # 7. GPU Cache 统计
        self.gpu_cache_hits = 0
        self.gpu_cache_misses = 0

    def _load_expert_config(self, state_path: str) -> Dict:
        """从 config.json 自动加载 expert 配置并识别模型类型"""
        import json
        import os

        config_path = os.path.join(state_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # 检测模型类型
        model_type = config.get('model_type', '')
        arch = config.get('architectures', [''])[0] if config.get('architectures') else ''

        # GPT-OSS 检测
        if 'gpt_oss' in model_type.lower() or 'GptOss' in arch:
            hidden_size = config['hidden_size']
            intermediate_size = config.get('intermediate_size', hidden_size)
            swiglu_limit = config.get('swiglu_limit', 7.0)

            return {
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'structure': 'merged',
                'has_bias': True,
                'activation': 'swiglu',
                'swiglu_limit': swiglu_limit,
                'alpha': 1.702,
            }

        # Qwen3 MoE 检测
        elif 'qwen3_moe' in model_type.lower() or 'Qwen3Moe' in arch:
            hidden_size = config['hidden_size']
            intermediate_size = config.get('moe_intermediate_size', hidden_size * 4)

            return {
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'structure': 'separate',
                'has_bias': False,
                'activation': 'swiglu',
            }

        # 默认处理：根据 key 推断
        else:
            hidden_size = config.get('hidden_size', 4096)

            if 'moe_intermediate_size' in config:
                return {
                    'hidden_size': hidden_size,
                    'intermediate_size': config['moe_intermediate_size'],
                    'structure': 'separate',
                    'has_bias': False,
                    'activation': 'swiglu',
                }
            else:
                return {
                    'hidden_size': hidden_size,
                    'intermediate_size': config.get('intermediate_size', hidden_size),
                    'structure': 'merged',
                    'has_bias': True,
                    'activation': 'swiglu',
                    'swiglu_limit': config.get('swiglu_limit', 7.0),
                    'alpha': 1.702,
                }

    def _allocate_physical_buffers(self):
        """分配物理 GPU buffer 池"""
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        structure = self.expert_config.get('structure', 'separate')
        has_bias = self.expert_config.get('has_bias', False)

        # 计算单个专家的参数数量
        if structure == 'merged':
            # GPT-OSS: gate_up_proj + bias + down_proj + bias
            gate_up_params = hidden_size * intermediate_size * 2
            gate_up_bias_params = intermediate_size * 2 if has_bias else 0
            down_params = intermediate_size * hidden_size
            down_bias_params = hidden_size if has_bias else 0
            total_expert_params = gate_up_params + gate_up_bias_params + down_params + down_bias_params
        else:
            # Qwen3: gate + up + down (no bias)
            gate_params = hidden_size * intermediate_size
            up_params = hidden_size * intermediate_size
            down_params = intermediate_size * hidden_size
            total_expert_params = gate_params + up_params + down_params

        self.params_per_expert = total_expert_params

        # 分配连续的 GPU 内存池
        total_gpu_memory = total_expert_params * self.buffer_size

        self.gpu_memory_pool = torch.empty(total_gpu_memory, dtype=torch.bfloat16, device=self.device)

        # 创建 buffer 信息结构
        self.gpu_buffers = []
        for buffer_idx in range(self.buffer_size):
            start_offset = buffer_idx * total_expert_params
            gpu_buffer = self._create_buffer_info(start_offset, structure, has_bias)
            self.gpu_buffers.append(gpu_buffer)

    def _create_buffer_info(self, start_offset: int, structure: str, has_bias: bool) -> Dict:
        """创建 buffer 元信息"""
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        current_offset = start_offset

        gpu_buffer = {'memory_pool': self.gpu_memory_pool}

        if structure == 'merged':
            # GPT-OSS 结构
            gate_up_params = hidden_size * intermediate_size * 2
            gate_up_bias_params = intermediate_size * 2 if has_bias else 0
            down_params = intermediate_size * hidden_size
            down_bias_params = hidden_size if has_bias else 0

            gpu_buffer['gate_up_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size * 2),
                'size': gate_up_params
            }
            current_offset += gate_up_params

            if has_bias:
                gpu_buffer['gate_up_proj_bias'] = {
                    'offset': current_offset,
                    'shape': (intermediate_size * 2,),
                    'size': gate_up_bias_params
                }
                current_offset += gate_up_bias_params

            gpu_buffer['down_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size),
                'size': down_params
            }
            current_offset += down_params

            if has_bias:
                gpu_buffer['down_proj_bias'] = {
                    'offset': current_offset,
                    'shape': (hidden_size,),
                    'size': down_bias_params
                }
        else:
            # Qwen3 结构
            gate_params = hidden_size * intermediate_size
            up_params = hidden_size * intermediate_size
            down_params = intermediate_size * hidden_size

            gpu_buffer['gate_proj'] = {
                'offset': current_offset,
                'shape': (intermediate_size, hidden_size),
                'size': gate_params
            }
            current_offset += gate_params

            gpu_buffer['up_proj'] = {
                'offset': current_offset,
                'shape': (intermediate_size, hidden_size),
                'size': up_params
            }
            current_offset += up_params

            gpu_buffer['down_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size),
                'size': down_params
            }

        return gpu_buffer

    # ========== 虚拟索引相关方法 ==========

    def _is_cache_slot(self, virtual_idx: int) -> bool:
        """判断虚拟索引是否指向 GPU Cache slot"""
        if self.gpu_cache_manager is None:
            return False
        return virtual_idx >= self.buffer_size

    def _cache_slot_to_virtual_idx(self, cache_slot_idx: int) -> int:
        """将 cache slot 索引映射为虚拟索引"""
        return self.buffer_size + cache_slot_idx

    def _virtual_idx_to_cache_slot(self, virtual_idx: int) -> int:
        """虚拟索引转回 cache slot 索引"""
        return virtual_idx - self.buffer_size

    def get_expert_view_for_computation(self, virtual_idx: int) -> Dict:
        """
        根据虚拟索引获取 expert view

        Args:
            virtual_idx: 虚拟索引 (可能是 temp buffer 或 cache slot)

        Returns:
            expert view dict
        """
        if self._is_cache_slot(virtual_idx):
            # 从 GPU Cache 获取
            cache_slot_idx = self._virtual_idx_to_cache_slot(virtual_idx)
            return self.gpu_cache_manager.get_expert_view(cache_slot_idx)
        else:
            # 从临时 buffer 获取
            return self.gpu_buffers[virtual_idx]

    # ========== 核心加载方法 ==========

    def load_experts_for_current_layer(
        self,
        layer_idx: int,
        expert_ids: List[int],
        expert_weights_dict: Dict[int, Any],
        router_logits: Optional[Dict[int, float]] = None
    ) -> Dict[int, int]:
        """
        为当前层加载 experts，优先使用 GPU Cache

        Args:
            layer_idx: 当前层索引
            expert_ids: 需要加载的专家 ID 列表
            expert_weights_dict: expert_id -> packed_tensor/dict 映射
            router_logits: (可选) expert_id -> max_logit_score 映射

        Returns:
            expert_id -> virtual_idx 映射表
        """
        # 释放旧层的临时 buffer (不释放 cache slot)
        for expert_id, buffer_idx in self.current_layer_mapping.items():
            if not self._is_cache_slot(buffer_idx):
                self.buffer_status[buffer_idx] = False
        self.current_layer_mapping.clear()

        loaded_mapping = {}

        with nvtx_range(f"Current_Layer_Availability_Check_Layer{layer_idx}"):
            # 处理 prefetch 状态迁移
            if self.prefetch_in_progress or self.prefetch_mapping:
                for expert_id, buffer_idx in list(self.prefetch_in_progress.items()):
                    self.prefetch_mapping[expert_id] = buffer_idx
                self.prefetch_in_progress.clear()

            # 1. 优先检查 GPU Cache
            experts_need_temp_load = []
            for expert_id in expert_ids:
                if self.gpu_cache_manager is not None:
                    cache_slot_idx = self.gpu_cache_manager.lookup(layer_idx, expert_id)
                    if cache_slot_idx is not None:
                        # GPU Cache 命中
                        virtual_idx = self._cache_slot_to_virtual_idx(cache_slot_idx)
                        loaded_mapping[expert_id] = virtual_idx
                        self.current_layer_mapping[expert_id] = virtual_idx
                        self.gpu_cache_hits += 1
                        continue
                experts_need_temp_load.append(expert_id)

            self.gpu_cache_misses += len(experts_need_temp_load)

            # 2. 检查 prefetch (仅对需要 temp load 的检查)
            remaining_experts = []
            prefetch_hits = 0

            for expert_id in experts_need_temp_load:
                if expert_id in self.prefetch_mapping:
                    buffer_idx = self.prefetch_mapping[expert_id]
                    self.current_layer_mapping[expert_id] = buffer_idx
                    loaded_mapping[expert_id] = buffer_idx
                    del self.prefetch_mapping[expert_id]
                    prefetch_hits += 1
                else:
                    remaining_experts.append(expert_id)

            # 释放未命中的 prefetch
            for expert_id in list(self.prefetch_mapping.keys()):
                buffer_idx = self.prefetch_mapping.pop(expert_id)
                if buffer_idx is not None:
                    self.buffer_status[buffer_idx] = False

            self.prefetch_hits += prefetch_hits

        # 3. 加载剩余需要的专家到临时 buffer
        if remaining_experts:
            with nvtx_range(f"Current_Layer_Miss_Load_Layer{layer_idx}"):
                for expert_id in remaining_experts:
                    buffer_idx = self.find_free_buffer()
                    if buffer_idx is None:
                        raise RuntimeError(f"No free buffers available! All {self.buffer_size} buffers are in use.")

                    expert_weights = expert_weights_dict[expert_id]
                    self._load_expert_to_buffer(buffer_idx, expert_weights)
                    self.allocate_buffer(buffer_idx)

                    self.current_layer_mapping[expert_id] = buffer_idx
                    loaded_mapping[expert_id] = buffer_idx

                    if buffer_idx not in self.wrapper_cache:
                        self.wrapper_cache[buffer_idx] = self._create_expert_wrapper(buffer_idx)

                    self.total_experts_loaded += 1
                    self.compute_loads += 1

        # 4. Cache update: 将 swap buffer 中的 experts 更新到 cache
        if (self.gpu_cache_manager is not None and
            router_logits is not None and
            remaining_experts):
            self._update_topk_lru_cache(layer_idx, remaining_experts, router_logits)

        return loaded_mapping

    def _update_topk_lru_cache(
        self,
        layer_idx: int,
        loaded_expert_ids: List[int],
        router_logits: Dict[int, float]
    ):
        """
        将 swap buffer 中刚加载的 experts 更新到 topk_lru cache
        """
        # 构建 swap_buffer_infos: (expert_id, buffer_info, logit_score)
        swap_buffer_infos = []
        for expert_id in loaded_expert_ids:
            if expert_id in self.current_layer_mapping:
                buffer_idx = self.current_layer_mapping[expert_id]
                # 只处理 swap buffer 中的（不是 cache slot）
                if not self._is_cache_slot(buffer_idx):
                    buffer_info = self.gpu_buffers[buffer_idx]
                    logit_score = router_logits.get(expert_id, 0.0)
                    swap_buffer_infos.append((expert_id, buffer_info, logit_score))

        if swap_buffer_infos:
            self.gpu_cache_manager.update_cache_from_buffers(
                layer_idx, swap_buffer_infos, self.gpu_memory_pool
            )

    def _load_expert_to_buffer(self, buffer_idx: int, expert_weights):
        """
        将 expert 权重加载到指定 buffer

        Args:
            expert_weights: 可以是打包的 torch.Tensor (GPT-OSS)
                           或分离的 dict {'gate': tensor, 'up': tensor, 'down': tensor} (Qwen3)
        """
        gpu_buffer = self.gpu_buffers[buffer_idx]

        # 判断权重格式
        if isinstance(expert_weights, torch.Tensor):
            # 打包格式 (GPT-OSS)
            if 'gate_proj' in gpu_buffer:
                start_offset = gpu_buffer['gate_proj']['offset']
            elif 'gate_up_proj' in gpu_buffer:
                start_offset = gpu_buffer['gate_up_proj']['offset']
            else:
                raise ValueError("Cannot find gate_proj or gate_up_proj in gpu_buffer")

            total_size = expert_weights.numel()
            dst = self.gpu_memory_pool[start_offset:start_offset + total_size]
            dst.copy_(expert_weights, non_blocking=True)
        else:
            # 分离格式 (Qwen3): dict with 'gate', 'up', 'down'
            # Gate projection
            gate_view = self.gpu_memory_pool[
                gpu_buffer['gate_proj']['offset']:
                gpu_buffer['gate_proj']['offset'] + gpu_buffer['gate_proj']['size']
            ].view(gpu_buffer['gate_proj']['shape'])
            gate_view.copy_(expert_weights['gate'], non_blocking=True)

            # Up projection
            up_view = self.gpu_memory_pool[
                gpu_buffer['up_proj']['offset']:
                gpu_buffer['up_proj']['offset'] + gpu_buffer['up_proj']['size']
            ].view(gpu_buffer['up_proj']['shape'])
            up_view.copy_(expert_weights['up'], non_blocking=True)

            # Down projection
            down_view = self.gpu_memory_pool[
                gpu_buffer['down_proj']['offset']:
                gpu_buffer['down_proj']['offset'] + gpu_buffer['down_proj']['size']
            ].view(gpu_buffer['down_proj']['shape'])
            down_view.copy_(expert_weights['down'], non_blocking=True)

    def _create_expert_wrapper(self, buffer_idx: int) -> ExpertWrapper:
        """创建 ExpertWrapper"""
        gpu_buffer = self.gpu_buffers[buffer_idx]
        return ExpertWrapper(gpu_buffer, self.gpu_memory_pool, self.expert_config)

    # ========== Buffer 管理 ==========

    def find_free_buffer(self) -> Optional[int]:
        """查找空闲 buffer"""
        try:
            return self.buffer_status.index(False)
        except ValueError:
            return None

    def allocate_buffer(self, buffer_idx: int):
        """分配 buffer"""
        if buffer_idx < len(self.buffer_status):
            self.buffer_status[buffer_idx] = True

    def free_buffer(self, buffer_idx: int):
        """释放 buffer"""
        if buffer_idx < len(self.buffer_status):
            self.buffer_status[buffer_idx] = False

    # ========== Prefetch ==========

    def prefetch_expert(self, layer_idx: int, expert_id: int, packed_tensor: torch.Tensor) -> bool:
        """
        预取 expert 到空闲 buffer

        如果 expert 已在 GPU cache 中，则跳过预取（不影响统计）
        """
        # 检查是否已在 GPU cache 中（使用 contains 不影响 hit/miss 统计）
        if self.gpu_cache_manager is not None:
            if self.gpu_cache_manager.contains(layer_idx, expert_id):
                return True  # 已在 cache 中，不需要 prefetch

        if expert_id in self.prefetch_mapping or expert_id in self.prefetch_in_progress:
            return True
        if self.prefetch_terminated:
            return False

        buffer_idx = self.find_free_buffer()
        if buffer_idx is None:
            return False

        with torch.cuda.stream(self.prefetch_stream):
            self._load_expert_to_buffer(buffer_idx, packed_tensor)
            self.prefetch_in_progress[expert_id] = buffer_idx
            self.allocate_buffer(buffer_idx)

            if buffer_idx not in self.wrapper_cache:
                self.wrapper_cache[buffer_idx] = self._create_expert_wrapper(buffer_idx)

        return True

    def get_prefetch_stream(self) -> torch.cuda.Stream:
        """获取 prefetch stream"""
        return self.prefetch_stream

    # ========== 统计信息 ==========

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_requests = self.gpu_cache_hits + self.gpu_cache_misses
        return {
            'gpu_cache_hits': self.gpu_cache_hits,
            'gpu_cache_misses': self.gpu_cache_misses,
            'cache_hit_rate': self.gpu_cache_hits / total_requests if total_requests > 0 else 0,
            'prefetch_hits': self.prefetch_hits,
            'compute_loads': self.compute_loads,
            'total_experts_loaded': self.total_experts_loaded,
        }

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'gpu_memory_pool'):
            del self.gpu_memory_pool
        if hasattr(self, 'wrapper_cache'):
            self.wrapper_cache.clear()
        if hasattr(self, 'prefetch_stream'):
            del self.prefetch_stream


# ========== 向后兼容的模型特定 Buffer Manager ==========

class GptOssExpertBufferManager(ExpertBufferManager):
    """
    GPT-OSS Expert Buffer Manager (向后兼容)
    继承统一 ExpertBufferManager
    """

    def __init__(self, state_path: str, device: torch.device, buffer_size: int = 32,
                 gpu_cache_manager: Optional[Any] = None):
        # 直接调用父类初始化 (方式1: 从 state_path 加载)
        super().__init__(state_path, device, buffer_size, gpu_cache_manager)


class Qwen3ExpertBufferManager(ExpertBufferManager):
    """
    Qwen3 Expert Buffer Manager (向后兼容)
    继承统一 ExpertBufferManager
    """

    def __init__(self, state_path: str, device: torch.device, buffer_size: int = 128,
                 gpu_cache_manager: Optional[Any] = None):
        # 直接调用父类初始化 (方式1: 从 state_path 加载)
        super().__init__(state_path, device, buffer_size, gpu_cache_manager)
