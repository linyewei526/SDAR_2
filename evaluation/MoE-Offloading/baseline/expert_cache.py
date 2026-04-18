"""
Unified Expert Cache - 统一的专家缓存管理器 (Baseline 整合版)

支持 GPT-OSS 和 Qwen3 两种模型架构，通过 expert_config 配置区分差异。

架构特点:
- CPU 端专家权重缓存
- 自动模型类型检测
- 统一的权重解析和打包
- 支持并行加载优化
- 支持 GPU Cache (可选)
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from baseline.expert_buffer_manager import ExpertBufferManager
from baseline.gpu_expert_cache import GPUExpertCacheManager


class ExpertCache:
    """
    统一的 Expert Cache - CPU 缓存 + GPU buffer 管理

    通过 expert_config 或自动检测区分不同模型架构:
    - GPT-OSS: 24层, 32专家, merged gate_up_proj + bias
    - Qwen3: 48层, 128专家, separate gate/up/down
    """

    def __init__(
        self,
        state_path_or_device,
        device_or_buffer_size,
        buffer_size_or_config=None,
        expert_config: Optional[Dict] = None,
        enable_gpu_cache: bool = True,
        cache_policy: str = "static",
        topk_lru_logit_percentile: float = 90.0,
        cache_slots_per_layer: int = 16,
    ):
        """
        初始化 Expert Cache

        方式1: ExpertCache(device, buffer_size, expert_config)
            device: GPU 设备
            buffer_size: GPU buffer 数量
            expert_config: 专家结构配置 dict

        方式2: ExpertCache(state_path, device, buffer_size)
            state_path: 模型路径，自动从 config.json 加载配置
            device: GPU 设备
            buffer_size: GPU buffer 数量
        """
        # 判断调用方式
        if isinstance(state_path_or_device, torch.device):
            # 方式1: 直接传入配置
            device = state_path_or_device
            buffer_size = device_or_buffer_size
            expert_config = buffer_size_or_config or expert_config
            self.state_path = None
        elif isinstance(state_path_or_device, str):
            # 方式2: 从 state_path 自动加载配置
            self.state_path = state_path_or_device
            device = device_or_buffer_size
            buffer_size = buffer_size_or_config
            expert_config = self._load_expert_config(self.state_path)
        else:
            raise TypeError(
                f"First argument must be torch.device or str (state_path), "
                f"got {type(state_path_or_device).__name__}"
            )

        self.device = device
        self.buffer_size = buffer_size
        self.expert_config = expert_config

        # 解析模型参数
        self.num_layers = expert_config.get('num_layers', 24)
        self.num_experts = expert_config.get('num_experts', 32)
        self.structure = expert_config.get('structure', 'separate')

        # 创建 Buffer Manager
        self.buffer_manager = self._create_buffer_manager(expert_config)

        # CPU 端专家缓存
        self.simple_expert_cache: Dict[Tuple, Any] = {}

        # Qwen3 专用: CPU pinned storage 预分配
        self.cpu_pinned_storage = None
        self.cpu_expert_offsets: Dict[Tuple, Tuple[int, Tuple]] = {}

        # Qwen3 专用: Pipeline streams
        self.cpu_to_gpu_stream = None
        self.gpu_to_cpu_stream = None

        # 向后兼容的变量
        self.gpu_buffers = self.buffer_manager.gpu_buffers
        self.gpu_memory_pool = self.buffer_manager.gpu_memory_pool
        self.buffer_used = [False] * buffer_size
        self.buffer_experts = [None] * buffer_size

        # 并行加载优化
        self.max_workers = 8

        # GPU Cache 配置 (baseline 中可选)
        self.enable_gpu_cache = enable_gpu_cache
        self.cache_policy = cache_policy
        self.topk_lru_logit_percentile = topk_lru_logit_percentile
        self.cache_slots_per_layer = cache_slots_per_layer
        self.gpu_cache_manager: Optional[GPUExpertCacheManager] = None

    def init_gpu_cache(self):
        """
        初始化 GPU Cache

        在 CPU expert cache 准备好后调用此方法，
        将从 CPU cache 加载 expert 到 GPU cache
        """
        if not self.enable_gpu_cache or self.gpu_cache_manager is not None:
            return

        # 创建 GPU Cache Manager，直接从 CPU cache 加载
        self.gpu_cache_manager = GPUExpertCacheManager(
            device=self.device,
            expert_config=self.expert_config,
            cpu_expert_cache=self.simple_expert_cache,
            cache_policy=self.cache_policy,
            topk_lru_logit_percentile=self.topk_lru_logit_percentile,
            cache_slots_per_layer=self.cache_slots_per_layer,
        )

        # 更新 BufferManager 的 gpu_cache_manager 引用
        self.buffer_manager.gpu_cache_manager = self.gpu_cache_manager

    def _load_expert_config(self, state_path: str) -> Dict:
        """从 config.json 自动加载配置并识别模型类型"""
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
            return {
                'model_type': 'gpt_oss',
                'hidden_size': config['hidden_size'],
                'intermediate_size': config.get('intermediate_size', config['hidden_size']),
                'num_layers': config.get('num_hidden_layers', 24),
                'num_experts': config.get('num_experts', 32),
                'structure': 'merged',
                'has_bias': True,
                'activation': 'swiglu',
                'swiglu_limit': config.get('swiglu_limit', 7.0),
            }

        # Qwen3 MoE 检测
        elif 'qwen3_moe' in model_type.lower() or 'Qwen3Moe' in arch:
            return {
                'model_type': 'qwen3_moe',
                'hidden_size': config['hidden_size'],
                'intermediate_size': config.get('moe_intermediate_size', config['hidden_size'] * 4),
                'num_layers': config.get('num_hidden_layers', 48),
                'num_experts': config.get('num_experts', 128),
                'structure': 'separate',
                'has_bias': False,
                'activation': 'swiglu',
            }

        # 默认处理
        if 'moe_intermediate_size' in config:
            return {
                'model_type': 'qwen3_moe',
                'hidden_size': config['hidden_size'],
                'intermediate_size': config['moe_intermediate_size'],
                'num_layers': config.get('num_hidden_layers', 48),
                'num_experts': config.get('num_experts', 128),
                'structure': 'separate',
                'has_bias': False,
                'activation': 'swiglu',
            }
        else:
            return {
                'model_type': 'gpt_oss',
                'hidden_size': config['hidden_size'],
                'intermediate_size': config.get('intermediate_size', config['hidden_size']),
                'num_layers': config.get('num_hidden_layers', 24),
                'num_experts': config.get('num_experts', 32),
                'structure': 'merged',
                'has_bias': True,
                'activation': 'swiglu',
                'swiglu_limit': config.get('swiglu_limit', 7.0),
            }

    def _create_buffer_manager(self, expert_config: Dict) -> ExpertBufferManager:
        """根据配置创建对应的 Buffer Manager"""
        # 直接使用统一的 ExpertBufferManager，传入 expert_config
        return ExpertBufferManager(
            self.device,
            self.buffer_size,
            expert_config
        )

    def _process_weights_batch(self, expert_weights_dict: dict, config):
        """
        批量处理专家权重，存入CPU缓存

        GPT-OSS: 打包成单个 tensor (gate_up_proj, bias, down_proj, down_bias)
        Qwen3: 分离存储 (gate, up, down)
        """
        if not expert_weights_dict:
            return 0

        if self.structure == 'merged':
            return self._process_weights_batch_gptoss(expert_weights_dict, config)
        else:
            return self._process_weights_batch_qwen3(expert_weights_dict, config)

    def _process_weights_batch_gptoss(self, expert_weights_dict: dict, config=None):
        """GPT-OSS: 打包成单个连续 tensor"""
        processed_count = 0
        expert_collector: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        for weight_name, weight_data in expert_weights_dict.items():
            # 解析权重名称: model.layers.X.mlp.experts.{proj_type}
            parts = weight_name.split('.')
            if len(parts) < 5:
                continue

            try:
                layer_idx = int(parts[2])
                proj_type = parts[5] if len(parts) > 5 else parts[4]
            except (ValueError, IndexError):
                continue

            if layer_idx >= self.num_layers:
                continue

            # 确定权重类型
            proj_key = self._parse_gptoss_proj_type(proj_type)
            if proj_key is None:
                continue

            # 确保正确的 dtype
            if weight_data.dtype != torch.bfloat16:
                weight_data = weight_data.to(torch.bfloat16)

            # 按 expert 拆分并收集
            if len(weight_data.shape) >= 2 and weight_data.shape[0] == self.num_experts:
                for expert_idx in range(self.num_experts):
                    expert_weight = weight_data[expert_idx].contiguous()
                    key = (layer_idx, expert_idx)
                    if key not in expert_collector:
                        expert_collector[key] = {}
                    expert_collector[key][proj_key] = expert_weight
                    processed_count += 1

        # 打包每个 expert 的所有权重成一个连续 tensor
        for (layer_idx, expert_idx), parts in expert_collector.items():
            packed_parts = []
            for key in ['gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias']:
                if key in parts:
                    packed_parts.append(parts[key].flatten())

            if packed_parts:
                packed_tensor = torch.cat(packed_parts)
                if not packed_tensor.is_pinned():
                    packed_tensor = packed_tensor.pin_memory()
                self.simple_expert_cache[(layer_idx, expert_idx)] = packed_tensor

        return processed_count

    def _parse_gptoss_proj_type(self, proj_type: str) -> Optional[str]:
        """解析 GPT-OSS 权重类型"""
        if 'gate_up_proj_bias' in proj_type:
            return 'gate_up_proj_bias'
        elif 'gate_up_proj' in proj_type:
            return 'gate_up_proj'
        elif 'down_proj_bias' in proj_type:
            return 'down_proj_bias'
        elif 'down_proj' in proj_type:
            return 'down_proj'
        return None

    def _process_weights_batch_qwen3(self, expert_weights_dict: dict, config):
        """
        Qwen3: 分离存储 gate/up/down，带 pipeline 传输优化
        """
        if not expert_weights_dict:
            return 0

        # 初始化 pinned memory 和 streams（只初始化一次）
        if self.cpu_pinned_storage is None and config is not None:
            self._preallocate_cpu_pinned_storage(config)
        if self.cpu_to_gpu_stream is None:
            self.cpu_to_gpu_stream = torch.cuda.Stream(priority=0)
            self.gpu_to_cpu_stream = torch.cuda.Stream(priority=0)

        # 按 layer+expert 分组
        expert_groups: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        processed_count = 0
        for weight_name, weight_data in expert_weights_dict.items():
            parts = weight_name.split('.')
            if len(parts) < 6:
                continue

            try:
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6] if len(parts) > 6 else parts[5]
            except (ValueError, IndexError):
                continue

            if layer_idx >= self.num_layers or expert_idx >= self.num_experts:
                continue

            proj_key = self._parse_qwen3_proj_type(proj_type)
            if proj_key is None:
                continue

            if weight_data.dtype != torch.bfloat16:
                weight_data = weight_data.to(torch.bfloat16)

            uid = (layer_idx, expert_idx)
            if uid not in expert_groups:
                expert_groups[uid] = {}
            expert_groups[uid][proj_key] = weight_data
            processed_count += 1

        # Pipeline 传输: 按层处理
        layers_experts: Dict[int, List[Tuple[int, Dict]]] = {}
        for (layer_idx, expert_idx), weights in expert_groups.items():
            if layer_idx not in layers_experts:
                layers_experts[layer_idx] = []
            layers_experts[layer_idx].append((expert_idx, weights))

        # 真正的 pipeline: 逐个 expert 处理
        for layer_idx, layer_experts in layers_experts.items():
            for expert_idx, expert_weights in layer_experts:
                buffer_idx = expert_idx % self.buffer_size

                # Pipeline Step 1: CPU -> GPU
                with torch.cuda.stream(self.cpu_to_gpu_stream):
                    for proj_key, weight_data in expert_weights.items():
                        proj_key_clean = proj_key.replace('_proj', '')
                        gpu_buffer = self.buffer_manager.gpu_buffers[buffer_idx]

                        if proj_key_clean == 'gate':
                            offset = gpu_buffer['gate_proj']['offset']
                            shape = gpu_buffer['gate_proj']['shape']
                        elif proj_key_clean == 'up':
                            offset = gpu_buffer['up_proj']['offset']
                            shape = gpu_buffer['up_proj']['shape']
                        else:  # down
                            offset = gpu_buffer['down_proj']['offset']
                            shape = gpu_buffer['down_proj']['shape']

                        gpu_memory_pool = self.buffer_manager.gpu_memory_pool
                        gpu_view = gpu_memory_pool[offset:offset + weight_data.numel()].view(shape)
                        gpu_view.copy_(weight_data, non_blocking=True)
                        gpu_weight = gpu_view

                        # Pipeline Step 2: GPU -> CPU pinned memory
                        with torch.cuda.stream(self.gpu_to_cpu_stream):
                            self.gpu_to_cpu_stream.wait_stream(self.cpu_to_gpu_stream)

                            cache_key = (layer_idx, expert_idx, proj_key_clean)
                            if cache_key in self.cpu_expert_offsets:
                                cpu_offset, cpu_shape = self.cpu_expert_offsets[cache_key]
                                storage_view = torch.as_tensor(
                                    self.cpu_pinned_storage[cpu_offset:cpu_offset + gpu_weight.numel() * 2],
                                    dtype=torch.bfloat16,
                                    device='cpu'
                                ).view(cpu_shape)
                                storage_view.copy_(gpu_weight, non_blocking=True)
                                self.simple_expert_cache[cache_key] = storage_view

        # 等待所有 stream 完成
        torch.cuda.current_stream().wait_stream(self.cpu_to_gpu_stream)
        torch.cuda.current_stream().wait_stream(self.gpu_to_cpu_stream)
        torch.cuda.synchronize()

        return processed_count

    def _preallocate_cpu_pinned_storage(self, config):
        """Qwen3: 预分配 CPU pinned memory"""
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
        element_size = 2  # bfloat16

        gate_size = intermediate_size * hidden_size * element_size
        up_size = intermediate_size * hidden_size * element_size
        down_size = hidden_size * intermediate_size * element_size
        expert_size = gate_size + up_size + down_size
        total_storage_size = expert_size * self.num_layers * self.num_experts

        self.cpu_pinned_storage = torch.UntypedStorage(
            total_storage_size, device='cpu'
        ).pin_memory(self.device)

        # 计算偏移量
        offset = 0
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.num_experts):
                gate_shape = (intermediate_size, hidden_size)
                up_shape = (intermediate_size, hidden_size)
                down_shape = (hidden_size, intermediate_size)

                self.cpu_expert_offsets[(layer_idx, expert_idx, 'gate')] = (offset, gate_shape)
                offset += gate_shape[0] * gate_shape[1] * element_size

                self.cpu_expert_offsets[(layer_idx, expert_idx, 'up')] = (offset, up_shape)
                offset += up_shape[0] * up_shape[1] * element_size

                self.cpu_expert_offsets[(layer_idx, expert_idx, 'down')] = (offset, down_shape)
                offset += down_shape[0] * down_shape[1] * element_size

    def _parse_qwen3_proj_type(self, proj_type: str) -> Optional[str]:
        """解析 Qwen3 权重类型"""
        if 'gate_proj' in proj_type:
            return 'gate'
        elif 'up_proj' in proj_type:
            return 'up'
        elif 'down_proj' in proj_type:
            return 'down'
        return None

    def batch_load_experts_continuous(
        self,
        layer_idx: int,
        expert_indices,
        router_logits: Optional[Dict[int, float]] = None
    ) -> Dict[int, int]:
        """
        为当前层加载 experts

        Args:
            layer_idx: 层索引
            expert_indices: (_, expert_id) 元组列表
            router_logits: (可选) expert_id -> max_logit_score 映射

        Returns:
            Dict[int, int]: expert_id → virtual_idx 的映射关系
        """
        # 延迟初始化 GPU Cache（在第一次调用时）
        if self.enable_gpu_cache and self.gpu_cache_manager is None:
            self.init_gpu_cache()

        if not expert_indices:
            return {}

        if self.structure == 'merged':
            return self._batch_load_gptoss(layer_idx, expert_indices, router_logits)
        else:
            return self._batch_load_qwen3(layer_idx, expert_indices, router_logits)

    # 向后兼容的别名
    batch_load_experts_continous = batch_load_experts_continuous

    def _batch_load_gptoss(
        self,
        layer_idx: int,
        expert_indices,
        router_logits: Optional[Dict[int, float]] = None
    ) -> Dict[int, int]:
        """GPT-OSS: 加载打包的 tensor"""
        expert_ids = []
        expert_weights_dict = {}

        for _, expert_id in expert_indices:
            packed_tensor = self.simple_expert_cache.get((layer_idx, expert_id))
            if packed_tensor is not None:
                expert_ids.append(expert_id)
                expert_weights_dict[expert_id] = packed_tensor

        buffer_mapping = self.buffer_manager.load_experts_for_current_layer(
            layer_idx, expert_ids, expert_weights_dict, router_logits
        )

        # 更新向后兼容的变量（只更新临时 buffer，跳过 GPU cache slot）
        for expert_id, virtual_idx in buffer_mapping.items():
            if not self.buffer_manager._is_cache_slot(virtual_idx):
                self.buffer_used[virtual_idx] = True
                self.buffer_experts[virtual_idx] = expert_id

        return buffer_mapping

    def _batch_load_qwen3(
        self,
        layer_idx: int,
        expert_indices,
        router_logits: Optional[Dict[int, float]] = None
    ) -> Dict[int, int]:
        """Qwen3: 加载分离的 gate/up/down"""
        expert_ids = []
        expert_weights_dict = {}

        for _, expert_id in expert_indices:
            gate_tensor = self.simple_expert_cache.get((layer_idx, expert_id, 'gate'))
            up_tensor = self.simple_expert_cache.get((layer_idx, expert_id, 'up'))
            down_tensor = self.simple_expert_cache.get((layer_idx, expert_id, 'down'))

            if gate_tensor is not None and up_tensor is not None and down_tensor is not None:
                expert_ids.append(expert_id)
                expert_weights_dict[expert_id] = {
                    'gate': gate_tensor,
                    'up': up_tensor,
                    'down': down_tensor
                }

        buffer_mapping = self.buffer_manager.load_experts_for_current_layer(
            layer_idx, expert_ids, expert_weights_dict, router_logits
        )

        # 更新向后兼容的变量（只更新临时 buffer，跳过 GPU cache slot）
        for expert_id, virtual_idx in buffer_mapping.items():
            if not self.buffer_manager._is_cache_slot(virtual_idx):
                self.buffer_used[virtual_idx] = True
                self.buffer_experts[virtual_idx] = expert_id

        return buffer_mapping


# ========== 向后兼容的模型特定 ExpertCache ==========

class GptOssExpertCache(ExpertCache):
    """
    GPT-OSS Expert Cache（向后兼容）
    继承统一 ExpertCache
    """

    def __init__(
        self,
        state_path: str,
        device: torch.device,
        buffer_size: int = 32,
        enable_gpu_cache: bool = True,
        cache_policy: str = "static",
        topk_lru_logit_percentile: float = 90.0,
        cache_slots_per_layer: int = 8,
    ):
        # 从 state_path 自动加载配置
        expert_config = self._load_config_for_gptoss(state_path)
        super().__init__(
            device, buffer_size, expert_config,
            enable_gpu_cache=enable_gpu_cache,
            cache_policy=cache_policy,
            topk_lru_logit_percentile=topk_lru_logit_percentile,
            cache_slots_per_layer=cache_slots_per_layer,
        )
        self.state_path = state_path

    def _load_config_for_gptoss(self, state_path: str) -> Dict:
        """加载 GPT-OSS 配置"""
        import json
        import os

        config_path = os.path.join(state_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        return {
            'model_type': 'gpt_oss',
            'hidden_size': config['hidden_size'],
            'intermediate_size': config.get('intermediate_size', config['hidden_size']),
            'num_layers': config.get('num_hidden_layers', 24),
            'num_experts': config.get('num_experts', 32),
            'structure': 'merged',
            'has_bias': True,
            'activation': 'swiglu',
            'swiglu_limit': config.get('swiglu_limit', 7.0),
        }


class Qwen3ExpertCache(ExpertCache):
    """
    Qwen3 Expert Cache（向后兼容）
    继承统一 ExpertCache
    """

    def __init__(
        self,
        state_path: str,
        device: torch.device,
        buffer_size: int = 128,
        enable_gpu_cache: bool = True,
        cache_policy: str = "static",
        topk_lru_logit_percentile: float = 90.0,
        cache_slots_per_layer: int = 16,
    ):
        # 从 state_path 自动加载配置
        expert_config = self._load_config_for_qwen3(state_path)
        super().__init__(
            device, buffer_size, expert_config,
            enable_gpu_cache=enable_gpu_cache,
            cache_policy=cache_policy,
            topk_lru_logit_percentile=topk_lru_logit_percentile,
            cache_slots_per_layer=cache_slots_per_layer,
        )
        self.state_path = state_path

    def _load_config_for_qwen3(self, state_path: str) -> Dict:
        """加载 Qwen3 配置"""
        import json
        import os

        config_path = os.path.join(state_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        return {
            'model_type': 'qwen3_moe',
            'hidden_size': config['hidden_size'],
            'intermediate_size': config.get('moe_intermediate_size', config['hidden_size'] * 4),
            'num_layers': config.get('num_hidden_layers', 48),
            'num_experts': config.get('num_experts', 128),
            'structure': 'separate',
            'has_bias': False,
            'activation': 'swiglu',
        }
