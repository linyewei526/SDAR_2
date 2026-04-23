"""
GPU Expert Cache Manager - GPU 显存中的 Expert 缓存管理器

支持缓存策略：
- static: 静态缓存，每层固定缓存前 N 个 expert
- lru: 基本的 LRU 缓存策略
- lfu: 基本的 LFU 缓存策略
- topk_lru: 继承 LRU，增加 logit 阈值准入控制
- tinylfu: 继承 LFU，使用 S-LRU 分段 + TinyLFU 准入控制
"""

import torch
from typing import Dict, Optional, List, Tuple, Any
from collections import OrderedDict, defaultdict
from baseline.debug_config import PRINT_BUFFER_INIT, PRINT_CACHE_UPDATE_DEBUG
from baseline.nvtx_utils import nvtx_range


class CachePolicy:
    """缓存策略基类"""
    def __init__(self, num_layers: int, num_experts: int, slots_per_layer: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.slots_per_layer = slots_per_layer
        self.total_slots = slots_per_layer * num_layers

        # 统计信息
        self.hits = 0
        self.misses = 0
        self.stats_enabled = True  # 默认启用统计
        self.cache_updates = 0

    def enable_stats(self):
        """启用统计"""
        self.stats_enabled = True

    def disable_stats(self):
        """禁用统计（hit/miss 不增加）"""
        self.stats_enabled = False

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        """查询 expert 是否在 cache 中，返回 slot_idx 或 None"""
        raise NotImplementedError

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """检查 expert 是否在 cache 中（不更新统计）"""
        raise NotImplementedError

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """更新 cache"""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'total_requests': total,
            'cache_updates': self.cache_updates,
        }


class StaticCachePolicy(CachePolicy):
    """
    静态缓存策略

    每层固定缓存 expert ID 从 0 到 slots_per_layer-1
    """
    def __init__(self, num_layers: int, num_experts: int, slots_per_layer: int):
        super().__init__(num_layers, num_experts, slots_per_layer)

        # 预计算缓存映射
        self.cached_experts: Dict[Tuple[int, int], int] = {}
        for layer_idx in range(num_layers):
            for offset in range(slots_per_layer):
                expert_id = offset
                slot_idx = layer_idx * slots_per_layer + offset
                self.cached_experts[(layer_idx, expert_id)] = slot_idx

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        result = self.cached_experts.get((layer_idx, expert_id))
        if self.stats_enabled:
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
        return result

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self.cached_experts

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """静态策略不更新 cache"""
        return []

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['policy'] = 'static'
        stats['expected_hit_rate'] = self.slots_per_layer / self.num_experts
        return stats


class LRUCachePolicy(CachePolicy):
    """
    基本的 LRU 缓存策略
    
    - 每层维护独立的 LRU cache
    - 淘汰最久未使用的 expert
    - 使用 OrderedDict 实现 O(1) 的 LRU 操作
    """
    def __init__(self, num_layers: int, num_experts: int, slots_per_layer: int):
        super().__init__(num_layers, num_experts, slots_per_layer)

        # 每层维护 LRU cache：expert_id -> slot_offset
        # OrderedDict 维护访问顺序，最近访问的在末尾
        self.layer_caches: List[OrderedDict] = [OrderedDict() for _ in range(num_layers)]

        # 反向映射：(layer_idx, slot_offset) -> expert_id
        self.slot_to_expert: Dict[Tuple[int, int], int] = {}

        # 预填充 static 缓存：每层缓存 expert 0 ~ slots_per_layer-1
        for layer_idx in range(num_layers):
            for offset in range(slots_per_layer):
                expert_id = offset
                self.layer_caches[layer_idx][expert_id] = offset
                self.slot_to_expert[(layer_idx, offset)] = expert_id

        self.in_warmup = True

    @property
    def cached_experts(self) -> Dict[Tuple[int, int], int]:
        """兼容接口：返回 (layer_idx, expert_id) -> global_slot_idx 映射"""
        result = {}
        for layer_idx, cache in enumerate(self.layer_caches):
            for expert_id, offset in cache.items():
                global_slot_idx = layer_idx * self.slots_per_layer + offset
                result[(layer_idx, expert_id)] = global_slot_idx
        return result

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        """查询 expert 是否在 cache 中，命中时更新 LRU 顺序和统计"""
        cache = self.layer_caches[layer_idx]
        if expert_id in cache:
            if self.stats_enabled:
                self.hits += 1
            # 移到最近使用（末尾）
            cache.move_to_end(expert_id)
            offset = cache[expert_id]
            return layer_idx * self.slots_per_layer + offset
        if self.stats_enabled:
            self.misses += 1
        return None

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return expert_id in self.layer_caches[layer_idx]

    def _evict_one(self, layer_idx: int) -> Tuple[int, int]:
        """淘汰一个 expert，返回 (evicted_expert_id, slot_offset)"""
        cache = self.layer_caches[layer_idx]
        evict_expert_id, evict_slot_offset = cache.popitem(last=False)
        del self.slot_to_expert[(layer_idx, evict_slot_offset)]
        return evict_expert_id, evict_slot_offset

    def _insert_one(self, layer_idx: int, expert_id: int, slot_offset: int):
        """插入一个 expert 到指定 slot"""
        cache = self.layer_caches[layer_idx]
        cache[expert_id] = slot_offset
        self.slot_to_expert[(layer_idx, slot_offset)] = expert_id

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """
        更新 cache - 基本 LRU 策略
        
        Args:
            layer_idx: 层索引
            new_experts: [(expert_id, logit_score), ...]  logit_score 在基本 LRU 中被忽略
        
        Returns:
            更新列表: [(expert_id, slot_offset, evicted_expert_id), ...]
        """
        cache = self.layer_caches[layer_idx]
        updates = []

        for expert_id, _ in new_experts:
            # 如果已在 cache 中，更新 LRU 顺序
            if expert_id in cache:
                cache.move_to_end(expert_id)
                continue

            # cache 已满，LRU 淘汰
            if len(cache) >= self.slots_per_layer:
                evict_expert_id, evict_slot_offset = self._evict_one(layer_idx)
                self._insert_one(layer_idx, expert_id, evict_slot_offset)
                updates.append((expert_id, evict_slot_offset, evict_expert_id))
                self.cache_updates += 1

        if self.in_warmup:
            self.in_warmup = False

        return updates

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['policy'] = 'lru'
        return stats


class LFUCachePolicy(CachePolicy):
    """
    基本的 LFU 缓存策略
    
    - 每层维护独立的 LFU cache
    - 淘汰访问频率最低的 expert
    - 同频率时按 LRU 淘汰
    """
    def __init__(self, num_layers: int, num_experts: int, slots_per_layer: int):
        super().__init__(num_layers, num_experts, slots_per_layer)

        # 每层: expert_id -> slot_offset
        self.layer_caches: List[Dict[int, int]] = [{} for _ in range(num_layers)]
        # 每层: expert_id -> frequency
        self.layer_freq: List[Dict[int, int]] = [{} for _ in range(num_layers)]
        # 每层: freq -> OrderedDict[expert_id -> True] (保持插入顺序用于同频 LRU)
        self.layer_freq_to_experts: List[Dict[int, OrderedDict]] = [defaultdict(OrderedDict) for _ in range(num_layers)]
        # 每层最小频率
        self.layer_min_freq: List[int] = [0] * num_layers

        # 反向映射
        self.slot_to_expert: Dict[Tuple[int, int], int] = {}

        # 预填充
        for layer_idx in range(num_layers):
            for offset in range(slots_per_layer):
                expert_id = offset
                self.layer_caches[layer_idx][expert_id] = offset
                self.layer_freq[layer_idx][expert_id] = 1
                self.layer_freq_to_experts[layer_idx][1][expert_id] = True
                self.slot_to_expert[(layer_idx, offset)] = expert_id
            self.layer_min_freq[layer_idx] = 1

        self.in_warmup = True

    @property
    def cached_experts(self) -> Dict[Tuple[int, int], int]:
        result = {}
        for layer_idx in range(self.num_layers):
            for expert_id, offset in self.layer_caches[layer_idx].items():
                result[(layer_idx, expert_id)] = layer_idx * self.slots_per_layer + offset
        return result

    @property
    def layer_caches_ordered(self) -> List[OrderedDict]:
        """兼容接口"""
        result = []
        for layer_idx in range(self.num_layers):
            merged = OrderedDict()
            for expert_id, offset in self.layer_caches[layer_idx].items():
                merged[expert_id] = offset
            result.append(merged)
        return result

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        cache = self.layer_caches[layer_idx]
        if expert_id in cache:
            if self.stats_enabled:
                self.hits += 1
            # 更新频率
            self._update_freq(layer_idx, expert_id)
            return layer_idx * self.slots_per_layer + cache[expert_id]
        if self.stats_enabled:
            self.misses += 1
        return None

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return expert_id in self.layer_caches[layer_idx]

    def _update_freq(self, layer_idx: int, expert_id: int):
        """增加 expert 的访问频率"""
        freq = self.layer_freq[layer_idx]
        freq_to_experts = self.layer_freq_to_experts[layer_idx]
        
        old_freq = freq[expert_id]
        new_freq = old_freq + 1
        
        # 从旧频率组移除
        del freq_to_experts[old_freq][expert_id]
        if len(freq_to_experts[old_freq]) == 0:
            del freq_to_experts[old_freq]
            if old_freq == self.layer_min_freq[layer_idx]:
                self.layer_min_freq[layer_idx] = new_freq
        
        # 加入新频率组
        freq[expert_id] = new_freq
        freq_to_experts[new_freq][expert_id] = True

    def _evict_one(self, layer_idx: int) -> Tuple[int, int]:
        """淘汰最低频率的 expert (同频率时 LRU)"""
        cache = self.layer_caches[layer_idx]
        freq = self.layer_freq[layer_idx]
        freq_to_experts = self.layer_freq_to_experts[layer_idx]
        min_freq = self.layer_min_freq[layer_idx]
        
        # 获取最低频率组中最老的 expert
        evict_expert_id = next(iter(freq_to_experts[min_freq]))
        evict_slot_offset = cache[evict_expert_id]
        
        # 移除
        del freq_to_experts[min_freq][evict_expert_id]
        if len(freq_to_experts[min_freq]) == 0:
            del freq_to_experts[min_freq]
        del cache[evict_expert_id]
        del freq[evict_expert_id]
        del self.slot_to_expert[(layer_idx, evict_slot_offset)]
        
        return evict_expert_id, evict_slot_offset

    def _insert_one(self, layer_idx: int, expert_id: int, slot_offset: int):
        """插入新 expert"""
        cache = self.layer_caches[layer_idx]
        freq = self.layer_freq[layer_idx]
        freq_to_experts = self.layer_freq_to_experts[layer_idx]
        
        cache[expert_id] = slot_offset
        freq[expert_id] = 1
        freq_to_experts[1][expert_id] = True
        self.layer_min_freq[layer_idx] = 1
        self.slot_to_expert[(layer_idx, slot_offset)] = expert_id

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """更新 cache - 基本 LFU 策略"""
        cache = self.layer_caches[layer_idx]
        updates = []

        for expert_id, _ in new_experts:
            if expert_id in cache:
                self._update_freq(layer_idx, expert_id)
                continue

            if len(cache) >= self.slots_per_layer:
                evict_expert_id, evict_slot_offset = self._evict_one(layer_idx)
                self._insert_one(layer_idx, expert_id, evict_slot_offset)
                updates.append((expert_id, evict_slot_offset, evict_expert_id))
                self.cache_updates += 1

        if self.in_warmup:
            self.in_warmup = False

        return updates

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['policy'] = 'lfu'
        return stats


class TopKLRUCachePolicy(LRUCachePolicy):
    """
    TopK + LRU 缓存策略（继承自 LRU）

    在 LRU 基础上增加 logit 阈值准入控制：
    - Warmup 阶段：允许所有 experts 准入
    - Decode 阶段：只有 logit >= percentile 阈值的 expert 才能准入
    
    当 logit_threshold_percentile=0 时，退化为纯 LRU 策略
    """
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        slots_per_layer: int,
        logit_threshold_percentile: float = 90.0
    ):
        super().__init__(num_layers, num_experts, slots_per_layer)
        self.logit_threshold_percentile = logit_threshold_percentile

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """
        更新 cache - 带 logit 阈值准入控制的 LRU
        """
        cache = self.layer_caches[layer_idx]
        updates = []

        # 计算 logit 阈值
        if self.in_warmup:
            logit_threshold = float('-inf')
        else:
            logit_threshold = float('inf')
            if new_experts:
                logits = [e[1] for e in new_experts]
                if logits:
                    logits.sort()
                    threshold_idx = int(len(logits) * self.logit_threshold_percentile / 100)
                    threshold_idx = min(threshold_idx, len(logits) - 1)
                    logit_threshold = logits[threshold_idx]

        for expert_id, max_logit in new_experts:
            # 检查 logit 阈值准入
            if max_logit < logit_threshold:
                continue

            if expert_id in cache:
                cache.move_to_end(expert_id)
                continue

            if len(cache) >= self.slots_per_layer:
                evict_expert_id, evict_slot_offset = self._evict_one(layer_idx)
                self._insert_one(layer_idx, expert_id, evict_slot_offset)
                updates.append((expert_id, evict_slot_offset, evict_expert_id))
                self.cache_updates += 1

        if self.in_warmup:
            self.in_warmup = False

        return updates

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['policy'] = 'topk_lru'
        stats['logit_threshold_percentile'] = self.logit_threshold_percentile
        return stats


class TinyLFUCachePolicy(LFUCachePolicy):
    """
    TinyLFU + S-LRU 缓存策略（继承自 LFU）
    
    在 LFU 基础上改进：
    1. 使用 S-LRU (Segmented LRU) 分段：Protected (75%) + Probation (25%)
    2. 使用 TinyLFU 准入控制：新 entry 频率 > victim 频率才准入
    3. 定期衰减频率防止历史污染
    """
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        slots_per_layer: int,
        protected_ratio: float = 0.75
    ):
        # 不调用 LFUCachePolicy.__init__，完全自己管理
        CachePolicy.__init__(self, num_layers, num_experts, slots_per_layer)
        self.protected_ratio = protected_ratio
        
        # S-LRU 分段大小
        self.protected_cap = max(1, int(slots_per_layer * protected_ratio))
        self.probation_cap = slots_per_layer - self.protected_cap
        
        # 每层两个 LRU 队列
        self.layer_protected: List[OrderedDict] = [OrderedDict() for _ in range(num_layers)]
        self.layer_probation: List[OrderedDict] = [OrderedDict() for _ in range(num_layers)]
        
        # 全局频率计数 (TinyLFU sketch)
        self.freq: Dict[int, int] = defaultdict(int)
        self.freq_decay_counter = 0
        self.freq_decay_threshold = 1000
        
        # slot 映射
        self.slot_to_expert: Dict[Tuple[int, int], int] = {}
        
        # 预填充
        for layer_idx in range(num_layers):
            for offset in range(slots_per_layer):
                expert_id = offset
                if offset < self.probation_cap:
                    self.layer_probation[layer_idx][expert_id] = offset
                else:
                    self.layer_protected[layer_idx][expert_id] = offset
                self.slot_to_expert[(layer_idx, offset)] = expert_id

        self.in_warmup = True

    @property
    def cached_experts(self) -> Dict[Tuple[int, int], int]:
        result = {}
        for layer_idx in range(self.num_layers):
            for expert_id, offset in self.layer_protected[layer_idx].items():
                result[(layer_idx, expert_id)] = layer_idx * self.slots_per_layer + offset
            for expert_id, offset in self.layer_probation[layer_idx].items():
                result[(layer_idx, expert_id)] = layer_idx * self.slots_per_layer + offset
        return result

    @property
    def layer_caches(self) -> List[OrderedDict]:
        """兼容接口"""
        result = []
        for layer_idx in range(self.num_layers):
            merged = OrderedDict()
            for expert_id, offset in self.layer_probation[layer_idx].items():
                merged[expert_id] = offset
            for expert_id, offset in self.layer_protected[layer_idx].items():
                merged[expert_id] = offset
            result.append(merged)
        return result

    def _estimate_freq(self, expert_id: int) -> int:
        return min(self.freq.get(expert_id, 0), 15)

    def _increment_freq(self, expert_id: int):
        self.freq[expert_id] = min(self.freq.get(expert_id, 0) + 1, 15)
        self.freq_decay_counter += 1
        if self.freq_decay_counter >= self.freq_decay_threshold:
            for k in list(self.freq.keys()):
                self.freq[k] //= 2
                if self.freq[k] == 0:
                    del self.freq[k]
            self.freq_decay_counter = 0

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        protected = self.layer_protected[layer_idx]
        probation = self.layer_probation[layer_idx]
        
        self._increment_freq(expert_id)
        
        if expert_id in protected:
            if self.stats_enabled:
                self.hits += 1
            protected.move_to_end(expert_id)
            return layer_idx * self.slots_per_layer + protected[expert_id]
        
        if expert_id in probation:
            if self.stats_enabled:
                self.hits += 1
            # 从 probation 升级到 protected
            offset = probation[expert_id]
            del probation[expert_id]
            
            if len(protected) >= self.protected_cap:
                demoted_id, demoted_offset = protected.popitem(last=False)
                probation[demoted_id] = demoted_offset
                if len(probation) > self.probation_cap:
                    evict_id, evict_offset = probation.popitem(last=False)
                    del self.slot_to_expert[(layer_idx, evict_offset)]
            
            protected[expert_id] = offset
            return layer_idx * self.slots_per_layer + offset
        
        if self.stats_enabled:
            self.misses += 1
        return None

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return (expert_id in self.layer_protected[layer_idx] or 
                expert_id in self.layer_probation[layer_idx])

    def update_cache(
        self,
        layer_idx: int,
        new_experts: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int]]:
        """更新 cache - TinyLFU 准入控制"""
        protected = self.layer_protected[layer_idx]
        probation = self.layer_probation[layer_idx]
        updates = []

        for expert_id, _ in new_experts:
            self._increment_freq(expert_id)

            if expert_id in protected:
                protected.move_to_end(expert_id)
                continue

            if expert_id in probation:
                # 升级到 protected
                offset = probation[expert_id]
                del probation[expert_id]
                
                if len(protected) >= self.protected_cap:
                    demoted_id, demoted_offset = protected.popitem(last=False)
                    probation[demoted_id] = demoted_offset
                    if len(probation) > self.probation_cap:
                        evict_id, evict_offset = probation.popitem(last=False)
                        del self.slot_to_expert[(layer_idx, evict_offset)]
                
                protected[expert_id] = offset
                continue

            # 新 entry: TinyLFU 准入控制
            if len(probation) >= self.probation_cap:
                victim_id, victim_offset = probation.popitem(last=False)
                
                if self._estimate_freq(expert_id) > self._estimate_freq(victim_id):
                    del self.slot_to_expert[(layer_idx, victim_offset)]
                    probation[expert_id] = victim_offset
                    self.slot_to_expert[(layer_idx, victim_offset)] = expert_id
                    updates.append((expert_id, victim_offset, victim_id))
                    self.cache_updates += 1
                else:
                    probation[victim_id] = victim_offset
            else:
                # 找空闲 slot
                used_offsets = set(probation.values()) | set(protected.values())
                for offset in range(self.slots_per_layer):
                    if offset not in used_offsets:
                        probation[expert_id] = offset
                        self.slot_to_expert[(layer_idx, offset)] = expert_id
                        updates.append((expert_id, offset, -1))
                        self.cache_updates += 1
                        break

        if self.in_warmup:
            self.in_warmup = False

        return updates

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['policy'] = 'tinylfu'
        stats['protected_ratio'] = self.protected_ratio
        return stats


class GPUExpertCacheManager:
    """
    GPU Expert Cache 管理器

    功能：
    - 支持静态缓存策略
    - 按配置比例占用 GPU 显存作为 cache
    - 提供 lookup 和 get_expert_view 接口
    - 提供 cache hit 统计
    """

    def __init__(
        self,
        device: torch.device,
        expert_config: Dict[str, Any],
        cpu_expert_cache: Dict[Tuple, Any],
        cache_policy: str = "static",
        topk_lru_logit_percentile: float = 90.0,
        cache_slots_per_layer: int = 16,
    ):
        """
        初始化 GPU Expert Cache Manager

        Args:
            device: GPU 设备
            expert_config: Expert 配置
            cpu_expert_cache: CPU 端 expert 缓存
            cache_policy: 缓存策略 ("static" 或 "topk_lru")
            topk_lru_logit_percentile: topk_lru 策略下的 logit 百分位阈值
            cache_slots_per_layer: 每层缓存的 expert 数量
        """
        self.device = device
        self.expert_config = expert_config
        self.cpu_expert_cache = cpu_expert_cache
        self.cache_policy_name = cache_policy
        self.topk_lru_logit_percentile = topk_lru_logit_percentile
        self.cache_slots_per_layer = cache_slots_per_layer

        # 解析模型配置
        self.num_layers = expert_config.get('num_layers', 24)
        self.num_experts = expert_config.get('num_experts', 32)
        self.hidden_size = expert_config['hidden_size']
        self.intermediate_size = expert_config['intermediate_size']
        self.structure = expert_config.get('structure', 'separate')
        self.has_bias = expert_config.get('has_bias', False)

        # 计算单个 expert 的参数数量
        self.params_per_expert = self._compute_params_per_expert()
        self.expert_size_bytes = self.params_per_expert * 2  # bfloat16 = 2 bytes

        if PRINT_BUFFER_INIT:
            print(f"📦 Initializing GPUExpertCacheManager")
            print(f"   Cache policy: {cache_policy}")
            print(f"   Model: {self.num_layers} layers, {self.num_experts} experts per layer")
            print(f"   Expert size: {self.expert_size_bytes / 1024**2:.2f} MB")

        # 计算 cache 容量
        self._compute_cache_capacity()

        # 分配 GPU 内存池
        self._allocate_gpu_cache_memory()

        # 初始化缓存策略
        self._init_cache_policy(cache_policy)

        # 预加载初始 cache 内容 (所有动态策略都需要)
        if cache_policy in ("static", "lru", "lfu", "topk_lru", "tinylfu"):
            self._load_experts_to_gpu_cache()

        if PRINT_BUFFER_INIT:
            print("✅ GPUExpertCacheManager initialized!")

    def _init_cache_policy(self, policy_name: str):
        """初始化缓存策略"""
        if policy_name == "static":
            self.policy = StaticCachePolicy(
                self.num_layers, self.num_experts, self.slots_per_layer
            )
        elif policy_name == "lru":
            self.policy = LRUCachePolicy(
                self.num_layers, self.num_experts, self.slots_per_layer
            )
        elif policy_name == "lfu":
            self.policy = LFUCachePolicy(
                self.num_layers, self.num_experts, self.slots_per_layer
            )
        elif policy_name == "topk_lru":
            self.policy = TopKLRUCachePolicy(
                self.num_layers, self.num_experts, self.slots_per_layer,
                logit_threshold_percentile=self.topk_lru_logit_percentile
            )
        elif policy_name == "tinylfu":
            self.policy = TinyLFUCachePolicy(
                self.num_layers, self.num_experts, self.slots_per_layer,
                protected_ratio=0.75
            )
        else:
            raise ValueError(f"Unknown cache policy: {policy_name}. Supported: 'static', 'lru', 'lfu', 'topk_lru', 'tinylfu'")

        # cached_experts 映射 (兼容旧代码)
        self.cached_experts = self.policy.cached_experts

        if PRINT_BUFFER_INIT:
            cached_ratio = (self.slots_per_layer / self.num_experts) * 100
            print(f"📋 Cache policy: {policy_name}")
            print(f"   Slots per layer: {self.slots_per_layer} ({cached_ratio:.1f}% of {self.num_experts})")
            if policy_name == "topk_lru":
                print(f"   Logit percentile threshold: {self.topk_lru_logit_percentile}%")
            if policy_name == "tinylfu":
                print(f"   Protected ratio: {self.policy.protected_ratio}")

    def _compute_params_per_expert(self) -> int:
        """计算单个 expert 的参数数量"""
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size

        if self.structure == 'merged':
            # GPT-OSS: gate_up_proj + bias + down_proj + bias
            gate_up_params = hidden_size * intermediate_size * 2
            gate_up_bias_params = intermediate_size * 2 if self.has_bias else 0
            down_params = intermediate_size * hidden_size
            down_bias_params = hidden_size if self.has_bias else 0
            return gate_up_params + gate_up_bias_params + down_params + down_bias_params
        else:
            # Qwen3: gate + up + down (no bias)
            gate_params = hidden_size * intermediate_size
            up_params = hidden_size * intermediate_size
            down_params = intermediate_size * hidden_size
            return gate_params + up_params + down_params

    def _compute_cache_capacity(self):
        """
        计算 GPU cache 容量（多少个 expert slots）
        使用固定 slots 模式：直接指定每层缓存的 expert 数量
        """
        self.slots_per_layer = min(self.cache_slots_per_layer, self.num_experts)
        self.total_cache_slots = self.slots_per_layer * self.num_layers
        self.actual_cache_memory = self.total_cache_slots * self.expert_size_bytes

        if PRINT_BUFFER_INIT:
            print(f"💾 GPU Memory (Fixed slots mode):")
            print(f"   Slots per layer: {self.slots_per_layer} (specified)")
            print(f"   Total cache slots: {self.total_cache_slots}")
            print(f"   Cache allocation: {self.actual_cache_memory / 1024**3:.2f} GB")

    def _allocate_gpu_cache_memory(self):
        """分配 GPU cache 内存池"""
        total_params = self.total_cache_slots * self.params_per_expert

        # 分配连续的 GPU 内存
        self.gpu_memory_pool = torch.empty(
            total_params,
            dtype=torch.bfloat16,
            device=self.device
        )

        # 为每个 cache slot 创建 buffer info
        self.cache_buffer_infos: List[Dict] = []
        for slot_idx in range(self.total_cache_slots):
            start_offset = slot_idx * self.params_per_expert
            buffer_info = self._create_buffer_info(start_offset)
            self.cache_buffer_infos.append(buffer_info)

        if PRINT_BUFFER_INIT:
            print(f"✅ Allocated GPU cache memory pool: {total_params:,} params ({total_params * 2 / 1024**3:.2f} GB)")

    def _create_buffer_info(self, start_offset: int) -> Dict:
        """创建 buffer 元信息（与 ExpertBufferManager 兼容的格式）"""
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        current_offset = start_offset

        buffer_info = {'memory_pool': self.gpu_memory_pool}

        if self.structure == 'merged':
            # GPT-OSS 结构
            gate_up_params = hidden_size * intermediate_size * 2
            gate_up_bias_params = intermediate_size * 2 if self.has_bias else 0
            down_params = intermediate_size * hidden_size
            down_bias_params = hidden_size if self.has_bias else 0

            buffer_info['gate_up_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size * 2),
                'size': gate_up_params
            }
            current_offset += gate_up_params

            if self.has_bias:
                buffer_info['gate_up_proj_bias'] = {
                    'offset': current_offset,
                    'shape': (intermediate_size * 2,),
                    'size': gate_up_bias_params
                }
                current_offset += gate_up_bias_params

            buffer_info['down_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size),
                'size': down_params
            }
            current_offset += down_params

            if self.has_bias:
                buffer_info['down_proj_bias'] = {
                    'offset': current_offset,
                    'shape': (hidden_size,),
                    'size': down_bias_params
                }
        else:
            # Qwen3 结构
            gate_params = hidden_size * intermediate_size
            up_params = hidden_size * intermediate_size
            down_params = intermediate_size * hidden_size

            buffer_info['gate_proj'] = {
                'offset': current_offset,
                'shape': (intermediate_size, hidden_size),
                'size': gate_params
            }
            current_offset += gate_params

            buffer_info['up_proj'] = {
                'offset': current_offset,
                'shape': (intermediate_size, hidden_size),
                'size': up_params
            }
            current_offset += up_params

            buffer_info['down_proj'] = {
                'offset': current_offset,
                'shape': (hidden_size, intermediate_size),
                'size': down_params
            }

        return buffer_info

    def _load_experts_to_gpu_cache(self):
        """从 CPU cache 加载 expert 权重到 GPU cache"""
        if not self.cpu_expert_cache:
            if PRINT_BUFFER_INIT:
                print("⚠️  CPU expert cache is empty, skipping GPU cache initialization")
            return

        loaded_count = 0
        missing_count = 0

        for (layer_idx, expert_id), cache_slot_idx in self.cached_experts.items():
            # 查找 CPU cache 中的权重
            cpu_weight = self._get_cpu_weight(layer_idx, expert_id)

            if cpu_weight is not None:
                # 加载到 GPU cache
                self._load_expert_to_cache_slot(cache_slot_idx, cpu_weight)
                loaded_count += 1
            else:
                missing_count += 1

        if PRINT_BUFFER_INIT:
            print(f"📥 Loaded {loaded_count} experts to GPU cache, {missing_count} missing")

    def load_from_cpu_cache(self, cpu_expert_cache: Dict[Tuple, Any]):
        """
        从 CPU cache 加载权重到 GPU cache

        在 ExpertCache 的 CPU cache 准备好后调用此方法
        """
        self.cpu_expert_cache = cpu_expert_cache
        self._load_experts_to_gpu_cache()

    def _get_cpu_weight(self, layer_idx: int, expert_id: int) -> Optional[Any]:
        """从 CPU cache 获取 expert 权重"""
        if self.structure == 'merged':
            # GPT-OSS: 打包的 tensor
            return self.cpu_expert_cache.get((layer_idx, expert_id))
        else:
            # Qwen3: 分离的 gate/up/down
            gate = self.cpu_expert_cache.get((layer_idx, expert_id, 'gate'))
            up = self.cpu_expert_cache.get((layer_idx, expert_id, 'up'))
            down = self.cpu_expert_cache.get((layer_idx, expert_id, 'down'))
            if gate is not None and up is not None and down is not None:
                return {'gate': gate, 'up': up, 'down': down}
            return None

    def _load_expert_to_cache_slot(self, cache_slot_idx: int, expert_weights):
        """将 expert 权重加载到指定的 cache slot"""
        buffer_info = self.cache_buffer_infos[cache_slot_idx]

        if isinstance(expert_weights, torch.Tensor):
            # 打包格式 (GPT-OSS)
            if 'gate_proj' in buffer_info:
                start_offset = buffer_info['gate_proj']['offset']
            elif 'gate_up_proj' in buffer_info:
                start_offset = buffer_info['gate_up_proj']['offset']
            else:
                raise ValueError("Cannot find gate_proj or gate_up_proj in buffer_info")

            total_size = expert_weights.numel()
            dst = self.gpu_memory_pool[start_offset:start_offset + total_size]
            dst.copy_(expert_weights, non_blocking=True)
        else:
            # 分离格式 (Qwen3)
            # Gate projection
            gate_view = self.gpu_memory_pool[
                buffer_info['gate_proj']['offset']:
                buffer_info['gate_proj']['offset'] + buffer_info['gate_proj']['size']
            ].view(buffer_info['gate_proj']['shape'])
            gate_view.copy_(expert_weights['gate'], non_blocking=True)

            # Up projection
            up_view = self.gpu_memory_pool[
                buffer_info['up_proj']['offset']:
                buffer_info['up_proj']['offset'] + buffer_info['up_proj']['size']
            ].view(buffer_info['up_proj']['shape'])
            up_view.copy_(expert_weights['up'], non_blocking=True)

            # Down projection
            down_view = self.gpu_memory_pool[
                buffer_info['down_proj']['offset']:
                buffer_info['down_proj']['offset'] + buffer_info['down_proj']['size']
            ].view(buffer_info['down_proj']['shape'])
            down_view.copy_(expert_weights['down'], non_blocking=True)

    # ========== 公共接口 ==========

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[int]:
        """
        查询 (layer_idx, expert_id) 是否在 GPU cache 中

        Returns:
            cache_slot_idx 如果在 cache 中，否则 None
        """
        return self.policy.lookup(layer_idx, expert_id)

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """检查 expert 是否在 cache 中（不更新统计和 LRU 顺序）"""
        return self.policy.contains(layer_idx, expert_id)

    def enable_stats(self):
        """启用 cache hit/miss 统计（用于 decode 阶段）"""
        self.policy.enable_stats()

    def disable_stats(self):
        """禁用 cache hit/miss 统计（用于 prefill 阶段）"""
        self.policy.disable_stats()

    def get_expert_view(self, cache_slot_idx: int) -> Dict:
        """获取指定 cache slot 的 buffer view"""
        return self.cache_buffer_infos[cache_slot_idx]

    def update_cache_from_buffers(
        self,
        layer_idx: int,
        swap_buffer_infos: List[Tuple[int, Dict, float]],  # (expert_id, buffer_info, logit_score)
        swap_memory_pool: torch.Tensor
    ):
        """
        从 swap buffer 更新 cache (topk_lru 和 tinylfu 策略)

        Args:
            layer_idx: 当前层索引
            swap_buffer_infos: swap buffer 中的 expert 信息列表
                              [(expert_id, buffer_info, logit_score), ...]
            swap_memory_pool: swap buffer 的 GPU 内存池
        """
        # 支持所有动态缓存策略
        if self.cache_policy_name not in ("lru", "lfu", "topk_lru", "tinylfu"):
            return

        # 检查 policy 是否支持 update_cache
        if not hasattr(self.policy, 'update_cache'):
            return

        with nvtx_range(f"Cache_Promotion_Layer{layer_idx}"):
            # 按 logit score 排序（从大到小）
            sorted_experts = sorted(swap_buffer_infos, key=lambda x: x[2], reverse=True)

            # 需要 (expert_id, max_logit)
            experts_to_update = [(eid, score) for eid, _, score in sorted_experts]

            # 让 policy 计算需要更新的 slots
            updates = self.policy.update_cache(layer_idx, experts_to_update)

            # Debug: 打印 cache 更新统计
            if PRINT_CACHE_UPDATE_DEBUG and updates:
                new_experts = [u[0] for u in updates if u[2] == -1]  # slot 未满时添加
                replaced_experts = [(u[0], u[2]) for u in updates if u[2] != -1]  # 替换
                if new_experts or replaced_experts:
                    print(f"📦 Layer {layer_idx:2d} [{self.cache_policy_name}] cache update: "
                          f"swap_in={len(swap_buffer_infos)}, "
                          f"new={len(new_experts)}, "
                          f"replaced={len(replaced_experts)}")
                    if replaced_experts:
                        for new_eid, old_eid in replaced_experts[:5]:  # 只显示前5个
                            new_logit = next((s for e, _, s in sorted_experts if e == new_eid), 0.0)
                            print(f"   e{old_eid} -> e{new_eid} (logit={new_logit:.4f})")

            # 执行 GPU-to-GPU memcpy
            for expert_id, slot_offset, evicted_expert_id in updates:
                # 找到这个 expert 在 swap_buffer_infos 中的 buffer_info
                src_buffer_info = None
                for eid, buf_info, _ in sorted_experts:
                    if eid == expert_id:
                        src_buffer_info = buf_info
                        break

                if src_buffer_info is None:
                    continue

                # 计算 cache slot 的 global index
                cache_slot_idx = layer_idx * self.slots_per_layer + slot_offset
                dst_buffer_info = self.cache_buffer_infos[cache_slot_idx]

                # GPU-to-GPU copy
                self._copy_buffer_to_cache_slot(
                    src_buffer_info, swap_memory_pool,
                    dst_buffer_info, self.gpu_memory_pool
                )

        # 更新 cached_experts 映射（兼容旧代码）
        self.cached_experts = self.policy.cached_experts

    def _copy_buffer_to_cache_slot(
        self,
        src_buffer_info: Dict, src_pool: torch.Tensor,
        dst_buffer_info: Dict, dst_pool: torch.Tensor
    ):
        """执行 GPU-to-GPU memcpy: swap buffer -> cache slot"""
        if self.structure == 'merged':
            # GPT-OSS: gate_up_proj + down_proj
            src_offset = src_buffer_info['gate_up_proj']['offset']
            dst_offset = dst_buffer_info['gate_up_proj']['offset']
            total_size = self.params_per_expert

            dst_pool[dst_offset:dst_offset + total_size].copy_(
                src_pool[src_offset:src_offset + total_size],
                non_blocking=True
            )
        else:
            # Qwen3: gate_proj + up_proj + down_proj
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                src_offset = src_buffer_info[proj]['offset']
                src_size = src_buffer_info[proj]['size']
                dst_offset = dst_buffer_info[proj]['offset']

                dst_pool[dst_offset:dst_offset + src_size].copy_(
                    src_pool[src_offset:src_offset + src_size],
                    non_blocking=True
                )

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取 cache 统计信息"""
        policy_stats = self.policy.get_stats()
        return {
            'total_slots': self.total_cache_slots,
            'slots_per_layer': self.slots_per_layer,
            'cache_memory_gb': self.actual_cache_memory / 1024**3,
            'cached_experts_count': len(self.cached_experts),
            'cache_policy': self.cache_policy_name,
            **policy_stats,
        }

    def print_cache_stats(self):
        """打印 cache 统计信息"""
        stats = self.get_cache_stats()
        print(f"\n📊 GPU Cache Statistics:")
        print(f"   Policy: {stats['cache_policy']}")
        print(f"   Slots: {stats['total_slots']} ({stats['slots_per_layer']} per layer)")
        print(f"   Memory: {stats['cache_memory_gb']:.2f} GB")
        print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"   Hit Rate: {stats['hit_rate']:.2%}")
        cache_rate = stats['slots_per_layer'] / self.num_experts
        alpha = stats['hit_rate'] / cache_rate if cache_rate > 0 else 0
        print(f"   Alpha (Hit Rate / Cache Rate): {alpha:.2f}")
        if 'expected_hit_rate' in stats:
            print(f"   Expected Hit Rate: {stats['expected_hit_rate']:.2%}")
        if 'cache_updates' in stats:
            print(f"   Cache Updates: {stats['cache_updates']}")
        if 'logit_threshold_percentile' in stats:
            print(f"   Logit Threshold Percentile: {stats['logit_threshold_percentile']}%")
