# SDAR MoE-Offloading 仿真器构建与测评全链路说明

本文档说明 `/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading` 中当前 SDAR MoE-Offloading 仿真器的组织方式、逐级调用链、模型构建逻辑、offloading runtime 执行逻辑、测评流程、输出指标，以及两个主要入口脚本 `test_sdar_offloading.py` 和 `run_sdar_offloading_benchmarks.py` 的区别。

本文默认模型为：

```text
/data_3/wly/.cache/huggingface/hub/models--JetLM--SDAR-30B-A3B-Chat-b32/snapshots/c351bbc37d240aa6871f167e8f92d694281b0c22
```

默认运行环境为：

```text
/data_3/wly/miniconda3/envs/sdar
```

## 一、仿真器总体目标

这个仿真器的目标不是重新实现 SDAR 解码，而是在保持 SDAR 原始 block diffusion 解码路径不变的前提下，把 SDAR-30B-A3B MoE 层替换成一个可控的 MoE-Offloading runtime，从而测量：

- SDAR 块间自回归、块内并行迭代解码下的端到端生成速度。
- MoE expert 从 CPU pinned cache 到 GPU swap buffer 的搬运开销。
- GPU expert cache 的命中率、替换次数、cache policy 效果。
- 下一层 expert prefetch 对 CPU miss load 的覆盖效果。
- 每个 benchmark 的 accuracy / pass@k / score 等 OpenCompass 指标。
- 每条样本的生成 latency、生成 token 数、tokens/s。
- 可选的 decode 阶段 expert 摘要和 layer 内时间分解摘要。

因此它是一个“保持真实 SDAR 推理主循环 + 替换 MoE 执行后端”的仿真和测评框架。

## 二、项目目录组织

核心目录如下：

```text
evaluation/MoE-Offloading/
├── tests/
│   ├── test_sdar_offloading.py
│   ├── run_sdar_offloading_benchmarks.py
│   ├── sdar_offloading_runner.py
│   └── sdar_offloading_utils.py
├── baseline/
│   ├── sdar_builder.py
│   ├── sdar_layers.py
│   ├── expert_cache.py
│   ├── expert_buffer_manager.py
│   ├── gpu_expert_cache.py
│   ├── sdar_runtime_trace.py
│   ├── nvtx_utils.py
│   └── debug_config.py
├── docs/
└── profiles/
```

OpenCompass 侧还有一个本地模型入口文件：

```text
evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe_offloading.py
```

它负责把 OpenCompass / HuggingFace 的 `from_pretrained()` 加载请求转交给 MoE-Offloading builder。

## 三、两个测评入口

### 3.1 `tests/test_sdar_offloading.py`

这是单 benchmark 入口。它适合调试、smoke test、单数据集全量评测、指定样本区间评测。

它本身很薄，只做三件事：

1. 创建 `argparse.ArgumentParser`。
2. 调用 `add_common_sdar_offloading_arguments(...)` 注册公共参数。
3. 调用 `run_sdar_offloading_evaluation(...)` 跑一个 benchmark。

默认特点：

- 默认样本数是 `1`。
- 默认 GPU 筛选较宽松：`min_free_memory_gib=40.0`、`max_gpu_utilization=20`。
- 默认不启用显存占位。
- 暴露 `--benchmark`，也暴露底层数据集参数 `--dataset-module`、`--dataset-var-name`、`--dataset-index`。
- 暴露 `--results-output`、`--record-output`，方便单次实验手工指定结果路径。

### 3.2 `tests/run_sdar_offloading_benchmarks.py`

这是多 benchmark suite 入口。它适合正式批量评测。

它也很薄，主要做三件事：

1. 额外注册 `--benchmarks` 参数。
2. 调用 `add_common_sdar_offloading_arguments(...)` 注册公共参数。
3. 调用 `run_sdar_offloading_benchmark_suite(...)` 依次跑多个 benchmark。

默认 benchmark 顺序：

```text
humaneval -> sanitized_mbpp -> gsm8k -> math
```

默认特点：

- 默认样本数是 `0`，表示每个 benchmark 跑当前 split 从 `start_idx` 开始的剩余全部样本。
- 默认 GPU 筛选更严格：`min_free_memory_gib=60.0`、`max_gpu_utilization=5`。
- 默认启用显存占位。
- 不暴露单个 benchmark 的 dataset args，因为它通过 `--benchmarks` 和内置 preset 自动解析。
- 不暴露 `--results-output`、`--record-output`，因为 suite 需要统一写入同一个时间戳实验目录。

## 四、公共 runner 的职责

真正的测评逻辑集中在：

```text
tests/sdar_offloading_runner.py
```

它负责：

- 注册公共 CLI 参数。
- 创建时间戳实验目录。
- 选择可用 GPU。
- 可选地占用 GPU 空闲显存，避免其他任务插入。
- 加载 OpenCompass 数据集配置、prompt template、postprocessor、evaluator。
- 构造 SDAR generation kwargs。
- 构造 `BD3withChatTemplate` 模型 wrapper。
- 跑样本循环。
- 调用 OpenCompass evaluator 计算指标。
- 收集 buffer manager / GPU cache 统计。
- 可选导出 compact expert / latency summary。
- 可选导出 GPU memory snapshot csv。
- 写入 `*_results.json`、`*_summary.json`、`experiment_config.json`。

## 五、从入口到 SDAR 解码的完整调用链

单 benchmark 入口调用链：

```text
tests/test_sdar_offloading.py
  -> run_sdar_offloading_evaluation(args)
    -> load_dataset_bundle(...)
    -> BD3withChatTemplate(...)
      -> _load_model(...)
        -> import configs.sdar_local_models.modeling_sdar_moe_offloading
        -> SDARMoeForCausalLM.from_pretrained(...)
          -> sdar_build_model(...)
    -> _run_samples(...)
      -> model_wrapper.generate_from_template([prompt], max_out_len=...)
        -> BD3withChatTemplate.generate(...)
          -> block_diffusion_generate(...)
            -> model(...)
              -> SDARMoeForCausalLM.forward(...)
                -> SDARMoeModel.forward(...)
                  -> SDAR decoder layer
                    -> attention
                    -> SDARSparseMoeWrapper.forward(...)
```

多 benchmark suite 入口调用链：

```text
tests/run_sdar_offloading_benchmarks.py
  -> run_sdar_offloading_benchmark_suite(args)
    -> 选择一次 GPU
    -> 可选地 suite 级显存占位一次
    -> for benchmark in benchmarks:
         -> copy args
         -> 设置 bench_args.benchmark
         -> _resolve_dataset_args(bench_args)
         -> run_sdar_offloading_evaluation(
              selected_gpu=同一张 GPU,
              memory_reservation_tensors=同一批占位 tensor,
              write_config=False,
              cleanup_model=True,
            )
    -> 写 all_benchmarks_summary.json
```

注意：suite 入口每个 benchmark 都会重新构建一次模型，并在该 benchmark 完成后清理模型。GPU 和显存占位张量在 suite 级别复用。

## 六、OpenCompass 数据集加载与 prompt 构造

数据集加载由 `tests/sdar_offloading_utils.py` 中的 `load_dataset_bundle(...)` 完成。

它接收：

- `dataset_module`
- `dataset_var_name`
- `dataset_index`

然后从 OpenCompass config 中取出对应 dataset 配置，实例化：

- dataset
- prompt template
- pred postprocessor
- dataset postprocessor
- evaluator

如果使用 `--benchmark`，则先通过内置 `BENCHMARK_PRESETS` 自动映射到 OpenCompass 数据集配置。

当前 preset 包括：

| benchmark | dataset module | 默认 split | 推荐生成长度 |
|---|---|---|---|
| `gsm8k` | `opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799` | `test` | 128 |
| `math` | `opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5` | `test` | 128 |
| `humaneval` | `opencompass.configs.datasets.humaneval.humaneval_gen` | `test` | 512 |
| `sanitized_mbpp` | `opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416` | `test` | 512 |

实际正式评测通常仍使用：

```text
--gen-length 4096 --max-out-len 4096
```

## 七、SDAR generation kwargs

runner 中 `_generation_kwargs(args)` 生成 SDAR 解码参数：

```python
{
    "mask_id": args.mask_id,
    "gen_length": args.gen_length,
    "block_length": args.block_length,
    "denoising_steps": args.denoising_steps or args.block_length,
    "temperature": args.temperature,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "remasking": args.remasking,
    "threshold": args.threshold,
}
```

默认值对应 SDAR-30B-A3B-Chat-b32 常用配置：

```text
mask_id=151669
gen_length=4096
block_length=32
denoising_steps=32
temperature=1.0
top_k=1
top_p=1.0
remasking=low_confidence
threshold=0.95
```

这些参数会传给 `BD3withChatTemplate.generate()`，最终进入 OpenCompass 侧的 `block_diffusion_generate()`。

## 八、SDAR 解码主循环保持不变

offloading 仿真器没有改 SDAR 的解码范式。实际生成仍然走：

```text
BD3withChatTemplate.generate()
  -> block_diffusion_generate()
```

SDAR 解码逻辑仍然是：

1. 将 prompt token 放在序列前部。
2. 后续生成区域初始化为 `<|MASK|>`。
3. 构造块级下三角 attention mask：
   - 块间自回归。
   - 块内双向可见。
4. prompt 完整块先做 prefill，并把 KV cache 写入 `past_key_values`。
5. 逐块生成后续内容。
6. 每个 block 内做 `denoising_steps` 次并行去噪。
7. 每步对当前 block 中仍为 mask 的位置同时预测 logits。
8. 用 confidence 和 `threshold` 选择本步接受哪些 token。
9. 未接受的位置保持 mask，进入下一步重新预测。
10. 当前 block 全部填满后，用 `store_kv=True` 提交该 block 的 KV cache。
11. 检查 stop words，必要时提前结束。

因此 offloading runtime 只影响每一层 MoE 的执行方式，不改变 SDAR 的块级扩散推理语义。

## 九、offloading 版 modeling 入口

OpenCompass 通过 `local_modeling_module` 选择模型实现。offloading 测评默认使用：

```text
configs.sdar_local_models.modeling_sdar_moe_offloading
```

对应文件：

```text
evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe_offloading.py
```

这个文件做两件关键事情：

1. 从默认 `modeling_sdar_moe.py` 导入 SDAR 基础类，例如 `SDARMoeModel`、`SDARMoeAttention`、`SDARMoeSparseMoeBlock`。
2. 定义新的 `SDARMoeForCausalLM.from_pretrained()`，把加载请求转交给 `baseline.sdar_builder.sdar_build_model(...)`。

也就是说，权重路径仍然是 HuggingFace snapshot，但构建模型的控制权交给了 MoE-Offloading builder。

## 十、offloading 模型构建逻辑

模型构建由：

```text
baseline/sdar_builder.py
```

中的 `sdar_build_model(...)` 完成。

核心步骤如下。

### 10.1 monkey patch 原始 SDAR 类

builder 会对原始 SDAR 类做局部 monkey patch：

- 给 attention forward 加 NVTX / timing range 包裹。
- 修改 `SDARMoeDecoderLayer.__init__`。
- 修改 `SDARMoeSparseMoeBlock.__init__`。

关键目的：构建模型骨架时不要创建完整的 GPU expert 模块。

原始 SDAR MoE 层会包含 `num_experts=128` 个 expert，如果直接加载到 GPU，会占用大量 HBM。offloading builder 会：

1. 记录原始 expert 数 `original_num_experts`。
2. 临时把 `config.num_experts = 0`。
3. 把真实 expert 数保存到 `config._target_experts`。
4. 创建模型骨架。
5. 后续用自定义 wrapper 替换 sparse MoE 层。

### 10.2 加载 dense 权重和 expert 权重

`_load_all_weights_unified(...)` 会读取 safetensors index。

对于每个权重文件：

- 非 expert 权重加载到 GPU 模型主体中。
- expert 权重不加载到模型主体，而是交给 `ExpertCache._process_weights_batch(...)`。

对当前 SDAR-30B-A3B，它会被识别为 Qwen3-style separate expert 结构：

- `gate_proj`
- `up_proj`
- `down_proj`

这些 expert 权重被组织进 CPU pinned cache，后续按需搬到 GPU。

### 10.3 初始化 ExpertCache

builder 创建：

```python
ExpertCache(
    state_path,
    device,
    original_num_experts,
    enable_gpu_cache=...,
    cache_policy=...,
    topk_lru_logit_percentile=...,
    cache_slots_per_layer=...,
)
```

这里 `original_num_experts` 同时作为临时 swap buffer 的 buffer 数量。对于 SDAR-30B-A3B，通常是 128。

### 10.4 可选初始化 GPU expert cache

如果 `enable_gpu_cache=True`，则调用：

```python
expert_cache.init_gpu_cache()
```

它会创建 `GPUExpertCacheManager`，按照每层 `cache_slots_per_layer` 分配 GPU cache slot，并从 CPU expert cache 预加载初始 expert。

默认：

```text
cache_policy=topk_lru
cache_slots_per_layer=16
topk_lru_logit_percentile=90.0
```

### 10.5 替换 MoE 层

最后 builder 遍历所有 sparse layer：

```python
layer.mlp = SDARSparseMoeWrapper(config, layer_idx, gate, expert_cache)
```

替换后：

- attention、RMSNorm、lm_head、rotary embedding 等 dense 模块仍在 GPU。
- MoE expert 权重不再作为原始 `ModuleList` 常驻模型主体。
- 每次 MoE forward 时通过 offloading runtime 查 cache、搬权重、计算。

## 十一、ExpertCache 组织逻辑

`baseline/expert_cache.py` 负责 CPU expert cache 和 GPU buffer manager 的统一管理。

它主要包含：

- `simple_expert_cache`
  - CPU 端 expert 权重缓存。
  - 对 SDAR/Qwen3-style expert，key 形如 `(layer_idx, expert_id, "gate")`、`(layer_idx, expert_id, "up")`、`(layer_idx, expert_id, "down")`。
- `cpu_pinned_storage`
  - CPU pinned memory 预分配区域。
  - 用于提高 CPU->GPU non-blocking copy 的可行性。
- `buffer_manager`
  - 管理 GPU 临时 swap buffer。
- `gpu_cache_manager`
  - 可选的 GPU expert cache。

ExpertCache 的核心接口是：

```python
batch_load_experts_continuous(layer_idx, expert_indices, router_logits)
```

它会把当前层需要的 active experts 准备到 GPU 上，并返回：

```text
expert_id -> virtual_idx
```

这里的 `virtual_idx` 可能指向：

- 临时 swap buffer。
- GPU cache slot。

上层 MoE wrapper 不需要关心 expert 来自哪里，只要拿到 `virtual_idx` 后通过 buffer manager 取出可计算的 GPU view。

## 十二、GPU 临时 buffer 管理

`baseline/expert_buffer_manager.py` 负责 GPU swap buffer。

它在初始化时分配一个连续 GPU memory pool。对 Qwen3-style expert，每个 expert buffer 内部 layout 是：

```text
gate_proj
up_proj
down_proj
```

核心功能：

- `load_experts_for_current_layer(...)`
  - 当前层 expert 准备主入口。
- `get_expert_view_for_computation(virtual_idx)`
  - 根据 virtual idx 返回 expert 权重 view。
- `prefetch_expert(layer_idx, expert_id, weights)`
  - 在 prefetch stream 上把下一层预测 expert 搬入空闲 buffer。
- `get_stats()`
  - 返回 GPU cache hit/miss、prefetch hit、compute load 等统计。

当前层 expert 准备顺序非常重要：

1. 释放上一层临时 buffer。
2. 检查 GPU cache。
3. 对 GPU cache miss 的 expert 检查 prefetch buffer。
4. 对仍未命中的 expert，从 CPU pinned cache 加载到 GPU 临时 buffer。
5. 对刚加载的 expert，根据 cache policy 做可选 cache promotion。
6. 记录 `last_load_trace`，供 compact expert summary 使用。

## 十三、GPU Expert Cache

`baseline/gpu_expert_cache.py` 负责 GPU expert cache。

支持策略：

- `static`
  - 每层固定缓存前 N 个 expert。
- `lru`
  - 每层独立 LRU。
- `lfu`
  - 每层独立 LFU。
- `topk_lru`
  - LRU + router logit percentile 准入控制。
- `tinylfu`
  - TinyLFU + S-LRU 分段。

默认策略是 `topk_lru`。

GPU cache 的容量由：

```text
cache_slots_per_layer * num_layers
```

决定。对于 SDAR-30B-A3B：

```text
num_layers = 48
cache_slots_per_layer = 16
total_cache_slots = 768
```

每个 cache slot 保存一个 expert 的完整权重。对 Qwen3-style expert，就是 gate/up/down 三个矩阵。

MoE forward 时，当前层 active expert 先调用：

```python
gpu_cache_manager.lookup(layer_idx, expert_id)
```

命中则直接返回 cache slot；未命中才进入 prefetch / CPU miss load 路径。

## 十四、SDAR MoE wrapper 执行逻辑

MoE 层替换为：

```text
baseline/sdar_layers.py::SDARSparseMoeWrapper
```

其核心 forward 逻辑如下。

### 14.1 Routing

输入 hidden states 形状：

```text
[batch, sequence_length, hidden_dim]
```

会先 flatten 成：

```text
[batch * sequence_length, hidden_dim]
```

然后过当前层 gate：

```python
router_logits_tensor = self.gate(hidden_states_flat)
full_routing_weights = softmax(router_logits_tensor)
routing_weights, selected_experts = topk(full_routing_weights, self.top_k)
```

当前 SDAR-30B-A3B 通常是：

```text
num_experts = 128
top_k = 8
```

### 14.2 下一层 expert 预测和 prefetch

如果启用 `PREFETCH_ENABLED`，且当前层不是最后一层，则 wrapper 会取下一层 gate：

```python
next_layer_gate = GateRegistry.get_gate(self.layer_idx + 1)
```

用当前 hidden states 预测下一层可能使用的 experts，默认取前 `PREFETCH_TOPK=4`。

这些 expert id 会在当前层 expert 准备之后，通过 `_parallel_prefetch(...)` 提交到 prefetch stream。

prefetch 的作用是：尽量让下一层需要的 expert 在下一层真正执行前已经在 GPU swap buffer 中。

### 14.3 统计当前层 active experts

wrapper 会对 `selected_experts` 做 bincount，得到当前层真正被至少一个 token 选中的 expert 集合：

```python
active_expert_ids = torch.where(expert_counts > 0)[0].tolist()
```

这和原始 SDAR MoE 实现有明显差异。原始实现通常遍历全部 128 个 expert；offloading wrapper 只处理 active experts。

### 14.4 准备 active experts

wrapper 调用：

```python
expert_to_buffer_mapping = expert_cache.batch_load_experts_continuous(
    self.layer_idx,
    expert_indices,
    router_logits,
)
```

返回：

```text
expert_id -> virtual_idx
```

每个 expert 可能来自：

- GPU cache。
- prefetch buffer。
- CPU pinned cache 刚搬到 GPU temp buffer。

### 14.5 Reorder / Gather

为了做 batched expert compute，wrapper 会把 token 按 expert id 排序：

```python
sorted_experts, perm = torch.sort(flat_experts)
sorted_tokens = token_indices[perm]
sorted_ranks = k_ranks[perm]
```

然后 gather：

```python
all_input_states = hidden_states_flat[sorted_tokens]
all_routing_weights = routing_weights[sorted_tokens, sorted_ranks]
```

### 14.6 Batched Expert Compute

如果 `BMM_ENABLED=True`，则会把所有 active experts 的输入 pad 到同一个 `max_tok`，形成：

```text
batched_inputs:  [num_active_experts, max_tok, hidden_dim]
gate_w:          [num_active_experts, intermediate_dim, hidden_dim]
up_w:            [num_active_experts, intermediate_dim, hidden_dim]
down_w:          [num_active_experts, hidden_dim, intermediate_dim]
```

然后用 `torch.bmm` 批量计算：

```python
gate_out = silu(bmm(batched_inputs, gate_w^T))
up_out = bmm(batched_inputs, up_w^T)
exp_out = bmm(gate_out * up_out, down_w^T)
```

这比原始 SDAR 中对 128 个 expert 逐个 Python 循环的方式更适合仿真 offloading runtime。

### 14.7 Scatter

最后用 `scatter_add_` 把 expert outputs 加回 token 维度：

```python
final_hidden_states.scatter_add_(...)
```

输出 reshape 回：

```text
[batch, sequence_length, hidden_dim]
```

并返回：

```python
return output_tensor, router_logits
```

## 十五、测评样本循环

样本循环在 `_run_samples(...)` 中。

对每个 sample：

1. 从 OpenCompass dataset 取一条 entry。
2. 用 prompt template 生成 prompt。
3. 如果启用 compact record，则调用 `begin_sample(...)`。
4. 可选记录 sample 前 GPU memory snapshot。
5. 调用：

```python
output_text = model_wrapper.generate_from_template([prompt], max_out_len=max_out_len)[0]
```

6. 可选调用 `end_sample(...)` 完成本 sample trace 聚合。
7. tokenizer 统计生成 token 数。
8. pred postprocessor 处理输出。
9. dataset postprocessor 处理 reference。
10. 保存 sample 结果：
    - raw prediction
    - processed prediction
    - reference
    - latency
    - generated token count
    - tokens/s
11. 可选记录 sample 后 GPU memory snapshot。

样本循环结束后调用：

```python
score_predictions(...)
```

得到 benchmark 评测指标。

## 十六、进度显示

当前 runner 使用 `tqdm` 显示进度。

单 benchmark：

- 显示当前数据集 sample 进度条。

多 benchmark suite：

- 外层显示 benchmark suite 进度条。
- 内层显示当前 benchmark 的 sample 进度条。

默认不会逐 sample 打印 `sample_idx`、latency、token 数等冗余信息。只有显式开启：

```text
--verbose-samples
```

才会打印逐样本详情。

每个数据集结束后仍会打印有限的汇总信息，例如：

- evaluation
- aggregate
- buffer manager stats
- GPU cache stats
- results path
- summary path

## 十七、compact record 逻辑

compact record 由：

```text
baseline/sdar_runtime_trace.py
```

负责。

通过 CLI 控制：

```text
--record-mode none|experts|latency|both
--record-scope none|all|first_k
--record-first-k K
```

### 17.1 experts 记录

只统计 decode 阶段。每个 decode layer 聚合：

- active expert 数。
- GPU cache hit 数。
- prefetch hit 数。
- CPU miss load 数。
- miss load 前 prefetch buffer 中可用 expert 数。
- GPU cache replacement 数。

最终输出 sample 级平均和全局 aggregate。

### 17.2 latency 记录

只统计 decode 阶段的 denoise step，不统计 prefill 和 finalize。

记录的操作类别包括：

- attention
- routing
- current_layer_availability_check
- current_layer_miss_load
- next_layer_prefetch
- reorder
- gather
- expert_compute
- scatter
- cache_promotion

它用 CUDA event 记录时间，因此能反映不同 stream 上操作的重叠关系。注意这些操作时间不是互斥墙钟片段，不能简单相加。

## 十八、输出文件

单 benchmark 常见输出：

```text
profiles/<timestamp>_sdar_offloading_single[_run_name]/
├── experiment_config.json
├── <benchmark>_results.json
├── <benchmark>_summary.json        # 启用 record 时
└── <benchmark>_memory.csv          # 启用 --track-gpu-memory 时
```

多 benchmark suite 常见输出：

```text
profiles/<timestamp>_sdar_offloading_suite[_run_name]/
├── experiment_config.json
├── all_benchmarks_summary.json
├── humaneval_results.json
├── humaneval_summary.json
├── sanitized_mbpp_results.json
├── sanitized_mbpp_summary.json
├── gsm8k_results.json
├── gsm8k_summary.json
├── math_results.json
└── math_summary.json
```

`*_results.json` 中主要字段：

- `model_path`
- `local_modeling_module`
- `benchmark`
- `gpu`
- `build_time_s`
- `generation_kwargs`
- `gpu_memory_reservation`
- `offloading`
- `dataset`
- `recording`
- `aggregate`
- `evaluation`
- `samples`
- `buffer_manager`
- `gpu_cache`

`aggregate` 统计端到端生成速度：

- sample count
- total generation latency
- total generated tokens
- average latency
- overall tokens/s

`buffer_manager` 统计 offloading runtime：

- gpu cache hits
- gpu cache misses
- cache hit rate
- prefetch hits
- compute loads
- total experts loaded

`gpu_cache` 统计 GPU expert cache：

- total slots
- slots per layer
- cache memory GB
- cached experts count
- cache policy
- hits
- misses
- hit rate
- cache updates

## 十九、显存占位逻辑

runner 支持显存占位：

```text
--reserve-gpu-memory
--reserve-gpu-memory-stage pre_build|post_build
--reserve-free-memory-gib 24
```

单 benchmark 默认不启用。suite 默认启用。

占位逻辑是：

1. 选择 GPU 后查询当前 free memory。
2. 按目标剩余空闲显存计算需要占用多少。
3. 分配若干 `torch.empty(..., dtype=torch.uint8, device=cuda)` tensor。
4. 这些 tensor 不参与计算，只持有 HBM。
5. 进程退出或释放引用后自动释放。

目的：减少其他任务在评测期间进入同一张 GPU 造成干扰。

## 二十、两个入口的核心区别

| 维度 | `test_sdar_offloading.py` | `run_sdar_offloading_benchmarks.py` |
|---|---|---|
| 定位 | 单 benchmark 入口 | 多 benchmark suite 入口 |
| 典型用途 | 调试、smoke run、单数据集全量评测 | 正式批量评测 |
| 默认样本数 | `1` | `0`，即每个 benchmark 跑剩余全部样本 |
| benchmark 控制 | `--benchmark` 选择一个 benchmark | `--benchmarks` 选择多个 benchmark |
| dataset args | 暴露 `--dataset-module`、`--dataset-var-name`、`--dataset-index` | 不暴露，使用 preset |
| 输出路径控制 | 可显式传 `--results-output`、`--record-output` | 统一写入 suite run dir |
| GPU 筛选默认 | `40GiB` free、`20%` util | `60GiB` free、`5%` util |
| 显存占位默认 | 关闭 | 开启 |
| 模型构建次数 | 一次 | 每个 benchmark 一次 |
| GPU 选择 | 每次单独选择 | suite 开始时选择一次 |
| suite 汇总 | 无 | 写 `all_benchmarks_summary.json` |

一句话总结：

- `test_sdar_offloading.py` 是单项实验入口，适合精确控制某一个 benchmark。
- `run_sdar_offloading_benchmarks.py` 是批量总控入口，适合在同一张 GPU 和同一个实验目录下连续评测多个 benchmark，并生成总汇总。

## 二十一、理解仿真结果时的注意事项

1. `overall_tokens_per_second` 是端到端样本生成吞吐，包含 SDAR block diffusion 的全部 decode 过程。
2. `build_time_s` 不计入 sample latency aggregate。
3. `buffer_manager` 的 hit/miss 统计来自实际 MoE layer forward 的 expert 准备路径。
4. compact latency summary 中不同操作可能跨 stream 重叠，不能把各项 duration 简单相加成 layer wall time。
5. offloading runtime 不只是“把原始 MoE 加上 CPU->GPU copy”，它同时替换了 MoE 执行方式：
   - 只处理 active experts。
   - 使用 reorder/gather/batched compute/scatter。
   - 使用 prefetch stream。
   - 使用 GPU expert cache。
6. 因此对比 pure SDAR 时，要区分：
   - 权重常驻 GPU 的优势。
   - 原始 MoE Python expert loop 的开销。
   - offloading copy 的开销。
   - offloading runtime 对 MoE compute 的优化。

## 二十二、最小运行示例

单 benchmark smoke run：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark gsm8k --split test --start-idx 0 --num-samples 1 --gen-length 128 --max-out-len 128 --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --record-mode none --record-scope none
```

多 benchmark smoke run：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/run_sdar_offloading_benchmarks.py --benchmarks gsm8k,math --split test --start-idx 0 --num-samples 2 --gen-length 128 --max-out-len 128 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --record-mode none --record-scope none --run-name smoke_gsm8k_math
```

正式 suite 评测：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/run_sdar_offloading_benchmarks.py --split test --start-idx 0 --num-samples 0 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode both --record-scope all --run-name suite_both
```
