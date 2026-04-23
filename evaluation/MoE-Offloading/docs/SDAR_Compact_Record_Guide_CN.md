# SDAR 精简记录使用说明（中文）

本文档说明如何在 `tests/test_sdar_offloading.py` 中控制新的精简记录功能。新的记录逻辑只输出按 sample 聚合后的摘要，不再落盘逐 `block/step/layer` 的大 JSON。

## 一、记录功能概览

当前脚本把“正常评测”和“摘要记录”分开控制。

- 不记录时：
  - 使用 `--record-mode none --record-scope none`
  - 不执行专家摘要记录
  - 不执行延迟摘要记录
  - 不生成记录文件
  - 除了代码路径上的条件判断外，不做额外的数据收集

- 专家摘要记录：
  - 只统计 `decode` 阶段
  - `prefill` 不统计
  - `finalize` step 会统计
  - 统计口径是“这个 sample 的所有 decode step 的所有 layer 直接取平均”
  - 每个 sample 只输出 6 个平均值和 3 个比率：
    - 平均每个 decode layer 激活的不同专家数
    - 平均每个 decode layer 的 GPU cache 命中专家数
    - 平均每个 decode layer 的 prefetch 命中专家数
    - 平均每个 decode layer 的 CPU miss load 专家数
    - 平均每个 decode layer 在 miss load 之前 swap buffer 中已经可用的 prefetch 专家数
    - 平均每个 decode layer 的 GPU cache 换入专家数
    - 以及上面后三类各自相对激活专家数的比率

- 延迟摘要记录：
  - 只统计 `decode` 阶段中的 `denoise` step
  - `prefill` 不统计
  - `finalize` step 不统计
  - 每个 sample 只输出“平均一个 decode layer”的时间分解
  - 对 10 类操作分别输出：
    - 平均开始时间
    - 平均结束时间
    - 平均墙钟跨度
    - 平均累计时长
    - 平均占 layer 总时长的比例
  - 这 10 类操作分别是：
    - `attention`
    - `routing`
    - `current_layer_availability_check`
    - `current_layer_miss_load`
    - `next_layer_prefetch`
    - `reorder`
    - `gather`
    - `expert_compute`
    - `scatter`
    - `cache_promotion`

## 二、命令行参数

### 1. 记录类型

- `--record-mode none`
  - 不做记录
- `--record-mode experts`
  - 只记录专家摘要
- `--record-mode latency`
  - 只记录延迟摘要
- `--record-mode both`
  - 专家摘要和延迟摘要都记录

### 2. 记录样本范围

- `--record-scope none`
  - 不记录任何 sample
- `--record-scope all`
  - 记录当前评测范围内的全部 sample
- `--record-scope first_k --record-first-k K`
  - 只记录当前评测范围内前 `K` 个 sample

这里的“当前评测范围”指的是本次命令真正跑到的样本区间，也就是由：

- `--start-idx`
- `--num-samples`

共同决定的那一段。

### 3. 记录输出路径

- `--record-output /absolute/path/to/file.json`
  - 指定精简记录 JSON 的输出路径

如果启用了记录但不传 `--record-output`，脚本会默认写到：

- `profiles/sdar_record_summary_<record_mode>_start<start_idx>_n<num_samples>.json`

## 三、输出文件结构

精简记录文件是一个 JSON，大致包含三部分：

- `metadata`
  - 本次模型、数据集、生成参数、记录模式等元信息
- `samples`
  - 每个被记录 sample 的摘要
- `aggregate`
  - 所有被记录 sample 的总体平均摘要

### 1. 专家摘要字段

每个 sample 的 `expert_summary` 包含：

- `recorded_decode_layer_count`
- `average_active_unique_experts_per_decode_layer`
- `average_gpu_cache_hits_per_decode_layer`
- `average_prefetch_hits_per_decode_layer`
- `average_cpu_miss_loads_per_decode_layer`
- `average_prefetch_available_experts_before_miss_load_per_decode_layer`
- `average_gpu_cache_replacements_per_decode_layer`
- `gpu_cache_hit_ratio`
- `prefetch_hit_ratio`
- `cpu_miss_load_ratio`

### 2. 延迟摘要字段

每个 sample 的 `latency_summary` 包含：

- `recorded_denoise_layer_count`
- `average_decode_layer_total_ms`
- `operations`

其中 `operations.<op_name>` 下包含：

- `occurrence_layer_count`
- `occurrence_ratio`
- `average_start_ms`
- `average_end_ms`
- `average_wall_span_ms`
- `average_duration_ms`
- `average_share_of_layer_percent`

说明：

- `average_start_ms` / `average_end_ms` 是相对“平均 layer 起点”的位置
- 因此它可以直接拿来观察 10 类操作的串并行关系
- `next_layer_prefetch` 在最后一层不会出现，所以它的 `occurrence_ratio` 通常小于 `1.0`

## 四、单行命令示例

下面所有命令都写成单行形式。

### 1. 纯 baseline，不做任何记录

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 1 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode none --record-scope none --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_baseline_results.json
```

### 2. 记录全部 sample 的专家摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 10 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode experts --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_expert_summary_all.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_expert_results_all.json
```

### 3. 只记录前 3 个 sample 的专家摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 10 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode experts --record-scope first_k --record-first-k 3 --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_expert_summary_first3.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_expert_results_first3.json
```

### 4. 记录全部 sample 的延迟摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 10 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode latency --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_latency_summary_all.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_latency_results_all.json
```

### 5. 只记录前 2 个 sample 的专家摘要和延迟摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 10 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode both --record-scope first_k --record-first-k 2 --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_first2.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_results_first2.json
```

### 6. 记录当前评测范围内全部 sample 的专家摘要和延迟摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 10 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_all.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_results_all.json
```

```bash
conda activate sdar
python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --num-samples 5 --start-idx 0 --gen-length 128 --max-out-len 128 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_all.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_results_all.json
```
--min-free-memory-gib 和 --max-gpu-utilization 都是用来筛选“当前可用 GPU”的，作用发生在真正加载模型之前。
--min-free-memory-gib 控制的是显存空闲下限。脚本会查询候选 GPU 当前还剩多少空闲显存，只有空闲显存不少于这个值的卡才会被认为可用。比如设成 40，就表示至少要有 40 GiB 空闲显存。
--max-gpu-utilization 控制的是 GPU 计算利用率上限。脚本会看 nvidia-smi 里的当前 utilization.gpu，只有利用率不高于这个阈值的卡才会被认为可用。比如设成20，就表示只接受当前算力占用不超过 20% 的卡。

## 五、和 nsys 的关系

新的精简记录本身不依赖 `nsys`。它内部直接基于运行时聚合得到 sample 级摘要。

如果你还要另外配合 `nsys` 做外部 profiler 分析，可以继续手动加：

- `--enable-nvtx-ranges`
- `--nsys-use-cuda-profiler-api`

但这和精简摘要文件是两件独立的事：

- 精简摘要文件：用于批量对比 sample 级平均行为
- `nsys`：用于外部 profiler 时间线排查

## 六、建议用法

- 做纯延迟对比：
  - 用 `--record-mode none --record-scope none`

- 想看 offloading 命中情况：
  - 用 `--record-mode experts`

- 想看平均 layer 时间分解和串并行关系：
  - 用 `--record-mode latency`

- 想同时保留两类摘要：
  - 用 `--record-mode both`
