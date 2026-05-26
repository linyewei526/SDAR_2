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

## 七、遍历整个数据集时怎么写

如果你要遍历“整个数据集 split”，当前脚本不能继续写成 `--num-samples 10 --start-idx 0`。原因很直接：

- `--start-idx` 决定起始样本
- `--num-samples` 决定本次真正要跑多少条
- 脚本会检查 `start_idx + num_samples <= len(dataset[split])`

所以“遍历整个 split”的写法是：

- `--start-idx 0`
- `--num-samples` 设为当前 split 的实际样本总数

对当前默认数据集配置：

- `--dataset-module opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799`
- `--dataset-var-name gsm8k_datasets`
- `--dataset-index 0`

其样本数是：

- `test` split: `1319`
- `train` split: `7473`

下面给出直接可用的单行命令。

### 1. 遍历整个 GSM8K test split，不做记录

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --split test --start-idx 0 --num-samples 1319 --gen-length 128 --max-out-len 128 --record-mode none --record-scope none --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_test_all_results.json
```

### 2. 遍历整个 GSM8K test split，记录全部 sample 的专家摘要和延迟摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --split test --start-idx 0 --num-samples 1319 --gen-length 128 --max-out-len 128 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_test_all_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_test_all_results.json
```

### 3. 遍历整个 GSM8K train split，不做记录

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --split train --start-idx 0 --num-samples 7473 --gen-length 128 --max-out-len 128 --record-mode none --record-scope none --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_train_all_results.json
```

### 4. 遍历整个 GSM8K train split，记录全部 sample 的专家摘要和延迟摘要

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --candidate-gpus 0,1,2,3 --min-free-memory-gib 40 --max-gpu-utilization 20 --split train --start-idx 0 --num-samples 7473 --gen-length 128 --max-out-len 128 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_train_all_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_train_all_results.json
```

如果你后面换了别的数据集或别的 `split`，原则不变：

- `--start-idx 0`
- `--num-samples` 改成该 `split` 的真实长度

## 八、`--gen-length` 和 `--max-out-len` 的区别

这两个参数名字看起来相近，但在当前这条 SDAR 自定义生成链路里，职责并不完全一样。

### 1. `--gen-length`

`--gen-length` 会进入 `generation_kwargs['gen_length']`，最终直接传给：

- `opencompass/opencompass/models/huggingface_bd3.py` 里的 `block_diffusion_generate(...)`

它决定的是：

- SDAR 解码阶段预留多少生成长度
- `num_blocks = (prompt_len + gen_length + block_length - 1) // block_length`
- 也就是 block diffusion 实际最多会往后生成多少 token 空间

因此在当前实现里，**真正控制 SDAR 实际生成上限的核心参数是 `--gen-length`**。

### 2. `--max-out-len`

在 `tests/test_sdar_offloading.py` 里：

- `max_out_len = args.max_out_len or args.gen_length`

然后传给：

- `model_wrapper.generate_from_template(..., max_out_len=max_out_len)`

在 `huggingface_bd3.py` 里，`max_out_len` 主要用于：

- 在 `mode == 'mid'` 时，为 prompt 截断预留空间
- 写入 `generation_kwargs['max_new_tokens']`

但当前这条本地 SDAR 路径最终调用的是自定义的 `block_diffusion_generate(...)`，它实际消费的是 `gen_length`，不是 `max_new_tokens`。

所以对当前项目这条 SDAR 链路来说，可以把它理解成：

- `--gen-length`: 真正的 SDAR 生成长度控制参数
- `--max-out-len`: 外层接口参数，默认最好和 `--gen-length` 保持一致

### 3. 实际建议

为了避免歧义，当前项目里建议始终写成相同值，例如：

- `--gen-length 128 --max-out-len 128`

如果两者写成不同值，在当前本地 SDAR 实现里，应该优先按 `--gen-length` 理解实际生成上限。

## 九、4 个 benchmark 的调用方式

当前 `tests/test_sdar_offloading.py` 已经支持通过 `--benchmark` 快捷切换到 4 个常用 benchmark：

- `gsm8k`
- `math`
- `humaneval`
- `sanitized_mbpp`

它的作用是自动填充对应的：

- `--dataset-module`
- `--dataset-var-name`
- `--dataset-index`

也就是说，你不再需要手动写一长串 OpenCompass 数据集模块路径。  
但生成相关参数仍然由你自己控制，尤其是：

- `--gen-length`
- `--max-out-len`

短 smoke run 可以按任务类型缩短生成长度：

- 数学题类：`gsm8k`、`math` 使用 `128`
- 代码生成类：`humaneval`、`sanitized_mbpp` 使用 `512`

下面正式 benchmark 命令统一使用 `--gen-length 4096 --max-out-len 4096`，与 OpenCompass SDAR 配置保持一致。

下面给出 4 个 benchmark 的 test split 全量单行命令示例。为了让结果结构统一，下面都用：

- `--record-mode both --record-scope all`

如果你只想测纯 baseline 延迟，把它们改成：

- `--record-mode none --record-scope none`

### 1. GSM8K

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark gsm8k --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_results.json
```

### 2. MATH

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark math --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --start-idx 0 --num-samples 500 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_math_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_math_results.json
```

### 3. HumanEval

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark humaneval --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --start-idx 0 --num-samples 164 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_humaneval_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_humaneval_results.json
```

说明：

- `humaneval` 的评估会复用 `evaluation/opencompass/human-eval` 这套执行评测后端
- 当前脚本已经补上了 `test_set` 传递，所以可以直接走 OpenCompass 原生的 `HumanEvalEvaluator`

### 4. Sanitized MBPP

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark sanitized_mbpp --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --start-idx 0 --num-samples 257 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_sanitized_mbpp_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_sanitized_mbpp_results.json
```

说明：

- `sanitized_mbpp` 当前走的是 OpenCompass 的 `MBPPEvaluator`
- 如果本地数据缓存目录空间不足，首次构造数据集时可能会报磁盘空间不足，这不是 MoE-Offloading 代码路径的问题，而是本地缓存空间问题

如果你不想用 `--benchmark` 快捷方式，仍然可以手动指定：

- `--dataset-module`
- `--dataset-var-name`
- `--dataset-index`

两种方式是等价的；`--benchmark` 只是把这 3 个数据集参数预先替你填好。

## 十、带 GPU 显存占位的 4 个 benchmark 调用方式

为了避免其他任务在评测过程中进入同一张 GPU 干扰效率测试，`tests/test_sdar_offloading.py` 现在支持显存占位参数：

- `--reserve-gpu-memory`
- `--reserve-gpu-memory-stage pre_build`
- `--reserve-free-memory-gib 24`

含义是：脚本选中 GPU 后、模型构建前，先占住大部分空闲显存，只留下约 `24GiB` 给当前 SDAR MoE-Offloading 测试使用。实现里还会额外保留约 `0.5GiB` allocator margin，避免占位本身卡在边界上 OOM。占位张量只持有 HBM，不参与计算；进程退出后自动释放。实际占用信息会写入结果 JSON 的 `gpu_memory_reservation` 字段。

如果后续出现 OOM，可以把 `--reserve-free-memory-gib 24` 调大到 `28` 或 `32`。

### 1. GSM8K，占位后测试

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/
python tests/test_sdar_offloading.py --benchmark gsm8k --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_reserved_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_gsm8k_reserved_results.json
```

### 2. MATH，占位后测试

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark math --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --start-idx 0 --num-samples 500 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_math_reserved_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_math_reserved_results.json
```

### 3. HumanEval，占位后测试

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark humaneval --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --start-idx 0 --num-samples 164 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_humaneval_reserved_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_humaneval_reserved_results.json
```

### 4. Sanitized MBPP，占位后测试

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading && /data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark sanitized_mbpp --split test --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --start-idx 0 --num-samples 257 --gen-length 4096 --max-out-len 4096 --record-mode both --record-scope all --record-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_sanitized_mbpp_reserved_summary.json --results-output /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_sanitized_mbpp_reserved_results.json
```
