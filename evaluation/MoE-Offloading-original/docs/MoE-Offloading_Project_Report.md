# MoE-Offloading 项目详解与使用报告

本文基于以下代码目录整理：

- 项目根目录：`/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading`
- 整理时间：`2026-04-18`

目标是把这个项目当前的组织方式、MoE offloading 仿真逻辑、延迟/缓存开销评估方式、以及实际可执行的使用方法讲清楚，方便下一步把 SDAR 的块级解码接进来。

---

## 1. 一句话结论

这个项目本质上不是一个通用“包”，而是一套面向 **自回归 MoE 模型单卡推理** 的实验型仿真框架。它的核心做法是：

1. 用 Hugging Face/Transformers 正常构建模型骨架，但 **不在 GPU 上真正实例化 experts**。
2. 把所有 expert 权重预处理后常驻在 **CPU 端 pinned memory**。
3. 在每层 MoE 前向时，根据 router 选中的 active experts，按需把对应 expert 权重搬到 GPU 的 **临时 swap buffer**。
4. 可选地在 GPU 上额外维持一层 **专家缓存（GPU cache）**，支持 `static / lru / lfu / topk_lru / tinylfu`。
5. 可选地用当前层 router 输入去猜下一层可能会用到的 experts，提前做 **prefetch**。
6. 整个评估目前是围绕 **AR 解码** 写的：prefill 一次，之后每次只解 1 个 token，统计整体 decode 时间和 GPU cache 命中情况。

因此它更准确地说是：

- 一个 **专家权重 CPU->GPU demand loading + GPU cache + prefetch** 的 MoE 推理实验器；
- 而不是完整意义上的“边端系统仿真器”。

---

## 2. 项目目录组织

项目根目录下的关键结构如下：

```text
MoE-Offloading/
├── baseline/
│   ├── qwen3_builder.py
│   ├── gptoss_builder.py
│   ├── qwen3_layers.py
│   ├── gptoss_layers.py
│   ├── expert_cache.py
│   ├── expert_buffer_manager.py
│   ├── gpu_expert_cache.py
│   ├── debug_config.py
│   └── utils.py
├── tests/
│   ├── test_baseline.py
│   ├── baseline_utils.py
│   └── adapeagle_backup/
│       ├── baseline_benchmark.py
│       └── theoretical_tps.py
├── benchmark/
│   ├── gsm8k/
│   ├── openai_humaneval/
│   └── CNN-DM.parquet
├── run_BS.sh
├── requirements.txt
└── docs/
```

各目录职责如下。

### 2.1 `baseline/`

这里是核心实现层，真正决定 offloading 仿真怎么做。

- `qwen3_builder.py`
  负责构建 Qwen3-MoE 版本的 offloading 模型。
- `gptoss_builder.py`
  负责构建 GPT-OSS 版本的 offloading 模型。
- `qwen3_layers.py`
  Qwen3 的自定义 MoE wrapper，包含 router、active expert 收集、expert load、prefetch、BMM 计算等逻辑。
- `gptoss_layers.py`
  GPT-OSS 的对应版本。
- `expert_cache.py`
  统一的专家缓存入口。负责：
  - 从 `config.json` 推断模型类型；
  - 把 expert 权重组织到 CPU cache；
  - 初始化 GPU cache；
  - 对外提供“按层批量加载 experts”的统一接口。
- `expert_buffer_manager.py`
  统一管理 GPU 临时 swap buffer、prefetch stream、当前层映射、prefetch 命中等。
- `gpu_expert_cache.py`
  GPU 侧专家缓存及缓存策略实现。
- `debug_config.py`
  各种实验开关，例如 `PREFETCH_ENABLED`、`BMM_ENABLED`、`PRELAUNCH_ENABLED`、打印选项等。

### 2.2 `tests/`

这里不是传统意义上的单元测试，而是运行入口和评估脚本。

- `test_baseline.py`
  当前主入口。根据 `--model` 选择 Qwen3 或 GPT-OSS，解析参数后调用统一测试基类。
- `baseline_utils.py`
  真正的通用评测流程实现：
  - 构建模型；
  - 读取 benchmark prompt；
  - 进行 prefill + AR decode；
  - 统计 decode TPS；
  - 打印 GPU cache 命中率。
- `adapeagle_backup/baseline_benchmark.py`
  旧版脚本，仍然很有参考价值。它比当前主脚本更“可移植”，因为 benchmark 路径使用的是项目内相对路径。
- `adapeagle_backup/theoretical_tps.py`
  一个纯理论估算脚本，用 PCIe 带宽粗略估计“若每个 token 都需要从 CPU 搬全部激活 experts”时的 TPS 上限。

### 2.3 `benchmark/`

存放本地 benchmark 数据文件。

- `gsm8k/main/test-00000-of-00001.parquet`
- `openai_humaneval/openai_humaneval/test-00000-of-00001.parquet`
- `CNN-DM.parquet`

这些文件在当前仓库中是存在的。

### 2.4 `run_BS.sh`

一个批量跑 baseline 的 shell 脚本，会分别跑：

- Qwen3-MoE on GSM8K / HumanEval / CNN-DM
- GPT-OSS on GSM8K / HumanEval / CNN-DM

但它依赖当前代码中的默认路径，因此在你这台机器上不能直接无修改运行。

---

## 3. 当前主运行链路

主入口链路如下：

```text
tests/test_baseline.py
    -> tests/baseline_utils.py::BaselineTestBase.run_test()
        -> baseline/qwen3_builder.py 或 baseline/gptoss_builder.py
            -> baseline/expert_cache.py
                -> baseline/expert_buffer_manager.py
                -> baseline/gpu_expert_cache.py
            -> baseline/qwen3_layers.py 或 baseline/gptoss_layers.py
        -> tokenizer.apply_chat_template()
        -> target_model(input_ids, use_cache=True)        # prefill
        -> target_model(input_ids=last_token, past_key_values=..., use_cache=True)  # decode loop
```

### 3.1 `tests/test_baseline.py` 做什么

它先只解析 `--model`，然后：

- `qwen3moe` 时使用 `baseline.qwen3_builder.qwen3_build_model`
- `gpt-oss` 时使用 `baseline.gptoss_builder.gptoss_build_model`

再把通用参数交给 `BaselineTestBase.run_test()`。

### 3.2 `BaselineTestBase.run_test()` 做什么

它是实际测试框架，流程是：

1. 根据命令行参数构建模型。
2. 用 `AutoTokenizer.from_pretrained()` 加载 tokenizer。
3. 从 benchmark 里取若干条 prompt。
4. 对每个样本串行执行：
   - 重置模型/类级状态；
   - prefill 一次；
   - 进入 AR decode 循环，每步只生成 1 token；
   - 记录 `prefill_time`、`decode_time`、`new_tokens`、`decode_tps`。
5. 汇总所有样本的总体 `Overall Decode TPS`。
6. 打印 GPU cache 统计。

注意：

- 这里的测试是 **batch size = 1**。
- 这里的核心统计是 **decode 阶段** 的 tokens/s。
- prefill 时间单独记录，但不会被并入 `Overall Decode TPS` 的分母。

---

## 4. 这个项目是怎么“搭起来”的

这个项目并不是重写一个完整模型，而是通过 **monkey patch + 自定义 MoE wrapper** 的方式，复用 Transformers 的主干实现，只把 MoE experts 那部分替换成按需加载。

### 4.1 模型骨架仍然来自 Transformers

Qwen3 使用：

- `transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM`

GPT-OSS 使用：

- `transformers.models.gpt_oss.modeling_gpt_oss.GptOssForCausalLM`

也就是说：

- attention、RMSNorm、decoder 主体、KV cache、logits head 这些都还是官方实现；
- 只是在构建阶段，把 MoE 部分重新接管。

### 4.2 Qwen3 的构建策略

`baseline/qwen3_builder.py` 的核心思路：

1. 先读原始 `config.json`。
2. 临时把 `config.num_experts = 0`，这样构造模型时不会真正实例化 128 个 experts。
3. 通过 monkey patch 保证每层仍保留 MoE 结构和 gate。
4. 建一个 `Qwen3ExpertCache`：
   - CPU 端存 expert 权重；
   - GPU 端维护临时 buffer 和可选 GPU cache。
5. 扫描 `model.safetensors.index.json`：
   - 非 expert 权重直接放到 GPU 模型参数里；
   - expert 权重批量处理后放进 CPU cache。
6. 再把每层 `layer.mlp` 换成 `Qwen3SparseMoeWrapper`。

结果是：

- 模型“看起来”仍是正常 Qwen3-MoE；
- 但真正 expert 的参数不在模型模块里，而是在 CPU cache / GPU buffer / GPU cache 里按需取用。

### 4.3 GPT-OSS 的构建策略

`baseline/gptoss_builder.py` 做法相似，但适配 GPT-OSS 的结构：

1. 给 config 打 `_skip_experts = True` 标记。
2. 构建时只保留 router，不实例化 experts。
3. 用 `GptOssExpertCache` 管理 expert 权重。
4. 加载非 expert 权重到 GPU，expert 权重打包后放到 CPU cache。
5. 单独加载所有 router 权重，注册到 `GateRegistry`。
6. 把每层 `mlp` 替换成 `GptOssSparseMoeWrapper`。

Qwen3 和 GPT-OSS 的差异主要在 expert 权重结构：

- Qwen3：`gate_proj / up_proj / down_proj` 三块分离
- GPT-OSS：`gate_up_proj + bias + down_proj + bias` 的打包结构

---

## 5. MoE offloading 仿真到底是怎么实现的

这是这套代码最核心的部分。

### 5.1 权重放置策略

项目把权重分成两类。

#### 一类：非 expert 权重

这些权重直接常驻 GPU：

- attention
- layernorm
- embedding
- lm head
- router / gate
- 其他 dense 模块

#### 一类：expert 权重

这些不直接挂在模型模块里，而是走专门的缓存/加载路径：

- 先存到 CPU cache
- 再按需拷贝到 GPU swap buffer
- 或者命中 GPU cache 时直接从 GPU cache 取

因此，这里仿真的 offloading 粒度是：

- **expert 权重级别**

不是：

- 全层 offloading
- 整模型 offloading
- KV cache offloading

### 5.2 CPU 端专家缓存

`baseline/expert_cache.py` 负责把专家权重组织成 CPU 端缓存。

两种格式：

#### GPT-OSS

每个 `(layer_idx, expert_idx)` 会被打包成一个连续 tensor：

- `gate_up_proj`
- `gate_up_proj_bias`
- `down_proj`
- `down_proj_bias`

这样 CPU->GPU 拷贝时只要拷一段连续内存。

#### Qwen3

每个 `(layer_idx, expert_idx)` 在 CPU cache 中拆成三块：

- `gate`
- `up`
- `down`

它还会提前预分配一块大的 CPU pinned storage，并用两个 CUDA stream 做 CPU/GPU 间的 pipeline 式搬运。

### 5.3 GPU 临时 swap buffer

`baseline/expert_buffer_manager.py` 管理 GPU 临时 expert buffer。

它会分配一整块连续的 GPU memory pool，然后按 expert 结构切成若干个 buffer slot。

当前 builder 的默认设置是：

- Qwen3：`buffer_size = 128`
- GPT-OSS：`buffer_size = 32`

也就是都等于“单层 expert 总数”。

这意味着：

- 一个 layer 当前所有可能 expert 都有机会被装到临时区；
- 但这块 buffer 是跨层复用的；
- 当前层用完后，下一层会覆盖它。

因此它更像：

- “一层级别的 swap 区”

而不是非常小的页式缓存。

### 5.4 GPU cache

`baseline/gpu_expert_cache.py` 负责长期驻留在 GPU 上的专家缓存。

它的特点是：

- 按层分配固定数量 `slots_per_layer`
- 每个 slot 能容纳一个 expert
- cache 内存和临时 swap buffer 内存是分开的

当前默认参数：

- Qwen3：`cache_slots_per_layer = 16`
- GPT-OSS：`cache_slots_per_layer = 8`

GPU cache 可以理解为：

- 跨 decode 步、跨样本 forward 都尽量复用的热点 expert 驻留区

### 5.5 缓存策略

支持五种策略。

#### `static`

- 每层固定缓存 expert `0 ~ slots_per_layer - 1`
- 不更新

#### `lru`

- 每层独立 LRU
- 命中时更新顺序
- 不在 cache 中的 expert 触发替换

#### `lfu`

- 每层独立 LFU
- 同频率时退化为 LRU

#### `topk_lru`

- 在 LRU 上增加 logit 准入阈值
- warmup 阶段允许全部准入
- 之后只让 logit 足够高的 swapped-in experts 进入 GPU cache

这里要注意一点：

- 它的 `percentile` 不是“保留前 90%”，而更接近“只允许高于当前层活跃 expert 中第 90 分位阈值的那些 experts 进入 cache”。
- 当当前层活跃 expert 数很少时，这个准入会非常保守。

#### `tinylfu`

- 用 TinyLFU + S-LRU 做更激进的准入/淘汰

### 5.6 一次前向里 expert 是怎么被加载的

真正的加载入口是：

- `ExpertCache.batch_load_experts_continuous()`
- `ExpertBufferManager.load_experts_for_current_layer()`

一次当前层 MoE forward 的逻辑是：

1. 先拿到本层所有 `active_expert_ids`。
2. 先查 GPU cache。
   - 命中就直接用 cache slot。
3. 未命中的再查 prefetch buffer。
   - 命中就直接复用 prefetch 结果。
4. 剩余的 experts 从 CPU cache 拷到 GPU 临时 buffer。
5. 如果使用动态 cache 策略，则把这次刚 load 到 swap buffer 的 experts 视为候选，尝试写回 GPU cache。

所以实际优先级是：

```text
GPU cache > prefetch buffer > CPU cache 临时加载
```

### 5.7 Prefetch 是怎么做的

Qwen3 和 GPT-OSS 都实现了跨层 prefetch，但本质上是启发式。

当前层会：

1. 用当前层输入 hidden states 先算本层 router；
2. 再直接把“当前层的 hidden states”喂给“下一层 gate/router”；
3. 预测下一层可能会用到哪些 experts；
4. 把这些 expert 的权重提前搬到 `prefetch_stream` 上。

这里的关键点是：

- 它预测下一层 expert 时，并没有真的先跑完当前层再拿到下一层真实输入；
- 而是用“当前层输入”去猜“下一层会选谁”。

所以：

- 这是一个 **近似 prefetch**；
- 不是严格准确的下一层 oracle。

### 5.8 Expert 计算怎么做

项目没有调用原始 Transformers expert module，而是直接从 GPU memory pool 上取权重 view 做计算。

流程是：

1. router 得到 `selected_experts`
2. 把所有 token 的 expert 分配展平
3. 在 CPU 上做 `bincount` 得到每个 expert 被分到多少 token
4. 收集 `active_expert_ids`
5. 一次性加载/映射对应 expert 权重
6. 根据 expert 分组，把输入 gather 成 batched layout
7. 用两种方式之一算 expert MLP
   - `BMM_ENABLED=True` 时，走 batched BMM
   - 否则逐 expert `F.linear`
8. 最后再 `scatter_add_` 回输出

这里的优化重点是：

- 把 per-expert 的小 kernel 尽量并成批量矩阵乘；
- 把“token -> expert”的 gather/scatter 集中处理；
- 减少 Python 循环里的 GPU kernel launch 数量。

### 5.9 仿真了什么，没仿真什么

#### 已仿真

- expert 权重不常驻 GPU
- expert 权重 CPU->GPU demand loading
- GPU resident cache
- 跨层 prefetch
- cache hit/miss 与替换
- 实际 wall-clock 解码时间
- NVTX 标注的 CUDA profile 路径

#### 没有仿真

- KV cache offloading
- attention/embedding/dense 层 offloading
- 多 batch 调度
- 多请求并发
- NUMA/CPU 内存带宽模型
- PCIe 竞争建模
- NVLink/UMA 差异建模
- 更细粒度的页式 expert 分块传输

换句话说，它更偏：

- “真实搬运 + 真实 CUDA kernel + 真实 wall-clock”的单机实验框架

而不是：

- 离散事件模拟器

---

## 6. 当前代码里的“开销评估”到底怎么看

### 6.1 主脚本当前直接输出的指标

`BaselineTestBase.run_test()` 当前会打印：

#### 每个样本

- `new_tokens`
- `decode_time`
- `decode_tps`
- `prefill_time`

#### 全部样本汇总

- `Total new tokens`
- `Total decode time`
- `Overall Decode TPS`

#### GPU cache 汇总

- `Policy`
- `Slots`
- `Memory`
- `Hits`
- `Misses`
- `Hit Rate`
- `Alpha (Hit Rate / Cache Rate)`
- `Cache Updates`
- `Logit Threshold Percentile`

### 6.2 这些指标分别是什么意思

#### `Overall Decode TPS`

定义是：

```text
所有样本生成的新 token 总数 / 所有样本 decode 阶段总耗时
```

注意：

- 它不包含 prefill 时间；
- 所以它更像“steady-state decode 吞吐”。

#### `Hits / Misses / Hit Rate`

这个命中率不是 token 级，而是：

- **当前层 forward 中 unique active experts 的请求级命中率**

也就是说，如果同一层同一次 forward 里某个 expert 被多个 token 命中，它只会算一次 active expert 请求。

#### `Alpha`

代码里定义为：

```text
alpha = hit_rate / cache_rate
cache_rate = slots_per_layer / num_experts
```

可以理解成：

- 你的 cache 命中效果，相比“按容量比例随机命中”的一个归一化指标。

直觉上：

- `alpha > 1` 说明缓存策略明显比“随机撞上”更有效；
- `alpha ~= 1` 说明缓存策略收益一般；
- `alpha < 1` 说明策略很差。

#### `Cache Updates`

动态策略实际发生的缓存替换/插入次数。

### 6.3 当前统计口径的一个重要细节

`GPUExpertCacheManager` 提供了 `enable_stats()` / `disable_stats()`，但当前主脚本没有显式在 prefill 前后切换统计开关。

这意味着当前打印出来的 cache 统计大概率是：

- **prefill + decode 混合统计**

不是纯 decode 统计。

这点非常关键，因为：

- prefill 的 sequence_length 往往很长；
- 一次 prefill forward 的活跃 expert 并集会远大于单 token decode；
- 会明显影响 hit/miss 与 cache update 的统计口径。

如果你后面想看纯 decode cache 行为，最好在代码里明确：

1. prefill 前 `disable_stats()`
2. decode 前 `enable_stats()`

### 6.4 当前代码里“存在但未直接打印”的额外统计

`ExpertBufferManager.get_stats()` 还维护了：

- `prefetch_hits`
- `compute_loads`
- `total_experts_loaded`

但主脚本当前没有把它们打印出来。

如果你想看，可以在 `tests/baseline_utils.py` 的 `run_test()` 里，在 cleanup 前加类似逻辑：

```python
expert_cache = target_model.model.layers[0].mlp.expert_cache
print(expert_cache.buffer_manager.get_stats())
```

### 6.5 理论上限估算

`tests/adapeagle_backup/theoretical_tps.py` 用的是非常粗的上界模型：

```text
每 token 需要搬运的字节数
  = num_layers * experts_per_layer * expert_size

理论 TPS 上限
  = PCIe 带宽 / 每 token 搬运量
```

这个脚本：

- 不考虑 GPU cache 命中；
- 不考虑 prefetch overlap；
- 不考虑 kernel 时间；
- 只把 CPU->GPU 传输当作瓶颈。

所以它只能作为一个 **无 cache / 无 overlap 的粗上界**。

### 6.6 代码中已埋好的 NVTX profiling 点

这套代码非常适合用 `nsys` 看分阶段开销，因为已经加了大量 NVTX range。

典型范围包括：

- `Attention_LayerX`
- `MoE_Routing_LayerX`
- `MoE_Expert_Load_Prep`
- `Prefetch_Start_LayerX`
- `Global_Expert_Routing_Prep_GPU_Sort`
- `Expert_Input_Gather_Prep`
- `Batched_Expert_Compute`
- `Per_Expert_Compute`
- `Batch_Result_Scatter`
- `Load_Additional_Experts_Layer_X_Count_Y`

因此你不仅能看到总 decode 时间，还能拆出：

- router 计算
- expert load
- GPU cache update
- prefetch overlap
- expert GEMM/BMM
- scatter/gather

---

## 7. 当前代码状态下的实际使用指南

这一节很重要，因为项目现在并不是“拿来就能跑”。

### 7.1 运行环境

当前机器上已验证：

- Conda 环境：`/data_3/wly/miniconda3/envs/sdar`
- Python 包版本可用：
  - `torch 2.7.1+cu126`
  - `transformers 4.53.3`
  - `pyarrow`
  - `pandas`
- GPU：4 张 `NVIDIA H100 PCIe 80GB`
- `nsys` 可用：`/usr/local/cuda-12.1/bin/nsys`

推荐所有命令都显式用这个 Python：

```bash
/data_3/wly/miniconda3/envs/sdar/bin/python ...
```

### 7.2 先看帮助信息

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py --help
```

当前可用参数为：

- `--model {qwen3moe,gpt-oss}`
- `--base-model-path`
- `--benchmark {gsm8k,humaneval,cnndm}`
- `--start-idx`
- `--num-samples`
- `--max-new-tokens`
- `--max-length`
- `--temperature`
- `--enable-gpu-cache`
- `--cache-policy {static,lru,lfu,topk_lru,tinylfu}`
- `--cache-slots-per-layer`
- `--topk-lru-logit-percentile`

### 7.3 当前代码直接运行前必须知道的三个问题

#### 问题 1：默认模型路径在你这台机器上不存在

当前默认值是：

- `/data/home/tianjianyang/models/moe/Qwen3-30B-A3B`
- `/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16`

这两个路径在当前机器上都不存在。

所以你必须显式传：

- `--base-model-path`

例如，本机已存在一个可用的 Qwen3-MoE 类模型缓存路径：

```text
/data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
```

#### 问题 2：当前 `baseline_utils.py` 的 benchmark 路径也写死成外部路径

当前代码读取的是：

- `/data/home/tianjianyang/code/adapeagle/benchmark/...`

这些路径在当前机器上也都不存在。

但项目内本地 benchmark 文件是存在的：

- `/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/benchmark/gsm8k/main/test-00000-of-00001.parquet`
- `/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/benchmark/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet`
- `/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/benchmark/CNN-DM.parquet`

因此当前推荐做法是：

- 先把 `tests/baseline_utils.py` 中的三个 `benchmark_path` 改成项目内相对/绝对路径。

也就是把：

```python
"/data/home/tianjianyang/code/adapeagle/benchmark/gsm8k/main/test-00000-of-00001.parquet"
```

改成：

```python
"/data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/benchmark/gsm8k/main/test-00000-of-00001.parquet"
```

另外两个 benchmark 同理。

#### 问题 3：当前 CLI 实际上无法关闭 GPU cache

参数定义是：

```python
parser.add_argument("--enable-gpu-cache", action="store_true", default=True)
```

这意味着：

- 默认就是 `True`
- 传不传 `--enable-gpu-cache` 都是 `True`
- 当前没有 `--disable-gpu-cache`

所以如果你想测试“无 GPU cache，只保留 CPU->GPU on-demand loading”的情形，当前 CLI 不能直接做到，必须改代码。

### 7.4 当前推荐的最小可运行方式

#### 方案 A：先修 benchmark 路径，再直接跑主脚本

如果你把 `tests/baseline_utils.py` 里的 benchmark 路径修成项目内路径，那么可以直接执行：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading

/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 1 \
  --start-idx 0 \
  --max-new-tokens 64 \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90
```

如果要跑静态缓存：

```bash
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 1 \
  --max-new-tokens 64 \
  --cache-policy static \
  --cache-slots-per-layer 16
```

如果你有 GPT-OSS 权重目录，则命令形态是：

```bash
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model gpt-oss \
  --base-model-path /your/path/to/gpt-oss-20b-BF16 \
  --benchmark humaneval \
  --num-samples 1 \
  --start-idx 0 \
  --max-new-tokens 64 \
  --cache-policy topk_lru \
  --cache-slots-per-layer 8 \
  --topk-lru-logit-percentile 90
```

注意：

- 当前主脚本对 GPT-OSS 仍然使用 `--base-model-path`，不是 `--gptoss-model-path`。

#### 方案 B：参考旧版脚本的路径处理方式

`tests/adapeagle_backup/baseline_benchmark.py` 是旧版脚本，但它有一个优点：

- benchmark 路径使用的是项目根目录相对路径

所以它在“本地 benchmark 可移植性”上比当前主脚本更合理。

如果你只是想快速理解路径组织，它值得参考。

### 7.5 批量跑法

当前 `run_BS.sh` 只是把几组命令串起来，但它依赖默认外部路径，不适合直接使用。

更稳妥的做法是自己显式写：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading

/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 80 \
  --max-new-tokens 256 \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90 | tee qwen3_gsm8k_topk_lru.txt
```

然后换 benchmark / cache policy / slots 继续跑即可。

### 7.6 如何获得延迟开销评估

最直接的方法是看主脚本输出：

1. `prefill_time`
2. `decode_time`
3. `decode_tps`
4. `Overall Decode TPS`
5. `GPU Cache Statistics`

如果你要做横向比较，建议固定：

- 同一个模型
- 同一组 benchmark 样本
- 同样的 `max_new_tokens`
- 同样的 `start_idx`

然后只改变：

- `cache_policy`
- `cache_slots_per_layer`
- `topk_lru_logit_percentile`

例如：

```bash
# static
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 20 \
  --max-new-tokens 128 \
  --cache-policy static \
  --cache-slots-per-layer 16 | tee static.txt

# lru
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 20 \
  --max-new-tokens 128 \
  --cache-policy lru \
  --cache-slots-per-layer 16 | tee lru.txt

# topk_lru
/data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 20 \
  --max-new-tokens 128 \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90 | tee topk_lru.txt
```

比较时主要看：

- `Overall Decode TPS`
- `Hit Rate`
- `Alpha`
- `Cache Updates`

### 7.7 如何做 NVTX + Nsight Systems 分析

因为代码里已经埋了 NVTX，推荐直接这样跑：

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading

/usr/local/cuda-12.1/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o qwen3_gsm8k_topk_lru \
  /data_3/wly/miniconda3/envs/sdar/bin/python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path /data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --benchmark gsm8k \
  --num-samples 1 \
  --start-idx 0 \
  --max-new-tokens 32 \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90
```

这样你能在 Nsight Systems 里直接看到：

- attention 时间
- routing 时间
- expert load 时间
- prefetch 是否与 compute overlap
- BMM 计算时间
- scatter/gather 时间

如果想看更细的 kernel 级信息，可以继续用 `nv-nsight-cu-cli`，但一般第一步先看 `nsys` 就够了。

### 7.8 如何做理论上界估算

当前 `tests/adapeagle_backup/theoretical_tps.py` 由于路径写死，直接跑也要先改路径。

它适合用来回答的问题是：

- 如果完全不命中 GPU cache
- 每个 token 都要搬完整的激活 experts
- PCIe 带宽是固定值

那么理论上 TPS 上限大约是多少。

这个值适合当：

- “最悲观无 cache 基线”

不适合当：

- 真实最终速度预测。

---

## 8. 一个更工程化的理解：这个项目如何“仿真 MoE offloading 场景”

你可以把它抽象成下面这个三层结构。

### 8.1 第 1 层：真实 Transformer 主干

这部分完全正常：

- embedding
- attention
- norm
- residual
- KV cache
- AR decode loop

### 8.2 第 2 层：被替换的 MoE 计算层

这里不再调用原始 expert 模块，而是：

- router 选专家
- active experts 汇总
- 从 cache/buffer 获取权重 view
- 用 F.linear 或 batched BMM 做计算

### 8.3 第 3 层：offloading/caching/prefetch 仿真层

这里决定 expert 权重从哪里来：

- GPU cache 命中
- prefetch buffer 命中
- CPU cache demand load

也就是说，这个项目仿真的并不是“router 怎么选专家”，而是：

- **专家已经选好了之后，这些专家权重如何在 CPU/GPU 之间放置、搬运、缓存与预取。**

---

## 9. 当前实现的几个关键限制与坑点

### 9.1 路径硬编码问题

这是当前最直接的使用障碍。

- 主脚本默认模型路径不存在
- `baseline_utils.py` 的 benchmark 路径也不存在

所以当前仓库状态下，文档比代码更接近“真实使用说明”。

### 9.2 无法用 CLI 关闭 GPU cache

当前只能测试：

- 带 GPU cache 的 offloading

不能直接测试：

- 纯 CPU->GPU on-demand loading、无 GPU cache

### 9.3 Cache 统计口径混入 prefill

当前主脚本没有对 prefill/decode 分相统计 cache hit。

这在 AR 场景下已经会有偏差；
到了 SDAR 场景下，这个偏差会更明显。

### 9.4 统计是“unique active expert per layer forward”，不是“per token”

这一点必须记住。

因为在单 token AR decode 时：

- 每层最多只会有 `top_k` 个 active experts

但在 SDAR 的块内并行或 prefill 场景下：

- 同一次 forward 的 `active_expert_ids` 会变成整个 block 内所有 token 的并集
- unique expert 数会显著放大

### 9.5 Prefetch 是启发式，不是 oracle

当前 prefetch 预测下一层专家时，用的是“当前层输入 hidden states + 下一层 gate/router”。

这对于 AR 单 token decode 也只是近似；
对于 SDAR 块内并行更会偏离真实下一层输入分布。

### 9.6 `layer_times` 和 `layer_expert_counts` 不是跨整个 decode 的累计

这些类级变量会在每次 forward 开始时被重置。

因此它们更像：

- “最近一次 forward 的逐层统计”

而不是：

- “整个样本 decode 过程的逐层累计统计”

### 9.7 这套实现默认对 AR decode 更友好

当前很多设计都默认：

- prefill 是长序列
- decode 是每步 1 token

而不是：

- 每步多 token 并行、并且同一位置会反复重算多次的扩散式/块式解码

---

## 10. 对你下一步接 SDAR 的直接启示

这一节专门从“把 SDAR 接进来”这个角度看。

### 10.1 这套仿真器最适合替换的位置

如果要接 SDAR，最自然的切入点不是重写 builder，而是：

- 保留 `expert_cache.py / expert_buffer_manager.py / gpu_expert_cache.py`
- 保留 Qwen3/GPT-OSS 的自定义 MoE wrapper
- 改“上层 decode 驱动方式”

也就是把现在的：

```text
prefill 1 次 + 每步 1 token AR decode
```

替换成：

```text
prompt prefill + block 级并行迭代 decode
```

### 10.2 SDAR 会改变 active expert 的统计形态

在当前 AR decode 中，每一步是单 token，因此每层大致只会激活：

- `top_k` 个 experts

而在 SDAR 中，块内多个位置并行：

- 一次 forward 会覆盖整个 block
- 每层 active experts 是 block 内所有位置的并集
- 这个并集通常明显大于单 token 的 top-k

直接后果是：

- CPU->GPU 搬运量会更大
- GPU cache hit/miss 统计会和 AR 非常不同
- prefetch 精度也会变化

### 10.3 SDAR 里同一块会反复迭代

AR 的 decode 特征是：

- 一步生成后不会回头

SDAR 的特征是：

- 同一个 block 会反复迭代多次
- 只是 mask 和 confidence 接收状态在变化

这意味着你未来很可能要额外统计：

- 同一 block 的第几次迭代
- 该迭代中 unique active experts 数
- 该迭代相对前一轮的 expert 集合重叠度
- cache hit/miss 是按“迭代”还是按“接受 token”统计

### 10.4 你大概率需要把统计口径重新拆开

建议未来至少区分：

1. prompt prefill
2. SDAR 每个 block 的第一次迭代
3. SDAR 每个 block 的后续迭代
4. 块完成后进入下一块

否则最终只看一个 `Overall Decode TPS`，很难解释：

- 是 block 内反复重算拖慢了速度
- 还是 expert 搬运拖慢了速度
- 还是 cache 命中率变差了

### 10.5 当前项目里最值得复用的组件

如果下一步要和 SDAR 结合，最值得保留的部分是：

- `baseline/expert_cache.py`
- `baseline/expert_buffer_manager.py`
- `baseline/gpu_expert_cache.py`
- `baseline/qwen3_layers.py` 或 `baseline/gptoss_layers.py` 里的 expert load / prefetch / batched compute 逻辑

最不应该直接照搬的部分是：

- `tests/baseline_utils.py` 里假设“decode 每步只有 1 token”的测试框架

---

## 11. 我建议你后续怎么用这份项目

如果你的下一步目标是把 SDAR 解码接进来，我建议分三步。

### 第一步：先把当前 AR 仿真器跑通

先修两个地方：

1. `tests/baseline_utils.py` 的 benchmark 路径
2. 显式传入正确的 `--base-model-path`

先确认：

- Qwen3 版本能跑
- cache 统计能正常打印
- `nsys` profile 能看到 NVTX

### 第二步：把“统计口径”改得更适合做研究

建议新增：

- 纯 decode cache stats
- `buffer_manager.get_stats()` 的打印
- 每个 token / 每次 forward 的 unique active experts 统计
- per-step / per-iteration 的耗时拆分

### 第三步：再把 AR 驱动替换成 SDAR 驱动

也就是把：

- `for step in range(max_new_tokens):` 的单 token 循环

替换成：

- `for block in ...`
- `for denoise_iter in ...`

并保持底层 MoE wrapper 不变，观察：

- block 内并行是否导致 active expert 并集变大
- cache 策略是否仍然有效
- prefetch 对 SDAR 是否还有收益

---

## 12. 附：当前代码状态下的几个可直接记住的事实

### 12.1 当前机器上已确认存在的本地资源

- 本地 benchmark 文件存在于项目内 `benchmark/`
- `sdar` 环境可用
- `nsys` 可用
- 本地有一个 Qwen3-MoE 类模型缓存：
  - `/data_3/wly/.cache/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`

### 12.2 当前机器上已确认不存在的默认资源

- `/data/home/tianjianyang/models/moe/Qwen3-30B-A3B`
- `/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16`
- `/data/home/tianjianyang/code/adapeagle/benchmark/...`

### 12.3 按当前本机可见 Qwen3 配置粗估的显存与传输量

以下仅作为“Qwen3-MoE 类模型”的量级参考。

按当前本机缓存模型 config：

- `hidden_size = 2048`
- `moe_intermediate_size = 768`
- `num_hidden_layers = 48`
- `num_experts = 128`
- `num_experts_per_tok = 8`

则有：

- 单 expert 约 `9.0 MB`
- 若临时 buffer `buffer_size = 128`，则 swap buffer 约 `1.125 GB`
- 若 GPU cache `16 slots/layer`，则 GPU cache 约 `6.75 GB`
- 若完全无 cache、每 token 都要搬 `48 * 8` 个 experts，则理论传输量约 `3456 MB/token`
- 按 `11.5 GB/s` PCIe 粗略估算，理论上限约 `3.41 tokens/s`

这组数值说明了一个很现实的结论：

- 没有高命中率 GPU cache 和有效 prefetch 时，专家搬运会极其昂贵。

### 12.4 按 GPT-OSS builder 注释配置粗估的量级

按代码注释中的 GPT-OSS 配置：

- `hidden_size = 2880`
- `intermediate_size = 2880`
- `num_hidden_layers = 24`
- `num_experts_per_tok = 4`

粗估得到：

- 单 expert 约 `47.48 MB`
- 若临时 buffer `buffer_size = 32`，则 swap buffer 约 `1.48 GB`
- 若 GPU cache `8 slots/layer`，则 GPU cache 约 `8.90 GB`
- 若完全无 cache，则理论上限约 `2.58 tokens/s`

这组数值没有用本机实际 GPT-OSS config 校验，只能作为代码默认设定的量级参考。

---

## 13. 总结

这套 `MoE-Offloading` 代码的核心价值，不在于它已经是一个成熟的产品化仿真器，而在于它已经把下面这件事做通了：

- 在不改主干 Transformer 推理框架的前提下，把 MoE expert 从“常驻 GPU 参数”改造成“CPU 常驻 + GPU 按需加载 + GPU cache + prefetch”的运行模式。

对你下一步接 SDAR 来说，它最有价值的部分是：

- `expert_cache`
- `expert_buffer_manager`
- `gpu_expert_cache`
- Qwen/GPT-OSS 的自定义 MoE wrapper

而当前最需要警惕的部分是：

- 上层测试框架强绑定 AR 单 token decode
- cache 统计口径和路径配置都还比较实验化

如果只想先把项目跑起来，优先修路径；
如果想把它变成 SDAR 的 MoE offloading 基线，优先重构统计口径和 decode 驱动层。

