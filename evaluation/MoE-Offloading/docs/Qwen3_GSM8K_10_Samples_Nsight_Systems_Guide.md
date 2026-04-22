# Qwen3 + GSM8K 10条测试与 Nsight Systems 操作指南

本文基于以下项目状态整理：

- 项目目录：`/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading`
- 整理日期：`2026-04-21`
- 目标模型路径：`/data/models/Qwen3-30B-A3B`

这份文档的目标很简单：

1. 让你先正常跑通一次 `Qwen3 + GSM8K 前 10 条`。
2. 让你再用 `Nsight Systems` 跑一次 profile。
3. 让你知道结果文件在哪、看什么、怎么理解。

如果你完全没用过这个项目，也没用过 Nsight Systems，按本文顺序做就行。

---

## 1. 先说结论：现在这个项目怎么用

当前仓库已经做了最小修复，**你现在不需要再手改代码**，就可以按本文命令直接跑：

- 已修复 `tests/test_baseline.py` 的本地导入问题。
- 已修复 `tests/baseline_utils.py` 的 benchmark 路径，改为读取项目内置的 `benchmark/`。
- 本文推荐的主入口是 `tests/test_baseline.py`。
- 本文**不使用** `run_BS.sh`，也**不使用** `tests/adapeagle_backup/` 下的旧脚本。
- 已补充外层 NVTX 标记，方便 Nsight Systems 查看：
  - `Benchmark_Run`
  - `Sample_1_Total`
  - `Sample_1_Prefill`
  - `Sample_1_Decode`
  - 一直到 `Sample_10_*`

本文里的命令已经直接使用你的真实模型目录：

- `/data/models/Qwen3-30B-A3B`

---

## 2. 你需要准备什么

你的 Qwen3 模型目录 `/data/models/Qwen3-30B-A3B` 至少应该包含这些文件：

- `config.json`
- `model.safetensors.index.json`
- 对应的 `*.safetensors` 分片
- tokenizer 相关文件，例如 `tokenizer.json`、`tokenizer_config.json`

这个项目是单卡 CUDA 跑法，所以你还需要：

- 可用的 NVIDIA GPU
- 能正常执行 `nvidia-smi`
- 能正常执行 `nsys`
- Python 环境里有：
  - `torch`
  - `transformers`
  - `pyarrow`
  - `pandas`

本文默认使用的环境方式是：

```bash
conda activate sdar
```

如果你在某些 shell 里直接执行 `conda activate sdar` 提示需要先 `conda init`，就先执行：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sdar
```

这样最稳。

---

## 2.1 我已经帮你检查过这个模型目录

我已经实际检查了 `/data/models/Qwen3-30B-A3B`，结论如下：

- `config.json` 存在
- `model.safetensors.index.json` 存在
- `model-00001-of-00016.safetensors` 到 `model-00016-of-00016.safetensors` 全部存在
- 权重索引里引用的 16 个 shard 都存在，没有缺失
- `tokenizer.json` 存在
- `tokenizer_config.json` 存在
- `generation_config.json` 存在
- `AutoConfig.from_pretrained(...)` 可以正常读取
- `AutoTokenizer.from_pretrained(...)` 可以正常读取
- tokenizer 自带 `chat_template`

我确认到的关键配置是：

- `model_type = qwen3_moe`
- `hidden_size = 2048`
- `moe_intermediate_size = 768`
- `num_hidden_layers = 48`
- `num_experts = 128`
- `num_experts_per_tok = 8`

补充说明：

- 这个目录里没有 `special_tokens_map.json`
- 但这不是阻塞项，因为 tokenizer 已经能正常加载，`eos_token` 和 `chat_template` 也正常

所以从本文测试目标来看：

- **这个模型目录是完整且可用的**

---

## 2.2 默认 GPU cache 和 GPU swap buffer 有多大

按当前文档里推荐的默认 Qwen3 测试配置：

- `GPU swap buffer` 默认是 `128` 个 expert slot
- `GPU cache` 默认是每层 `16` 个 slot

这两个默认值来自：

- Qwen3 builder 里把 `buffer_size` 设成了 `original_num_experts = 128`
- 主测试脚本里把 `cache_slots_per_layer` 默认设成了 `16`

对应代码：

- `baseline/qwen3_builder.py`
- `tests/test_baseline.py`

### 2.2.1 单个 expert 大小

当前这个 Qwen3 模型配置是：

- `hidden_size = 2048`
- `moe_intermediate_size = 768`
- `num_hidden_layers = 48`
- `num_experts = 128`

Qwen3 每个 expert 包含三块权重：

- `gate`
- `up`
- `down`

所以单个 expert 的参数量是：

```text
3 * hidden_size * moe_intermediate_size
= 3 * 2048 * 768
= 4,718,592 parameters
```

当前项目用的是 `bfloat16`，每个参数 `2 bytes`，所以：

```text
单个 expert 大小 = 4,718,592 * 2 bytes
               = 9,437,184 bytes
               = 9 MiB
```

### 2.2.2 默认 `GPU swap buffer` 容量

`GPU swap buffer` 默认有 `128` 个 expert slot。

所以总大小是：

```text
128 * 9 MiB = 1152 MiB = 1.125 GiB
```

你可以把它理解成：

- GPU 上的临时周转区
- 当前层需要临时加载的 expert 放在这里
- prefetch 预取到的 expert 也先放在这里

### 2.2.3 默认 `GPU cache` 容量

`GPU cache` 默认是每层 `16` 个 slot。

Qwen3 有 `48` 层，所以总 slot 数是：

```text
48 * 16 = 768 slots
```

总大小是：

```text
768 * 9 MiB = 6912 MiB = 6.75 GiB
```

你可以把它理解成：

- GPU 上的热点常驻区
- 某层经常被访问的 expert 会长期留在这里
- 后续 token 再命中时就不需要重新 CPU->GPU 搬运

### 2.2.4 合起来一共多大

默认配置下，这两块 offloading 专用 GPU 区域合计约：

```text
1.125 GiB + 6.75 GiB = 7.875 GiB
```

注意这只是：

- `GPU swap buffer`
- `GPU cache`

两者之和。

它**不包含**：

- dense 主干权重
- attention / decoder 其他模块显存
- KV cache
- activation
- CUDA runtime / workspace

所以实际整卡显存占用会比这个更大。

### 2.2.5 这是不是在模拟真实 GPU HBM 的划分

可以说：

- **是概念上的模拟**

但不是完整的整卡 HBM 精确预算器。

更准确地说，这个项目会在 GPU 上单独预分配两块连续内存池：

- 一块给 `swap buffer`
- 一块给 `GPU cache`

因此它确实是在模拟：

- HBM 里有一块临时 expert 周转区
- HBM 里有一块热点 expert 常驻区

但它没有做这些事：

- 不会从“整张卡总 HBM 容量”自动反推这两块该占多少
- 不会把 dense 权重、KV cache、activation 一起做统一预算
- 不会做真实部署系统里那种更复杂的全局显存调度

所以最合适的理解是：

- **它在软件里手工预留两块 GPU 内存区域，来模拟 offloading runtime 对 HBM 的划分**

而不是：

- **它精确复现了真实部署时整张 GPU HBM 的完整内存管理**

### 2.2.6 为什么 `swap buffer` 默认会开到 128 个 slot

这是因为 builder 直接把它设成了“这一层最多可能出现的 expert 总数”：

- `buffer_size = original_num_experts = 128`

直觉上看，decode 单 token 场景里每层只会选 `top_k = 8` 个 expert，好像用不到 128 个 slot。

但这里这样配是为了让它也能覆盖：

- prefill 长序列
- 某一层多个 token 合并后的 unique expert working set 变大

也就是说，它不是按“单 token decode 通常只需要 8 个”来抠得很紧，而是按“这一层理论最多可能用到 128 个不同 expert”来留足空间。

---

## 2.3 在 decode 过程中，GPU cache 和 GPU swap buffer 怎么工作

下面用一个具体例子说明。

假设现在在 decode 某个 token，走到第 `10` 层 MoE。

这一层 router 最终选出的 active experts 是：

```text
{3, 9, 18, 27, 51, 80, 93, 101}
```

也就是这一层这次真正要用的 8 个 expert。

### 2.3.1 第一步：先查 GPU cache

系统先看这些 expert 里，哪些已经常驻在第 10 层的 GPU cache 里。

比如当前命中了：

```text
{3, 18, 51, 80}
```

这 4 个 expert 不需要再从 CPU 搬运，后面直接参与计算。

### 2.3.2 第二步：再查 prefetch / swap buffer

假设上一层已经预测过下一层可能要用到这些 expert，并且提前把其中两个预取到了 `swap buffer`：

```text
{27, 93}
```

那么这两个也不需要临时现拷，它们会直接从 `swap buffer` 命中。

到这里为止：

- cache 命中 4 个
- prefetch 命中 2 个

### 2.3.3 第三步：剩余 miss 从 CPU 搬到 GPU swap buffer

现在还剩两个 expert 没在 GPU 上：

```text
{9, 101}
```

这时系统就会：

1. 从 CPU expert cache 取这两个 expert 的权重
2. 把它们拷到 `GPU swap buffer` 的空闲 slot
3. 建立 `expert_id -> GPU 位置` 的映射

这一步才是真正的：

- CPU -> GPU 按需加载

### 2.3.4 第四步：开始 expert 计算

接下来进入真正的 expert MLP 计算阶段。

对计算代码来说，它并不关心某个 expert 来自哪里，它只关心：

- 这个 expert 当前对应哪个 GPU view

于是：

- `3, 18, 51, 80` 从 `GPU cache` 取权重
- `27, 93, 9, 101` 从 `swap buffer` 取权重

然后统一进入 batched expert compute。

也就是说，在计算阶段：

- `GPU cache` 和 `GPU swap buffer` 会一起为当前层提供可计算的 expert 权重

### 2.3.5 第五步：可选地把新加载热点提升进 GPU cache

如果缓存策略是动态策略，比如这里默认的 `topk_lru`，那么像：

```text
{9, 101}
```

这种“这次刚从 CPU 搬上来的 expert”，有可能会被进一步提升进 `GPU cache`。

这一步不是 CPU->GPU，而是：

- `swap buffer -> GPU cache`

也就是一次 GPU 内部复制。

如果它们被提升成功，那么下一个 token 再访问这些 expert 时，就可能直接命中 `GPU cache`。

### 2.3.6 第六步：顺便预取下一层

当前层在执行时，还会预测下一层可能会用到哪些 expert。

比如它预测第 11 层大概率会用：

```text
{5, 12, 33, 90}
```

那它就会在 prefetch stream 上，提前把这些 expert 尝试塞进 `swap buffer`。

这样等第 11 层真正开始时：

- 如果这些预测是对的
- 就能直接命中 prefetch
- 少掉一部分临时 CPU->GPU copy

### 2.3.7 一句话总结两者分工

可以把它记成：

- `GPU cache`：长期热点常驻区
- `GPU swap buffer`：当前层临时周转区 + prefetch 承接区

在 decode 里，标准顺序就是：

```text
查 GPU cache
  -> 查 prefetch / swap buffer
  -> 剩余 miss 从 CPU 拷到 swap buffer
  -> 做 expert 计算
  -> 可选提升进 GPU cache
  -> 预取下一层到 swap buffer
```

如果你后面用 Nsight Systems 去看，就可以把：

- HtoD copy
- prefetch
- batched expert compute

都按这个顺序去理解。

---

## 3. 第一次操作前，先做环境检查

### 3.1 进入项目目录

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
```

### 3.2 激活 `sdar` 环境

推荐直接执行：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sdar
```

### 3.3 定义几个变量，后面命令直接复用

```bash
export PROJECT=/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
export MODEL=/data/models/Qwen3-30B-A3B
```

### 3.4 检查模型目录是否完整

```bash
ls $MODEL/config.json
ls $MODEL/model.safetensors.index.json
```

再检查分片数量：

```bash
ls $MODEL/model-*.safetensors | wc -l
```

你这里应该看到：

```text
16
```

### 3.5 检查 GPU、Python 包、Nsight Systems

```bash
nvidia-smi
```

```bash
python -c "import torch, transformers, pyarrow, pandas; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('transformers', transformers.__version__)"
```

```bash
python -c "from transformers import AutoConfig, AutoTokenizer; m='/data/models/Qwen3-30B-A3B'; c=AutoConfig.from_pretrained(m); t=AutoTokenizer.from_pretrained(m); print(c.model_type, c.hidden_size, c.moe_intermediate_size, c.num_hidden_layers, c.num_experts, c.num_experts_per_tok); print(type(t).__name__, t.eos_token, bool(getattr(t, 'chat_template', None)))"
```

```bash
nsys --version
```

### 3.6 看一下脚本帮助信息

```bash
python tests/test_baseline.py --help
```

如果这一步能正常打印参数说明，说明主入口已经可用。

### 3.7 要不要指定 `CUDA_VISIBLE_DEVICES`

**建议指定。**

原因是当前这套代码没有单独提供 `--device` 或 `--gpu-id` 参数，而是直接使用：

- `torch.device("cuda")`
- `tensor.to("cuda")`

也就是说，它默认会使用“当前进程可见的第 1 张 GPU”。

因此：

- 如果你机器上只有 1 张 GPU，可以不写 `CUDA_VISIBLE_DEVICES`
- 如果你机器上有多张 GPU，**建议你始终显式写**

最常见的写法是：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py ...
```

如果你想用第 3 张物理 GPU，就写：

```bash
CUDA_VISIBLE_DEVICES=2 python tests/test_baseline.py ...
```

这里有个很重要的细节：

- 当你写 `CUDA_VISIBLE_DEVICES=2` 时
- 程序内部看到的仍然是 `cuda:0`
- 只是这个 `cuda:0` 实际上映射到物理 GPU 2

所以这是**当前项目选择 GPU 的正确方式**。

---

## 4. 先正常跑一遍 Qwen3 + GSM8K 10条

这一步先不要上 Nsight Systems。原因很简单：

- 先确认项目本身能正常跑。
- 先拿一份不带 profiler 开销的基线结果。
- 后面再单独跑 profile，比较清楚。

### 4.1 创建日志目录

```bash
mkdir -p logs
```

### 4.2 运行命令

下面这条命令会跑：

- 模型：`qwen3moe`
- benchmark：`gsm8k`
- 样本范围：从第 0 条开始，共 10 条
- GPU cache：开启
- cache policy：`topk_lru`
- 每层 cache slot：`16`

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path "$MODEL" \
  --benchmark gsm8k \
  --start-idx 0 \
  --num-samples 10 \
  --max-new-tokens 128 \
  --max-length 8192 \
  --enable-gpu-cache \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90.0 | tee logs/qwen3_gsm8k_10.log
```

### 4.3 这条命令会经历什么

你会看到大致这几类输出：

- `Building qwen3moe model with offloading...`
- `Loaded 10 prompts`
- `Processing Sample 1/10`
- `Sample 1 completed: ... tokens, ... tokens/s`
- 最后的总表
- 最后的 GPU Cache Statistics

第一次跑通常会比较久，因为它要先：

- 构建模型骨架
- 从 safetensors 加载 dense 权重
- 把 expert 权重整理进 CPU pinned memory
- 初始化 GPU cache

这部分时间**不计入**最后的 `Overall Decode TPS`。

---

## 5. 普通运行结果怎么看

你最需要看的是控制台结尾的两块。

### 5.1 第一块：总体吞吐

你会看到类似：

```text
OVERALL STATISTICS (All Samples)
Total samples: 10
Total new tokens: ...
Total decode time: ...
Overall Decode TPS: ...
```

这里最重要的是：

- `Total samples`
  - 应该是 `10`
- `Total new tokens`
  - 10 条样本一共生成了多少新 token
- `Total decode time`
  - 10 条样本 decode 阶段总耗时
- `Overall Decode TPS`
  - 计算方式是：
  - `总生成 token 数 / 总 decode 时间`

注意两点：

- 这里看的是 **decode 吞吐**，不是端到端总耗时。
- 它**不包含**模型构建时间，也**不包含** prefill 时间。

### 5.2 第二块：GPU Cache Statistics

你会看到类似：

```text
GPU Cache Statistics:
  Policy: topk_lru
  Slots: ...
  Memory: ...
  Hits: ...
  Misses: ...
  Hit Rate: ...
  Alpha (Hit Rate / Cache Rate): ...
  Cache Updates: ...
  Logit Threshold Percentile: 90.0%
```

这些字段的意思如下。

`Policy`

- 当前 GPU expert cache 的策略。
- 这里应当是 `topk_lru`。

`Slots`

- GPU cache 一共多少个 slot。
- 例如 Qwen3 48 层、每层 16 个 slot，那么就是 `48 * 16 = 768`。

`Memory`

- GPU cache 自己占用多少显存。

`Hits / Misses`

- 在每层 MoE 运行时，查 GPU cache 命中了多少次、没命中多少次。

`Hit Rate`

- `Hits / (Hits + Misses)`。

`Alpha`

- `Hit Rate / Cache Rate`
- 其中 `Cache Rate = 每层缓存 slots / 每层总 experts`
- 这个值大于 1，说明你的热点专家分布比“完全均匀随机”更集中，cache 有效果。

`Cache Updates`

- 动态 cache 策略下，发生了多少次 cache 更新或替换。

`Logit Threshold Percentile`

- `topk_lru` 的准入阈值。
- 这里是 `90.0%`，表示只让本次活跃专家里 logit 分数较高的一部分进入 cache。

### 5.3 当前主脚本有一个你必须知道的统计口径

当前这份主脚本：

- 会在 prefill 和 decode 两个阶段都访问 GPU cache
- 但是最终只打印一份合并后的 cache hit/miss 统计

所以你要知道：

- `Overall Decode TPS` 只看 decode
- 但 `Hits / Misses / Hit Rate` 是 **prefill + decode 混合统计**

不要把这两个统计口径当成同一件事。

### 5.4 当前主脚本没有直接打印 `prefill_time`

代码内部记录了 `prefill_time`，但控制台默认没有逐样本打印出来。

所以如果你只是看当前控制台输出：

- 你能直接看到 `decode_tps`
- 你不能直接看到每个样本的 `prefill_time`

对“先跑通项目”这件事没有影响。

---

## 6. 建议你把普通运行结果保存好

你前面已经用 `tee` 保存了日志：

```text
logs/qwen3_gsm8k_10.log
```

后面如果你只想快速看关键结果，可以直接：

```bash
grep -E "Overall Decode TPS|GPU Cache Statistics|Hits:|Misses:|Hit Rate:|Alpha|Cache Updates" logs/qwen3_gsm8k_10.log
```

---

## 7. 再跑一遍 Nsight Systems

这一步的目标不是重新测一份最准确的 TPS，而是看：

- CUDA 时间线
- CPU 到 GPU 的 expert copy
- NVTX 标记的各阶段
- 每层 Attention / MoE / Prefetch / Batched Expert Compute 的耗时分布

要记住一条基本原则：

- **被 Nsight Systems profile 时，程序通常会变慢**

所以：

- 普通跑一遍，用来记 benchmark 结果
- 再单独 profile 一遍，用来做性能分析

不要把二者混在一起理解。

---

## 8. Nsight Systems 最简单的用法

### 8.1 创建 profile 输出目录

```bash
mkdir -p profiles
```

### 8.2 运行 profile 命令

这是最推荐的新手命令：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10 \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path "$MODEL" \
    --benchmark gsm8k \
    --start-idx 0 \
    --num-samples 10 \
    --max-new-tokens 128 \
    --max-length 8192 \
    --enable-gpu-cache \
    --cache-policy topk_lru \
    --cache-slots-per-layer 16 \
    --topk-lru-logit-percentile 90.0
```

这条命令里的关键参数如下。

`--trace=cuda,nvtx,osrt`

- 采集 CUDA、NVTX、OS runtime。
- 对这个项目最重要。

`--sample=none`

- 不做 CPU 采样。
- 可以减少额外开销，先看 CUDA/NVTX 就够了。

`--cpuctxsw=none`

- 不采集 CPU context switch。
- 同样是为了先简化结果。

`--force-overwrite=true`

- 如果输出文件已存在，允许覆盖。

`--output profiles/qwen3_gsm8k_10`

- 输出前缀。
- 生成的主文件会是：
  - `profiles/qwen3_gsm8k_10.nsys-rep`

### 8.3 什么时候说明这一步成功了

profile 正常结束后，目录里至少应当有：

```bash
ls profiles/qwen3_gsm8k_10.nsys-rep
```

如果这个文件存在，说明 profile 已经抓到了。

---

## 9. 这次 profile 里你会看到哪些 NVTX 标记

当前项目里已经有很多内部 NVTX 标记，例如：

- `Attention_Layer0`
- `MoE_Routing_Layer0`
- `MoE_Expert_Load_Prep`
- `Prefetch_Start_Layer0`
- `Batched_Expert_Compute`
- `Batch_Result_Scatter`

此外，当前主脚本还额外加了更适合新手看的外层标记：

- `Benchmark_Run`
- `Sample_1_Total`
- `Sample_1_Prefill`
- `Sample_1_Decode`
- `Sample_2_Total`
- `Sample_2_Prefill`
- `Sample_2_Decode`
- 一直到 `Sample_10_*`

这几个外层标记很重要，因为它们让你能快速回答这些问题：

- 整个 benchmark 真正运行阶段在哪里
- 某一条样本的 prefill 在哪里
- 某一条样本的 decode 在哪里
- 某一条样本 decode 里，Attention 和 MoE 各占多大比例

---

## 10. 不开 GUI，先用 `nsys stats` 看汇总

就算你还不会看 GUI，先用命令行也能得到很多信息。

### 10.1 看默认汇总

```bash
nsys stats profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.stats.txt
```

这会生成一份默认摘要，包括：

- NVTX summary
- CUDA API summary
- CUDA kernel summary
- CUDA memcopy summary

### 10.2 只看 NVTX 汇总

```bash
nsys stats \
  --report nvtx_sum \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.nvtx.txt
```

你会看到各个 NVTX range 的累计时间和次数。

第一次看时，建议重点找：

- `Benchmark_Run`
- `Sample_1_Prefill`
- `Sample_1_Decode`
- `Batched_Expert_Compute`
- `MoE_Routing_Layer*`
- `Attention_Layer*`

### 10.3 只看 CUDA kernel 汇总

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.cuda_kern.txt
```

这可以帮助你看：

- 哪些 kernel 总时间最长
- 是 GEMM/BMM 更重，还是别的 kernel 更重

### 10.4 只看 GPU memcopy 时间汇总

```bash
nsys stats \
  --report cuda_gpu_mem_time_sum \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.mem_time.txt
```

这一步很适合看 offloading，因为它能帮助你判断：

- HtoD copy 是否很多
- memcpy 时间是否明显

### 10.5 只看 GPU memcopy 大小汇总

```bash
nsys stats \
  --report cuda_gpu_mem_size_sum \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.mem_size.txt
```

这一步更偏“搬了多少数据”。

---

## 11. 用 `--filter-nvtx` 只看某一段

这是 Nsight Systems 非常好用的一点。

因为现在脚本里有外层 NVTX range，所以你可以只分析某一段。

### 11.1 只看整个 benchmark 真正运行区间

```bash
nsys stats \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --filter-nvtx Benchmark_Run \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.benchmark_only.txt
```

这会把模型构建阶段排除掉，更接近“正式推理阶段”的分析。

### 11.2 只看第 1 条样本的 prefill

```bash
nsys stats \
  --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --filter-nvtx Sample_1_Prefill \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.sample1_prefill.txt
```

### 11.3 只看第 1 条样本的 decode

```bash
nsys stats \
  --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --filter-nvtx Sample_1_Decode \
  profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.sample1_decode.txt
```

这一步对新手特别有用，因为你可以直接比较：

- `Sample_1_Prefill`
- `Sample_1_Decode`

通常你会看到：

- prefill 更长
- prefill 的活跃 expert 并集更大
- decode 的每步更短，但调用层数更多、次数也更多

---

## 12. 如果你要看 GUI，应该怎么做

在真正打开 GUI 之前，你先要知道：

- `Nsight Systems` 到底是什么
- 它的 GUI 和 `nsys stats` 有什么区别
- 对这个 MoE offloading 项目来说，它为什么值得看
- 如果你关心的是推理延迟和 GPU 端显存占用，GUI 里最应该盯哪些信息

### 12.1 `Nsight Systems GUI` 是什么

你可以把 `Nsight Systems` 理解成一个“把一次程序运行录成时间线”的工具：

- `nsys profile` 负责采集 profile，生成 `.nsys-rep`
- `nsys stats` 负责把 profile 汇总成文本统计
- `Nsight Systems GUI` 负责把 profile 变成可缩放、可展开、可对齐 CPU/GPU 事件的图形时间线

它最擅长回答的是这类问题：

- 程序时间到底花在哪
- CPU 和 GPU 有没有并行起来
- H2D copy 和 compute 有没有重叠
- 哪一段在等待、同步、空转
- 某个慢样本到底慢在 prefill、decode，还是某些 memcopy

要注意：

- `Nsight Systems` 主要看“系统级时间线和瓶颈位置”
- `Nsight Compute` 才更偏向看“单个 CUDA kernel 内部执行效率”

对你这个项目，第一步几乎总是应该先看 `Nsight Systems`。

### 12.2 为什么要看 GUI，而不只是看 `nsys stats`

只看 `stats.txt` 时，你拿到的是聚合统计。例如它能告诉你：

- `Host-to-Device` 一共搬了多少数据
- `Host-to-Device` 一共花了多少时间
- `Device-to-Device` 占 memop 的多少比例

但它不能直接回答：

- 这些 copy 是发生在模型构建阶段，还是 benchmark 推理阶段
- 是 prefill 慢，还是 decode 慢
- copy 有没有和 compute overlap
- 哪个样本、哪个 token 前后出现了异常长的等待

GUI 的价值就在于：

- 你能把 `Benchmark_Run -> Sample_i_Prefill -> Sample_i_Decode` 这些 NVTX 范围
- 和下面的 CUDA stream、memcpy、kernel、CPU 线程活动
- 在同一条时间轴上对齐起来看

对于 MoE offloading，这一点尤其重要。因为瓶颈往往不是“某个 kernel 算得慢”，而是：

- CPU expert cache 到 GPU 的 H2D copy 太多
- prefetch 没有覆盖住下一层真正要用的专家
- copy 和 expert compute 没有重叠好
- GPU cache promotion 带来了额外 DtoD
- Python / CPU 调度让 GPU 出现空洞

这些问题都属于系统级时序问题，正是 `Nsight Systems GUI` 最擅长定位的。

### 12.3 平常 `Nsight Systems` 是怎么用的

通常是这个顺序：

1. 用 `nsys profile` 录一次代表性的运行
2. 用 `nsys stats` 先看粗粒度统计
3. 用 GUI 打开 `.nsys-rep`
4. 在 GUI 里定位慢区间，放大时间线看细节
5. 如果最后发现是某个 kernel 本身有问题，再考虑 `Nsight Compute`

所以对大多数 CUDA 项目来说：

- `Nsight Systems` 先回答“哪里慢”
- `Nsight Compute` 再回答“为什么这个 kernel 慢”

对于这个 Qwen3 MoE offloading 项目，你现在最需要的是前者。

### 12.4 对这个 MoE offloading 项目，GUI 的价值是什么

如果你关心的是：

- 推理延迟
- GPU 端显存占用

那么 `Nsight Systems GUI` 的价值主要有两块。

第一块是看“延迟是怎么形成的”。

对 MoE offloading，延迟通常由这些部分叠加而成：

- CPU 侧调度 / Python 调用
- router 选 expert
- CPU -> GPU 的 expert 权重搬运
- GPU 上的 expert compute
- GPU cache promotion
- 下一层 prefetch
- 各种同步、等待、空转

`stats.txt` 只能给你总量；GUI 能让你看到：

- `Sample_i_Prefill` 里是不是 H2D 非常密集
- `Sample_i_Decode` 里是不是每层都在做小块 H2D copy
- copy 和 compute 有没有 overlap
- GPU stream 上是否有明显空白
- CPU 线程是否在 launch 之间卡住了

第二块是看“显存不是只看占了多少，还要看怎么用的”。

这里要分清两类概念：

- 常驻容量，例如 `GPU cache` 和 `GPU swap buffer` 预留了多少
- 动态活动，例如运行时到底搬了多少数据，什么时候搬，是否造成峰值上涨

你前面看到的：

- `cuda_gpu_mem_time_sum`
- `cuda_gpu_mem_size_sum`

反映的是“内存操作的耗时和累计流量”，不是“当前时刻 GPU 上占了多少显存”。

如果你想直接看显存占用曲线或峰值，通常要在 profile 时额外打开：

```bash
--cuda-memory-usage=true
```

所以，对显存问题，GUI 的价值不只是看“占用”，还包括：

- 看峰值出现在 prefill 还是 decode
- 看 H2D / DtoD 是否异常密集
- 看缓存策略是否引入了很多额外 DtoD
- 看动态行为和你设定的 `GPU cache` / `GPU swap buffer` 容量是否匹配

### 12.5 如果你关心推理延迟和显存，GUI 里优先看什么

如果你第一次用，不要一下子看所有轨迹。先盯下面几类。

第一类：`NVTX Range`

- `Benchmark_Run`
- `Sample_i_Total`
- `Sample_i_Prefill`
- `Sample_i_Decode`

这是入口。先把模型加载阶段、benchmark 阶段、每个样本的 prefill / decode 切开。

第二类：`CUDA memcpy`

重点看：

- `Host-to-Device`
- `Device-to-Device`
- `Device-to-Host`

对 MoE offloading 来说，`Host-to-Device` 最关键。你要看：

- 它是不是集中出现在 decode 阶段
- 单次 copy 大小是否接近一个 expert projection
- copy 是不是串行堆积
- copy 能不能和 compute 重叠

第三类：`CUDA kernels`

重点不是做 kernel 微架构分析，而是看：

- kernel 前面有没有长时间等 copy
- kernel 是否被很多小 copy 切碎
- GPU 有没有明显 idle / gap

第四类：`CPU 线程 / CUDA API`

这部分能帮你判断：

- Python 主线程是否在频繁同步
- CUDA API 调用之间是否有大空洞
- 是 GPU 算得慢，还是 CPU 没及时把工作发下去

第五类：`GPU Memory Usage`

只有在 profile 时开启了 `--cuda-memory-usage=true` 以后，这一轨才会特别有价值。它能回答：

- benchmark 阶段显存峰值是多少
- 峰值是在 prefill 还是 decode
- decode 时显存是否稳定，还是明显抖动
- 是否已经接近物理 HBM 上限

如果没开这个选项，你现在这份 profile 仍然可以很好地看“流量”和“时间线”，但不能直接给你完整的显存占用曲线。

更具体的 GUI 时间线阅读顺序，见下一节 `## 13. 在 GUI 里你应该先看什么`。

### 12.6 先说明当前机器状态

当前机器上我确认有：

- `nsys` CLI

但没有确认到：

- `nsys-ui` 命令

这意味着：

- 你现在一定可以在命令行生成 `.nsys-rep`
- 但不一定能在这台机器上直接用 GUI 打开

### 12.7 如果你本机或桌面环境装了 Nsight Systems GUI

最简单的做法是：

1. 找到这个文件：

```text
/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/qwen3_gsm8k_10.nsys-rep
```

2. 在装有 Nsight Systems GUI 的机器上打开它。

常见方法有两种：

- 直接双击 `.nsys-rep`
- 在 GUI 里 `File -> Open`

如果你的机器上有 `nsys-ui`，也可以尝试：

```bash
nsys-ui /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/qwen3_gsm8k_10.nsys-rep
```

### 12.8 如果服务器没有 GUI，怎么处理

你有两种办法。

办法 A：只用 CLI

- 继续用 `nsys stats` 看汇总
- 对大多数“先了解项目热点在哪”已经够用了

办法 B：把 `.nsys-rep` 拷到你自己的桌面机器再看

例如从别的机器执行：

```bash
scp <server_user>@<server_host>:/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/qwen3_gsm8k_10.nsys-rep .
```

然后在本地 GUI 打开。

---

## 13. 在 GUI 里你应该先看什么

如果你第一次打开 Nsight Systems GUI，不要一上来就看所有细节。按下面顺序看最快。

### 13.1 先看时间线最上层

先找这些 NVTX range：

- `Benchmark_Run`
- `Sample_1_Total`
- `Sample_1_Prefill`
- `Sample_1_Decode`

先确认：

- 哪一段是 prefill
- 哪一段是 decode

### 13.2 再展开 CUDA 行和 NVTX 行

你会看到内部范围，例如：

- `Attention_Layer*`
- `MoE_Routing_Layer*`
- `Prefetch_Start_Layer*`
- `Batched_Expert_Compute`

先问自己三个问题：

1. `Attention` 和 `MoE`，谁更重
2. `Batched_Expert_Compute` 是否连续、是否很多
3. `Prefetch_Start_Layer*` 是否真的与计算发生重叠

### 13.3 再看 memcopy

如果你在时间线上看到了大量 HtoD copy，说明：

- CPU expert cache 到 GPU swap buffer 的搬运很重

如果 HtoD copy 分散且频繁，通常意味着：

- cache hit 不够高
- 或者 active expert working set 变化大

### 13.4 再回到控制台结果对照

把 GUI 和控制台日志对起来看：

- 控制台的 `Overall Decode TPS` 高不高
- GPU Cache 的 `Hit Rate` 高不高
- GUI 里 memcpy 时间多不多
- GUI 里 `Batched_Expert_Compute` 占比大不大

这样你就能把“数值结果”和“时间线原因”对应起来。

---

## 14. 推荐的实际操作顺序

如果你第一次做，最推荐按这个顺序。

### 第一步：先跑普通版本

```bash
cd $PROJECT
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sdar
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path "$MODEL" \
  --benchmark gsm8k \
  --start-idx 0 \
  --num-samples 10 \
  --max-new-tokens 128 \
  --max-length 8192 \
  --enable-gpu-cache \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90.0 | tee logs/qwen3_gsm8k_10.log
```

### 第二步：记录普通结果

```bash
grep -E "Overall Decode TPS|Hits:|Misses:|Hit Rate:|Alpha|Cache Updates" logs/qwen3_gsm8k_10.log
```

### 第三步：再跑 profile

```bash
mkdir -p profiles
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10 \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path "$MODEL" \
    --benchmark gsm8k \
    --start-idx 0 \
    --num-samples 10 \
    --max-new-tokens 128 \
    --max-length 8192 \
    --enable-gpu-cache \
    --cache-policy topk_lru \
    --cache-slots-per-layer 16 \
    --topk-lru-logit-percentile 90.0
```

### 第四步：先用 CLI 看摘要

```bash
nsys stats profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.stats.txt
```

```bash
nsys stats --report nvtx_sum profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.nvtx.txt
```

```bash
nsys stats --report cuda_gpu_mem_time_sum profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.mem_time.txt
```

### 第五步：如果有 GUI，再打开 `.nsys-rep`

- 打开 `profiles/qwen3_gsm8k_10.nsys-rep`
- 先看 `Benchmark_Run`
- 再看 `Sample_1_Prefill`
- 再看 `Sample_1_Decode`

---

## 15. 常见问题

### 问题 1：`--help` 都跑不起来

先确认你在项目目录下：

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
```

再确认你已经激活了 `sdar` 环境：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sdar
python tests/test_baseline.py --help
```

### 问题 2：提示找不到模型文件

先检查：

```bash
ls $MODEL/config.json
ls $MODEL/model.safetensors.index.json
```

如果这两项不存在，说明模型路径不对，或者模型还没准备完整。

### 问题 3：提示找不到 benchmark 文件

当前主脚本已经改为读取项目内置 benchmark。

你可以手动检查：

```bash
ls $PROJECT/benchmark/gsm8k/main/test-00000-of-00001.parquet
```

### 问题 4：profile 后感觉速度比普通运行慢很多

这是正常的。

- 普通运行看 benchmark 结果
- profile 运行看时间线和热点

不要直接拿 profile 时的 TPS 去和普通运行做严格对比。

### 问题 5：服务器上没有 GUI

先用：

```bash
nsys stats profiles/qwen3_gsm8k_10.nsys-rep
```

如果还想看时间线，就把 `.nsys-rep` 拷到装有 Nsight Systems GUI 的机器上。

### 问题 6：显存不够

可以先尝试把 cache 规模降下来：

```bash
--cache-slots-per-layer 8
```

或者先把生成长度降下来：

```bash
--max-new-tokens 64
```

---

## 16. 你最终应该拿到哪些文件

正常情况下，你最后会得到这些文件：

普通运行日志：

- `logs/qwen3_gsm8k_10.log`

Nsight Systems 主报告：

- `profiles/qwen3_gsm8k_10.nsys-rep`

Nsight Systems 命令行摘要：

- `profiles/qwen3_gsm8k_10.stats.txt`
- `profiles/qwen3_gsm8k_10.nvtx.txt`
- `profiles/qwen3_gsm8k_10.mem_time.txt`

这几份文件已经足够你做第一次完整实验记录。

---

## 17. 最后一段最短版操作清单

如果你已经理解上面内容，实际只要执行下面这些命令。

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
export PROJECT=/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sdar
export MODEL=/data/models/Qwen3-30B-A3B
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs profiles
```

普通跑：

```bash
python tests/test_baseline.py \
  --model qwen3moe \
  --base-model-path "$MODEL" \
  --benchmark gsm8k \
  --start-idx 0 \
  --num-samples 10 \
  --max-new-tokens 128 \
  --max-length 8192 \
  --enable-gpu-cache \
  --cache-policy topk_lru \
  --cache-slots-per-layer 16 \
  --topk-lru-logit-percentile 90.0 | tee logs/qwen3_gsm8k_10.log
```

Nsight Systems：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10 \
  env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path "$MODEL" \
    --benchmark gsm8k \
    --start-idx 0 \
    --num-samples 10 \
    --max-new-tokens 128 \
    --max-length 8192 \
    --enable-gpu-cache \
    --cache-policy topk_lru \
    --cache-slots-per-layer 16 \
    --topk-lru-logit-percentile 90.0
```

看摘要：

```bash
nsys stats profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.stats.txt
nsys stats --report nvtx_sum profiles/qwen3_gsm8k_10.nsys-rep | tee profiles/qwen3_gsm8k_10.nvtx.txt
```
