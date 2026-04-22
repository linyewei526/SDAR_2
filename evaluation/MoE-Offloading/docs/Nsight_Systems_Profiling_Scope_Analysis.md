# Nsight Systems 采集范围与精简建议

本文专门回答下面几个问题：

1. 当前这次 `Nsight Systems` profile 到底采了哪些信息。
2. 为什么 `/profiles/qwen3_gsm8k_10.nsys-rep` 会达到 `419M`，GUI 打开时容易卡在 `Loading report from file 100%...`。
3. 对这个 `MoE offloading` 项目来说，如果你真正关心的是：
   - 推理延迟
   - GPU 端显存占用
   - 最好还能看到峰值和变化过程
   那么哪些信息应该保留，哪些信息可以不采。
4. 便于后续把 profile 命令改成“只采你真正需要的指标”。

本文基于当前项目状态和下面这次实际 profile 文件整理：

- 项目目录：`/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading`
- profile 文件：`profiles/qwen3_gsm8k_10.nsys-rep`
- stats 文件：`profiles/qwen3_gsm8k_10.stats.txt`
- 整理日期：`2026-04-21`

---

## 1. 当前 profile 命令到底采了什么

你在操作指南里跑的命令是这一条：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10 \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
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

这意味着当前 profile 实际上采集了三大类信息：

- `cuda`
  - CUDA Runtime API
  - CUDA kernel 执行
  - CUDA memcpy / memset
  - CUDA event / synchronization
- `nvtx`
  - 代码里主动打的 NVTX 范围
- `osrt`
  - 操作系统运行库调用，例如 `poll`、`pthread_cond_wait`、`ioctl`、`read`

同时，它明确没有采这些内容：

- 没有 CPU sampling，因为 `--sample=none`
- 没有 CPU context switch，因为 `--cpuctxsw=none`
- 没有 GPU memory usage 轨迹，因为没有加 `--cuda-memory-usage=true`
- 没有 OpenMP / OpenGL / Vulkan / Unified Memory 等专项采集

所以当前这份 `.nsys-rep` 的核心内容，其实是：

- CUDA 事件很多
- NVTX 标记很多
- OS runtime 调用少量

而不是：

- CPU 采样数据很多
- 回溯栈很多
- 显存占用曲线很多

---

## 2. 代码里主动打了哪些 NVTX

当前项目代码除了 `nsys` 自动采集的 CUDA / OSRT 事件，还会主动打很多 NVTX 范围。

### 2.1 外层 benchmark 范围

在 `tests/baseline_utils.py` 里，现在有这些外层范围：

- `Benchmark_Run`
- `Sample_i_Total`
- `Sample_i_Prefill`
- `Sample_i_Decode`

这几层范围的价值很高，因为它们能把：

- 模型构建阶段
- benchmark 真正运行阶段
- 每个样本的 prefill
- 每个样本的 decode

在时间线上明确切开。

### 2.2 Qwen3 MoE 内部范围

在 `baseline/qwen3_layers.py` 和 `baseline/expert_buffer_manager.py` 里，还打了很多内部范围，例如：

- `Attention_Layer{idx}`
- `MoE_Routing_Layer{idx}`
- `MoE_Expert_Load_Prep`
- `Prefetch_Start_Layer{idx}`
- `Global_Expert_Routing_Prep_GPU_Sort`
- `Expert_Input_Gather_Prep`
- `Batched_Expert_Compute`
- `Per_Expert_Compute`
- `Batch_Result_Scatter`
- `Prefetch_Validity_Check`
- `Load_Additional_Experts_Layer_{layer_idx}_Count_{n}`

这些范围的价值是：

- 帮你在 GUI 里直接看到每层 routing / prefetch / gather / batched expert compute 的位置
- 对定位“offloading copy 和 compute 是否重叠”“哪一步最慢”很有帮助

但它们的代价是：

- 范围实例数很多
- GUI 时间线会更密、更杂
- 文件会变大一些

### 2.3 当前这 10 条数据为什么会出现这么多层级范围

这次配置是：

- `10` 条 GSM8K 样本
- 每条最多 `128` 个 decode token
- 每次前向都要过 `48` 层

所以一个很粗的量级是：

- 每条样本大约有 `1` 次 prefill 前向 + 最多 `128` 次 decode 前向
- 大约就是 `129` 次 transformer 前向
- `10` 条样本就是 `1290` 次前向
- 每次前向都过 `48` 层

于是像：

- `Attention_Layer*`
- `MoE_Routing_Layer*`
- `Prefetch_Start_Layer*`

这类按层打的范围，实例数都会接近：

```text
1290 * 48 = 61,920
```

这和实际 `stats.txt` 里的实例数是对得上的。

---

## 3. 当前 `.stats.txt` 实际证明采到了什么

这次 `nsys stats` 实际生成了这些主要报告：

- `NVTX Range Summary`
- `OS Runtime Summary`
- `CUDA API Summary`
- `CUDA GPU Kernel Summary`
- `CUDA GPU MemOps Summary (by Time)`
- `CUDA GPU MemOps Summary (by Size)`

这说明当前最主要的有效采集内容正是：

- NVTX
- OSRT
- CUDA API
- CUDA kernel
- CUDA memcopy / memset

而后面那些 `SKIPPED` 条目，只是 `nsys stats` 默认也会尝试检查一些别的报告类型，但当前 profile 里没有对应数据，并不是错误。

---

## 4. 为什么这个 `.nsys-rep` 会到 419M

先看实际文件大小：

- `qwen3_gsm8k_10.nsys-rep`：`419M`
- `qwen3_gsm8k_10.sqlite`：`1.2G`

这里最重要的一点是：

- `.nsys-rep` 不是坏掉了
- 它大，是因为这次真的采到了“非常多的事件”

GUI 打开时卡在 `Loading report from file 100%...`，通常不是因为文件损坏，而是因为：

- 需要解析和索引的事件太多
- 本地机器内存或 I/O 跟不上
- GUI 需要把大量 CUDA / NVTX 时间线对象建出来

### 4.1 当前真正的“大头”是什么

我直接查了这次 `.sqlite` 里的主要事件表行数和占用，大头如下：

| 表 | 行数 | 在 sqlite 中约占空间 |
| --- | ---: | ---: |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `9,399,564` | `395 MB` |
| `CUPTI_ACTIVITY_KIND_KERNEL` | `5,571,365` | `519 MB` |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `3,439,826` | `201 MB` |
| `NVTX_EVENTS` | `607,719` | `44 MB` |
| `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` | `265,305` | `15 MB` |
| `OSRT_API` | `42,182` | `1.6 MB` |

从这个表可以直接看出：

- **最大头不是 `osrt`**
- **最大头也不是 NVTX**
- **真正的大头是 CUDA Runtime API、CUDA Kernel、CUDA Memcpy**

也就是说，文件大的核心原因不是“你打了太多 NVTX”，而是：

- 这次运行里本来就发生了海量 CUDA API 调用
- 海量 kernel 启动
- 海量 memcpy

NVTX 会让文件更大、GUI 更密，但不是最大的主因。

### 4.2 当前 CUDA 事件为什么会这么多

这次 `stats.txt` 里最关键的几个数字是：

#### CUDA API 调用量

- `cudaLaunchKernel`: `5,507,412` 次
- `cudaMemcpyAsync`: `3,439,826` 次
- `cudaStreamSynchronize`: `246,825` 次

#### GPU memcopy 次数

- `Host-to-Device`: `914,824` 次
- `Device-to-Device`: `2,320,820` 次
- `Device-to-Host`: `204,182` 次

#### GPU kernel 实例数

- `CUPTI_ACTIVITY_KIND_KERNEL`: `5,571,365` 条

这对 MoE offloading 是非常符合预期的，因为当前实现会频繁发生：

- CPU expert cache -> GPU swap buffer 的 H2D
- swap buffer -> GPU cache 的 DtoD promotion
- 很多小粒度的 PyTorch / CUDA kernel
- 每一层 routing / gather / scatter / batched expert compute 带来的大量 kernel launch

也就是说：

- 当前文件大，不是 Nsight Systems 乱采
- 而是这个 workload 本来就是“高事件密度”的

### 4.3 当前 profile 还把“模型构建阶段”也一起录进来了

这是另一个很关键的原因。

当前 profile 命令没有使用：

- `--capture-range=nvtx`

所以它录的是：

- Python 进程启动
- 模型加载
- expert 权重整理
- benchmark 正式运行
- 程序退出

而不是只录：

- `Benchmark_Run`

这会带来两个问题：

第一，文件更大。

因为模型构建阶段也会产生很多 CUDA 事件，尤其是：

- pinned memory 分配
- expert 权重搬运
- 构建 GPU cache / swap buffer

第二，GUI 更乱。

因为你真正关心的是“推理 latency 和推理显存”，但 timeline 前面还混着一大段模型 build。

### 4.4 `osrt` 不是这次体量的主因

当前 `OSRT_API` 只有：

- `42,182` 条事件
- 在 sqlite 里约 `1.6 MB`

而且从 `OS Runtime Summary` 来看，主要是：

- `pthread_cond_wait`
- `poll`
- `pthread_cond_timedwait`
- `ioctl`
- `read`

所以结论很明确：

- 如果你先去掉 `osrt`，文件会小一点，GUI 会清爽一点
- 但它不会从根本上把 `419M` 变成 `50M`

真正的根因仍然是：

- 全程采集
- 大量 CUDA kernel
- 大量 CUDA memcpy

### 4.5 NVTX 会增加体量，但不是第一主因

当前 `NVTX_EVENTS` 大约：

- `607,719` 条
- sqlite 中约 `44 MB`

所以：

- 去掉很细的内部 NVTX，确实可以减少一些体量和 GUI 杂乱度
- 但它也不是这次 419M 的第一主因

你可以把它理解成：

- NVTX 主要影响“GUI 可读性”和“几十 MB 级别体量”
- CUDA API / kernel / memcpy 才决定“几百 MB 级别体量”

---

## 5. 如果你真正关心的是推理延迟，哪些信息必须保留

如果你的目标是分析：

- `prefill` 和 `decode` 哪个更慢
- `decode` 慢是不是因为 H2D copy
- copy 和 compute 是否 overlap
- GPU 有没有空转

那下面这些信息是必须保留的。

### 5.1 必须保留：`cuda`

这是最核心的。

没有 `cuda` trace，你就看不到：

- kernel 时间线
- memcpy 时间线
- CUDA API 调用

而这些正是 MoE offloading latency 的主体。

### 5.2 必须保留：外层 `nvtx`

至少要保留：

- `Benchmark_Run`
- `Sample_i_Prefill`
- `Sample_i_Decode`

因为没有这些范围，你很难把：

- 模型构建
- benchmark
- prefill
- decode

在 GUI 里快速切开。

### 5.3 强烈建议保留：部分中层 NVTX

例如：

- `MoE_Routing_Layer*`
- `Prefetch_Start_Layer*`
- `Batched_Expert_Compute`

这些对分析 offloading 的价值很高，因为它们能直接回答：

- routing 多慢
- prefetch 发起得早不早
- batched expert compute 是否被 copy 卡住

### 5.4 不一定必须：`osrt`

如果你现在主要关心的是 GPU 侧 latency 形成过程，`osrt` 不是第一优先级。

只有当你怀疑：

- Python / CPU 调度有明显阻塞
- 文件 I/O、线程等待、poll / cond wait 有问题

时，`osrt` 才值得保留。

换句话说：

- 第一次看 latency：可以不采 `osrt`
- 怀疑 CPU-side stall 时：再把 `osrt` 打开

---

## 6. 如果你真正关心的是 GPU 端显存占用，哪些信息必须保留

这里先要区分两类东西：

- **内存流量**：搬了多少、搬了多久
- **显存占用**：某一时刻 HBM 上实际占了多少

当前这次 profile 只采到了前者，没采到后者的完整轨迹。

### 6.1 当前已经采到的：memcopy 流量

你现在已经有：

- `cuda_gpu_mem_time_sum`
- `cuda_gpu_mem_size_sum`

它们告诉你的是：

- H2D / DtoD / DtoH 花了多少时间
- H2D / DtoD / DtoH 搬了多少累计数据量

它们很适合分析：

- offloading copy 重不重
- DtoD promotion 多不多

但它们**不能直接告诉你**：

- 当前显存峰值是多少
- 平均显存使用是多少
- 显存曲线怎么变化

### 6.2 如果你要显存占用曲线，必须打开 `--cuda-memory-usage=true`

这是最关键的一点。

如果你希望在 GUI 里看到：

- benchmark 阶段显存怎么变化
- 峰值是多少
- 峰值出现在 prefill 还是 decode

那 profile 时必须加：

```bash
--cuda-memory-usage=true
```

没有这个选项时：

- 你只能看到 memcopy 流量
- 不能看到完整的 GPU memory usage 轨迹

### 6.3 关于“平均显存使用”的现实说明

`Nsight Systems` 对“显存峰值”和“曲线变化”的支持是比较直接的，GUI 里很适合看：

- 峰值
- 变化趋势
- 某一段时间的上升 / 下降

但如果你要一个非常明确的文本指标，例如：

- `Benchmark_Run` 平均显存占用 = `xx.xx GiB`

这通常不是 `nsys stats` 默认现成给你的摘要项。

更实际的做法通常有两个：

1. 在 GUI 里直接看 memory usage 轨迹和峰值
2. 若必须要“平均值”这个数字，再额外导出或后处理 memory usage 数据

所以对当前需求，最重要的是先做到：

- 有显存曲线
- 能看到峰值
- 能对齐 `prefill` / `decode`

这比一开始执着于“平均值文本摘要”更重要。

---

## 7. 对这个项目来说，哪些采集项值得保留，哪些可以先删

下面给一个最实用的结论。

### 7.1 建议保留

如果你想分析推理延迟和 GPU 显存：

- `cuda`
- `nvtx`
- `--sample=none`
- `--cpuctxsw=none`
- `--capture-range=nvtx`
- `--nvtx-capture=Benchmark_Run`
- `--capture-range-end=stop`
- `--cuda-memory-usage=true`

它们的作用分别是：

- `cuda`
  - 看 kernel / memcpy / API
- `nvtx`
  - 给 benchmark / prefill / decode 分段
- `--sample=none`
  - 避免 CPU sampling 让文件暴涨
- `--cpuctxsw=none`
  - 避免额外线程调度噪音
- `--capture-range=nvtx`
  - 只录 `Benchmark_Run`，把模型构建阶段排掉
- `--cuda-memory-usage=true`
  - 看显存曲线和峰值

### 7.2 可以先不保留

第一次做 latency + memory 分析时，下面这些可以先不采：

- `osrt`
- CPU sampling
- CPU context switch
- 各种和当前任务无关的图形 / OpenMP / Unified Memory 项

理由是：

- 它们对当前两个核心问题帮助不大
- 会让 GUI 更乱
- 有些还会明显增加体量或开销

### 7.3 代码里的细粒度 NVTX 可以考虑分级保留

如果你后面想进一步精简，还可以把当前 NVTX 分成三层看待。

第一层：强烈建议永久保留

- `Benchmark_Run`
- `Sample_i_Total`
- `Sample_i_Prefill`
- `Sample_i_Decode`

第二层：建议默认保留

- `Attention_Layer*`
- `MoE_Routing_Layer*`
- `Prefetch_Start_Layer*`
- `Batched_Expert_Compute`

第三层：可以考虑作为“深度调试模式”再打开

- `Prefetch_Validity_Check`
- `Load_Additional_Experts_Layer_*_Count_*`
- `Per_Expert_Compute`

原因是第三层虽然有价值，但：

- 实例数很多
- 名字很多
- GUI 最容易被它们塞满

它们更适合“已经知道问题大概在 expert loading 这一块，再去深挖”的阶段。

---

## 8. 最推荐的精简 profile 方案

如果你现在的目标是：

- 文件别太大
- GUI 能打开
- 重点看推理 latency
- 重点看 GPU memory 变化和峰值

我最推荐下面这条命令。

### 8.1 推荐命令：只抓 `Benchmark_Run`，同时开启显存占用轨迹

注意：

- 这一节保留的是之前按 `named NVTX capture` 写的思路
- 我后面在 `13.1` 做了当前环境下的最小复现实验，确认 `torch.cuda.nvtx.range(...)` 记录本身没问题，但 `nsys profile --capture-range=nvtx --nvtx-capture=...` 在这台机器上不稳定，可能直接出现 `No reports were generated`
- 所以如果你现在要立刻稳定跑通，优先用 `13.1` 里基于 `cudaProfilerApi` 的命令
- 这一节可以继续当“采集范围应该怎么设计”的参考，但不要再优先照抄成最终命令

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=Benchmark_Run \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --show-output=false \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10_benchmark_only \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
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

这条命令相比当前版本，做了四个关键精简：

- 去掉了 `osrt`
- 只抓 `Benchmark_Run`
- 开启了 `--cuda-memory-usage=true`
- 关闭了 `--show-output`

这样做的结果通常会是：

- 文件明显更小
- GUI 更容易打开
- 时间线更集中在真正的推理区间
- 能看到更有价值的显存使用轨迹

### 8.2 如果你只想看 latency，甚至可以先不开 memory usage

如果你先只关心：

- copy / compute overlap
- prefill / decode 慢在哪里

那还可以再轻一点：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=Benchmark_Run \
  --capture-range-end=stop \
  --show-output=false \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10_latency_only \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
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

### 8.3 如果你怀疑 CPU-side stall，再加回 `osrt`

只有在你发现：

- GPU timeline 有大空白
- 但看不清是不是 CPU 线程没及时发工作

时，再把 `osrt` 打开：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=Benchmark_Run \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --show-output=false \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_10_with_osrt \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
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

---

## 9. 结论

对当前这次 `419M` 的 `.nsys-rep`，最核心的判断是：

- 文件大是正常现象，不像损坏
- 真正的大头是 CUDA Runtime / Kernel / Memcpy 事件
- `osrt` 和 NVTX 不是主要体量来源
- 当前命令把模型构建阶段也录进来了，这会明显放大文件

如果你下一步只想围绕：

- 推理延迟
- GPU 端显存峰值和变化

来做 profile，那么最值得改的不是先去纠结某一两个小 NVTX，而是优先做这四件事：

1. 只采 `Benchmark_Run`
2. 保留 `cuda,nvtx`
3. 去掉 `osrt`
4. 打开 `--cuda-memory-usage=true`

这样得到的 profile 会更适合这个项目，也更容易被 GUI 打开和使用。

---

## 10. 现在代码里已经支持的两种新能力

当前代码已经额外支持了两组新参数：

- `--nsys-profile-every-k`
- `--nsys-profile-first-sample`
- `--nsys-capture-range-name`
- `--track-gpu-memory`
- `--gpu-memory-output`

### 10.1 每隔 `k` 个 sample 才让 Nsight Systems 捕获一次

这组参数的作用不是“程序内部自己做 profile”，而是：

- 在被选中的 sample 外层自动打一层专用 NVTX 范围
- 让你可以配合 `nsys --capture-range=nvtx`，只抓每隔 `k` 个 sample 的时间线

例如：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --nvtx-capture=Nsys_Profile_Capture \
  --capture-range-end=repeat:2:sync \
  --cuda-memory-usage=true \
  --show-output=false \
  --force-overwrite=true \
  --output profiles/qwen3_gsm8k_sparse \
  env CUDA_VISIBLE_DEVICES=0 python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
    --benchmark gsm8k \
    --start-idx 0 \
    --num-samples 10 \
    --max-new-tokens 128 \
    --max-length 8192 \
    --enable-gpu-cache \
    --cache-policy topk_lru \
    --cache-slots-per-layer 16 \
    --topk-lru-logit-percentile 90.0 \
    --nsys-profile-every-k 5 \
    --nsys-profile-first-sample 1
```

这条命令的含义是：

- Python 仍然跑完整个 `10` 条 sample
- 但只会给第 `1` 条和第 `6` 条 sample 打 `Nsys_Profile_Capture`
- `nsys` 只抓这两个 capture window

如果你想抓第 `3, 8, 13, ...` 条，就写：

```bash
--nsys-profile-every-k 5 --nsys-profile-first-sample 3
```

### 10.2 全局显存变化信息

如果你加上：

```bash
--track-gpu-memory
```

当前脚本会额外记录一份轻量级显存快照 CSV，默认写到：

```text
profiles/<model>_<benchmark>_start<start_idx>_n<num_samples>_gpu_memory.csv
```

它会在这些关键点记录快照：

- `pre_build`
- `post_build`
- `benchmark_start`
- `sample_start`
- `prefill_end`
- `sample_end`
- `benchmark_end`
- `post_cleanup`

每条记录里会包含：

- `device_used_bytes`
- `device_free_bytes`
- `torch_allocated_bytes`
- `torch_reserved_bytes`
- `torch_max_allocated_bytes`
- `torch_max_reserved_bytes`

以及对应的 `GiB` 列。

这份 CSV 的价值是：

- 即使你只让 `nsys` 抓每隔 `k` 个 sample
- 你仍然能拿到“整个 benchmark 全程”的粗粒度显存变化信息

换句话说：

- `nsys --cuda-memory-usage=true` 负责给你被捕获窗口里的精细显存曲线
- `--track-gpu-memory` 负责给你全程 sample 级别的全局显存变化

两者是互补的。

---

## 11. 三个容易混淆的补充澄清

下面专门回答三个最容易让人误解的点。

### 11.1 `### 4.3` 和 `### 5.2` 看起来矛盾，到底该保留哪个外层 NVTX

这两处说的不是同一个层面，所以并不矛盾。

#### 第一层：哪一段时间会被 `nsys` 真正录进去

这是 **capture 边界** 的问题。

`### 4.3 当前 profile 还把“模型构建阶段”也一起录进来了` 这一段说的是：

- 当前那条 `nsys profile` 命令没有用 `--capture-range=nvtx`
- 所以 `nsys` 从进程启动就开始录
- 因而录进去了：
  - Python 进程启动
  - 模型加载
  - expert 权重整理
  - benchmark 正式运行
  - 程序退出

如果你想让 `nsys` 只录“真正的推理阶段”，那最自然的做法是：

- 用 `Benchmark_Run` 作为 **capture trigger**

也就是：

```bash
--capture-range=nvtx
--nvtx-capture=Benchmark_Run
--capture-range-end=stop
```

这样 `nsys` 录进去的是：

- `Benchmark_Run` 这一整段

而不会把模型构建阶段也录进去。

所以，`### 4.3` 这一段的核心意思是：

- **如果你要限定“录哪一大段时间”，用 `Benchmark_Run` 作为 capture 边界最合适。**

#### 第二层：在已经录进去的推理区间内部，时间线还要怎么分段

这是 **时间线标记可读性** 的问题。

`### 5.2 必须保留：外层 nvtx` 说的是：

- 一旦你已经决定录 `Benchmark_Run`
- 那在这段推理时间里，最好还要继续保留：
  - `Sample_i_Prefill`
  - `Sample_i_Decode`
  - 以及通常也保留 `Sample_i_Total`

因为不然你在 GUI 里虽然知道“这一整段是 benchmark”，但你看不出来：

- 第几条 sample 在哪里开始、哪里结束
- 哪一段是 prefill
- 哪一段是 decode

所以，最推荐的理解方式是：

- `Benchmark_Run`：负责回答“**哪一大段时间要被 Nsight Systems 录进去**”
- `Sample_i_Prefill / Sample_i_Decode / Sample_i_Total`：负责回答“**录进去以后，内部该怎么切段分析**”

这两者不是二选一，而是：

- **最好同时保留**

#### 结论

如果你现在只关心推理过程，也就是 `prefill + decode`，最推荐的是：

1. 用 `Benchmark_Run` 做 capture 边界  
   这样不录模型构建阶段。
2. 在 `Benchmark_Run` 内部继续保留  
   - `Sample_i_Total`
   - `Sample_i_Prefill`
   - `Sample_i_Decode`

所以更准确地说：

- `### 4.3` 讨论的是“`nsys` 在 **capture 层面** 录哪一大段”
- `### 5.2` 讨论的是“外层 `nvtx` 在 **时间线标记层面** 应该怎么保留”

不是“CUDA 层面只保留 `Benchmark_Run`，NVTX 层面再保留别的”这种区分；  
而是：

- `Benchmark_Run` 更适合做 **capture 边界**
- `Sample_i_Prefill / Decode` 更适合做 **capture 内部的分析分段**

---

### 11.2 Qwen3 MoE 内部 NVTX 到底分别是什么意思，顺序是什么，哪些可能并行

这里专门说当前 **Qwen3 路径** 里的手工 NVTX。

#### 11.2.1 先说完整的层内大顺序

Qwen3 decoder layer 的顺序是标准的：

1. `Attention_Layer{i}`
2. `MoE_Routing_Layer{i}`
3. `MoE_Expert_Load_Prep`
4. `Global_Expert_Routing_Prep_GPU_Sort`
5. `Expert_Input_Gather_Prep`
6. `Batched_Expert_Compute` 或 `Per_Expert_Compute`
7. `Batch_Result_Scatter`

其中：

- `Attention_Layer{i}` 是该层 self-attention 的整个前向
- 后面这些是该层 MLP 部分被替换成 MoE offloading 逻辑后的内部阶段

也就是说，在同一层里：

- `Attention_Layer{i}` 先发生
- 然后才进入这个层的 MoE routing / loading / compute / scatter

#### 11.2.2 每个 NVTX 分别是什么意思

下面按执行顺序解释。

##### `Attention_Layer{i}`

含义：

- 第 `i` 层 attention 整体前向
- 是在 `qwen3_builder.py` 里给 `Qwen3MoeAttention.forward` 打的 monkey patch 标记

对应真实执行阶段：

- 当前层 attention 计算
- 包括 attention 内部的 Q/K/V、FlashAttention、输出投影等

它的作用主要是：

- 让你在 GUI 里直接对比 attention 和 MoE 哪边更重

##### `MoE_Routing_Layer{i}`

含义：

- 当前层 router 计算阶段

里面实际做的事包括：

- `router_logits = self.gate(hidden_states_flat)`
- `softmax`
- 当前层 `top-k` 专家选择
- 如果开启 prefetch，还会：
  - 用下一层 gate 对当前 hidden states 做一次“下一层专家预测”
  - 取下一层预测的 `PREFETCH_TOPK`
  - 把下一层预测 expert 列表从 GPU 搬到 CPU，供后续 prefetch 使用

对应真实执行阶段：

- “当前层要用哪些 expert”
- “猜下一层可能要用哪些 expert”

##### `MoE_Expert_Load_Prep`

含义：

- 当前层 expert 装载前的准备阶段

里面实际做的事包括：

- 把 `selected_experts` 展平
- DtoH 到 CPU
- 在 CPU 上做 `bincount`
- 统计当前层 active experts、每个 expert 对应 token 数量、offset
- 结合 GPU cache 策略准备 `router_logits`
- 调用 `batch_load_experts_continuous(...)`
- 在这个阶段末尾，如果开启 prefetch，会发起 `Prefetch_Start_Layer{i}`

对应真实执行阶段：

- “当前层到底要装哪些 expert”
- “这些 expert 在 cache / prefetch / swap buffer 里怎么分配”

这个范围本身是个大包围框，里面还会嵌套更细的范围。

##### `Prefetch_Validity_Check`

含义：

- 检查前一层提前预取过来的 expert 现在是否可直接复用

里面实际做的事：

- 把 `prefetch_in_progress` 转成可消费的 `prefetch_mapping`
- 对当前层需要的 experts 检查：
  - 有没有 GPU cache hit
  - 有没有 prefetch hit

对应真实执行阶段：

- “下一层提前搬来的东西，现在能不能直接用”

##### `Load_Additional_Experts_Layer_{layer_idx}_Count_{n}`

含义：

- 当前层仍然缺的 expert，要从 CPU 真正搬到 GPU swap buffer

这里的 `{n}` 表示：

- **还需要临时加载的 expert 数量**
- 它已经排除了：
  - GPU cache 命中
  - prefetch 命中

里面实际做的事：

- 找空闲 swap buffer slot
- 对每个 miss expert 执行三块 H2D copy：
  - `gate`
  - `up`
  - `down`

对应真实执行阶段：

- CPU expert cache -> GPU swap buffer 的“当前层按需加载”

这是 offloading 延迟里最关键的阶段之一。

##### `Prefetch_Start_Layer{i}`

含义：

- 当前层已经把本层要用的 expert 准备好了，现在开始为下一层发起预取

里面实际做的事：

- 取刚才在 `MoE_Routing_Layer{i}` 里得到的下一层预测 expert 列表
- 去重
- 等待上一个 prefetch stream 完成
- 在单独的 prefetch stream 上调用 `prefetch_expert(...)`

对应真实执行阶段：

- 为下一层准备热点 expert

##### `Global_Expert_Routing_Prep_GPU_Sort`

含义：

- 在 GPU 上对 `(token, expert)` 分配做全局排序和重排准备

里面实际做的事：

- 根据 `selected_experts` 把所有 `(token, top_k_rank)` 展开
- 按 expert ID 全局排序
- 得到 `sorted_experts / sorted_tokens / sorted_ranks`

对应真实执行阶段：

- 把“哪些 token 去哪个 expert”整理成后面 gather/compute/scatter 好处理的格式

##### `Expert_Input_Gather_Prep`

含义：

- 真正 expert 计算前的输入 gather 和批结构准备

里面实际做的事：

- 把属于每个 expert 的 token 输入 gather 出来
- gather 对应的 routing weights
- 准备 batched BMM 所需的输入张量
- 准备每个 active expert 的权重 view

对应真实执行阶段：

- “把 token 送到对应 expert 前，整理成可批量计算的布局”

##### `Batched_Expert_Compute`

含义：

- 当前层 active experts 的批量专家计算

只有在 `BMM_ENABLED=True` 时出现。

里面实际做的事：

- stack active expert 的 `gate/up/down` 权重
- 用 `torch.bmm` 做 batched expert MLP
- 乘上 routing weights

对应真实执行阶段：

- 当前层 MoE expert 的主算子阶段

##### `Per_Expert_Compute`

含义：

- 不用 batched BMM 时，逐个 expert 做计算

只有在 `BMM_ENABLED=False` 时出现。

它和 `Batched_Expert_Compute` 是二选一关系，不会同时是主路径。

##### `Batch_Result_Scatter`

含义：

- 把 expert 输出按 token 累加回原始 token 位置

里面实际做的事：

- `scatter_add_` 把 `expert_outputs` 写回 `final_hidden_states`

对应真实执行阶段：

- MoE 输出重组

#### 11.2.3 它们之间的关系和嵌套关系

最重要的关系可以这样记：

- `Attention_Layer{i}` 是该层 attention 的整个前向
- 后面的 `MoE_*` / `Prefetch_*` / `Gather/Compute/Scatter` 是该层 MLP 被替换成 MoE 后的内部阶段

嵌套关系上：

- `MoE_Expert_Load_Prep` 是个大范围
  - 里面可能出现 `Prefetch_Validity_Check`
  - 里面可能出现 `Load_Additional_Experts_Layer_*`
  - 里面可能出现 `Prefetch_Start_Layer*`

- `Expert_Input_Gather_Prep` 也是个大范围
  - 里面会嵌套 `Batched_Expert_Compute`
  - 或者嵌套多个 `Per_Expert_Compute`

#### 11.2.4 哪些可能并行，哪些基本不并行

要分“默认 stream 主路径”和“prefetch stream”两类看。

##### 基本不并行的部分

这些大体都在当前层主路径上，按顺序推进：

- `Attention_Layer{i}`
- `MoE_Routing_Layer{i}`
- `MoE_Expert_Load_Prep` 里的当前层按需加载
- `Global_Expert_Routing_Prep_GPU_Sort`
- `Expert_Input_Gather_Prep`
- `Batched_Expert_Compute` / `Per_Expert_Compute`
- `Batch_Result_Scatter`

对同一个 token、同一层来说，它们整体是顺序依赖的。

尤其是：

- `Load_Additional_Experts_Layer_*`

这一步是当前层真正缺 expert 时的现搬运，属于当前层关键路径。

##### 可能并行的部分

真正有并行意图的是：

- `Prefetch_Start_Layer{i}`

它在单独的 prefetch CUDA stream 上发起，因此它**可能**与当前层后面的这些默认 stream 工作重叠：

- `Global_Expert_Routing_Prep_GPU_Sort`
- `Expert_Input_Gather_Prep`
- `Batched_Expert_Compute`
- `Batch_Result_Scatter`

也就是说，理想目标是：

- 当前层还在算
- 下一层专家已经在另一条 stream 上往 GPU 搬

但这里有两个限制：

1. 代码在启动新的 prefetch 前，会先 `synchronize()` 上一个 prefetch stream  
   所以同时只允许一个 prefetch 链路在飞，不会无限并行叠加。

2. 当前层真正缺的 expert 装载并不在单独 stream 上异步隐藏  
   它属于当前层关键路径，不能像下一层 prefetch 那样轻松藏到后面。

#### 11.2.5 一个具体 decode 例子

假设 decode 某个 token 到第 `10` 层时：

- router 选中了 `8` 个 expert：
  - `{3, 9, 18, 27, 51, 80, 93, 101}`

那么时间线大致会这样走：

1. `Attention_Layer10`
   - 第 10 层 attention 完成

2. `MoE_Routing_Layer10`
   - 当前层 top-k 选出上述 `8` 个 expert
   - 同时预测第 11 层可能会用到的 expert，例如 `{7, 22, 41, 90}`

3. `MoE_Expert_Load_Prep`
   - 先检查 cache 和 prefetch
   - 假设：
     - GPU cache 命中：`3, 18, 51, 80`
     - prefetch 命中：`27, 93`
     - 仍然 miss：`9, 101`

4. `Load_Additional_Experts_Layer_10_Count_2`
   - 把 `9` 和 `101` 从 CPU 搬到 GPU swap buffer

5. `Prefetch_Start_Layer10`
   - 在 prefetch stream 上，为第 11 层尝试预取 `{7, 22, 41, 90}`

6. `Global_Expert_Routing_Prep_GPU_Sort`
   - 对这 8 个 expert 对应的 token 分派关系做 GPU 上重排

7. `Expert_Input_Gather_Prep`
   - gather 本层 expert 输入

8. `Batched_Expert_Compute`
   - 当前层 active experts 一起算

9. `Batch_Result_Scatter`
   - 把输出写回 token 位置

这时理想情况下，步骤 `5` 的下一层 prefetch 与 `6-9` 的当前层计算是有机会重叠的。

#### 11.2.6 这是不是当前 Qwen3 路径里全部的手工 NVTX

对当前 **Qwen3 runtime 路径** 来说，基本可以认为是。

当前手工 NVTX 主要来自四处：

1. `tests/baseline_utils.py`
   - `Benchmark_Run`
   - `Sample_i_Total`
   - `Sample_i_Prefill`
   - `Sample_i_Decode`
   - 可选的 `Nsys_Profile_Capture`

2. `baseline/qwen3_builder.py`
   - `Attention_Layer{i}`

3. `baseline/qwen3_layers.py`
   - `MoE_Routing_Layer{i}`
   - `MoE_Expert_Load_Prep`
   - `Prefetch_Start_Layer{i}`
   - `Global_Expert_Routing_Prep_GPU_Sort`
   - `Expert_Input_Gather_Prep`
   - `Batched_Expert_Compute`
   - `Per_Expert_Compute`
   - `Batch_Result_Scatter`

4. `baseline/expert_buffer_manager.py`
   - `Prefetch_Validity_Check`
   - `Load_Additional_Experts_Layer_{layer_idx}_Count_{n}`

所以，如果问题限定为：

- “当前 Qwen3 baseline 路径里，代码手工打了哪些 NVTX”

那基本就是这些。

但要注意：

- 这不是整个仓库所有路径的全部 NVTX，因为 GPT-OSS 路径也有自己一套
- 这也不是 Nsight Systems 时间线里的全部信息，因为时间线上还有大量非 NVTX 的：
  - CUDA kernels
  - memcpy
  - CUDA API
  - OSRT

#### 11.2.7 为什么 `### 5.3` 里优先建议保留那几个中层 NVTX

文档里举的那几个：

- `MoE_Routing_Layer*`
- `Prefetch_Start_Layer*`
- `Batched_Expert_Compute`

之所以优先，是因为它们是 **“信息密度高、和 offloading 机制最直接相关、但不会像更细粒度标签那样太吵”** 的三类。

分别对应：

- `MoE_Routing_Layer*`
  - 当前层选 expert 的决策入口
  - 也是下一层 prefetch 预测的入口

- `Prefetch_Start_Layer*`
  - 你能直接看到“下一层预取什么时候开始”
  - 很适合和当前层 compute 做 overlap 分析

- `Batched_Expert_Compute`
  - 当前层 expert 真正做主计算的窗口
  - 很适合和 memcpy / prefetch 做重叠分析

这三个加起来，已经足够回答很多核心问题：

- 当前层 routing 慢不慢
- 下一层 prefetch 发起得够不够早
- 当前层真正的 expert compute 有多长
- prefetch 和 compute 有没有 overlap

相比之下：

- `Prefetch_Validity_Check`
  - 更偏内部 bookkeeping
  - 信息有用，但太细

- `Load_Additional_Experts_Layer_*`
  - 很重要，但事件数多、名字碎
  - 适合在你已经怀疑“cache miss / 按需加载太重”时再重点看

- `Expert_Input_Gather_Prep`
  - 更偏张量整理
  - 不是最核心的 offloading 决策点

- `Batch_Result_Scatter`
  - 重要，但通常不是 offloading 分析的第一入口

补充一点：

- `Attention_Layer*` 其实也很有价值
- 只是 `### 5.3` 那里特意强调的是“**offloading 机制最直接相关**”的几个中层范围

如果你的问题是“attention 和 MoE 到底谁更重”，那 `Attention_Layer*` 也应该保留。

---

### 11.3 `osrt` 到底记录什么，不记会漏掉什么

`osrt` 是 `Operating System Runtime` 的缩写。

它记录的不是 CUDA kernel，也不是显存占用，而是：

- 程序在用户态调用的一些操作系统运行库/线程/等待相关 API

当前这次 profile 里，`OS Runtime Summary` 里主要出现的是：

- `pthread_cond_wait`
- `poll`
- `pthread_cond_timedwait`
- `ioctl`
- `read`
- `open64`
- `mmap / munmap`
- `pthread_create`
- `sleep`

可以把它理解成：

- “CPU 线程在等什么、睡什么、读什么、调什么系统接口”

#### 它对整体 latency 和显存占用影响大吗

对 **分析价值** 来说：

- 对显存占用：几乎没有直接帮助
- 对 GPU 侧 latency 主结论：通常不是第一优先级

对 **profile 文件体量** 来说：

- 也不是主要来源
- 当前这次 `.sqlite` 里 `OSRT_API` 只有大约 `4.2 万` 条，约 `1.6 MB`
- 和几百万级别的 CUDA Runtime / Kernel / Memcpy 相比，小很多

所以：

- `osrt` 不是当前 `419M` 文件的主因
- 去掉它会稍微清爽一点，但不是决定性缩小文件的手段

#### 不记录它，会对分析遗漏什么

如果你不记 `osrt`，通常不会影响你对下面这些核心问题的判断：

- H2D copy 重不重
- DtoD promotion 多不多
- expert compute 多长
- prefetch 和 compute 有没有 overlap
- 显存峰值和变化趋势如何

因为这些主要看：

- CUDA timeline
- NVTX
- `--cuda-memory-usage=true`

就够了。

但如果你不记 `osrt`，你会少掉一类“CPU 侧等待原因”的信息，例如：

- 某段 GPU idle，到底是不是主线程/工作线程在 `poll`
- 某段等待是不是线程在 `pthread_cond_wait`
- 启动阶段是不是有文件读取/映射开销
- 某些停顿是不是 `ioctl` 这种驱动交互导致

也就是说：

- **不记 `osrt`，通常不会影响你做 MoE offloading 的第一轮 GPU latency / memory 分析**
- **但如果你后来发现 GPU timeline 有空洞，而且 CUDA/NVTX 不足以解释，`osrt` 会有帮助**

#### 一句话结论

对当前项目，如果你关注的是：

- 推理延迟
- H2D / prefetch / compute 的关系
- GPU 显存峰值和变化

那 `osrt` 可以先不记。

只有在你进一步怀疑：

- CPU 线程调度
- Python 侧阻塞
- 文件 I/O / 驱动等待

时，再把 `osrt` 打开更合理。

---

## 12. 两个进一步的概念澄清

下面再专门回答两个容易混淆的问题：

1. `dispatch/reorder` 到底是什么，和 routing / loading / compute 的关系是什么。
2. `cache promotion` 里的 `DtoD` 到底是什么，是不是 HBM -> SRAM 的实际计算加载。

### 12.1 `dispatch/reorder` 具体指什么

这里的 `dispatch/reorder` 不是在说“加载 expert 权重”，也不是在说“expert MLP 真正开始算”。

它更准确地指：

- 把 `routing` 阶段决定好的 token -> expert 分配关系
- 重新整理成适合 expert 批量计算的张量布局
- 以及在计算结束后，再把 expert 输出重新加回 token 位置

所以它夹在：

- `routing`
- `expert loading`
- `expert compute`

之间，起的是“数据重排和搬运布局整理”的作用。

#### 12.1.1 它和其他阶段的关系

这几类阶段可以这样理解：

- `routing`
  - 决定“哪些 token 应该去哪些 expert”
- `expert loading`
  - 决定“这些 expert 的权重从哪里拿，需不需要从 CPU 搬到 GPU”
- `dispatch/reorder`
  - 决定“把 token 的输入怎么整理成 expert 能高效吃的布局”
- `expert compute`
  - 真正用 expert 权重做 MLP 计算

换句话说：

- `routing` 解决的是“逻辑分配”
- `dispatch/reorder` 解决的是“张量排布”
- `expert compute` 解决的是“数值计算”

#### 12.1.2 token -> expert 重排是什么意思

先举一个最小例子。

假设当前层有 3 个 token：

- `token0`
- `token1`
- `token2`

并且每个 token 选择 `top_k = 2` 个 expert。

`routing` 结果假设是：

```text
token0 -> expert3, expert9
token1 -> expert3, expert18
token2 -> expert9, expert18
```

这时按 token 顺序展开后，原始分配关系大致像这样：

```text
flat_experts = [3, 9, 3, 18, 9, 18]
token_ids     = [0, 0, 1,  1, 2,  2]
```

这说明：

- 第 1 个分配项：`token0 -> expert3`
- 第 2 个分配项：`token0 -> expert9`
- 第 3 个分配项：`token1 -> expert3`
- 第 4 个分配项：`token1 -> expert18`
- 第 5 个分配项：`token2 -> expert9`
- 第 6 个分配项：`token2 -> expert18`

但这样的顺序并不适合专家批量计算，因为同一个 expert 对应的 token 是分散的。

所以代码会做一次“按 expert 排序”的重排：

```text
sorted_experts = [3, 3, 9, 9, 18, 18]
sorted_tokens  = [0, 1, 0, 2,  1,  2]
```

这一步的意义就是：

- 把原来“按 token 排列”的 expert 分配
- 改成“按 expert 分组排列”

这样后面就能很方便地说：

- expert3 需要处理 `token0, token1`
- expert9 需要处理 `token0, token2`
- expert18 需要处理 `token1, token2`

这就是我前面说的 `token -> expert 重排`。

对应当前代码的 NVTX 主要是：

- `Global_Expert_Routing_Prep_GPU_Sort`

#### 12.1.3 gather 输入是什么意思

有了上面的重排信息后，下一步要真正把 token 的 hidden states 收集出来。

还是上面的例子。

假设当前层输入是：

```text
hidden_states = [h0, h1, h2]
```

那么根据重排后的 `sorted_tokens = [0, 1, 0, 2, 1, 2]`，代码会 gather 出：

```text
all_input_states = [h0, h1, h0, h2, h1, h2]
```

这表示：

- 给 expert3 的输入是 `[h0, h1]`
- 给 expert9 的输入是 `[h0, h2]`
- 给 expert18 的输入是 `[h1, h2]`

也就是说：

- 同一个 token 可能会被复制多份
- 每一份送到不同的 expert

这一步就是 `gather 输入`。

它不是在搬 expert 权重，而是在搬：

- token 的 activation / hidden state

对应当前代码里主要落在：

- `Expert_Input_Gather_Prep`

#### 12.1.4 scatter 输出是什么意思

expert 计算结束后，每个 expert 会给自己收到的 token 产出一份输出。

还是上面的例子。

假设：

- expert3 输出 `[o(0,3), o(1,3)]`
- expert9 输出 `[o(0,9), o(2,9)]`
- expert18 输出 `[o(1,18), o(2,18)]`

那么最终要恢复到 token 视角：

- `token0` 的总输出 = `o(0,3) + o(0,9)`
- `token1` 的总输出 = `o(1,3) + o(1,18)`
- `token2` 的总输出 = `o(2,9) + o(2,18)`

这一步就是 `scatter 输出`，当前代码里是：

- 用 `scatter_add_` 按 token 索引把各 expert 的输出加回去

对应 NVTX 是：

- `Batch_Result_Scatter`

#### 12.1.5 为什么我建议把它单独列成一类

因为它和 `routing`、`loading`、`compute` 都不一样：

- 它不是选 expert
- 它不是搬 expert 权重
- 它也不是 expert MLP 本身

但它会产生真实开销，而且在 token 数多、active experts 多时可能不小。

如果你不单列，最容易出现两种误判：

1. 把它错算到 `routing`
   - 会让你以为 router 很慢，其实一部分时间花在 GPU 排序和 gather/scatter 上

2. 把它错算到 `expert compute`
   - 会让你以为 expert 算得慢，其实一部分时间花在输入整理和结果回填上

所以从分析角度看，`dispatch/reorder` 单列是合理的。

### 12.2 `cache promotion` 的 DtoD 到底是什么

这里的 `cache promotion` 指的是：

- 某个 expert 这次先被临时加载到了 `GPU swap buffer`
- 然后缓存策略决定“这个 expert 值得长期留下来”
- 于是把它从 `swap buffer` 复制到 `GPU cache` 的常驻 slot

当前代码里这一步就是：

- `swap buffer -> cache slot`
- 使用 `copy_`
- 属于 GPU-to-GPU memcpy

也就是：

- **HBM 里的一个区域，复制到 HBM 里的另一个区域**

当前实现对应代码是：

- `expert_buffer_manager.py` 里的 `_update_topk_lru_cache(...)`
- `gpu_expert_cache.py` 里的 `update_cache_from_buffers(...)`
- `gpu_expert_cache.py` 里的 `_copy_buffer_to_cache_slot(...)`

#### 12.2.1 它不是 HBM -> SRAM 的实际计算 load

不是。

这个区别非常重要。

我这里说的 `DtoD` 指的是：

- 一个显式的 CUDA memory copy
- 从 GPU 全局内存里的 `swap buffer` 区域
- 复制到 GPU 全局内存里的 `cache slot` 区域

这两个区域都在：

- GPU HBM / 全局显存

它不是在说：

- kernel 执行时，硬件自动把某段权重从 HBM 拉到 L2 / shared memory / register

后者属于：

- GPU 微架构内部的数据层次访问
- 是 kernel 内部的 load/store 行为

通常不会在 `Nsight Systems` 里表现成：

- 一条显式的 `[CUDA memcpy Device-to-Device]`

如果你想看那种 HBM -> SRAM / shared memory / register 的利用效率，通常要用：

- `Nsight Compute`

而不是主要靠 `Nsight Systems`。

#### 12.2.2 一个直观例子

假设第 `10` 层当前需要 expert `9`。

这次它不在 GPU cache 里，于是发生：

1. CPU expert cache -> GPU swap buffer  
   这是 `Host-to-Device`

2. 当前层计算直接用 swap buffer 里的这份权重  
   这里如果 kernel 需要，会在执行中由硬件自己把数据从 HBM 拉进片上缓存/寄存器  
   但这不是 `Nsight Systems` 里那种显式 memcopy

3. 缓存策略判断 expert `9` 很热，值得常驻  
   于是执行：

```text
GPU swap buffer slot X -> GPU cache slot Y
```

这一步才是我说的：

- `cache promotion`
- `DtoD`

所以这里有两个完全不同层面的“load”：

- `swap buffer -> GPU cache`
  - 显式 DtoD memcpy
  - 能在 `Nsight Systems` 的 memops 里看到

- `HBM -> SRAM / register`
  - kernel 内部访存
  - 不会作为单独的 DtoD memcpy 出现在 `Nsight Systems` 里

#### 12.2.3 为什么它值得关注

因为它会带来额外的 GPU 内部流量和时间。

如果 cache policy 比较激进，或者 working set 波动很大，那么：

- 很多 expert 会反复被 promotion
- `Device-to-Device` 总流量会升高
- decode 阶段可能会被额外 DtoD 拖慢

所以它和单纯的 H2D 不同，但同样可能影响 latency。

不过也要注意：

- `DtoD` 不是都等于 `cache promotion`
- 只是当前项目里，`cache promotion` 是一个很明确、很重要的 DtoD 来源

#### 12.2.4 一句话总结

`dispatch/reorder` 说的是：

- token 和 expert 之间的数据排布整理
- 包括按 expert 分组、gather 输入、scatter 输出

`cache promotion` 说的是：

- 已经在 GPU HBM 里的 expert
- 从 `swap buffer` 再复制到 `GPU cache`

它是：

- **GPU HBM 内部的显式 DtoD memcpy**

而不是：

- **kernel 内部 HBM -> SRAM 的硬件访存加载**

---

## 13. 当前代码里已经落地的 Nsight 配置

当前代码里，针对 Qwen3 路径，内部 NVTX 已经收敛为下面这 10 类：

- `Attention_Layer{i}`
- `Routing_Layer{i}`
- `Current_Layer_Availability_Check_Layer{i}`
- `Current_Layer_Miss_Load_Layer{i}`
- `Next_Layer_Prefetch_Layer{i}`
- `Reorder_Layer{i}`
- `Gather_Layer{i}`
- `Expert_Compute_Layer{i}`
- `Scatter_Layer{i}`
- `Cache_Promotion_Layer{i}`

此外，外层 sample 范围仍然保留：

- `Benchmark_Run`
- `Sample_i_Total`
- `Sample_i_Prefill`
- `Sample_i_Decode`

而外部全局显存记录是一个命令行可选项：

- 默认关闭
- 开启参数：`--track-gpu-memory`
- 可选输出路径：`--gpu-memory-output <csv_path>`

### 13.1 只抓第一个 sample，总范围，不录模型构建，不录 osrt

如果你要的是：

- 只抓第一个 sample
- 允许范围是 `Sample_1_Total`
- 不录模型构建
- 开启 `--cuda-memory-usage=true`
- 不采 `osrt`

推荐命令如下：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --output profiles/qwen3_sample1_only \
  -e CUDA_VISIBLE_DEVICES=0 \
  python tests/test_baseline.py \
    --model qwen3moe \
    --base-model-path /data/models/Qwen3-30B-A3B \
    --benchmark gsm8k \
    --start-idx 0 \
    --num-samples 10 \
    --max-new-tokens 128 \
    --max-length 8192 \
    --enable-gpu-cache \
    --cache-policy topk_lru \
    --cache-slots-per-layer 16 \
    --topk-lru-logit-percentile 90.0 \
    --nsys-profile-single-sample 1 \
    --nsys-use-cuda-profiler-api
```

说明：

- 这里不再使用 `--capture-range=nvtx --nvtx-capture=Sample_1_Total`
- 我已经在当前环境里做过最小复现实验，`torch.cuda.nvtx.range(...)` 能被正常记录，但不能稳定触发 `nsys profile` 的 named NVTX capture，所以会出现 `Processing events... Generated: No reports were generated`
- 现在改成 `cudaProfilerStart/Stop` 作为 capture 触发，可靠得多
- 这里也刻意去掉了 `--show-output=false`，这样你还能在终端看到原本的
  - `Building qwen3moe model with offloading...`
  - `Loaded 10 prompts`
  - `Processing Sample 1/10`
  - `OVERALL STATISTICS`
  - `GPU Cache Statistics`

### 13.2 如果还想顺便记录全程的外部显存快照

就在 Python 命令末尾加：

```bash
--track-gpu-memory
```

如果还想指定 CSV 输出路径，再加：

```bash
--gpu-memory-output profiles/qwen3_gsm8k_10_gpu_memory.csv
```

### 13.3 可直接复制的一行命令

当前代码里，**不启用 `nsys` 且不手动传 `--enable-nvtx-ranges` 时，NVTX 默认关闭**，因此下面这条就是现在的纯净 baseline 单行命令：

```bash
CUDA_VISIBLE_DEVICES=1 python -u tests/test_baseline.py --model qwen3moe --base-model-path /data/models/Qwen3-30B-A3B --benchmark gsm8k --start-idx 0 --num-samples 10 --max-new-tokens 128 --max-length 8192 --enable-gpu-cache --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 | tee logs/qwen3_gsm8k_10.log
```

如果你要跑 **带 `nsys`、只监测第一个 sample** 的单行命令，用下面这条：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --force-overwrite=true --output profiles/qwen3_sample1_only -e CUDA_VISIBLE_DEVICES=1 python -u tests/test_baseline.py --model qwen3moe --base-model-path /data/models/Qwen3-30B-A3B --benchmark gsm8k --start-idx 0 --num-samples 10 --max-new-tokens 128 --max-length 8192 --enable-gpu-cache --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --nsys-profile-single-sample 1 --nsys-use-cuda-profiler-api
```

如果你还想同时做外部全局显存快照记录，就在第二条命令末尾继续加上：

```bash
--track-gpu-memory --gpu-memory-output profiles/qwen3_gsm8k_10_gpu_memory.csv
```
