# Pure SDAR 与 SDAR MoE-Offloading 对照分析

## 实验口径

- 数据集: `opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799`
- 样本范围: `start_idx=0`, `num_samples=5`
- 生成参数: `gen_length=128`, `block_length=32`, `denoising_steps=32`, `threshold=0.95`
- Pure SDAR TPS 文件: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_dense_original_all_results.json`
- Pure SDAR profile 文件: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_dense_baseline_all.json`
- Offloading TPS 文件: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_offloading_unprofiled_all_results.json`
- Offloading summary 文件: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_summary_all.json`

## 关键结论

1. 原始 pure SDAR 的无埋点 TPS 不是和 offloading 差不多，而是明显更高。
2. Pure SDAR 无埋点 TPS = `3.239192`，Offloading 无埋点 TPS = `2.518023`。
3. 也就是说，Pure SDAR 相比 Offloading 约快 `28.64%`。
4. 如果把总生成时间折算到平均一个 decode layer 的墙钟时间，Pure SDAR 约为 `15.501 ms/layer`，Offloading 约为 `19.822 ms/layer`，Pure SDAR 约快 `27.88%`。
5. 因此，“提前把所有权重放到 GPU 上的 pure SDAR 没有比 offloading 快”这个判断，在无埋点 TPS 上并不成立。

## 为什么你会产生“看起来差不多”的感觉

### 1. Offloading summary 里的操作占比不是互斥墙钟占比

在 Offloading summary 里，`current_layer_miss_load` 和 `next_layer_prefetch` 都很大，但它们不是互斥的墙钟片段。

- `next_layer_prefetch` 运行在单独的 prefetch stream 上。
- 它会和当前层后半段的 `reorder/gather/expert_compute/scatter` 明显重叠。
- 所以虽然 summary 里它们的“平均累计时长”很大，但不能把这些时长直接加起来理解成 wall-clock 总耗时。

这也是为什么 Offloading summary 里会看到：

- `current_layer_miss_load` 平均占 layer `52.42%`
- `next_layer_prefetch` 平均占 layer `64.96%`

两者已经明显超过 `100%`，说明这里本来就在统计重叠执行的区间，而不是互斥时间片。

### 2. Offloading 并不是每层都要从 CPU 重新搬运全部激活专家

Offloading 的平均专家统计是：

- 每层激活专家数: `57.864345`
- GPU cache 命中: `14.109176`
- prefetch 命中: `27.428992`
- 真正 CPU miss load: `16.326177`

也就是说，平均每层大约只有 `28%` 左右的激活专家需要真正同步 miss load，剩下大约 `72%` 已经被 `GPU cache + prefetch` 覆盖。

所以 Offloading 的真实代价不是“把 57 个专家全都从 CPU 搬到 GPU”，而更接近“每层只对其中约 16 个专家支付真正暴露在关键路径上的 miss load 成本”。

### 3. 原始 pure SDAR 本身也不是一个轻量解码路径

从 Pure SDAR 的 profiled 分解看，平均一个 decode layer 里：

- `attention`: `0.900 ms`
- `routing`: `0.119 ms`
- `moe_dispatch`: `7.630 ms`
- `moe_expert_compute_hbm_fetch`: `8.382 ms`
- `moe_scatter`: `2.105 ms`
- `other_layer_overhead`: `4.731 ms`

Pure SDAR 的主要时间根本不在 attention，而是在 MoE 路径本身。

更关键的是，原始 `modeling_sdar_moe.py` 的 MoE 实现本身就比较低效：

- 它在 `SDARMoeSparseMoeBlock.forward()` 里对 `range(self.num_experts)` 的 `128` 个专家逐个循环。
- 每个专家都会做 `torch.where`、token gather、expert MLP、`index_add_` scatter。
- 这条路径即使所有权重常驻 GPU，也仍然会产生大量小粒度 kernel、dispatch/scatter 开销和 HBM 读权重开销。

所以 pure SDAR 并不是“没有 CPU->GPU copy，所以只剩 attention 和一点点 compute”。它本身就是一个 MoE-dominated 的推理路径。

### 4. Offloading runtime 在 MoE 执行层面比原始 dense baseline 更高效

这是这次对照里最重要的一点。

Offloading 版本并不只是“把原始 SDAR 再加上 CPU->GPU copy”。它实际上还把原始的 MoE 执行方式换成了一套更高效的 runtime：

- 只针对激活专家处理，不再像原始实现那样对 128 个专家逐个 Python 循环
- 使用 `reorder/gather/batched expert compute/scatter`
- 同时把下一层 prefetch 放到独立 stream 上和当前层计算重叠

所以 Offloading 版本一方面多了专家搬运，另一方面也显著优化了原始 SDAR 里那条低效的 MoE 执行路径。这两部分效应会互相抵消一部分。

## 应该怎样解读这次结果

- 如果看真正无埋点 TPS：Pure SDAR 还是明显快于 Offloading，约快 `28.64%`。
- 如果看 profile summary：会看到 Offloading 的 copy 类操作很重，但它们有较强重叠，而且 Offloading 的 MoE compute 路径本身更高效。
- 因此最终不会出现“CPU->GPU 搬运很重，所以 Offloading 必然比 pure SDAR 慢很多很多”这种线性关系。

## 注意事项

1. Pure SDAR 的层级分解来自 `modeling_sdar_moe_profiled.py`，它在每个专家循环上都插了 CUDA event，因此 profile 版的绝对耗时会高于原始 pure SDAR 无埋点 TPS；这里适合看结构，不适合直接拿 profile 版 TPS 当真实吞吐。
2. Offloading summary 的操作时长包含跨 stream 重叠，不能简单把各项时长相加解释为墙钟总时间。
3. 真正用于吞吐对照的应当是：
   - Pure SDAR 无埋点: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_dense_original_all_results.json`
   - Offloading 无埋点: `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading/profiles/sdar_offloading_unprofiled_all_results.json`
