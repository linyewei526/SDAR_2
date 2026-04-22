# Qwen3 MoE Offloading 中 GPU Cache / GPU Swap Buffer 说明与研究备注

本文回答四个相关问题，并统一整理到一份文档里：

1. 当前默认测试配置下，`GPU cache` 和 `GPU swap buffer` 的容量分别是多少，怎么计算。
2. 这两块 GPU 区域是不是在模拟真实 GPU HBM 的划分。
3. 在一次 decode 过程中，`GPU cache -> prefetch/swap buffer -> CPU->GPU 临时加载 -> 计算 -> cache promotion -> 下一层 prefetch` 是怎么工作的。
4. 当 `GPU swap buffer` 同时承载“当前层计算”和“下一层 prefetch”时，如果极端情况下空间不够，当前代码怎么处理；以及“热点专家缓存 + 预取”这种组合在 MoE offloading 研究里是不是常见做法。

本文基于当前代码目录：

- `/data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading`

以及当前 Qwen3 测试路径：

- `/data/models/Qwen3-30B-A3B`

---

## 1. 直接结论

如果只看当前文档里推荐的默认 Qwen3 测试配置：

- `GPU swap buffer` 默认是 `128` 个 expert slot，大约 `1.125 GiB`
- `GPU cache` 默认是每层 `16` 个 slot，48 层共 `768` 个 slot，大约 `6.75 GiB`

两者合起来大约：

```text
7.875 GiB
```

这表示：

- `GPU swap buffer` 是 GPU 上的临时 expert 周转区
- `GPU cache` 是 GPU 上的热点 expert 常驻区

它们确实是在**概念上模拟** HBM 里的“两块不同用途区域”，但不是一个完整的“整卡 HBM 统一预算器”。

对于你问的极端情况：

- 当前代码**保证当前层最多 128 个 unique experts 能装下**
- 但**不保证**“当前层 + 下一层 prefetch”同时都能装下
- 空间不够时，当前代码的处理方式是：
  - 当前层计算优先
  - 超出的 prefetch 直接失败并被丢弃
  - 不会扩容，不会 spill，不会做复杂腾挪，也不会抛异常中止

另外，`热点 expert 常驻缓存 + 预取` 这个思路**不是这份代码独有的创新点**，而是 MoE offloading/serving 研究中的一个常见设计方向；只是不同工作在“缓存保留多久、预取多远、是否完全去缓存”这几个维度上选择不同。

---

## 2. 当前代码里的默认容量

### 2.1 默认 `GPU swap buffer` 容量

Qwen3 builder 里直接把 `buffer_size` 设成了该层的总专家数：

- `buffer_size = original_num_experts = 128`

对应代码：

- `baseline/qwen3_builder.py:97-106`

也就是说，默认的 `GPU swap buffer` 有：

- `128` 个 slot

### 2.2 默认 `GPU cache` 容量

主脚本里 Qwen3 的默认参数是：

- `cache_policy = topk_lru`
- `cache_slots_per_layer = 16`

对应代码：

- `tests/test_baseline.py:28-35`

所以：

- 每层缓存 `16` 个 expert
- Qwen3 一共有 `48` 层
- 因此总 cache slot 数是：

```text
48 * 16 = 768
```

### 2.3 单个 expert 大小怎么算

当前 Qwen3 模型的关键配置是：

- `hidden_size = 2048`
- `moe_intermediate_size = 768`
- `num_experts = 128`
- `num_hidden_layers = 48`
- `num_experts_per_tok = 8`

Qwen3 一个 expert 有三块权重：

- `gate`
- `up`
- `down`

所以单 expert 参数量是：

```text
3 * hidden_size * moe_intermediate_size
= 3 * 2048 * 768
= 4,718,592 parameters
```

当前实现里用的是 `bfloat16`，即每个参数 `2 bytes`，所以：

```text
4,718,592 * 2 bytes = 9,437,184 bytes = 9 MiB
```

### 2.4 最终容量数字

因此：

`GPU swap buffer`

```text
128 * 9 MiB = 1152 MiB = 1.125 GiB
```

`GPU cache`

```text
768 * 9 MiB = 6912 MiB = 6.75 GiB
```

两者合计：

```text
1.125 GiB + 6.75 GiB = 7.875 GiB
```

注意这个数只包含 offloading 相关的 expert 区域，不包含：

- dense 主干权重
- KV cache
- activation
- CUDA runtime/workspace

所以实际总显存占用一定比 `7.875 GiB` 更大。

---

## 3. 这是不是在模拟真实 GPU HBM 的划分

可以说：

- **是概念上的模拟**

但不是完整精确的 HBM 总预算模型。

更准确地讲，这份代码在 GPU 上显式预分配了两块连续内存池：

1. `GPU swap buffer`
2. `GPU cache`

它对应的现实系统含义是：

- 在 HBM 里预留一块临时 expert 周转区
- 再预留一块长期保留热点 expert 的常驻区

因此它确实体现了真实系统里常见的内存分层思路：

- CPU pinned memory 放全量 experts
- GPU HBM 里留一块临时区做 CPU->GPU demand loading / prefetch
- GPU HBM 里再留一块热点缓存区降低重复搬运

但它没有做这些更完整的事：

- 不会从整张卡总 HBM 容量自动推导 `swap buffer` 和 `cache` 的最佳大小
- 不会把 dense 权重、KV cache、activation 等一起纳入统一预算
- 不会做真实部署系统里那种全局显存调度与竞争控制

所以最合适的理解是：

- **它在软件里手工划出两块 GPU 区域，模拟 offloading runtime 对 HBM 的分工**

而不是：

- **它精确重建了真实 GPU HBM 的全部组成和调度**

---

## 4. 一次 decode 里，GPU cache 和 GPU swap buffer 怎么工作

先看当前 Qwen3 路径的几个关键事实：

- 当前测试主脚本是单样本、单 token 自回归 decode
- 每层真实路由会选 `top_k = 8` 个 expert
- 下一层预取只取 `PREFETCH_TOPK = 4`

对应代码：

- `baseline/qwen3_layers.py:18-20`
- `baseline/qwen3_layers.py:132-149`

### 4.1 一个具体例子

假设某个 decode token 走到第 `10` 层时，router 最终选中：

```text
{3, 9, 18, 27, 51, 80, 93, 101}
```

也就是这一层真正要计算的 8 个 experts。

这时实际顺序如下。

### 4.2 第一步：先查 GPU cache

系统先检查这 8 个 expert 里，哪些已经常驻在第 10 层的 GPU cache 里。

比如命中了：

```text
{3, 18, 51, 80}
```

这些 expert 直接就能用于计算，不需要再从 CPU 搬。

对应代码：

- `baseline/expert_buffer_manager.py:471-485`

### 4.3 第二步：再查 prefetch 结果

假设上一层已经预取过：

```text
{27, 93}
```

那么这两个会从 `prefetch_mapping` 命中，也不需要现拷。

到这里为止：

- `GPU cache` 命中 4 个
- `prefetch/swap buffer` 命中 2 个

对应代码：

- `baseline/expert_buffer_manager.py:487-507`

### 4.4 第三步：剩余 miss 从 CPU 拷到 swap buffer

现在还剩：

```text
{9, 101}
```

这两个不在 GPU 上，于是运行时会：

1. 找 `GPU swap buffer` 的空闲 slot
2. 从 CPU expert cache 取权重
3. 复制到 `swap buffer`
4. 记录 `expert_id -> buffer_idx`

这才是真正的：

- CPU -> GPU 按需加载

对应代码：

- `baseline/expert_buffer_manager.py:509-528`

### 4.5 第四步：统一开始 expert 计算

这一步对计算逻辑来说，不再区分这些 expert 来自哪里，只看它们当前在 GPU 上的 view：

- cache 命中的，从 `GPU cache` 取 view
- 临时加载的，从 `swap buffer` 取 view

然后一起进入 batched expert compute。

对应代码：

- `baseline/qwen3_layers.py:239-241`
- `baseline/qwen3_layers.py:297-327`
- `baseline/expert_buffer_manager.py:417-433`

### 4.6 第五步：可选地做 cache promotion

如果策略是动态策略，比如当前默认的 `topk_lru`，那么像：

```text
{9, 101}
```

这种刚从 CPU 搬到 `swap buffer` 的 expert，有可能会被提升进 `GPU cache`。

这一步不是 CPU->GPU，而是：

- `swap buffer -> GPU cache`

也就是一次 GPU 内部复制。

对应代码：

- `baseline/expert_buffer_manager.py:530-535`
- `baseline/gpu_expert_cache.py:981-1050`

### 4.7 第六步：预取下一层到 swap buffer

当前层运行时，还会预测下一层可能用到哪些 expert。

比如它预测第 11 层可能会用：

```text
{5, 12, 33, 90}
```

那系统就会在 prefetch stream 上，尝试把它们提前放进 `GPU swap buffer`。

这样等第 11 层开始时，就有机会直接命中 prefetch。

对应代码：

- `baseline/qwen3_layers.py:252-255`
- `baseline/qwen3_layers.py:422-470`
- `baseline/expert_buffer_manager.py:635-663`

### 4.8 一句话总结

可以把两者记成：

- `GPU cache`：长期热点常驻区
- `GPU swap buffer`：当前层临时周转区 + prefetch 承接区

标准执行顺序就是：

```text
查 GPU cache
  -> 查 prefetch / swap buffer
  -> 剩余 miss 从 CPU 拷到 swap buffer
  -> 做 expert 计算
  -> 可选提升进 GPU cache
  -> 预取下一层到 swap buffer
```

---

## 5. 你担心的极端情况：当前层和下一层加起来超过 128 个 slot，会怎样

这是一个很好的问题。

你举的例子是：

- 当前层已经在 `GPU swap buffer` 里占了 `70` 个 expert
- 下一层 prefetch 还想再放 `65` 个
- 合计 `135`
- 超过了 `swap buffer` 的 `128`

### 5.1 先说当前默认 benchmark 里会不会发生

在**当前默认测试脚本**里，实际上几乎不会发生。

原因是当前 decode 是：

- 单 token 自回归

因此一层里：

- 当前层 active experts 最多 `8`
- 下一层预取 experts 最多 `4`

所以在默认 decode 路径下，最坏也就：

```text
8 + 4 = 12
```

离 `128` 还很远。

也就是说：

- **你举的 70 + 65 这种情况，不会出现在当前默认单 token decode benchmark 中**

### 5.2 但在 prefill 或多 token 并行场景里，确实可能发生

如果换成：

- prefill
- 多 token 并行解码
- 未来你接 SDAR 的块内并行场景

那当前层的 unique active experts 可能非常大，下一层预取的 unique experts 也会变大。

这时你说的“当前层 + 下一层 prefetch 一起超过 128”就完全可能发生。

### 5.3 当前代码是怎么处理的

当前代码的处理方式并不复杂：

1. `swap buffer` 是一个固定大小的共享池
2. 当前层加载完后，当前层占用的 temp slots 会一直保留到下一层真正开始
3. prefetch 再从这个共享池里继续找空闲 slot
4. 如果还有空位，就继续塞
5. 如果没有空位，`prefetch_expert()` 直接返回 `False`

对应代码：

- `baseline/expert_buffer_manager.py:513-519`
- `baseline/expert_buffer_manager.py:651-653`

也就是说：

- **当前层计算优先**
- **prefetch 只是“能塞多少塞多少”**

### 5.4 会不会报错或崩掉

对 prefetch 来说：

- 不会因为空间不够而报错中止
- 它只是装不下的那些 expert 不再预取

对当前层真正需要计算的 expert 来说：

- 如果当前层本身就超过 `buffer_size`，才可能报 `No free buffers available`

但 Qwen3 一层总共就 `128` 个 expert，而 `swap buffer` 也配成 `128`，所以：

- **当前层自身的 unique expert 上限是能保证装下的**

因此当前实现的逻辑是：

- 当前层容量上限有保证
- “当前层 + 下一层 prefetch”的联合容量没有保证

### 5.5 空间不够时，代码有没有更高级的处理

没有。

当前实现**没有**这些机制：

- 没有给 prefetch 预留固定配额
- 没有为了 prefetch 去主动驱逐当前层临时 expert
- 没有对 prefetch 做二次重排或重试
- 没有额外的 spill 区
- 没有因为“buffer 满了”而给出明确 warning

它本质上就是：

- best-effort prefetch

### 5.6 一个容易忽略的细节：overflow 时 prefetch 也不够“聪明”

当前 Qwen3 路径里，下一层预取集合是这样得到的：

1. 每个 token 先取下一层 top-`PREFETCH_TOPK`
2. 然后 flatten
3. 再做 `unique().tolist()`

对应代码：

- `baseline/qwen3_layers.py:146-149`
- `baseline/qwen3_layers.py:434`

这意味着：

- 当 prefetch 候选很多时，它并没有再根据分数做一次严格的全局优先级裁剪
- 如果 buffer 满了，最终能被装进去的只是“在当前遍历顺序下先拿到空位”的那些 experts

所以即使在多 token 场景里真的出现 buffer 紧张，当前实现也只是：

- **部分预取**
- 而不是“最优预取”

### 5.7 这说明了什么

说明这份代码更偏：

- 单 token AR decode 友好的工程原型

而不是：

- 为多 token 并行 decode / SDAR 精细设计过的严格 runtime

如果你后面把这套东西接到 SDAR 上，这里很可能就是一个必须重新设计的点。

---

## 6. `GPU cache + GPU swap buffer` 这样的组合，在研究里常见吗

### 6.1 结论

**常见。**

但要更精确一点地说：

- “CPU 放全量 experts，GPU 上保留一部分热点 experts，再配合预取或按需加载”是当前 MoE offloading/serving 研究里的一个常见设计方向
- 但它不是唯一方向

大致可以分成三类。

### 6.2 一类：按需加载 / 预取优先

这类工作尽量减少长期驻留的 GPU expert 状态，更强调：

- 当前层用谁就搬谁
- 下一层要用谁就预取谁

典型例子是 **Pre-gated MoE**。

它强调：

- 当前 block 和下一 block 的 activated experts
- 通过提前知道下一 block 的 active experts 来隐藏迁移延迟

它在内存分析里直接把峰值 GPU expert 内存写成“当前 block + 下一 block 的 active experts 之和”，而不是一个长期热点 cache 模型。

这更接近：

- prefetch/on-demand 路线

### 6.3 一类：缓存 + 预取结合

这类工作更接近当前仓库的思路。

典型例子是 **MoE-Infinity**。

它明确提出：

- sparse activation 下存在明显的 expert reuse
- 因此需要 expert cache
- 同时还要用预测来指导 replacement 和 prefetch

也就是说：

- 不是“只有预取”
- 也不是“只有缓存”
- 而是缓存和预取配合

**ProMoE** 也属于这个方向。

它直接把自己定义成：

- 在现有 cache-based offloading 之上，把 reactive miss 变成 proactive caching

这说明在研究语境里：

- “GPU 上留热点 experts”
- “再提前取下一批专家”

本来就是一条常见路线，不是这份代码独有。

### 6.4 还有一类：反过来，想把 cache 去掉

近一点的工作比如 **OD-MoE**，反而把“GPU 缓存一部分热门 experts”当成现有主流 baseline，然后尝试：

- 用更强预测
- 用更激进的 just-in-time loading

去实现接近 fully GPU-cached 的效果，同时减少 cache 占用。

这说明：

- cache + prefetch 很常见
- 甚至常见到后续工作开始专门思考“能不能不要 cache”

### 6.5 所以当前仓库是不是“创新点”

如果只看**大方向**：

- 不是独创

因为：

- 热点 expert 缓存
- expert prefetch
- CPU pinned memory + GPU hot cache + demand loading

这些都已经是相关研究和工程系统里反复出现的通用思路。

如果看**具体实现细节**：

当前仓库的实现仍然有自己的工程选择，例如：

- per-layer 固定 `16` 个 cache slot
- 一个全局共享的 `128` 槽 swap buffer
- 单步 next-layer prefetch
- `topk_lru / lru / lfu / tinylfu` 这些策略

这些具体参数和 runtime 组织方式，是这份代码自己的工程实现，不等于论文里某一家的原样复刻。

所以更准确的说法是：

- **“热点缓存 + 预取”不是这份代码的独有创新点**
- **这份代码的价值在于，把这类通行思路落成了一个可实验、可改造、适合你继续接 SDAR 的原型**

---

## 7. 你说“保留热点专家看起来比单纯 prefetch 更好”，这个判断对吗

我认为：

- **在单序列 AR decode、且 expert reuse 明显时，这个判断通常是对的**

原因是纯 prefetch 主要解决的是：

- 把“下一次马上要用”的 expert 提前搬上来

它更偏重：

- 隐藏延迟

而热点 cache 解决的是：

- 同一个 expert 在很多 token 上反复出现时，不要每次都重新搬

它同时影响：

- 延迟
- 总传输量

所以在 decode 中如果 expert reuse 很强：

- 热点 cache 往往比“只做一步 prefetch”更稳

但它也不是绝对更优，因为：

- cache 会占长期 HBM
- cache slot 给多了，会压缩临时区或别的显存预算
- 如果 reuse 并不强，长期保留热点的收益就可能不如更激进的 just-in-time prefetch

所以最终还是看：

- locality 强不强
- HBM 紧不紧
- 预测准不准

### 一个更准确的比较

`纯 prefetch`

- 优点：更轻、HBM 常驻开销小
- 缺点：主要针对“马上要用到谁”，不能很好利用跨很多 token 的长期复用

`热点 cache + prefetch`

- 优点：同时利用长期复用和短期预测
- 缺点：实现更复杂，占更多 HBM，需要 replacement policy

`cacheless / on-demand only`

- 优点：HBM 最省
- 缺点：更依赖预测和流水线，错误代价更高

所以当前仓库采取：

- `GPU cache + shared swap buffer + next-layer prefetch`

其实正是一个很典型的折中点。

---

## 8. 结合当前代码，最值得你记住的三点

### 8.1 对当前默认 benchmark

当前主脚本是：

- 单样本
- 单 token 自回归 decode

因此默认 decode 阶段里，`swap buffer` 的 128 槽其实是很宽裕的。

### 8.2 对 prefill 和未来多 token / SDAR 场景

当前实现没有认真处理：

- “当前层临时专家 + 下一层大规模预取”共同超过 `swap buffer` 容量

它的策略只是：

- 当前层优先
- prefetch best effort

### 8.3 对你后面接 SDAR 的直接启示

如果你下一步要做：

- 多 token 并行
- block 级并行验证
- 更深的跨层预测

那么你大概率需要重新设计：

1. `swap buffer` 的容量和分区
2. prefetch 的优先级排序
3. 当前层和下一层的 buffer quota
4. 命中 / miss / prefetch drop 的统计口径

---

## 9. 外部参考

下面这些工作能帮助你理解这份代码在研究图谱里的位置：

- MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache  
  https://arxiv.org/abs/2401.14361

- MoE-Infinity OpenReview PDF  
  https://openreview.net/pdf/683ee20a5f0b1a7b8c437a03ec6b0961d677527f.pdf

- ProMoE: Fast MoE-based LLM Serving using Proactive Caching  
  https://arxiv.org/abs/2410.22134

- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference  
  https://www.microsoft.com/en-us/research/uploads/prod/2024/05/isca24_pregated_moe_camera_ready.pdf

- OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference  
  https://arxiv.org/abs/2512.03927
