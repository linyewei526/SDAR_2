# Qwen3 MoE Offloading 代码与执行链路详解

本文只聚焦 `MoE-Offloading` 项目里的 **Qwen3** 路径，目标不是泛泛介绍 MoE，而是把下面四件事彻底讲清楚：

1. 在一个 benchmark 上，这套代码是怎么模拟和评估 Qwen3 的 GPU-CPU MoE offloading 的。
2. `CPU expert cache`、`GPU swap buffer`、`GPU cache` 分别是什么，容量怎么来的，参数在哪里设，对应真实系统里的什么组件。
3. 从命令行启动，到构建模型，到跑一个样本，到跑一个 token，到某一层 MoE 里一次 expert 加载，这条链路每一步代码到底在做什么，对应真实 GPU-CPU offloading 里的什么执行阶段。
4. 代码里评估了哪些指标，这些指标是怎么从执行链路里算出来的，具体代表什么意思。

这份文档假设你只知道“MoE offloading 的基础含义”，不知道 GPU-CPU 真正怎么跑。因此文档会同时讲：

- 真实系统视角
- 代码实现视角
- 二者的一一对应关系

---

## 0. 先给一个总图

如果只记住一句话，这套 Qwen3 offloading 代码的核心就是：

> **把 Qwen3 的 dense 主干权重常驻 GPU，把所有 expert 权重常驻 CPU pinned memory；推理时每一层先用 router 决定当前 token 需要哪些 experts，再按“GPU cache -> prefetch -> CPU->GPU 临时加载”的顺序把这些 experts 搬到 GPU 上计算。**

可以先看下面这个概念图：

```text
CPU 端（存全量 experts）
┌──────────────────────────────┐
│ CPU expert cache             │
│ 所有层、所有 experts 的权重    │
│ Qwen3 中是 gate/up/down 三块  │
└──────────────┬───────────────┘
               │ H2D copy（miss 时）
               ▼
GPU 端（临时周转区）
┌──────────────────────────────┐
│ GPU swap buffer              │
│ 当前层临时加载的 experts      │
│ 也承接 prefetch 预取结果      │
└──────────────┬───────────────┘
               │ 需要长期保留时，GPU→GPU copy
               ▼
GPU 端（热点常驻区）
┌──────────────────────────────┐
│ GPU cache                    │
│ 每层固定若干个热点 experts    │
│ 支持 static/lru/lfu/...      │
└──────────────────────────────┘
```

真实运行时，一层 MoE 的顺序可以概括成：

```text
router 选专家
    -> 查 GPU cache
    -> 查 prefetch 结果
    -> 缺的从 CPU expert cache 拷到 GPU swap buffer
    -> 用这些 experts 做 MLP 计算
    -> 可选：把这次新加载的 expert 提升进 GPU cache
    -> 可选：预测下一层专家并做 prefetch
```

这正是这份代码库在做的事情。

---

## 1. 真实 GPU-CPU 跑 MoE offloading，到底在做什么

这一节先不看代码，只讲真实系统。

### 1.1 先把 MoE 模型拆成两类权重

一个 MoE 大模型可以粗略拆成两部分：

1. **dense 主干**
   - embedding
   - attention
   - layernorm
   - residual
   - lm head
   - router/gate

2. **expert 权重**
   - MoE MLP 中的 experts

真实部署时，如果 GPU 放不下全部 experts，常见做法就是：

- dense 主干常驻 GPU
- experts 常驻 CPU 内存
- 每次只把当前 token 真正会用到的 experts 搬上 GPU

这就叫 **expert offloading**。

### 1.2 一次 token 生成时，真实系统会做什么

以一个新 token 的解码为例，真实 GPU-CPU offloading 系统通常会经历这些阶段：

1. 当前 token 进入某一层 MoE。
2. router 在 GPU 上算出“这个 token 该走哪几个 experts”。
3. 系统检查这几个 experts 现在是否已经在 GPU 上。
4. 已在 GPU 上的，直接用。
5. 不在 GPU 上的，从 CPU 把对应权重搬到 GPU。
6. 用这几个 experts 做 MLP 计算。
7. 为了让后续 token 更快，系统可能会把热点 expert 留在 GPU 上，或者提前搬运下一层可能会用到的 expert。

所以 offloading 的本质不是“每次把整层搬上来”，而是：

- **router 先决定要用谁**
- **系统再按需搬运这些 experts**

### 1.3 这套代码“仿真”的到底是什么

这里要非常明确：

- 这套代码并不是纯数学公式推延迟的“离散事件模拟器”
- 它是真实执行 CUDA 计算和真实执行 CPU->GPU 拷贝的实验框架

也就是说，它的“仿真”更准确地说是：

- **用真实 CUDA tensor 和真实 H2D copy，把“expert 常驻 CPU、按需上 GPU”的运行方式复现出来**

因此：

- benchmark 里的延迟是墙钟时间（wall-clock）
- CPU->GPU copy 也是实际发生的
- 只有 `tests/adapeagle_backup/theoretical_tps.py` 才是纯理论估算

这一点很重要。因为后面你看到的 `decode_time`、`TPS`、`hit rate`，都不是纸上推导，而是当前机器真跑出来的。

---

## 2. Qwen3 路径里最重要的三个“存放区”

用户最容易混淆的地方就是这三个名词。这里先彻底讲清楚。

---

## 2.1 CPU expert cache 是什么

### 从真实系统看

它对应真实部署里的：

- **CPU 端全量 expert 权重仓库**

也就是：

- 所有层的所有 experts 都放在 CPU 内存里
- GPU 上并不常驻这些 experts
- 需要时再从 CPU 搬上 GPU

### 从代码看

Qwen3 路径里，`CPU expert cache` 由 `baseline/expert_cache.py` 里的 `ExpertCache` 管理：

- `self.simple_expert_cache`
- `self.cpu_pinned_storage`
- `self.cpu_expert_offsets`

关键代码在：

- `baseline/expert_cache.py:84-109`
- `baseline/expert_cache.py:289-421`

### 为什么名字叫 cache，但它其实更像“仓库”

这个名字容易误导。

在当前 Qwen3 代码里，CPU expert cache **不是会淘汰的 cache**，而是：

- 存放全部 expert 权重的主仓库

更准确的理解应该是：

- `CPU expert repository`

只是代码里沿用了 `cache` 这个词。

### Qwen3 里存的是什么

Qwen3 expert 不是一个打包大 tensor，而是三块独立权重：

- `gate`
- `up`
- `down`

对应代码：

- `baseline/expert_cache.py:423-430`
- `baseline/expert_cache.py:493-518`

也就是说，在 Qwen3 路径下，CPU cache 里的 key 形态是：

```python
(layer_idx, expert_idx, "gate")
(layer_idx, expert_idx, "up")
(layer_idx, expert_idx, "down")
```

### 为什么还要用 pinned memory

代码里会预分配一整块 CPU pinned memory：

- `baseline/expert_cache.py:390-421`

真实系统里，pinned memory 的作用是：

- 让 CPU->GPU 异步 DMA copy 更高效
- 减少 pageable memory 带来的额外 staging 开销

所以从真实系统的对应关系看：

- `cpu_pinned_storage` 就是在模拟“用于 expert offloading 的 host-side pinned buffer”

### 容量怎么算

Qwen3 每个 expert 有三块矩阵：

- `gate_proj`: `intermediate_size x hidden_size`
- `up_proj`: `intermediate_size x hidden_size`
- `down_proj`: `hidden_size x intermediate_size`

单 expert 参数量：

```text
3 * hidden_size * intermediate_size
```

Qwen3 当前代码默认对应的典型配置是：

- `hidden_size = 2048`
- `moe_intermediate_size = 768`
- `num_layers = 48`
- `num_experts = 128`
- `dtype = bfloat16 = 2 bytes`

这些值在本机可见的同类 Qwen3 配置中可直接看到：

- `hidden_size`: `config.json:12`
- `moe_intermediate_size`: `config.json:19`
- `num_experts`: `config.json:22`
- `num_experts_per_tok`: `config.json:23`
- `num_hidden_layers`: `config.json:24`

所以：

- 单 expert 大小约 `9 MB`
- 全模型所有 experts 大小约 `54 GB`

这也正好解释了为什么必须 offload：

- 全量 experts 体量太大，不适合直接常驻单卡 GPU

### 一个重要实现细节

Qwen3 的构建阶段里，expert 权重整理到 CPU pinned storage 的过程并不是“checkpoint 直接读入 pinned memory”，而是：

1. 先从 safetensors 文件读到普通 CPU tensor
2. 再拷到 GPU 临时 buffer
3. 再从 GPU 临时 buffer 回写到 CPU pinned storage

对应代码：

- `baseline/expert_cache.py:342-386`

这一步是 **初始化阶段的一次性整理动作**，不是 decode 时每 token 的搬运路径。

也就是说：

- 这一步对应真实系统里的“启动时整理权重布局”
- 不对应在线推理阶段的 offloading 开销

---

## 2.2 GPU swap buffer 是什么

### 从真实系统看

它对应真实部署里的：

- **GPU 上的临时中转区 / staging area / scratch space**

作用是：

- 当某个 expert 不在 GPU cache 中时
- 先把它从 CPU 搬到这个临时区域
- 当前层算完后，这些位置可以被下一层覆盖复用

### 从代码看

它由 `baseline/expert_buffer_manager.py` 管理。

关键对象：

- `self.gpu_memory_pool`
- `self.gpu_buffers`
- `self.buffer_status`
- `self.current_layer_mapping`

对应代码：

- `baseline/expert_buffer_manager.py:202-228`
- `baseline/expert_buffer_manager.py:294-399`

### 它和 GPU cache 的区别

这个区别一定要搞清楚：

- `GPU swap buffer`：临时的、当前层用完就可以覆盖
- `GPU cache`：长期的、希望跨 token / 跨 forward 保留热点 expert

在真实系统里可以理解为：

- swap buffer 像“临时卸货区”
- GPU cache 像“常驻仓位”

### Qwen3 下它装什么

Qwen3 的每个 buffer slot 里放一整个 expert 的三块权重：

- `gate_proj`
- `up_proj`
- `down_proj`

对应代码：

- `baseline/expert_buffer_manager.py:374-397`

### 容量怎么定

Qwen3 builder 里，临时 buffer 数量不是命令行参数，而是硬编码成：

```python
buffer_size = original_num_experts
```

对应代码：

- `baseline/qwen3_builder.py:97-106`

也就是：

- 如果模型有 128 个 experts
- 那么 `buffer_size = 128`

这意味着：

- GPU swap buffer 一共能同时容纳 128 个 experts
- 正好等于“一整层的全部 experts”

### 这在真实系统里意味着什么

这说明当前实现不是特别“极限节省显存”的那种设计。

它的取舍更偏向：

- 简化实现
- 避免当前层 active experts 太多时 buffer 不够
- 给 prefetch 留足空间

而不是：

- 把 staging area 压到非常小

### 容量实际有多大

前面已经算过 Qwen3 单 expert 约 `9 MB`，所以：

- `128 * 9 MB = 1152 MB`
- 也就是大约 `1.125 GB`

这块显存完全是“为 expert 临时加载准备的额外显存池”。

### 运行时怎么用

真实运行时，如果某个 expert 不在 GPU cache 里，就会：

1. 在 `buffer_status` 里找空闲 slot
2. 把 CPU 里的 `gate/up/down` 权重 copy 到这个 slot
3. 在 `current_layer_mapping` 里记住“这个 expert 现在住在几号 buffer”

对应代码：

- `baseline/expert_buffer_manager.py:509-528`
- `baseline/expert_buffer_manager.py:563-607`

---

## 2.3 GPU cache 是什么

### 从真实系统看

它对应真实部署里的：

- **GPU 上的热点 expert 常驻区**

作用是：

- 把经常访问的 experts 留在 GPU
- 避免它们每次都从 CPU 再搬一遍

### 从代码看

它由 `baseline/gpu_expert_cache.py` 里的 `GPUExpertCacheManager` 管理：

- `self.gpu_memory_pool`
- `self.cache_buffer_infos`
- `self.policy`

关键代码：

- `baseline/gpu_expert_cache.py:631-701`
- `baseline/gpu_expert_cache.py:761-899`

### 它如何组织

当前实现是：

- **每层固定若干个 slot**
- 每个 slot 放一个 expert

也就是说它不是“所有层共享一个大全局 LRU”，而是：

- **layer-local cache**

### 容量参数在哪里

它由命令行参数 `--cache-slots-per-layer` 控制。

对 Qwen3，默认值来自 `tests/test_baseline.py`：

- `cache_slots_per_layer = 16`

对应代码：

- `tests/test_baseline.py:29-35`

### 容量实际有多大

对于 Qwen3：

- 每层 16 个 slot
- 一共 48 层

所以总 slot 数是：

```text
48 * 16 = 768
```

每个 slot 是一个 `9 MB` 的 expert，所以：

- 总 GPU cache 大约 `6.75 GB`

### 这对应真实系统里的什么

可以理解为：

- 你在 GPU 上额外划了一块大约 `6.75 GB` 的“热点专家停车位”
- 每层都有自己的 16 个停车位
- 热门 expert 可以留下来，后面 token 再次用到时直接命中

### 一个很有用的直觉

Qwen3 每个 token 每层只会选 `top_k = 8` 个 experts。

而默认 GPU cache 每层有 `16` 个 slot。

所以在 **单 token decode** 场景下：

- 当前一步真正要算的 active experts，最多就 8 个
- 16 个 slot 理论上足够放下“一次当前 working set”

但问题在于：

- 不同 token 访问的 expert 会变化
- cache 想优化的是“跨 token 的复用”

所以：

- `16 > 8` 不代表就一定高命中
- 真正关键是 expert 访问的时间局部性

---

## 2.4 这三个组件在真实系统里的对应关系

把上面三者合起来，可以形成一个非常清楚的对应表。

| 代码对象 | 真实系统中的对应组件 | 是否存全量 experts | 是否在 GPU | 生命周期 |
|---|---|---:|---:|---|
| CPU expert cache | 主机端全量 expert 仓库 | 是 | 否 | 长期 |
| GPU swap buffer | GPU 临时 staging/scratch 区 | 否 | 是 | 当前层/短期 |
| GPU cache | GPU 热点 expert 常驻区 | 否 | 是 | 跨 token/中长期 |

如果你只记住这一张表，后面读代码就容易多了。

---

## 3. Qwen3 路径上，程序从启动到跑 benchmark 的完整调用链

下面进入代码顺序。这里会把“代码动作”和“真实系统动作”一一对应起来。

---

## 3.1 阶段 A：命令行启动与测试框架初始化

### 代码入口

入口脚本是：

- `tests/test_baseline.py`

Qwen3 路径关键逻辑：

- `tests/test_baseline.py:25-35`

### 代码做了什么

1. 解析 `--model`
2. 如果是 `qwen3moe`：
   - 导入 `baseline.qwen3_builder.qwen3_build_model`
   - 设置默认参数
3. 把通用参数交给 `BaselineTestBase.run_test()`

### 对应真实系统阶段

这一步对应真实服务的：

- **启动配置阶段**

也就是：

- 你选择要跑哪个模型
- 你设定 GPU cache 策略和容量
- 你准备 benchmark 输入源

### 当前 Qwen3 默认参数

默认值是：

- `cache_policy = "topk_lru"`
- `cache_slots_per_layer = 16`
- `enable_gpu_cache = True`

注意：

- 对 Qwen3 来说，`temperature` 最后不会真正起作用，因为 Qwen3 在 `baseline_utils.py` 里走的是 greedy `argmax`
- 对应代码：`baseline_utils.py:247-253`

---

## 3.2 阶段 B：Benchmark prompt 读取

### 代码位置

- `tests/baseline_utils.py:189-191`
- `tests/baseline_utils.py:21-56`

### 代码做了什么

它会从 benchmark parquet 文件中取出文本 prompt。

比如 `gsm8k`：

- 读取 `question` 字段

比如 `humaneval`：

- 读取 `prompt` 字段

### 对应真实系统阶段

这一步对应真实推理系统里的：

- **请求到达 / 输入文本装载**

### 一个非常重要的理解

benchmark 本身并不直接改变 offloading 机制，它影响的是：

- prompt 的长度
- prompt 的内容
- 从而影响 router 选出的 expert 模式

也就是说：

- benchmark 的作用不是“告诉系统如何 offload”
- benchmark 的作用是“提供一批真实输入，观察这些输入导致什么 expert 访问轨迹”

举例：

- GSM8K：通常 prompt 较短，偏推理问答
- CNN-DM：prompt 往往更长，prefill 阶段会激活更多专家并集

因此不同 benchmark 的 offloading 开销不同，本质上是：

- **不同输入分布导致不同 expert locality**

---

## 3.3 阶段 C：构建 Qwen3 offloading 模型

这是最关键的一步。这里不是简单 `AutoModelForCausalLM.from_pretrained()`，而是“把原始 Qwen3 改造成 offloading 版”。

### 代码入口

- `tests/baseline_utils.py:168-184`
- `baseline/qwen3_builder.py:73-173`

### 总体上它做了什么

可以概括成六步：

1. 读 config
2. 造一个“不带 experts 的 Qwen3 骨架”
3. 建 CPU expert cache + GPU swap buffer
4. 把 dense 权重放到 GPU，把 expert 权重整理到 CPU cache
5. 建 GPU cache
6. 把原始 `mlp` 替换成自定义 offloading wrapper

下面逐步展开。

---

## 3.3.1 第一步：读 Qwen3 config

### 代码

- `baseline/qwen3_builder.py:82-90`

### 代码做了什么

```python
config = AutoConfig.from_pretrained(state_path)
original_num_experts = config.num_experts
config.num_experts = 0
config._force_moe_structure = True
config._target_experts = 128
config._original_num_experts = original_num_experts
```

### 对应真实系统阶段

这一步对应真实系统里的：

- **告诉运行时：主干要保留，expert 实体不要直接建在 GPU 上**

### 为什么要把 `config.num_experts = 0`

因为如果不这么做，Transformers 会在构建模型时真的把 experts 都实例化出来。

而当前 offloading 实现希望的是：

- GPU 上只有主干
- expert 权重自己另行管理

所以它等于在说：

> “我要保留 Qwen3 的结构，但 experts 不要按原始方式住进模型里。”

### `_target_experts = 128` 是什么意思

这是一个实现补丁。

因为：

- 真正的 router/gate 维度仍然应该是 128
- 虽然你把 `num_experts` 暂时改成了 0

所以代码额外保存：

- “目标 expert 数仍然是 128”

这样后面 gate 维度仍然能做对。

---

## 3.3.2 第二步：构建“不带 experts 的 Qwen3 骨架”

### 代码

- `baseline/qwen3_builder.py:22-66`
- `baseline/qwen3_builder.py:94-95`

### 代码做了什么

它先 monkey patch 两个地方：

1. `Qwen3MoeDecoderLayer.__init__`
   - 强制每层保留 MoE 结构
2. `Qwen3MoeSparseMoeBlock.__init__`
   - 不真正创建 experts
   - 只保留 gate

然后才：

```python
model = Qwen3MoeForCausalLM(config)
```

### 对应真实系统阶段

这一步对应真实系统里的：

- **构建没有 expert 常驻权重的推理骨架**

你可以把它理解为：

- 先造一台“主机”
- 这台主机里有 attention、norm、router、lm head
- 但机房仓库里的 experts 还没装进去

### 这一步结束后，模型里有什么

有：

- attention
- norm
- KV cache 路径
- gate/router
- 其他 dense 权重位置

没有：

- 真正可计算的常驻 experts

---

## 3.3.3 第三步：创建 CPU expert cache 和 GPU swap buffer

### 代码

- `baseline/qwen3_builder.py:97-113`
- `baseline/expert_cache.py:29-109`
- `baseline/expert_buffer_manager.py:144-228`

### 代码做了什么

```python
buffer_size = original_num_experts  # 128
expert_cache = Qwen3ExpertCache(...)
```

当 `Qwen3ExpertCache` 初始化时：

1. 读取 config 推断 Qwen3 结构
2. 创建 `ExpertBufferManager`
3. `ExpertBufferManager` 立刻分配 GPU swap buffer 的物理内存池

### 对应真实系统阶段

这一步对应真实部署里的：

- **准备 host-side expert 仓库**
- **准备 device-side 临时 staging area**

### 这一阶段结束后，已经有了什么

已经有：

- 一套 CPU 端 expert 存储管理对象
- 一块 GPU 临时 buffer 池

但这时 CPU cache 里还没有真正填入 checkpoint expert 权重。

---

## 3.3.4 第四步：加载 checkpoint，把 dense 权重放 GPU，把 expert 权重放 CPU cache

### 代码

- `baseline/qwen3_builder.py:115-116`
- `baseline/qwen3_builder.py:176-229`
- `baseline/expert_cache.py:289-421`

### 代码做了什么

builder 会遍历 `model.safetensors.index.json` 指向的所有权重文件。

对于每个权重：

- 如果名字里带 `experts.`：
  - 交给 `expert_cache._process_weights_batch(...)`
  - 最终放进 CPU expert cache

- 如果是非 expert 权重：
  - 直接拷到 GPU 模型参数里

### 对应真实系统阶段

这一步对应真实部署里的：

- **模型启动装载阶段**

可以理解成：

- dense 主干权重放入 GPU 常驻区
- 全量 expert 权重放入 CPU 端专家仓库

### Qwen3 expert 权重整理的特殊实现

Qwen3 路径里，expert 权重进入 CPU pinned storage 的过程是：

1. 先按 `(layer, expert)` 把 `gate/up/down` 收集起来
2. 如果还没分配 CPU pinned storage，就一次性分配整块空间
3. 用两个 stream 做：
   - CPU tensor -> GPU 临时 view
   - GPU 临时 view -> CPU pinned storage

对应代码：

- `baseline/expert_cache.py:296-386`

### 为什么它要绕这么一圈

从工程直觉上看，这有点“绕”：

- 本来权重就在 CPU
- 为什么不直接 copy 到 CPU pinned storage

比较合理的理解是：

- 作者想把权重统一整理成后面运行时需要的固定 layout
- 并复用已有 GPU buffer 的形状/偏移信息

所以这一段更像：

- **构建期的布局整理**

不是：

- **推理期的每 token offloading 路径**

这一点一定不要混淆。

---

## 3.3.5 第五步：初始化 GPU cache

### 代码

- `baseline/qwen3_builder.py:118-120`
- `baseline/expert_cache.py:111-132`
- `baseline/gpu_expert_cache.py:642-699`

### 代码做了什么

如果开启 GPU cache：

1. 创建 `GPUExpertCacheManager`
2. 根据策略分配 GPU cache 内存池
3. 根据策略的初始映射，把一部分 experts 从 CPU cache 预装到 GPU cache

### 对应真实系统阶段

这一步对应真实系统里的：

- **服务启动时的热点专家预热**

### 对于 Qwen3 默认配置意味着什么

默认是：

- `cache_policy = topk_lru`
- `cache_slots_per_layer = 16`

但即便是动态策略，它初始化时也会有一个初始内容：

- 每层先装 `expert 0 ~ 15`

原因是：

- `LRU/LFU/TopKLRU/TinyLFU` 都是从一个预填充状态开始

对应代码可见：

- `StaticCachePolicy`: `gpu_expert_cache.py:73-108`
- `LRUCachePolicy` 预填充：`gpu_expert_cache.py:129-136`
- `LFUCachePolicy` 预填充：`gpu_expert_cache.py:243-253`

---

## 3.3.6 第六步：把原始 MLP 换成 Qwen3 offloading wrapper

### 代码

- `baseline/qwen3_builder.py:122-171`

### 代码做了什么

builder 会遍历 48 层：

1. 找到原始 gate 权重
2. 构造一个维度正确的 `fixed_gate`
3. 用 checkpoint 中的 gate 权重填进去
4. 用 `Qwen3SparseMoeWrapper(...)` 替换原始 `layer.mlp`

### 对应真实系统阶段

这一步对应真实部署里的：

- **把原始 MoE 层切换到“按需加载版运行时”**

也就是：

- 从“标准 Qwen3 expert 调用”
- 切换成“router + cache lookup + buffer load + custom expert compute”

到这里，模型构建完成。

---

## 4. 跑一个 benchmark 样本时，链路是怎么走的

下面讲一次完整样本的推理过程。

为了易懂，可以脑补一个 GSM8K 样本：

> “If Tom has 3 apples and buys 5 more, how many apples does he have?”

真正内容不重要，重要的是：

- 它会被 tokenize 成 prompt
- prompt 和后续生成 token 会触发 router 访问不同 experts

---

## 4.1 样本进入：tokenize 与 chat template

### 代码

- `baseline_utils.py:209-215`

### 代码做了什么

它把 benchmark 里的一条文本包装成：

```python
messages = [{"role": "user", "content": prompt}]
```

再调用：

```python
tokenizer.apply_chat_template(..., add_generation_prompt=True)
```

得到 `input_ids`。

### 对应真实系统阶段

对应真实服务里的：

- **请求规范化**
- **tokenize**
- **构造模型实际输入**

这一步跟 offloading 还没有直接关系，但它决定了：

- prompt 长度
- prompt 语义
- 后面 router 的访问轨迹

---

## 4.2 Prefill 阶段：先把整个 prompt 跑一遍

### 代码

- `baseline_utils.py:221-225`

```python
outputs = target_model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values
prefill_time = ...
```

### 代码做了什么

它把完整 prompt 一次性送进模型，建立：

- hidden states
- KV cache

### 对应真实系统阶段

对应真实部署里的：

- **prefill / context encode**

### 为什么 offloading 在 prefill 中也会发生

因为 prompt 经过每一层时，仍然要经过每一层的 MoE。

所以 prefill 时也会触发：

- router 选 experts
- GPU cache lookup
- miss 时 CPU->GPU 搬运
- expert 计算

只是与 decode 相比：

- prefill 一次处理的是整段 prompt
- 每层 active experts 是所有 prompt token 的并集
- 因此 unique active experts 往往比单 token decode 大很多

所以通常：

- prefill 阶段的 offloading 更重
- decode 阶段的 expert working set 更小、更稳定

### 一个对小白很关键的直觉

在 AR 解码里，`past_key_values` 只优化了 attention 的历史复用；

但对 MoE 来说：

- **每来一个新 token，仍然要重新做 48 层 router + expert MLP**

也就是说：

- KV cache 不会让 expert 计算消失
- 它只减少 attention 重算

---

## 4.3 Decode 阶段：每次只生成 1 个 token

### 代码

- `baseline_utils.py:227-275`

### 代码做了什么

Prefill 之后，代码进入一个标准 HF 风格的 AR decode loop：

1. 取 prompt 的最后一个 token 作为起点
2. 每一步只输入一个 token：

```python
outputs = target_model(
    input_ids=current_input_ids,
    past_key_values=past_key_values,
    use_cache=True
)
```

3. 取最后位置的 logits
4. Qwen3 路径用 `argmax` 选下一个 token
5. 更新 `past_key_values`
6. 继续下一步

### 对应真实系统阶段

对应真实部署里的：

- **steady-state 单 token 递归解码**

### 对 offloading 来说，这一阶段意味着什么

每生成一个 token，都要再走一遍 48 层。

而每一层都要重新回答：

- 这个 token 本层该用哪 8 个 experts？
- 这 8 个 experts 现在在不在 GPU？
- 缺的从 CPU 还是 prefetch 里拿？

所以，decode 阶段的 offloading 开销核心就在这里。

---

## 5. 重点：Qwen3 某一层 MoE forward 内部，到底怎么执行

这是整套代码最重要的部分。

入口在：

- `baseline/qwen3_layers.py:108-470`

下面按严格顺序讲。

---

## 5.1 当前层拿到 hidden states，开始路由

### 代码

- `qwen3_layers.py:108-133`

### 代码做了什么

1. 输入是本层 hidden states，形状大致是：

```python
[batch, sequence_length, hidden_dim]
```

2. 展平成：

```python
hidden_states_flat = hidden_states.view(-1, hidden_dim)
```

3. 用 gate 算 router logits：

```python
router_logits = self.gate(hidden_states_flat)
```

4. 对 logits 做 softmax，再取 top-k：

```python
full_routing_weights = softmax(router_logits)
routing_weights, selected_experts = topk(full_routing_weights, self.top_k)
```

对于 Qwen3：

- `self.top_k = config.num_experts_per_tok = 8`

### 对应真实系统阶段

对应真实 GPU-CPU offloading 系统里的：

- **router / dispatch 阶段**

也就是：

- 先决定本层当前 token 要用哪几个 experts

### 一个直观例子

假设当前是 decode 单 token，那么这一层可能选出：

```text
[3, 17, 42, 61, 88, 99, 105, 120]
```

这 8 个 ID 就是接下来 offloading runtime 要服务的对象。

---

## 5.2 同时预测“下一层”可能会用哪些 experts，准备 prefetch

### 代码

- `qwen3_layers.py:135-161`

### 代码做了什么

如果 `PREFETCH_ENABLED = True`，当前层还会额外做一件事：

1. 取出下一层 gate
2. 用“当前层输入 hidden_states_flat”喂给下一层 gate
3. 取下一层预测的 top `PREFETCH_TOPK`

其中：

- `PREFETCH_TOPK = 4`
- 定义在 `qwen3_layers.py:18-20`

最终会得到：

- `next_layer_selected_experts_cpu`

### 对应真实系统阶段

对应真实系统里的：

- **下一层 expert 预取预测**

### 这个预测为什么只是“近似”

因为它并没有先真的算完当前层再拿到下一层真实输入，而是：

- 直接用当前层输入 hidden states 去猜下一层 router

所以它不是 oracle，而是：

- **启发式 prefetch**

### 一个直观例子

如果当前层预测下一层 top-4 是：

```text
[17, 42, 90, 121]
```

那么系统就会尝试提前把下一层这几个 expert 搬到 GPU swap buffer。

---

## 5.3 把本层 active experts 汇总出来

### 代码

- `qwen3_layers.py:216-228`

### 代码做了什么

`selected_experts` 里可能每个 token 都有 8 个 expert。

代码会：

1. 把它展平
2. copy 到 CPU
3. 在 CPU 上 `bincount`
4. 找出当前层实际被至少一个 token 用到的 expert 集合

得到：

- `active_expert_ids`
- `expert_counts_list`
- `expert_offsets_list`

### 对应真实系统阶段

对应真实 offloading runtime 里的：

- **计算本层真正需要准备的 expert working set**

### 为什么要先做“去重后的 working set”

因为真实系统不会因为同层有多个 token 都要 expert 17，就把 expert 17 载入多次。

它只会：

- 载入一次
- 然后让该 expert 处理所有分配给它的 token

所以这里做的是：

- **token 级选择 -> expert 级工作集去重**

### 一个单 token decode 的直觉

如果当前 forward 只有 1 个 token，那么：

- `selected_experts.shape == [1, 8]`
- `active_expert_ids` 最多就是 8 个

这就是为什么单 token decode 的 expert working set 通常很小。

---

## 5.4 为动态 cache 策略准备 logit 分数

### 代码

- `qwen3_layers.py:230-242`

### 代码做了什么

如果 GPU cache 策略是动态的，比如：

- `lru`
- `lfu`
- `topk_lru`
- `tinylfu`

那么代码还会为每个 active expert 计算一个 `max_logit`：

```python
max_logits_per_expert, _ = full_routing_weights.max(dim=0)
router_logits = {eid: max_logits_cpu[eid].item() for eid in active_expert_ids}
```

注意这里有个实现细节：

- `router_logits` 这个变量名在函数前半段是真正的 gate logits tensor
- 到这里被重用成了 `expert_id -> max_logit` 的 Python dict

所以读代码时不要把这两个概念混了。

### 对应真实系统阶段

对应真实系统里的：

- **给缓存准入策略提供“热点程度”分数**

尤其对 `topk_lru` 而言，它需要知道：

- 这次 miss 进来的 experts 中，哪些更值得留下来

---

## 5.5 调用 expert runtime：先查 GPU cache，再查 prefetch，再从 CPU 临时加载

### 代码入口

- `qwen3_layers.py:239-242`

```python
expert_to_buffer_mapping = self.expert_cache.batch_load_experts_continuous(
    self.layer_idx, expert_indices, router_logits
)
```

这行是整个 offloading runtime 的核心入口。

下面继续往下拆。

---

## 5.5.1 先把 CPU cache 中的权重组织成当前层可加载的字典

### 代码

- `baseline/expert_cache.py:493-518`

### 代码做了什么

它会从 `simple_expert_cache` 中取出当前层每个 active expert 的：

- `gate`
- `up`
- `down`

然后组成：

```python
expert_weights_dict[expert_id] = {
    "gate": ...,
    "up": ...,
    "down": ...
}
```

### 对应真实系统阶段

这一步对应真实系统里的：

- **从 host-side expert 仓库里找到这次要搬的 expert 权重句柄**

注意：

- 这一步只是“找到位置”
- 还没有真正发生 CPU->GPU copy

---

## 5.5.2 运行时先释放上一层的临时 buffer

### 代码

- `baseline/expert_buffer_manager.py:456-460`

### 代码做了什么

进入当前层前，会把上一层 `current_layer_mapping` 对应的临时 buffer 释放掉。

但注意：

- 如果某个映射是 GPU cache slot，就不会释放

### 对应真实系统阶段

对应真实系统里的：

- **清空当前层临时 staging 区**

这说明 swap buffer 是：

- 按层复用的
- 不是永久保留的

---

## 5.5.3 先检查 prefetch 是否已经完成

### 代码

- `baseline/expert_buffer_manager.py:464-470`

### 代码做了什么

如果上一层已经在 `prefetch_stream` 上把某些 experts 预取好了，那么这里会把：

- `prefetch_in_progress`

转移到：

- `prefetch_mapping`

表示：

- 这些 expert 现在已经真的可用了

### 对应真实系统阶段

对应真实部署里的：

- **把异步 DMA 完成的预取结果挂到运行时可见表里**

---

## 5.5.4 第一优先级：查 GPU cache

### 代码

- `baseline/expert_buffer_manager.py:471-485`

### 代码做了什么

对每个 active expert：

1. 调 `gpu_cache_manager.lookup(layer_idx, expert_id)`
2. 如果命中：
   - 返回对应 cache slot
   - 把它映射成一个 `virtual_idx`
   - 记到 `loaded_mapping`

### 对应真实系统阶段

对应真实部署里的：

- **查“这个 expert 是不是已经常驻 GPU 热点区”**

### 什么是 virtual_idx

这是实现里一个很重要的抽象。

因为 expert 可能住在两种地方：

1. GPU swap buffer
2. GPU cache

为了让后面的计算代码不用关心“它到底住哪”，运行时统一返回一个 `virtual_idx`。

然后计算时通过：

- `get_expert_view_for_computation(virtual_idx)`

再解析成真正的内存 view。

这对应真实系统里的：

- **统一的驻留句柄 / residency handle**

### cache hit 是在这里统计的

`lookup()` 调用时，如果命中，就会增加：

- `hits`

如果 miss，就增加：

- `misses`

对应代码在：

- `gpu_expert_cache.py:84-91`
- `gpu_expert_cache.py:148-160`
- `gpu_expert_cache.py:274-284`

注意：

- 统计单位不是 token，而是 **active expert request**

---

## 5.5.5 第二优先级：查 prefetch buffer

### 代码

- `baseline/expert_buffer_manager.py:487-507`

### 代码做了什么

对于 GPU cache miss 的 experts，再看它是不是已经被上一层预取到了 swap buffer：

```python
if expert_id in self.prefetch_mapping:
    ...
```

如果是：

- 直接拿这个 buffer
- 不需要再 CPU->GPU copy

### 对应真实系统阶段

对应真实部署里的：

- **查下一层预取是否命中**

### 这里为什么和 GPU cache 是分开的

因为 prefetch buffer 只是：

- 提前放到 GPU staging 区

它还不是：

- 真正进入长期驻留的 GPU cache

所以优先级才会是：

```text
GPU cache > prefetch > CPU->GPU 临时加载
```

### 一个很关键的统计口径点

即便 prefetch 命中，它在 GPU cache 统计里仍然是 miss。

因为：

- 它没有命中 GPU cache
- 只是命中了 prefetch 结果

这就是为什么：

- `cache hit rate` 和 `prefetch hit` 不是一回事

当前主脚本只打印前者，不打印后者。

---

## 5.5.6 第三优先级：把剩余 miss 从 CPU 搬到 GPU swap buffer

### 代码

- `baseline/expert_buffer_manager.py:509-528`
- `baseline/expert_buffer_manager.py:563-607`

### 代码做了什么

对于剩下既不在 GPU cache、也没有 prefetch 的 experts：

1. 找空闲 swap buffer slot
2. 从 CPU expert cache 取出 `gate/up/down`
3. `copy_` 到 GPU swap buffer 的对应位置
4. 记录 `current_layer_mapping`

### 对应真实系统阶段

对应真实部署里的：

- **真正的 CPU->GPU demand loading**

这就是 offloading 最核心的那一步。

### 一个具体例子

假设当前层 active experts 是：

```text
[3, 17, 42, 61, 88, 99, 105, 120]
```

其中：

- GPU cache 已有 `[3, 17, 42]`
- prefetch 已有 `[88]`

那真正需要 CPU->GPU 临时加载的就是：

```text
[61, 99, 105, 120]
```

代码就会把这 4 个 expert 的 `gate/up/down` 权重 copy 到 GPU swap buffer。

### 这一步统计了什么

会增加：

- `total_experts_loaded`
- `compute_loads`

对应代码：

- `expert_buffer_manager.py:527-528`

这两个统计当前主脚本没有打印，但它们非常重要，因为它们更接近：

- **真正发生了多少次 CPU->GPU load**

---

## 5.5.7 把这次新加载的 experts 提升进 GPU cache（如果策略允许）

### 代码

- `baseline/expert_buffer_manager.py:530-561`
- `baseline/gpu_expert_cache.py:981-1050`

### 代码做了什么

如果当前策略是动态策略，并且这次确实有新的 miss 被装入 swap buffer：

1. 收集这些刚装入 swap buffer 的 expert
2. 附上它们的 `logit_score`
3. 调 `gpu_cache_manager.update_cache_from_buffers(...)`
4. 根据策略决定：
   - 哪些能留下
   - 替换掉谁
5. 真的发生一次 GPU->GPU copy：

```text
swap buffer -> GPU cache slot
```

### 对应真实系统阶段

对应真实部署里的：

- **把这次访问过、看起来比较热的 expert 晋升为 GPU 常驻热点**

### 为什么是 GPU->GPU copy

因为这些 expert 已经被搬到 GPU swap buffer 了。

如果想把它留下，只需要：

- 从 swap buffer 再拷到 GPU cache

所以这一步不是 CPU->GPU，而是：

- **GPU 内部拷贝**

### 这一步和“算完立即释放”有什么区别

如果不做 cache promotion：

- 当前层算完，expert 只在 swap buffer 里短暂停留
- 下一层来了就可能被覆盖

如果做了 promotion：

- 它会被复制到 GPU cache
- 后续 token 再用到它时可以直接命中

---

## 5.6 如果允许 prefetch，就开始异步预取下一层 experts

### 代码

- `qwen3_layers.py:252-255`
- `qwen3_layers.py:422-469`
- `expert_buffer_manager.py:635-663`

### 代码做了什么

当前层在开始大规模 expert compute 前，会：

1. 同步上一轮 prefetch stream，避免多个 prefetch 重叠冲突
2. 对下一层预测到的 expert 列表去重
3. 从 CPU expert cache 里取下一层这些 experts 的权重
4. 在 `prefetch_stream` 上调用 `prefetch_expert(...)`
5. 把它们装入 GPU swap buffer，但挂在 `prefetch_in_progress` 里

### 对应真实系统阶段

对应真实部署里的：

- **用独立 DMA/传输 stream 提前为下一层备货**

### 一个非常重要的点

prefetch **没有单独的物理内存池**。

它用的还是：

- 同一块 `GPU swap buffer`

只是逻辑状态不同：

- 当前层正在使用的临时加载：`current_layer_mapping`
- 异步预取中的：`prefetch_in_progress`
- 已经预取完成、等待下一层使用的：`prefetch_mapping`

所以：

- prefetch buffer 不是第四种内存区域
- 它只是 swap buffer 的一种占用状态

这点读代码时非常重要。

---

## 5.7 Expert 真正开始计算

### 代码

- `qwen3_layers.py:257-386`

### 代码做了什么

当前层开始真正的 expert MLP 计算：

1. 先把 token-expert 分配按 expert 排序
2. gather 出所有 expert 输入
3. 根据 `expert_to_buffer_mapping` 找到每个 expert 的权重 view
4. 默认走 `BMM_ENABLED = True` 的批量 BMM 路径
5. 计算：

```text
silu(x @ gate^T) * (x @ up^T) -> @ down^T
```

6. 最后用 `scatter_add_` 把结果写回输出

### 对应真实系统阶段

对应真实部署里的：

- **拿到已驻留在 GPU 上的 expert 权重，执行实际 MLP 计算**

### 为什么前面那么复杂，最后还是要回到这里

因为 offloading 的本质只是：

- **决定 expert 权重怎么到 GPU**

一旦权重到了 GPU，最终目的还是：

- 用这些权重完成 expert MLP 计算

### 为什么有 BMM 优化

如果逐 expert 单独算：

- kernel 很碎
- launch 开销高

所以代码提供：

- `BMM_ENABLED = True`

对应 `debug_config.py:22-24`

意思是：

- 尽量把多个 active experts 的计算并成 batched matrix multiplication

这更接近真实高性能 runtime 会做的事。

---

## 5.8 当前层结束，进入下一层

当前层 expert 计算完成后，输出会回到 decoder 主干，继续：

- residual
- norm
- 下一层 attention / MoE

对 offloading runtime 来说，这意味着：

- 当前层 `current_layer_mapping` 的作用结束
- 下一层进来时会先释放这些临时 slot
- 只有 GPU cache 和 prefetch 结果能跨层延续

这就是为什么：

- swap buffer 是短期的
- GPU cache 是长期的

---

## 6. 现在把整条链路用“真实系统阶段 vs 代码阶段”对齐一次

下面给你一张最重要的对照表。

| 真实 GPU-CPU offloading 阶段 | 代码位置 | 代码动作 |
|---|---|---|
| 启动配置 | `tests/test_baseline.py` | 选择 Qwen3、设 cache policy 和 slot 数 |
| 构建无 expert 主干 | `qwen3_builder.py:82-95` | `num_experts=0`，只保留 Qwen3 骨架 |
| 建 host expert 仓库 | `expert_cache.py` | 创建 CPU expert cache / pinned storage |
| 建 GPU staging 区 | `expert_buffer_manager.py` | 分配 GPU swap buffer |
| 装载 dense 权重到 GPU | `qwen3_builder.py:176-221` | 非 expert 权重直接放 GPU |
| 装载全量 expert 到 CPU | `expert_cache.py:289-421` | Qwen3 gate/up/down 进 CPU cache |
| 预热 GPU 热点区 | `gpu_expert_cache.py:868-899` | 初始 experts 装进 GPU cache |
| 样本输入进入 | `baseline_utils.py:209-215` | tokenize + chat template |
| Prefill | `baseline_utils.py:221-225` | 整段 prompt 建 KV cache |
| 单 token decode | `baseline_utils.py:238-275` | 每步只输入 1 个 token |
| 本层 router 选专家 | `qwen3_layers.py:119-166` | top-8 experts |
| 预测下一层 prefetch | `qwen3_layers.py:135-161` | top-4 next-layer experts |
| 构造当前层 working set | `qwen3_layers.py:216-228` | active experts 去重 |
| 查 GPU cache | `expert_buffer_manager.py:471-485` | 看 expert 是否已常驻 |
| 查 prefetch 结果 | `expert_buffer_manager.py:487-507` | 看 expert 是否提前搬好了 |
| CPU->GPU demand load | `expert_buffer_manager.py:509-528` | miss 的 expert 装入 swap buffer |
| GPU->GPU cache promote | `gpu_expert_cache.py:981-1050` | 热门 miss 从 swap 升级到 cache |
| 异步预取下一层 | `qwen3_layers.py:422-469` | 用 prefetch stream 提前搬运 |
| expert MLP 计算 | `qwen3_layers.py:257-386` | BMM / gather / scatter |
| 统计 decode 时间与 TPS | `baseline_utils.py:277-315` | 汇总样本级和全局指标 |
| 打印 cache 命中统计 | `baseline_utils.py:324-335` | 打印 GPU cache stats |

如果你能把这张表读顺，说明你已经理解了这套代码在模拟什么。

---

## 7. 代码里到底评估了什么

下面进入评估指标。

---

## 7.1 样本级指标

### 代码位置

- `baseline_utils.py:277-289`

### 当前样本会记录

- `new_tokens`
- `decode_time`
- `decode_tps`
- `prefill_time`

### 各自是什么意思

#### `prefill_time`

从把完整 prompt 送进模型开始，到拿到 `past_key_values` 为止的时间。

对应真实系统里的：

- 首次建上下文成本

#### `decode_time`

这个样本在 decode loop 中生成新 token 的总时间。

不包括：

- 模型构建时间
- benchmark 文件读取时间

#### `new_tokens`

真正生成出的新 token 数量。

如果中间遇到 EOS 或达到 `max_new_tokens`/`max_length` 限制，就会停。

#### `decode_tps`

定义是：

```text
new_tokens / decode_time
```

表示这个样本单独的 decode 吞吐。

### 举个例子

假设某个样本：

- 生成了 80 个新 token
- decode 花了 20 秒

那么：

- `decode_tps = 80 / 20 = 4.0 tokens/s`

---

## 7.2 全局指标

### 代码位置

- `baseline_utils.py:299-309`

### 打印内容

- `Total samples`
- `Total new tokens`
- `Total decode time`
- `Overall Decode TPS`

### `Overall Decode TPS` 怎么算

定义是：

```text
所有样本的新 token 总数 / 所有样本 decode 时间总和
```

### 它代表什么

代表：

- 整组 benchmark 样本在当前 offloading 策略下的整体 decode 吞吐

### 举个例子

如果 20 个样本总共生成：

- `2000 tokens`

总 decode 时间：

- `500 s`

那么：

- `Overall Decode TPS = 2000 / 500 = 4.0 tokens/s`

---

## 7.3 GPU cache 指标

### 代码位置

- `baseline_utils.py:324-335`
- `gpu_expert_cache.py:1080-1109`

### 当前会打印

- `Policy`
- `Slots`
- `Memory`
- `Hits`
- `Misses`
- `Hit Rate`
- `Alpha`
- `Expected Hit Rate`（static 下）
- `Cache Updates`
- `Logit Threshold Percentile`（topk_lru 下）

下面逐个解释。

---

## 7.3.1 `Hits` / `Misses`

### 它们是怎么来的

每一层 MoE 在拿到 `active_expert_ids` 之后，会对每个 expert 做：

```python
gpu_cache_manager.lookup(layer_idx, expert_id)
```

如果命中：

- `hits += 1`

否则：

- `misses += 1`

### 统计单位是什么

不是：

- token 数
- 样本数

而是：

- **每层 forward 中，每个 active expert 的一次 cache 查询**

### 举个例子

当前是单 token decode，某一层选出 8 个 experts：

```text
[3, 17, 42, 61, 88, 99, 105, 120]
```

其中：

- 3,17,42,61 在 cache
- 88,99,105,120 不在 cache

那么这一层这一轮会记：

- `hits += 4`
- `misses += 4`

### 一个关键点

如果某个 miss 的 expert 恰好被 prefetch 了：

- 它仍然是 `cache miss`
- 只是后面会成为 `prefetch hit`

这两个统计不要混。

---

## 7.3.2 `Hit Rate`

定义是：

```text
hits / (hits + misses)
```

表示：

- 当前 GPU cache 对 expert 请求的命中比例

### 举个例子

如果累计：

- `hits = 30000`
- `misses = 10000`

那么：

- `hit_rate = 75%`

---

## 7.3.3 `Alpha`

### 代码

- `gpu_expert_cache.py:1101-1103`

### 定义

```text
cache_rate = slots_per_layer / num_experts
alpha = hit_rate / cache_rate
```

### 它代表什么

可以把它理解成：

- 你的缓存命中效果，相对“按容量比例随机猜中”的归一化倍数

### 举个例子

对于 Qwen3：

- `slots_per_layer = 16`
- `num_experts = 128`

所以：

- `cache_rate = 16 / 128 = 12.5%`

如果测出来：

- `hit_rate = 75%`

那么：

- `alpha = 75% / 12.5% = 6.0`

含义是：

- 你的 cache 命中率是“随机放 12.5% experts”预期命中率的 6 倍

这通常说明：

- expert 访问存在较强时间局部性
- 缓存策略在起作用

---

## 7.3.4 `Cache Updates`

表示：

- 动态策略实际发生了多少次 cache 插入/替换

### 举例

如果 `static`：

- 它固定装前 16 个 expert
- 不会更新
- 所以 `Cache Updates = 0`

如果 `lru`：

- miss 进来的新 expert 可能顶掉最老的一个
- 每发生一次替换，就记一次 update

如果 `topk_lru`：

- 不是每个 miss 都有资格进 cache
- 只有 logit 足够高的 miss 才可能引发 update

---

## 7.3.5 `Expected Hit Rate`

只在 `static` 策略下有意义。

代码里直接定义为：

```text
slots_per_layer / num_experts
```

对于 Qwen3 默认参数：

- `16 / 128 = 12.5%`

它更像一个粗基线：

- 如果 cache 只是固定放 12.5% 的 experts，且 expert 请求均匀随机，那么期望命中率大约就是 12.5%

---

## 7.3.6 当前代码里没有打印但很重要的指标

`ExpertBufferManager.get_stats()` 里还有：

- `prefetch_hits`
- `compute_loads`
- `total_experts_loaded`

对应代码：

- `expert_buffer_manager.py:671-681`

它们分别更接近：

- `prefetch_hits`
  - 多少次 GPU cache miss 最终被 prefetch 救回来了
- `compute_loads`
  - 多少个 expert 真正从 CPU 载入到了 swap buffer
- `total_experts_loaded`
  - 实际发生的临时载入总次数

当前主脚本没有打印这些值，所以：

- 你现在看到的评估主要聚焦 GPU cache，而不是 prefetch 本身

---

## 7.4 `layer_expert_counts` 和 `layer_times` 是什么

### 代码

- `qwen3_layers.py:73-84`
- `qwen3_layers.py:390-399`

### 它们记录什么

- `layer_expert_counts`
  - 本次 forward 的每层 active expert 数
- `layer_times`
  - 本次 forward 的每层 MoE 时间

### 当前问题

这些统计虽然被收集了，但主脚本没有输出，也没有跨整个 decode 累积保存。

所以当前它们更像：

- 调试信息

不是：

- 最终 benchmark 报表的一部分

---

## 8. 不同缓存策略到底是什么意思

下面专门解释你问到的不同 cache policy。

为了方便理解，假设：

- 某一层一共有 8 个 experts：`0~7`
- GPU cache 只有 2 个 slot

---

## 8.1 `static`

### 含义

每层固定缓存前 `N` 个 experts，不更新。

### 例子

如果 `slots_per_layer = 2`，那永远缓存：

```text
[0, 1]
```

无论后面访问序列是什么，都不变。

### 适合看什么

适合当最朴素基线：

- 不做智能替换
- 不做热点适应

---

## 8.2 `lru`

### 含义

最近最少使用淘汰。

### 例子

初始 cache：

```text
[0, 1]
```

访问顺序：

```text
1 -> 2 -> 3
```

过程：

1. 访问 1：命中，最近使用的是 1
2. 访问 2：miss，淘汰最久没用的 0，cache 变成 `[1, 2]`
3. 访问 3：miss，淘汰最久没用的 1，cache 变成 `[2, 3]`

### 对 Qwen3 的直觉

如果相邻 token 往往复用相近的 experts，LRU 往往有效。

---

## 8.3 `lfu`

### 含义

最少使用频率淘汰。

### 例子

cache 容量 2，当前频率：

```text
0 用过 10 次
1 用过 2 次
```

来了一个新 expert 2：

- LFU 更可能淘汰 1，而不是 0

### 适合什么情况

适合：

- 长时间存在固定热门 experts

但缺点是：

- 历史访问可能污染未来决策

---

## 8.4 `topk_lru`

这是当前 Qwen3 默认策略，也是最值得重点理解的。

### 含义

它是：

- `LRU + logit 准入门槛`

不是所有 miss 进来的 expert 都能晋升进 GPU cache，只有：

- 这次 router 分数足够高的 expert 才能进去

### 代码逻辑

对应：

- `gpu_expert_cache.py:372-440`

其思想是：

1. 当前层这次新载入了一批 miss experts
2. 给它们按 `max_logit` 排序
3. 算一个 percentile 阈值
4. 只有高于这个阈值的 expert 才有资格进 cache

### 例子

假设这层这次新载入的 miss experts 有：

```text
e17: 0.91
e42: 0.88
e61: 0.73
e88: 0.69
```

如果阈值 percentile 较高，那么最后可能只有：

```text
e17, e42
```

会被提升进 GPU cache，`e61/e88` 只是算完就走。

### 为什么这么做

因为系统默认认为：

- router 分数越高，说明这个 expert 对当前 token 更关键、更像热点
- 这类 expert 可能更值得留在 GPU

### 和纯 LRU 的区别

纯 LRU 是：

- miss 进来一个，就可能替换一个

`topk_lru` 是：

- 先问“这个 miss 值不值得进 cache”
- 值得才替换

所以它通常更保守。

---

## 8.5 `tinylfu`

### 含义

它比 LFU 更复杂，做了两件事：

1. 用频率估计决定是否准入
2. 用分段 LRU 管理“新来的”和“真正热点”

### 小白理解方式

你可以先把它想成：

- 一个比 LFU 更稳健的“热点筛选器”

它会尽量避免：

- 某个偶然访问一次的新 expert，把真正长期热点挤掉

### 对现在理解代码是否必须

不是必须。

如果你先把：

- `static`
- `lru`
- `topk_lru`

理解透，再回来看 `tinylfu` 会更容易。

---

## 9. Prefetch 到底是什么意思，代码具体怎么做

现在单独讲 prefetch。

---

## 9.1 prefetch 的直觉含义

prefetch 的目标是：

> **不要等到下一层真正要用 expert 时才从 CPU 搬；而是在当前层计算时，顺手把下一层大概率会用的 expert 先搬过去。**

这样做的好处是：

- 下一层真正开始时，更可能直接命中已搬好的 expert
- CPU->GPU copy 可以和当前层 compute 重叠

这就是它在真实系统里的意义。

---

## 9.2 Qwen3 当前 prefetch 怎么预测下一层专家

### 代码

- `qwen3_layers.py:135-161`

### 做法

当前层拿自己的 `hidden_states_flat`，直接喂给：

- 下一层 gate

然后取：

- `PREFETCH_TOPK = 4`

得到下一层预测 experts。

### 你要清楚的事实

它不是：

- 用下一层真实输入做预测

而是：

- 用当前层输入做近似预测

所以它是：

- 启发式 prefetch

不是：

- 精确预知

---

## 9.3 预取结果放在哪

放在：

- 同一块 `GPU swap buffer`

不是放在 GPU cache，也不是单独开一块“prefetch 专用池”。

这点对应代码：

- `expert_buffer_manager.py:635-663`

可以把它理解成：

- “我先在临时卸货区里替下一层占了几个位子”

---

## 9.4 prefetch 命中是什么意思

当下一层真正开始时，如果某个 expert：

- 不在 GPU cache
- 但已经在 `prefetch_mapping` 里

那么它就算：

- `prefetch hit`

这意味着：

- 本来这个 expert 应该从 CPU 再搬一次
- 但因为提前搬好了，这次直接复用

对应代码：

- `expert_buffer_manager.py:487-507`

---

## 9.5 一个完整的 prefetch 例子

假设当前是第 10 层。

### 第 10 层执行时

当前层预测：

```text
第 11 层大概率会用 [17, 42, 61, 90]
```

于是它在 `prefetch_stream` 上尝试把这 4 个 expert 搬进 GPU swap buffer。

### 第 11 层真正开始时

真实 router 选出来的是：

```text
[17, 42, 61, 88, 105, 120, 3, 7]
```

那么：

- `17, 42, 61` 命中 prefetch
- `88, 105, 120, 3, 7` 仍需正常处理

于是：

- 这次 CPU->GPU 临时加载次数就减少了 3 个

这就是 prefetch 的实际价值。

---

## 10. 你现在应该如何理解“这套代码是在 benchmark 上怎么评估 offloading”

把前面的内容收束成一句完整话：

> **它在 benchmark 上逐条取 prompt，真实跑 Qwen3 的 AR prefill + 单 token decode；每次进入某层 MoE 时，真实执行 router、GPU cache 查询、prefetch 查询、CPU->GPU expert copy、GPU cache promotion、expert MLP 计算和下一层 prefetch；最后用真实墙钟时间统计 decode TPS，并输出 GPU cache 命中情况。**

这句话可以再拆成四个层面。

### 第一层：benchmark 提供输入分布

benchmark 不负责 offloading，只负责给出一批 prompt，让模型产生真实的 expert 访问模式。

### 第二层：运行时复现真实 offloading

这套代码真的把 expert 权重放 CPU，真的发生 H2D copy，真的用 GPU 执行 expert 计算。

### 第三层：缓存和预取策略影响时间

不同 `cache_policy` / `cache_slots_per_layer` / `prefetch` 会改变：

- CPU->GPU copy 次数
- GPU cache 命中率
- 最终 decode TPS

### 第四层：指标输出是从真实执行链路中量出来的

不是拍脑袋定义出来的。

---

## 11. 对读代码最有帮助的几个“不要混淆”的点

最后列几个最容易混淆的点。

### 11.1 CPU expert cache 不是会淘汰的 cache

它本质上是：

- 全量 expert 的 host-side 仓库

### 11.2 prefetch 不是独立物理内存区

它只是：

- GPU swap buffer 的一种占用状态

### 11.3 GPU cache 才是长期驻留区

swap buffer 用完就可覆盖，GPU cache 才是希望跨 token 留下的 expert。

### 11.4 当前 benchmark 主要评估 decode，不计模型构建

`decode_time` 和 `Overall Decode TPS` 都不包含：

- builder 初始化
- 全量权重装载

所以它衡量的是：

- 运行期吞吐

不是：

- 服务冷启动耗时

### 11.5 当前主脚本没有把 prefetch_hits 打出来

所以你看到的报表主要是：

- GPU cache 视角

不是完整的：

- GPU cache + prefetch 联合视角

### 11.6 当前主脚本没有区分 prefill 和 decode 的 cache 统计

虽然 `GPUExpertCacheManager` 有 `enable_stats()` / `disable_stats()` 接口，但主脚本没有显式切换。

这意味着当前打印的 cache 统计很可能混入了：

- prefill 的请求
- decode 的请求

因此如果你要做更严谨的实验，下一步最好把这两段拆开统计。

---

## 12. 读完这份文档后，你应该已经能回答的几个问题

如果你现在已经能清楚回答下面这些问题，说明你已经从“小白”进到“能读懂这份代码”的状态了。

1. 为什么 Qwen3 的 experts 不直接挂在模型模块里，而是放到 CPU expert cache？
2. 为什么还需要 GPU swap buffer？为什么不能 miss 了直接算？
3. GPU cache 和 swap buffer 的区别是什么？
4. 为什么一个 token decode 时每层最多只会有 8 个 active experts？
5. 为什么 `cache_slots_per_layer = 16` 不等于“一定高命中”？
6. prefetch 命中了，为什么 GPU cache 统计里仍然可能是 miss？
7. `topk_lru` 为什么比纯 LRU 更保守？
8. 当前 benchmark 打印的 `decode_tps`、`hit_rate`、`alpha` 到底分别在说明什么？

如果其中还有哪一题不够清楚，那就说明应该回到对应章节再读一遍。

---

## 13. 最后一句总结

对 Qwen3 来说，这个项目最核心的思想可以压缩成下面这句话：

> **router 负责决定“用谁”，CPU expert cache 负责“权重都放在哪”，GPU swap buffer 负责“miss 时先搬到哪”，GPU cache 负责“热点留在哪”，而 benchmark 负责提供真实输入去激发这些专家访问模式；最终用真实运行时间和命中统计来评估这套 offloading 运行时。**

你后面如果要把 SDAR 接进来，真正要改的不是这三层存放区的思想，而是：

- 上层“每次 forward 是单 token AR 还是块内并行迭代”的驱动方式
- 以及由此带来的 expert working set、cache 命中模式和 prefetch 逻辑变化

