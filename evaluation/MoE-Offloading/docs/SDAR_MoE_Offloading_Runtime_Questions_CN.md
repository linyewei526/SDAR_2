# SDAR MoE-Offloading 运行时细节答疑

本文针对以下三个问题做执行层面的拆解：

1. 为什么 `current_layer_miss_load` 和 `next_layer_prefetch` 都是 CPU->GPU 权重搬运，却会在时间轴上部分重叠，而且 `next_layer_prefetch` 常常在 `current_layer_miss_load` 中途才开始。
2. Pure SDAR 的 profile 里，为什么 `moe_dispatch`、`moe_expert_compute_hbm_fetch`、`moe_scatter`、`other_layer_overhead` 都很显著，它们分别对应什么代码和机制。
3. 为什么在权重都已经到 GPU 之后，当前 MoE-Offloading runtime 的 MoE 执行仍然可能比原始 Pure SDAR 更快，`reorder/gather/batched expert compute/scatter` 具体是什么意思。

相关代码路径：

- 原始 SDAR 本地模型: `evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe.py`
- 原始 SDAR 的 profile 版: `evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe_profiled.py`
- SDAR offloading MoE wrapper: `baseline/sdar_layers.py`
- expert 临时 buffer 管理: `baseline/expert_buffer_manager.py`
- GPU cache 管理: `baseline/gpu_expert_cache.py`
- offloading 摘要记录: `baseline/sdar_runtime_trace.py`
- pure SDAR 摘要记录: `baseline/sdar_dense_profile.py`
- SDAR block diffusion 解码主循环: `evaluation/opencompass/opencompass/models/huggingface_bd3.py`

## 1. `current_layer_miss_load` 和 `next_layer_prefetch` 为什么会部分重叠

### 1.1 先看代码里的真实执行顺序

在 offloading 版 `SDARSimpleMoE.forward()` 里，相关顺序是：

1. 先做当前层 routing，得到 `selected_experts`。
2. 同时提前用下一层 gate 预测 `next_layer_selected_experts`。
3. 调用 `self.expert_cache.batch_load_experts_continuous(...)` 为当前层准备专家。
4. 如果 `PRELAUNCH_ENABLED=False`，这里会 `torch.cuda.synchronize()`；但当前配置里 `PRELAUNCH_ENABLED=True`。
5. 然后才调用 `_parallel_prefetch(...)`，把预测到的下一层专家预取到 prefetch buffer。
6. 之后才进入 `reorder -> gather -> expert compute -> scatter`。

对应代码位置是：

- 当前层专家准备: `baseline/sdar_layers.py` 里 `batch_load_experts_continuous(...)`
- `PRELAUNCH_ENABLED` 开关: `baseline/debug_config.py`
- 下一层预取入口: `baseline/sdar_layers.py::_parallel_prefetch`

所以从“CPU 代码顺序”看，`next_layer_prefetch` 的确是在当前层 `miss_load` 之后才发起的。这里没有矛盾。

### 1.2 为什么 GPU 时间轴上仍然会重叠

关键点在于：这里的拷贝是异步提交，不是同步阻塞拷贝。

#### 当前层 miss load

`current_layer_miss_load` 发生在 `ExpertBufferManager.load_experts_for_current_layer()` 里：

- 对于既不在 GPU cache、也不在 `prefetch_mapping` 里的专家，进入 `remaining_experts`
- 对每个 `remaining_expert` 调用 `_load_expert_to_buffer(...)`
- `_load_expert_to_buffer(...)` 内部是多个 `copy_(..., non_blocking=True)`

也就是说，CPU 做的不是“等一个专家完全搬完，再搬下一个，再等函数返回”，而是在当前 stream 上连续提交一串异步 H2D copy。

#### 下一层 prefetch

`_parallel_prefetch()` 的行为是：

1. 先 `prefetch_stream.synchronize()`，只保证“上一轮 prefetch stream 上的任务已经结束”。
2. 然后进入 `with torch.cuda.stream(prefetch_stream)`。
3. 对下一层预测专家逐个调用 `prefetch_expert(...)`。
4. `prefetch_expert(...)` 里同样调用 `_load_expert_to_buffer(...)`，也是 `copy_(..., non_blocking=True)`。

也就是说，下一层 prefetch 被提交到单独的 `prefetch_stream`，而不是当前计算 stream。

### 1.3 为什么会出现“miss load 已经开始了，但 prefetch 过一会儿才开始”

这是当前代码的预期行为，不是异常。

原因有三层：

1. `next_layer_prefetch` 的 launch 点本来就在 `batch_load_experts_continuous(...)` 之后。
2. 但 `batch_load_experts_continuous(...)` 返回，不代表当前层 miss load 在 GPU 上已经执行完；它只代表 CPU 已经把这一批异步 copy 提交出去了。
3. 因为 `PRELAUNCH_ENABLED=True`，代码不会在这两个阶段之间做 `torch.cuda.synchronize()`，所以 CPU 会继续往前走，马上把下一层 prefetch 的 copy 也提交到另一条 stream。

因此实际会出现这种时间线：

1. 当前 stream 上的 `current_layer_miss_load` 第一批 H2D copy 已经开始在 GPU/DMA 侧执行。
2. CPU 很快把这一批 copy 都提交完，函数返回。
3. CPU 继续进入 `_parallel_prefetch()`。
4. prefetch stream 上开始排队下一批 H2D copy。
5. 这时当前 stream 上上一批 miss load 还没完全排空，所以两者在 GPU 全局时间轴上部分重叠。

这也是你看到平均值里：

- `current_layer_miss_load` 平均约 `1.23 ~ 13.81 ms`
- `next_layer_prefetch` 平均约 `8.49 ~ 24.00 ms`

即 `prefetch` 常常在 `miss_load` 的中后段开始。

### 1.4 这是否意味着两批 H2D copy 在“满带宽并行”

不意味着。

这里需要严格区分两件事：

1. 软件层面是否允许并发提交
2. 硬件层面是否真的拿到独立带宽

当前代码只保证了第 1 点：

- `current_layer_miss_load` 在当前 stream 上提交
- `next_layer_prefetch` 在 `prefetch_stream` 上提交
- 两者之间没有显式跨 stream 依赖

这意味着两批 copy 可以同时“处于 in-flight / queued / executing”的状态。

但它不意味着：

- H2D 带宽翻倍
- 两批 copy 毫无竞争
- 两者都以各自单独执行时的峰值带宽完成

实际硬件行为更接近：

- 两个 stream 上的 H2D copy 都被提交到了设备
- 它们共享同一块 GPU、同一条 host-device 链路、同一套 host pinned memory 源数据、以及设备内部的 copy 资源
- GPU / DMA engine / PCIe 或 NVLink 会对它们做排队、交错或分时推进

所以“时间上部分重叠”和“有效带宽完全相加”不是一回事。

更准确的理解是：

- 当前 runtime 利用异步 stream，把下一层 prefetch 尽早提交出去
- 让它尽量和当前层剩余的 miss load、甚至后面的 compute 区间发生重叠
- 这样 wall-clock 关键路径会变短
- 但单个 copy 的完成时间通常会因为竞争而被拉长，不会白拿双倍带宽

### 1.5 为什么 `next_layer_prefetch` 不是在 `current_layer_miss_load` 一结束立刻开始

因为它的 start 点还受另外两个因素影响：

1. `_parallel_prefetch()` 开头有 `prefetch_stream.synchronize()`  
   这会先等“上一层遗留的 prefetch stream 任务”结束。

2. timing summary 记录的是“很多 layer、很多 step 平均后的区间”  
   它不是某一层的严格固定时序。平均后出现“中途开始”的现象，本质上反映的是：
   - 有些层 miss load 多，prefetch 晚启动
   - 有些层 miss load 少，prefetch 更早启动
   - 有些层还要先等上一层 prefetch stream 排空

因此平均 start 点落在 miss load 区间内部，是合理结果。

### 1.6 总结

一句话总结这一段机制：

- `current_layer_miss_load` 和 `next_layer_prefetch` 虽然都是 CPU->GPU 权重搬运，
- 但它们被提交到不同 stream，
- 且 `PRELAUNCH_ENABLED=True` 让两者之间没有同步栅栏，
- 所以 GPU 时间轴上会部分重叠；
- 这种重叠表示“异步并发提交 + 部分重叠执行”，不表示“各自独占满带宽同时传输”。

## 2. Pure SDAR 里 `moe_dispatch`、`moe_expert_compute_hbm_fetch`、`moe_scatter`、`other_layer_overhead` 分别是什么

### 2.1 先看原始 Pure SDAR 的 MoE 执行机制

原始 `SDARMoeSparseMoeBlock.forward()` 的主体逻辑是：

1. 对所有 token 做 gate，得到 `router_logits`
2. `softmax + topk`，得到每个 token 的 top-k 专家
3. `one_hot(selected_experts)` 生成 `expert_mask`
4. `for expert_idx in range(self.num_experts)` 遍历全部专家
5. 对每个专家：
   - `torch.where(expert_mask[expert_idx])` 找出分配到该专家的 token
   - gather 这些 token 的 hidden state
   - 跑该专家自己的 `gate_proj/up_proj/down_proj`
   - 乘 routing weight
   - `index_add_` 写回 `final_hidden_states`

注意这里最关键的一个特征：

- 它不是“只遍历活跃专家”
- 而是“固定遍历这一层的全部 `num_experts` 个专家”

对 SDAR-30B-A3B，这意味着每个 sparse layer 都会固定走一遍 128 专家的 Python 循环。

即使某个专家在这一层没有任何 token 命中，代码也仍然会进入这次循环，至少做一次 `torch.where(...)`。

### 2.2 `moe_dispatch` 的含义

在 profile 版里，`moe_dispatch` 包裹的是：

- `idx, top_x = torch.where(expert_mask[expert_idx])`
- `current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)`

它对应的含义是：

- 找出“这个专家当前要处理哪些 token / 哪个 top-k rank”
- 然后把这些 token 的 hidden states 从大张量里 gather 出来，组成本专家的输入 batch

这部分为什么不小：

1. 固定循环 128 次  
   就算只有 40~60 个专家活跃，其余 60~80 次循环仍然存在。

2. `expert_mask` 是先做 one-hot，再按专家维度 `permute`  
   这个中间表示本身就不轻。

3. `torch.where` 和高级索引不是免费操作  
   它们会启动额外 kernel，涉及索引张量生成和 gather。

4. 每个专家的 token 数通常不大  
   于是你会得到大量小粒度 dispatch，而不是少量大批量 dispatch。

可以用一个简化例子理解：

- 假设当前层有 16 个 token，每个 token top-k=2
- 实际只有专家 `3、7、20、41` 被用到
- 原始 Pure SDAR 仍然会循环 `0..127`
- 对 `0、1、2、4、5 ...` 这些没被用到的专家，也会各跑一次 `torch.where`

因此 `moe_dispatch` 不只是“真正有 token 的专家的 gather 时间”，还包括“大量空专家检查 + 小 batch gather”的代价。

### 2.3 `moe_expert_compute_hbm_fetch` 的含义

在 profile 版里，这一项包裹的是：

- `expert_layer(current_state) * routing_weights[...]`

其中 `expert_layer(current_state)` 就是 `SDARMoeMLP.forward()`：

- `gate_proj(x)`
- `up_proj(x)`
- `act_fn(gate) * up`
- `down_proj(...)`

我把它命名成 `moe_expert_compute_hbm_fetch`，是因为这个阶段同时包含两类事情：

1. expert 真正的 MLP 计算
2. kernel 读 expert 权重时，从 GPU HBM 把权重搬到片上 cache / register / SRAM 的代价

这里需要强调一个常见误区：

- “权重已经在 GPU 上”不等于“这一步没有权重加载成本”

即使所有专家参数都已经常驻显存，每次 `linear` 启动时，kernel 仍然必须：

- 从 HBM 读取对应权重
- 再送入 SM 上的寄存器 / cache / shared memory 参与 matmul

所以 Pure SDAR 没有 CPU->GPU copy，并不意味着 MoE 权重访问开销消失了。它只是没有跨 PCIe/NVLink 的 H2D copy，但仍然有 GPU 内部的 HBM 访问。

这部分时间为什么大：

1. 每个专家都单独跑自己的 MLP
2. 一个专家对应至少 3 个线性层算子
3. 每个专家的 token batch 往往很小，难以把 matmul 做大
4. 于是会变成很多小 GEMM / GEMV kernel
5. 每个 kernel 都要重新读取这位专家的权重

所以这部分既包含算术成本，也包含大量“小 kernel + 重复读 HBM 权重”的成本。

### 2.4 `moe_scatter` 的含义

在 profile 版里，`moe_scatter` 包裹的是：

- `final_hidden_states.index_add_(0, top_x, current_hidden_states)`

它的含义是：

- 把当前专家算出的输出，按 token 位置加回全局输出张量

为什么它也会明显占时：

1. 这是按专家重复执行的  
   原始实现不是“把全部专家输出拼好以后一次 scatter”，而是每个专家单独做一次 `index_add_`。

2. `index_add_` 属于稀疏式写回  
   写地址由 `top_x` 决定，不像普通 dense matmul 那样规则。

3. 同一个 token 可能会被多个专家写回  
   因为 top-k>1，所以 scatter 本质上带有聚合语义。

4. 小 batch、多次调用  
   这又会带来大量小 kernel 和额外 launch 开销。

可以继续用上面的例子理解：

- token 5 同时命中了专家 3 和专家 20
- 那么专家 3 的输出先对 `final_hidden_states[5]` 做一次加法
- 专家 20 的输出稍后还要再对同一个位置做一次加法

这类反复的按索引累加，本来就不是 GPU 最擅长的高吞吐模式。

### 2.5 `other_layer_overhead` 包含什么

`other_layer_overhead` 不是一个具体单独算子，而是“这个 layer 总时间减去前面已分类时间”后的剩余项。

对 Pure SDAR，它主要包括：

1. `input_layernorm`
2. attention 后 residual add
3. `post_attention_layernorm`
4. MLP 前后的张量 reshape / view / 临时张量分配
5. `final_hidden_states` 的初始化
6. `expert_mask = one_hot(...).permute(...)`
7. Python for-loop 本身和循环体里未被单独包住的零碎逻辑
8. tuple 解包、router logits 处理、残差连接等

所以你看到 `other_layer_overhead` 不小，并不意味着有一块“神秘的慢算子”；它更像是：

- layernorm + residual 这些 dense 小算子
- one-hot / permute / 分配中间张量
- Python 调度与碎片化小开销

被合并到了一个兜底项里。

### 2.6 为什么这四项都很可观

核心原因可以压缩成一句话：

原始 Pure SDAR 的 sparse MoE 实现，本质上是“Python 驱动的、按专家逐个 dispatch / compute / scatter 的细粒度执行方式”，它天然会产生：

- 很多次专家循环
- 很多次 `torch.where`
- 很多次小 gather
- 很多次小 linear
- 很多次小 `index_add_`
- 以及大量中间张量和 Python 调度开销

所以它不会呈现出“attention 占大头、MoE 很轻”的形态，而会呈现出：

- `dispatch` 很重
- `expert compute + HBM fetch` 很重
- `scatter` 也不轻
- 剩余杂项也不小

## 3. 为什么 Offloading runtime 的 MoE 执行层面比原始 Pure SDAR 更高效

### 3.1 先澄清一个前提

“在 prefetch 和 miss load 之后，权重都已经在 GPU 上了，所以计算应该和 Pure SDAR 一样快”这个判断并不成立。

原因是：

- 权重在 GPU 上，只说明“数据位置一样”
- 不说明“计算组织方式一样”

真正决定快慢的，除了数据是否在 GPU 上，还包括：

- 是否遍历全部专家
- 是否把同专家 token 聚成连续 batch
- 是否把多个专家计算合并成更大的 batched matmul
- scatter 是一次做还是很多次碎片化做
- kernel 启动数量有多少

当前 offloading runtime 恰恰在这些点上和原始 Pure SDAR 差别很大。

### 3.2 原始 Pure SDAR 的执行方式

原始实现是：

1. `selected_experts`
2. `one_hot -> expert_mask`
3. `for expert_idx in range(num_experts)`
4. 每个专家：
   - `torch.where`
   - gather token
   - 跑本专家 MLP
   - `index_add_` scatter 回去

特点是：

- 遍历所有 128 个专家
- dispatch / compute / scatter 都按专家碎片化执行
- kernel 数量很多
- 空专家也要过一遍循环

### 3.3 Offloading runtime 的执行方式

offloading 版在 `baseline/sdar_layers.py` 中，做法是：

1. 先拿到当前层所有 token 的 `selected_experts`
2. 把 `selected_experts` flatten 成一维 `flat_experts`
3. `bincount` 统计每个专家实际命中了多少 token
4. 只保留 `active_expert_ids`
5. `torch.sort(flat_experts)`，把同一专家对应的 token-rank 项排到一起
6. 用排好序的 `sorted_tokens / sorted_ranks` 一次性 gather 出 `all_input_states` 和 `all_routing_weights`
7. 对活跃专家构造连续的权重 view
8. 如果 `BMM_ENABLED=True`，把多个活跃专家堆成 batched tensor，用 `torch.bmm` 一次做一批专家
9. 最后用一次 `scatter_add_` 把所有专家输出写回

也就是：

- `reorder`
- `gather`
- `batched expert compute`
- `scatter`

这四步把原始“按专家逐个碎片化执行”的方式改成了“先分组，再批处理”的方式。

### 3.4 `reorder` 是什么

`reorder` 的作用是：

- 不再按“token 顺序”看专家选择结果
- 而是按“expert id 顺序”把所有 `(token, k-rank)` 对重新排好

举一个简单例子。

假设有 4 个 token，每个 token top-2：

- token0 -> expert `[7, 3]`
- token1 -> expert `[3, 9]`
- token2 -> expert `[7, 9]`
- token3 -> expert `[3, 7]`

flatten 后是：

- `[7, 3, 3, 9, 7, 9, 3, 7]`

`reorder` 之后会变成类似：

- `[3, 3, 3, 7, 7, 7, 9, 9]`

同时保留这些项各自原来对应的 token 索引和 top-k rank。

这样一来，属于同一个专家的 token 会被排成一段连续区间，后面就更容易做连续 gather 和 batched compute。

### 3.5 `gather` 是什么

`gather` 就是根据 `reorder` 后的结果，一次性把需要计算的 token hidden states 收集到连续张量里。

原始 Pure SDAR 是：

- 每个专家循环一次
- 每次只 gather 当前专家自己的 token

offloading runtime 是：

- 先把所有专家的 token assignment 排好
- 再一次 gather 出 `all_input_states`
- 每个专家只是在这块连续大张量里占据一个区间

好处是：

- 内存访问更规整
- 减少反复小 gather
- 为后面的 batched matmul 做准备

### 3.6 `batched expert compute` 是什么

offloading 里如果 `BMM_ENABLED=True`，会：

1. 对每个活跃专家，把它对应的 token 输入放进 `batched_inputs[row_idx, :count]`
2. 把每个专家的 `gate/up/down` 权重也分别 stack 成：
   - `gate_w`
   - `up_w`
   - `down_w`
3. 用 `torch.bmm` 做批量矩阵乘法：
   - `gate_out = bmm(batched_inputs, gate_w^T)`
   - `up_out = bmm(batched_inputs, up_w^T)`
   - `exp_out = bmm(gate_out * up_out, down_w^T)`

它不是再“每个专家单独发 3 个 linear kernel”，而是把多位专家打成一个 batch，一次发更大的 batched kernel。

这带来三个直接收益：

1. kernel 数量更少
2. 每个 kernel 的工作量更大，更容易吃满 GPU
3. 权重和输入访问更连续，整体吞吐更高

### 3.7 `scatter` 和原始 `index_add_` 的区别

原始 Pure SDAR 是：

- 每个专家算完后，立刻对 `final_hidden_states` 做一次 `index_add_`
- 所以是很多次小 scatter

offloading runtime 是：

- 所有活跃专家的输出先放进 `expert_outputs`
- 然后一次 `scatter_add_` 回 `final_hidden_states`

也就是把很多碎的写回，合并成一次更大的写回。

这会减少：

- kernel 数量
- 小规模稀疏 scatter 的反复开销

### 3.8 为什么“只遍历活跃专家”本身就很重要

这是最直观但也最重要的一点。

假设某一层：

- 总专家数 = 128
- 实际活跃专家数 = 58

那么：

- 原始 Pure SDAR 会固定循环 128 次
- offloading runtime 只处理这 58 个活跃专家

也就是说，offloading runtime 天然砍掉了：

- 70 次空专家循环
- 70 次空 `torch.where`
- 以及大量空分支、小张量调度和 Python 层开销

这不是一个小优化，而是执行范式上的变化。

### 3.9 所以为什么 Offloading 没有慢很多很多

综合起来，offloading 版本一层里同时发生了两件事：

1. 新增了 CPU->GPU 权重搬运
2. 把原始低效的 MoE 执行方式换成了更高效的 runtime

新增的代价是：

- `current_layer_miss_load`
- `next_layer_prefetch`

但被抵消掉的一部分原始代价是：

- 遍历所有 128 专家的 Python 循环
- 大量空专家 `torch.where`
- 大量小 gather / 小 linear / 小 index_add
- 更差的批量化程度

所以最终看到的不是：

- “offloading 因为多了搬运，所以一定比 pure SDAR 慢一大截”

而是：

- “offloading 的确更慢，但没有按 H2D copy 的累计时间线性变差”

因为 copy 有重叠，而 compute 组织方式又更优。

## 4. 读 timing summary 时要注意什么

### 4.1 Offloading summary 里的时间不是互斥切片

在 `baseline/sdar_runtime_trace.py` 里：

- 每个 operation 的 `average_duration_ms` 是该类区间在一个 layer 内的累计 duration 平均值
- 不同 operation 可能来自不同 stream
- 因此不同类 duration 可以重叠，不能直接相加

这正是为什么你会看到：

- `current_layer_miss_load` 占比很大
- `next_layer_prefetch` 占比也很大
- 两者加起来会超过 100%

### 4.2 Pure SDAR profile 里的 `dispatch/compute/scatter` 也是累计项

在 `baseline/sdar_dense_profile.py` 里：

- 每个 layer 内，同一 category 会有很多 interval
- 因为原始实现对 128 个专家逐个循环
- `average_duration_ms` 是这些 interval 的总和的平均值
- `average_start_ms ~ average_end_ms` 则是这一类 interval 的“包络区间”

所以你看到：

- `moe_dispatch.average_duration_ms = 7.63 ms`
- 但它的 wall span 约 `22.4 ms`

意思不是“dispatch 独占跑了 22ms”，而是：

- 这一层里很多次 dispatch 小片段分散地穿插在整个 MoE 循环里
- 它们加起来总共大约 7.63ms
- 从第一次 dispatch 到最后一次 dispatch 的包络覆盖了约 22ms

这和 offloading summary 的解读方式是一致的。

## 5. 最终结论

针对这三个问题，结论可以压缩成下面几句：

1. `current_layer_miss_load` 和 `next_layer_prefetch` 之所以会重叠，是因为当前代码把它们异步提交到不同 CUDA stream，且 `PRELAUNCH_ENABLED=True` 去掉了中间同步栅栏；这表示存在并发排队和部分重叠，不表示二者各自都拿到独占满带宽。
2. Pure SDAR 的 `moe_dispatch / moe_expert_compute_hbm_fetch / moe_scatter / other_layer_overhead` 都明显，是因为原始 sparse MoE 实现本身就是“遍历全部专家、逐专家做小粒度 dispatch/compute/scatter”的执行方式，MoE 路径天然很重。
3. Offloading runtime 在权重已经到 GPU 后仍然更高效，是因为它不仅改变了“权重在哪里”，还改变了“MoE 怎么算”：只处理活跃专家，把 token 先按专家重排，再做连续 gather、batched expert compute 和一次性 scatter，显著减少了空循环、小 kernel 和碎片化 dispatch/scatter。

这也是为什么本项目里看到的现象是：

- Pure SDAR 仍然更快
- 但 Offloading 没有慢到“远低于 pure SDAR”
- 因为 Offloading 新增了 H2D copy，同时也替换掉了原始 Pure SDAR 那条本来就不高效的 MoE 执行链路
