# SDAR MoE-Offloading 集成与推理链路说明

## 文档目标

这份文档不是使用说明，而是代码级链路说明。目标是把当前
`SDAR-30B-A3B` 在 `MoE-Offloading` 场景下的构建方式、调用关系、
与原始 SDAR 和原始 MoE-Offloading 代码的对应关系完整拆开，便于检查
移植是否正确。

## 一句话概括当前集成

当前实现不是重写一套新的 SDAR 解码器，而是保留原始 OpenCompass/SDAR
的 block-diffusion 解码主链，只把其中的 MoE 层从“模型内常驻 experts”
换成“CPU expert cache + GPU swap buffer + 可选 GPU expert cache”的
MoE-Offloading runtime。

也就是说：

- SDAR 的块间自回归、块内并行迭代、confidence 接收/remask 逻辑没有改
- SDAR 的 attention 和 `store_kv` 语义没有改
- OpenCompass 的 tokenizer、chat template、dataset config、prompt
  template 也没有改
- 真正改动的是模型构建方式和每层 MoE 的执行方式

## 一、原始两套代码分别负责什么

### 1. 原始 SDAR 代码负责什么

原始 SDAR 代码主要负责：

- OpenCompass 配置和数据集组织
- `BD3withChatTemplate` 封装
- `block_diffusion_generate()` 的块扩散解码逻辑
- `modeling_sdar_moe.py` 里的 SDAR 模型结构、attention、KV cache 提交语义
- 原始 `SDARMoeSparseMoeBlock` 的 router + experts 执行

原始 SDAR 的典型调用链是：

1. OpenCompass 配置文件选定模型和数据集
2. `GenInferencer.inference()` 调 `self.model.generate_from_template(...)`
3. `BD3withChatTemplate.generate_from_template()` 调 `generate()`
4. `generate()` 调 `block_diffusion_generate()`
5. `block_diffusion_generate()` 在每一步里调用 `model(...)`
6. `model(...)` 进入 `SDARMoeForCausalLM.forward() -> SDARMoeModel.forward()`
7. 每一层执行 `SDARMoeDecoderLayer.forward()`
8. 层内 attention 由 `SDARMoeAttention.forward()` 完成
9. 层内 MoE 由 `SDARMoeSparseMoeBlock.forward()` 完成

### 2. 原始 MoE-Offloading 代码负责什么

原始 `MoE-Offloading` 项目主要负责：

- expert 权重不常驻 GPU 的 runtime
- CPU expert cache
- GPU swap buffer
- 可选 GPU expert cache
- prefetch
- gather / batched expert compute / scatter
- buffer/cache 统计
- latency / memory benchmark 脚手架

原始项目最成熟的路径是 `Qwen3MoE` 和 `GPT-OSS`。它们的共同思路是：

1. 创建一个“没有真正 experts”的模型骨架
2. dense 权重加载到 GPU
3. expert 权重单独加载到 CPU cache
4. 用自定义 wrapper 替换原始 MoE 层
5. wrapper 在 forward 里按 routing 结果按需取 expert 权重并计算

## 二、这次移植的核心策略

这次移植没有去碰 `block_diffusion_generate()` 的算法本身，而是采用了下面的
策略：

1. 保留原始 `BD3withChatTemplate` 和 `block_diffusion_generate()`
2. 保留原始 `modeling_sdar_moe.py` 里的 SDAR attention 和 `store_kv` 语义
3. 额外提供一个新的本地模型入口
   `configs.sdar_local_models.modeling_sdar_moe_offloading`
4. 这个新入口不从 HF/transformers 默认路径构建完整 SDAR 模型，而是调用
   `baseline/sdar_builder.py`
5. `sdar_builder.py` 负责构建一个“dense 在 GPU、experts 在 CPU cache”的
   SDAR 模型
6. 原始 `SDARMoeSparseMoeBlock` 被替换成
   `baseline/sdar_layers.py` 里的 `SDARSparseMoeWrapper`

这意味着：

- SDAR 解码主链沿用原始代码
- offloading runtime 复用原始 `MoE-Offloading` 的缓存/缓冲区/计算链
- 新写的部分主要是“把两边接起来”的胶水层

## 三、当前评测脚本和原始 OpenCompass 推理链的关系

这里有一个很重要的点。

原始 OpenCompass 全量评测会经过：

1. 配置文件
2. `GenInferencer`
3. `generate_from_template`

而当前新增的评测脚本
`tests/test_sdar_offloading.py` 为了更直接测 latency 和 memory，没有启动
完整的 OpenCompass runner/inferencer，而是手工复用了同一套数据集配置和
prompt template。

具体做法是：

1. `tests/test_sdar_offloading.py`
2. `tests/sdar_offloading_utils.py::load_dataset_bundle()`
3. 直接 import OpenCompass 的数据集 config，例如
   `opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799`
4. 直接实例化其中的：
   - dataset
   - prompt_template
   - evaluator
   - pred_postprocessor
   - dataset_postprocessor
5. 对每条样本直接调用 `prompt_template.generate_item(entry)`
6. 然后再调用 `BD3withChatTemplate.generate_from_template([prompt], ...)`

所以这条测试链路与原始 OpenCompass 的关系是：

- 相同点：
  - 用的是同一个 dataset config
  - 用的是同一个 prompt template
  - 用的是同一个 `BD3withChatTemplate`
  - 用的是同一个 `block_diffusion_generate()`
  - 用的是同一个 SDAR 本地模型入口机制
  - 用的是同一个 evaluator / postprocessor
- 不同点：
  - 没有经过 `GenInferencer`
  - 没有经过 OpenCompass 的 runner/partitioner/task 调度层
  - 这样做是为了减少调度噪声，更适合单卡 latency/memory profiling

因此，当前脚本测的是“与 SDAR 实际使用数据集配置一致、但更轻量的评测入口”。

## 四、当前代码新增了哪些文件，各自职责是什么

### 1. `baseline/sdar_layers.py`

这是 SDAR 版的 MoE 执行 wrapper，职责是：

- 保留 router gate
- 计算 top-k experts
- 为下一层做 gate-based prefetch 预测
- 通过 `ExpertCache` / `ExpertBufferManager` 加载本层需要的 experts
- 调用共享的 swap buffer / GPU cache 逻辑
- 完成 gather / compute / scatter
- 输出 `(hidden_states, router_logits)`

它本质上是把原来 `baseline/qwen3_layers.py` 的 offloading 逻辑，改写成能
挂到 SDAR 模型上，并去掉了 Qwen3 的硬编码层号假设，改为：

- `self.num_hidden_layers = config.num_hidden_layers`
- “最后一层”判断使用 `self.num_hidden_layers - 1`

### 2. `baseline/sdar_builder.py`

这是本次集成的核心 builder。职责是：

- import 原始 `configs.sdar_local_models.modeling_sdar_moe`
- 对 SDAR 的 attention、decoder layer、sparse moe block 做 monkey patch
- 用一个“不真正创建 experts”的配置构造模型骨架
- 加载 dense 权重到 GPU
- 加载 expert 权重到 CPU cache
- 把所有 MoE 层替换成 `SDARSparseMoeWrapper`

### 3. `configs/sdar_local_models/modeling_sdar_moe_offloading.py`

这是 OpenCompass 会看到的本地模型入口。职责只有一个：

- 实现 `SDARMoeForCausalLM.from_pretrained()`
- 在这个类方法内部，不走默认 HF 构造，而是直接调用
  `baseline.sdar_builder.sdar_build_model()`

也就是说，这个文件是 OpenCompass 世界和 MoE-Offloading 世界的连接点。

### 4. `tests/sdar_offloading_utils.py`

这是评测辅助层，负责：

- 轮询 GPU 空闲状态
- 加载 OpenCompass dataset config
- 实例化 dataset / prompt template / evaluator
- 导出 memory snapshot CSV
- 导出结果 JSON

### 5. `tests/test_sdar_offloading.py`

这是实际评测入口，负责：

- 选择可用 GPU
- 加载真实 SDAR 数据集配置
- 构造 `BD3withChatTemplate`
- 触发真实 SDAR block-diffusion 解码
- 统计 per-sample latency
- 统计 token 数和 tokens/s
- 执行 evaluator
- 导出 JSON / CSV

## 五、模型构建阶段到底做了什么

下面按 `sdar_build_model()` 的真实顺序说明。

### 1. 先对原始 SDAR 类做 monkey patch

在 `baseline/sdar_builder.py` 里做了三类 patch：

1. 给 `SDARMoeAttention.forward()` 加 NVTX 包裹
2. 改写 `SDARMoeDecoderLayer.__init__()`
3. 改写 `SDARMoeSparseMoeBlock.__init__()`

这样做的目的不是改 SDAR 算法，而是为了在“构造模型骨架”的阶段避免真正创建
128 个 experts 的常驻参数。

### 2. 构造“空 experts”的 SDAR 模型骨架

builder 会：

- 从模型快照读取 `SDARMoeConfig`
- 保存 `original_num_experts`
- 把 `config.num_experts = 0`
- 再额外记录：
  - `config._target_experts = original_num_experts`
  - `config._original_num_experts = original_num_experts`

这里的语义是：

- `num_experts = 0`：构造骨架时不要真的创建 experts 参数
- `_target_experts = 128`：router gate 维度仍然要保持 128

也就是说，模型骨架里：

- attention、norm、embedding、lm_head 等 dense 模块照常创建
- MoE block 只保留 gate，不创建 `experts = ModuleList([...])`

### 3. 为什么这一步是必要的

如果沿用原始 SDAR 构造方式，48 层 * 128 experts 的所有权重会直接成为模型参数，
这就和 offloading 的目标相冲突了。

所以必须先创建一个“没有常驻 experts”的骨架，再把 expert 权重放进
CPU cache，由 runtime 按需搬运。

### 4. 初始化 ExpertCache

builder 调用统一的 `ExpertCache`：

- `state_path` 指向 SDAR 模型快照
- `device` 是当前 GPU
- `buffer_size = original_num_experts = 128`
- 可选参数包括：
  - `enable_gpu_cache`
  - `cache_policy`
  - `topk_lru_logit_percentile`
  - `cache_slots_per_layer`

这里没有新写一套 SDAR 专用 cache，而是复用了已有统一实现。因为
SDAR-30B-A3B 的 expert 权重布局与 Qwen3 一样属于：

- separate structure
- gate / up / down 三套权重分开存储
- 无 bias

所以统一的 `ExpertCache` 可以直接处理。

### 5. 加载权重时如何区分 dense 与 experts

builder 读取模型目录下的 `model.safetensors.index.json`，按文件分组，然后逐个
文件处理。

处理规则是：

- 如果权重名里包含 `experts.`，就交给 `expert_cache._process_weights_batch()`
  进入 CPU cache
- 否则就是 dense 权重，直接 `.to(device)` 放到模型参数里

这一步非常关键，因为它决定了：

- 模型对象本身只持有 dense 权重
- expert 权重不进入模型参数树，而是进入 cache runtime

### 6. SDAR 模型快照的文件组织与当前 builder 的适配关系

当前 `SDAR-30B-A3B-Chat-b32` 的模型快照不是单纯的
`model-00001-of-000xx.safetensors` 风格，还包含：

- `others.safetensors`
- `layer-{i}-ep-0-of-1.safetensors`
- `model.safetensors.index.json`

当前 builder 不依赖固定文件名，而是只依赖 `model.safetensors.index.json` 的
weight map，所以能正确适配这种按层拆分的权重布局。

### 7. 初始化 GPU expert cache

如果 `enable_gpu_cache=True`，builder 会在 CPU cache 准备完以后调用：

- `expert_cache.init_gpu_cache()`

这会创建 `GPUExpertCacheManager`，并为每层分配固定数量的 expert cache slots。

### 8. 把原始 MLP 替换成 offloading wrapper

最后，builder 遍历每一层：

- 拿到原始 `layer.mlp.gate`
- 用 `SDARSparseMoeWrapper(config, layer_idx, gate, expert_cache)` 替换
  `layer.mlp`

从这一刻起，SDAR 模型的每层 MoE 执行，不再走原始
`SDARMoeSparseMoeBlock.forward()`，而是走 `SDARSimpleMoE.forward()`。

## 六、执行阶段到底走了哪条链

下面按一次真实样本生成说明执行链。

### 1. 评测脚本构造 `BD3withChatTemplate`

`tests/test_sdar_offloading.py` 实例化 `BD3withChatTemplate` 时，传入了：

- `path=model_path`
- `local_modeling_module=configs.sdar_local_models.modeling_sdar_moe_offloading`
- `generation_kwargs=...`
- `model_kwargs=...offloading args...`

这里最关键的是 `local_modeling_module`。它让 `BD3withChatTemplate` 在加载模型时，
不是走默认 `AutoModelForCausalLM.from_pretrained()`，而是去 import 这个本地模块。

### 2. `BD3withChatTemplate._load_model()` 如何接入本地模型

`BD3withChatTemplate._load_model()` 会：

1. `_load_local_model_class(...)`
2. 从模型快照 `config.json` 里读 `architectures[0]`
3. 找到本地模块里的同名类 `SDARMoeForCausalLM`
4. 调用它的 `from_pretrained(...)`

### 3. 本地 `from_pretrained()` 做了什么

`modeling_sdar_moe_offloading.py` 里的 `SDARMoeForCausalLM.from_pretrained()`
并不直接返回 HF 默认模型，而是：

1. 解析 `device` / `device_map`
2. 读取 offloading 参数
3. 调用 `sdar_build_model(...)`

所以实际返回给 `BD3withChatTemplate` 的，是一份已经被 offloading 化的
SDAR 模型。

### 4. prompt 是如何构造的

当前测试脚本没有调用 `GenInferencer`，而是直接：

1. 从 OpenCompass GSM8K config 里取 `PromptTemplate`
2. 对样本 `entry` 调 `prompt_template.generate_item(entry)`
3. 再传给 `BD3withChatTemplate.generate_from_template([prompt], ...)`

因此 prompt 内容与 OpenCompass 配置保持一致。

## 七、`BD3withChatTemplate.generate()` 之后到底发生什么

### 1. 文本先被 tokenizer 和 chat template 处理

`generate()` 会把 prompt 包装成 chat messages，再用 tokenizer 编码成：

- `input_ids`
- `attention_mask`

### 2. 然后进入 `block_diffusion_generate()`

这一步没有改，仍然使用原始 SDAR 解码器。

它会：

1. 根据 `prompt_len`、`gen_length`、`block_length` 计算总块数
2. 构造块级下三角 attention mask
3. 初始化一个全长 `x`，把 prompt 放前面，其余位置填 `mask_id`
4. 计算 `prefill_blocks = prompt_len // block_length`
5. 只对完整 prompt blocks 执行 prefill，并把它们的 KV 写入 cache

### 3. 为什么 prefill 只处理完整块

这是原始 SDAR 的语义，不是这次移植加的。

因此：

- 如果 prompt 长度不能整除 `block_length`
- 多出来的 prompt 尾巴不会提前进入 KV cache
- 它会作为第一个待生成块的一部分参与块内迭代

### 4. 每个 block 的生成过程

对每个 block：

1. 取出当前 block 的 `cur_x`
2. 取出当前 block 对应的 block-diffusion attention mask
3. 进入 `for i in range(denoising_steps + 1)` 迭代

如果当前 block 还有 `MASK`：

- 调 `model(... store_kv=False)`
- 得到 logits
- 根据 `temperature / top_k / top_p` 采样
- 计算 confidence
- 根据 `threshold` 和 `num_transfer_tokens[i]` 决定哪些位置被接受
- 未接受的位置继续 remask

如果当前 block 已经没有 `MASK`：

- 调一次 `model(... store_kv=True)`
- 把这个 block 的最终 KV 提交进全局 `DynamicCache`
- 然后进入下一个 block

## 八、`store_kv` 为什么对 SDAR 很关键

这部分是理解当前移植是否正确的关键点之一。

原始 SDAR attention 的语义是：

- `store_kv=True`：
  - 当前 block 计算出的 K/V 会真正写入 `past_key_values`
- `store_kv=False`：
  - 只取历史 cache 中已经提交过的 past K/V
  - 再把当前 block 的 K/V 临时拼接进去用于当前轮 attention
  - 但不会把当前 block K/V 提交到 cache

这意味着：

- 在一个 block 还没最终确定之前，它可以反复改 token
- 但历史块的 KV 是稳定的
- 当前 block 一旦最终确定，再统一提交一次 KV

这正是“块间自回归，块内并行反复迭代”的实现基础。

这次移植完全保留了这个机制，没有去改 attention 或 cache 语义。

## 九、offloading 版 MoE 在层内是怎么执行的

下面是 `baseline/sdar_layers.py` 中 `SDARSimpleMoE.forward()` 的真实执行顺序。

### 1. router 计算

输入 `hidden_states` 先展平成 `[tokens, hidden_dim]`，然后：

- `router_logits_tensor = self.gate(hidden_states_flat)`
- `softmax`
- `topk`

得到：

- `routing_weights`
- `selected_experts`

### 2. 下一层 prefetch 预测

如果开启 prefetch，当前层会：

- 从 `GateRegistry` 取出下一层的 gate
- 用当前层输出前的 `hidden_states_flat` 直接过下一层 gate
- 对下一层做一次 top-k 预测
- 取 `PREFETCH_TOPK=4`
- 把这些 expert id 提前拉到 CPU->GPU prefetch 流程里

这部分沿用了原来 Qwen3 offloading 的思路。

### 3. 统计当前层有哪些 experts 真正活跃

当前实现不是在 GPU 上做复杂稀疏调度，而是：

- `selected_experts.reshape(-1)`
- 先搬到 CPU
- `torch.bincount`

得到：

- 每个 expert 被多少 token 命中
- 当前层活跃的 `active_expert_ids`

### 4. 调用 ExpertCache / BufferManager 加载 experts

然后调用：

- `expert_cache.batch_load_experts_continuous(self.layer_idx, expert_indices, router_logits)`

这里可能发生三种情况：

1. expert 已经在 GPU expert cache 里
2. expert 不在 GPU expert cache，但在 CPU cache 里，需要临时装入 swap buffer
3. 装入 swap buffer 后，如果策略允许，还会回写/更新 GPU cache

### 5. gather / reorder

加载完当前层需要的 expert 权重后，wrapper 会：

- 按 expert 对 token 重新排序
- gather 对应 token hidden states
- gather 对应 routing weights

这样后续计算可以按 expert 分组批量进行。

### 6. expert compute

这里没有再调用原始 `SDARMoeMLP` 模块对象，而是直接在 wrapper 里从 buffer
视图拿到：

- gate_proj
- up_proj
- down_proj

然后执行：

- SwiGLU
- down projection
- 乘 routing weight

如果启用了 BMM 路径，会把多个 active experts 组织成 batched 计算。

### 7. scatter

最后通过 `scatter_add_` 把所有 expert 输出加回 token 位置，恢复成：

- `[batch, seq_len, hidden_dim]`

并返回：

- `output_tensor`
- `router_logits`

这样就能与原始 `SDARMoeDecoderLayer.forward()` 保持兼容，因为原始 layer 已经支持：

- 如果 `self.mlp(hidden_states)` 返回 tuple
- 就按 `(hidden_states, router_logits)` 解释

## 十、当前实现和原始 SDAR MoE 的区别

原始 `SDARMoeSparseMoeBlock.forward()` 的逻辑是：

1. 计算 gate
2. 选择 experts
3. 直接从 `self.experts[expert_idx]` 取出常驻模块
4. 逐 expert 做 MLP
5. `index_add_` 聚合

当前 offloading 版的区别是：

1. `self.experts` 根本不创建
2. expert 权重不在模型参数里常驻
3. expert 权重在 CPU cache / GPU cache / swap buffer 中管理
4. 执行时按需取权重视图
5. MLP 计算由 wrapper 直接完成

所以二者在“数学上”执行的是同一种 expert 计算，但在“权重驻留位置”和“运行时调度方式”
上完全不同。

## 十一、当前实现和原始 Qwen3 MoE-Offloading 的关系

当前 SDAR offloading 不是从零写的，它与原始 Qwen3 路径的关系是：

- 复用的部分：
  - `ExpertCache`
  - `ExpertBufferManager`
  - `GPUExpertCacheManager`
  - prefetch / swap buffer / cache policy
  - gather / compute / scatter 组织方式
- 新写或改写的部分：
  - `baseline/sdar_layers.py`
  - `baseline/sdar_builder.py`
  - OpenCompass 本地入口
  - SDAR 专用评测脚本

最核心的差异在于：

- Qwen3 路径面对的是标准 AR `model.generate()`
- SDAR 路径面对的是 block-diffusion `block_diffusion_generate()`

但因为 SDAR 的 MoE 层接口仍然是“输入 hidden_states，输出 hidden_states”，所以
offloading runtime 仍然可以挂接进去。

## 十二、当前实现中哪些部分没有改

为了保证移植风险最低，下面这些部分没有改：

- `block_diffusion_generate()` 的算法逻辑
- SDAR attention 的主要实现
- `store_kv=True/False` 的语义
- tokenizer 和 chat template
- 数据集配置本身
- evaluator / postprocessor 的定义

这意味着当前性能结果仍然可以被解释为：

- 同一套 SDAR 解码算法
- 同一套数据与 prompt
- 只是 MoE 权重驻留和调度方式改成了 offloading

## 十三、当前测试里“100% 正确率”到底是什么意思

这里的 `100%` 不是全 GSM8K 测试集的正确率，也不是完整 benchmark 的最终结论。

它的准确含义是：

- 在我做的那两轮小样本真机验证里
- `num_samples=1` 那轮，测试了 1 条 GSM8K 样本，答对了 1 条
- `num_samples=2` 那轮，测试了 2 条 GSM8K 样本，答对了 2 条
- 所以 evaluator 返回的 accuracy 都是 `100.0`

而且这个 `accuracy` 不是直接比较原始文本，而是按 OpenCompass GSM8K 配置里的
评测链得到的：

1. `pred_postprocessor = math_postprocess_v2`
2. `dataset_postprocessor = gsm8k_dataset_postprocess`
3. `evaluator = MATHEvaluator(version='v2')`

也就是说，它表示“在所测试的小样本上，经过 OpenCompass 既定后处理后，最终答案匹配参考答案”。

因此，这个 `100%` 的意义只是：

- 当前移植在少量样本上功能正确
- 不代表完整数据集精度评估已经做完

## 十四、当前实现中值得你重点检查的点

如果你要检查这次移植是否正确，我建议重点看下面几类点。

### 1. 解码主链是否保持原样

重点确认：

- 仍然是 `BD3withChatTemplate.generate() -> block_diffusion_generate()`
- 没有另写一套 block decoder
- `store_kv` 语义没有被改坏

### 2. 模型构建是否真的避免了常驻 experts

重点确认：

- `config.num_experts = 0`
- `config._target_experts = 128`
- `SDARMoeSparseMoeBlock` 构造时只保留 gate，不创建真实 experts

### 3. dense 权重和 expert 权重是否分流正确

重点确认：

- `experts.` 权重进 CPU cache
- 非 `experts.` 权重进 GPU 模型参数
- gate 权重仍然进模型参数

### 4. wrapper 返回接口是否和 SDAR layer 兼容

重点确认：

- `self.mlp(hidden_states)` 返回 `(hidden_states, router_logits)`
- `SDARMoeDecoderLayer.forward()` 能原样接住这个返回值

### 5. 当前评测入口是否沿用了真实 SDAR 数据集配置

重点确认：

- 当前不是读 `MoE-Offloading/benchmark`
- 而是 import 真实 OpenCompass dataset config
- prompt 和 evaluator 都来自这套 config

## 十五、当前实现的边界与已知约束

### 1. 当前 builder 固定使用 bfloat16

即使外部传入 `torch_dtype=torch.float16`，builder 也会提示并强制落到
`torch.bfloat16`。这是因为当前 offloading cache/pinned storage 的实现按
这个 dtype 组织，先保证功能正确。

### 2. 当前“正确率 100%”只是小样本 smoke test

它的意义是功能验证，不是完整 benchmark 结论。

### 3. 当前评测脚本是轻量入口，不是完整 OpenCompass runner

这有意减少了框架调度噪声，更适合 latency/memory profiling。

## 十六、结论

当前这次移植的本质可以总结成一句话：

保留了 SDAR 的原始 block-diffusion 解码和 KV 语义，只把其每层 MoE 从
“模型内常驻 experts”替换成“MoE-Offloading runtime 按需取 expert 权重并执行”。

从代码结构上看，移植的关键正确性标准是：

- 解码链仍然是 SDAR 原链
- 模型接口仍然是 SDAR 原接口
- 运行时 expert 调度已经变成 MoE-Offloading 的缓存/缓冲区体系
- 测试入口已经切到 SDAR 实际使用的数据集配置

如果这四点成立，这次移植的方向就是正确的。当前实现满足这四点。
