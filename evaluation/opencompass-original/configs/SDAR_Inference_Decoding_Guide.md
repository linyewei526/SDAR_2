# SDAR 推理解码逻辑详解

## 目录
1. [论文核心思想概述](#1-论文核心思想概述)
2. [代码调用链分析](#2-代码调用链分析)
3. [核心解码函数详解](#3-核心解码函数详解)
4. [参数传递详解](#4-参数传递详解)
5. [完整生成流程示例](#5-完整生成流程示例)
6. [关键算法实现](#6-关键算法实现)

---

## 1. 论文核心思想概述

### 1.1 SDAR: Synergistic Diffusion-AutoRegression

SDAR是一种**协同扩散-自回归范式**，将自回归模型的高效训练与扩散模型的并行推理能力相结合。

#### 核心创新点：
- **块间自回归（Inter-block Autoregression）**：在块级别保持自回归结构，确保全局连贯性
- **块内并行扩散（Intra-block Parallel Diffusion）**：每个块内部通过离散扩散过程并行生成所有token
- **轻量级范式转换**：通过少量数据（~50B tokens）将预训练的AR模型转换为块级扩散模型

### 1.2 解码策略

#### 静态低置信度重掩码（Static Low Confidence Remasking）
- 每步解码固定数量的token
- 对于块大小为B、去噪步数为T，每步选择⌈B/T⌉个最高置信度的掩码位置解码
- 保证在固定次数的前向传播内完成生成

#### 动态低置信度重掩码（Dynamic Low Confidence Remasking）
- 自适应策略：当预测置信度超过阈值τ时，直接接受该token
- 加速解码：模型可以用更少的步数填充"简单"部分
- 保证进度：如果没有足够多的token超过阈值，至少选择置信度最高的⌈B/T⌉个位置

---

## 2. 代码调用链分析

### 2.1 完整调用链路图

```
eval_sdar_hf.py (配置入口)
    │
    ├── 配置模型参数
    │   ├── path = Hugging Face权重目录
    │   └── local_modeling_module =
    │       configs.sdar_local_models.modeling_sdar_moe
    │   └── generation_kwargs = {
    │           mask_id=151669,
    │           gen_length=4096,
    │           block_length=32,
    │           denoising_steps=32,
    │           temperature=1.0,
    │           top_k=1,
    │           top_p=1.0,
    │           remasking='low_confidence',
    │           threshold=0.95
    │       }
    │
    ▼
GenInferencer.inference() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py:82-204]
    │
    ├── 构建prompt列表
    ├── 遍历dataloader
    │   └── results = self.model.generate_from_template(entry, max_out_len=self.max_out_len, ...)
    │
    ▼
BD3withChatTemplate.__init__() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/opencompass/models/huggingface_bd3.py]
    │
    ├── _get_possible_max_seq_len()   # 从权重目录的config.json读取序列长度
    ├── _load_tokenizer(path)         # tokenizer仍从权重目录加载
    └── _load_model(path, local_modeling_module=...)
        │
        ├── importlib.import_module(local_modeling_module)
        ├── 读取本地类 SDARMoeForCausalLM
        └── SDARMoeForCausalLM.from_pretrained(path)  # 权重仍从path读取
    │
    ▼
BD3withChatTemplate.generate_from_template() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/opencompass/models/huggingface_bd3.py]
    │
    ├── inputs = self.parse_template(templates, mode='gen')
    └── return self.generate(inputs, max_out_len=max_out_len, **kwargs)
    │
    ▼
BD3withChatTemplate.generate() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/opencompass/models/huggingface_bd3.py]
    │
    ├── _convert_chat_messages(inputs)  # 转换聊天格式
    ├── tokenizer.batch_encode_plus()   # Tokenize
    │
    └── outputs = block_diffusion_generate(
            self.model,
            self.tokenizer,
            tokens,
            mask_id=...,
            denoising_steps=...,
            gen_length=...,
            block_length=...,
            temperature=...,
            top_k=...,
            top_p=...,
            remasking=...,
            threshold=...,
            stopping_criteria_idx=...
        )
    │
    ▼
block_diffusion_generate() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/opencompass/models/huggingface_bd3.py]
    │
    ├── Prefill阶段: 计算prompt的KV cache
    │   └── model(input_ids, attention_mask, position_ids, past_key_values, use_cache=True, store_kv=True)
    │
    └── 生成阶段: 逐块生成
        └── for num_block in range(prefill_blocks, num_blocks):
            └── for i in range(denoising_steps + 1):
                ├── logits = model(cur_x, attention_mask, position_ids, past_key_values, use_cache=True, store_kv=False).logits
                ├── sample_with_temperature_topk_topp()  # 采样
                └── 基于置信度的remasking策略
    │
    ▼
SDARMoeForCausalLM.forward() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe.py]
    │
    └── outputs = self.model(input_ids, attention_mask, position_ids, past_key_values, ...)
        └── logits = self.lm_head(hidden_states)
    │
    ▼
SDARMoeModel.forward() [~/dLLM-MoE/SDAR_2/evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe.py]
    │
    ├── inputs_embeds = self.embed_tokens(input_ids)
    ├── position_embeddings = self.rotary_emb(hidden_states, position_ids)
    │
    └── for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask, position_ids, ...)
            hidden_states = layer_outputs[0]
    │
    └── hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, ...)
```

### 2.2 文件路径汇总

| 文件 | 路径 | 功能 |
|------|------|------|
| 配置入口 | `eval_sdar_hf.py` | 定义模型参数和数据集 |
| 推理器 | `opencompass/openicl/icl_inferencer/icl_gen_inferencer.py` | GenInferencer类 |
| 模型包装 | `opencompass/models/huggingface_bd3.py` | BD3withChatTemplate类 + block_diffusion_generate |
| 本地模型实现 | `configs/sdar_local_models/modeling_sdar_moe.py` | 默认本地SDAR模型控制文件 |
| 本地模型变体 | `configs/sdar_local_models/modeling_sdar_moe_modified.py` | 可切换的本地修改版本 |
| 本地配置类 | `configs/sdar_local_models/configuration_sdar_moe.py` | 本地`SDARMoeConfig`定义 |
| 权重目录 | `models--JetLM--SDAR-30B-A3B-Chat-b32/.../` | 仅负责提供config / tokenizer / safetensors权重 |

### 2.3 本地modeling切换逻辑

现在 HF 推理的“代码控制权”和“权重来源”被拆开：

- `path` 仍然指向 Hugging Face 模型目录，用来读取 `config.json`、tokenizer 和 `.safetensors` 权重。
- `local_modeling_module` 指向项目内的本地 Python 模块，用来决定实际执行哪个 `SDARMoeForCausalLM / SDARMoeModel` 实现。

默认目录结构：

```text
evaluation/opencompass/configs/sdar_local_models/
├── __init__.py
├── configuration_sdar_moe.py
├── modeling_sdar_moe.py
└── modeling_sdar_moe_modified.py
```

在 `eval_sdar_hf.py` 及其它 `eval_sdar_hf*.py` 中，切换如下变量即可更换控制文件：

```python
LOCAL_SDAR_MODELING_MODULE = (
    'configs.sdar_local_models.modeling_sdar_moe'
)
```

如果想切换到 `modeling_sdar_moe_modified.py`，改成：

```python
LOCAL_SDAR_MODELING_MODULE = (
    'configs.sdar_local_models.modeling_sdar_moe_modified'
)
```

注意：

- 这里改的是 Python 模块路径，不是磁盘文件系统路径字符串。
- 如果你新建了例如 `modeling_sdar_moe_v2.py`，对应模块路径就是 `configs.sdar_local_models.modeling_sdar_moe_v2`。
- `configuration_sdar_moe.py` 需要与这些 `modeling_*.py` 文件放在同一目录，因为当前 `modeling` 文件内部通过相对导入 `from .configuration_sdar_moe import SDARMoeConfig` 依赖它。

---

## 3. 核心解码函数详解

### 3.1 block_diffusion_generate() 函数

位置：`huggingface_bd3.py`

```python
@torch.no_grad()
def block_diffusion_generate(
    model,
    tokenizer,
    prompt,
    mask_id,           # 掩码token ID = 151669
    gen_length=128,    # 生成长度 = 4096
    block_length=8,    # 块大小 = 32
    denoising_steps=8, # 去噪步数 = 32
    temperature=0.,    # 温度参数 = 1.0
    top_k=0,          # Top-k = 1
    top_p=1.0,        # Top-p = 1.0
    remasking='low_confidence',  # 重掩码策略
    threshold=1.0,    # 置信度阈值 = 0.95
    stopping_criteria_idx=None   # 停止词token ID列表
):
```

#### 函数参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_id` | 151669 | 特殊掩码token `<\|MASK\|>` 的ID |
| `gen_length` | 4096 | 最大生成长度 |
| `block_length` | 32 | 每个块包含的token数量 |
| `denoising_steps` | 32 | 每个块的去噪迭代次数 |
| `temperature` | 1.0 | 采样温度，1.0表示标准softmax |
| `top_k` | 1 | Top-k采样，1表示贪婪 |
| `top_p` | 1.0 | Nucleus采样阈值 |
| `remasking` | 'low_confidence' | 重掩码策略：'low_confidence'或'random' |
| `threshold` | 0.95 | 动态解码的置信度阈值 |

### 3.2 解码流程详解

#### Step 1: 初始化

```python
# 获取输入信息
input_ids = prompt['input_ids']  # shape: (1, prompt_length)
prompt_length = input_ids.shape[1]
attention_mask = prompt['attention_mask']
past_key_values = DynamicCache()  # KV缓存

# 计算块数量
num_blocks = (input_ids.shape[1] + gen_length + block_length - 1) // block_length
total_length = num_blocks * block_length

# 创建块级下三角注意力掩码
block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device), diagonal=0)
block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0) \
                                    .repeat_interleave(block_length, dim=1) \
                                    .unsqueeze(0)

# 初始化生成序列，全部填充为MASK
x = torch.full((input_ids.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
x[:, :input_ids.shape[1]] = input_ids.clone()  # 保留原始prompt
```

#### Step 2: Prefill阶段（预填充）

```python
prefill_blocks = input_ids.shape[1] // block_length
prefill_length = prefill_blocks * block_length

if prefill_length > 0:
    cur_x = x[:, :prefill_length]
    cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
    cur_position_ids = position_ids[:, :prefill_length]

    # 前向传播，计算并存储KV cache
    _ = model(
        input_ids=cur_x,
        attention_mask=cur_attn_mask,
        position_ids=cur_position_ids,
        past_key_values=past_key_values,  # 全局KV cache
        use_cache=True,
        store_kv=True  # 存储KV到cache
    )
```

#### Step 3: 块级自回归生成

```python
# 计算每步需要转换的token数量
num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
# 例如：block_length=32, denoising_steps=32 → 每步转换1个token

for num_block in range(prefill_blocks, num_blocks):
    # 当前块的位置
    cur_x = x[:, num_block * block_length: (num_block + 1) * block_length]
    cur_attn_mask = block_diffusion_attention_mask[:, num_block * block_length: (num_block + 1) * block_length, :(num_block + 1) * block_length]
    cur_position_ids = position_ids[:, num_block * block_length: (num_block + 1) * block_length]

    # 块内去噪循环
    for i in range(denoising_steps + 1):
        mask_index = (cur_x == mask_id)  # 找到仍是MASK的位置

        if mask_index.sum() == 0:
            # 所有位置都已解码，存储最后一个token的KV
            _ = model(..., store_kv=True)
            break
        else:
            # 获取logits（不存储KV）
            logits = model(..., store_kv=False).logits

            # 添加Gumbel噪声（用于采样）
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            # 采样得到预测token和置信度
            x0, x0_p = sample_with_temperature_topk_topp(logits, temperature, top_k, top_p)

            # 选择需要在本步解码的位置
            if threshold < 1.0:
                # 动态解码
                confidence = torch.where(mask_index, x0_p, -np.inf)
                # 选择高置信度的位置
                high_conf_mask = confidence[j] > threshold
                if num_high_confidence >= num_transfer_tokens[i]:
                    transfer_index[j] = high_conf_mask
                else:
                    # 不够时选择top-k
                    _, idx = torch.topk(confidence[j], num_transfer_tokens[i])
                    transfer_index[j, idx] = True
            else:
                # 静态解码：固定选择top-k
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[i])
                transfer_index[j, select_index] = True

            # 更新当前块
            cur_x[transfer_index] = x0[transfer_index]

    # 存储整个块到x
    x[:, num_block * block_length: (num_block + 1) * block_length] = cur_x

    # 检查停止条件
    for stop_idx in stopping_criteria_idx:
        if stop_idx in x[:, prompt_length:]:
            end_generate = True
            break
```

---

## 4. 参数传递详解

### 4.1 配置文件到函数的参数映射

```python
# eval_sdar_hf.py 中的配置
generation_kwargs = dict(
    mask_id=151669,           # 特殊token ID
    gen_length=4096,          # 最大生成长度
    block_length=32,          # 块大小
    denoising_steps=32,       # 去噪步数 = block_length
    temperature=1.0,          # 采样温度
    top_k=1,                  # Top-k采样（1=贪婪）
    top_p=1.0,               # Top-p采样
    cfg_scale=0.0,           # CFG缩放（未使用）
    remasking='low_confidence',  # 重掩码策略
    threshold=0.95           # 动态解码阈值
)
```

### 4.2 关键参数选择指南

| 参数 | 推荐值 | 影响 |
|------|--------|------|
| `block_length` | 4-32 | 越大并行度越高，但可能影响质量 |
| `denoising_steps` | = block_length | 每块的去噪迭代次数 |
| `threshold` | 0.90-0.95 | 动态解码阈值，越高越保守 |
| `temperature` | 1.0 | 控制采样多样性 |
| `top_k` | 1 | 贪婪解码时设为1 |

### 4.3 块大小与模型规模的关系

根据论文的scaling分析：

- **小模型（1.7B, 4B）**：对block_size敏感，建议B≤4
- **中等模型（8B）**：B∈{8,16}效果较好
- **大模型（30B）**：对block_size鲁棒性强，B可达32或64

---

## 5. 完整生成流程示例

### 5.1 示例设置

假设输入prompt：
```
"What is 2 + 3? Let's think step by step."
```

参数配置：
- `block_length = 4`
- `denoising_steps = 4`
- `threshold = 0.95`
- `gen_length = 32`

### 5.2 Token化后的处理

```
Prompt tokens: [151644, 8948, 198, 2610, 525, 220, 17, 220, 18, 9554, 499, 7288, 634, 499, 7288, 382, 151645, 198, 151644, 77091, 198]  # 21 tokens

# 应用chat template后可能扩展到约25 tokens
# 假设prefill_length = 24 (6 blocks * 4)
```

### 5.3 生成过程详解

#### Phase 1: Prefill（预填充）

```
Blocks 0-5: 完整的prompt（已知token）
┌─────────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5│
│ [已知]  │ [已知]  │ [已知]  │ [已知]  │ [已知]  │ [已知] │
└─────────────────────────────────────────────────────────┘
                         ↓
              计算KV Cache（存储所有block的KV）
```

#### Phase 2: 块级生成

**Block 6（第一个生成块）的生成过程：**

```
初始状态: [MASK, MASK, MASK, MASK]

Step 1: 全部是MASK
┌───────────────────────────────────────┐
│ Logits = model(MASK, MASK, MASK, MASK)│
│ 每个位置得到预测token和置信度          │
└───────────────────────────────────────┘
预测: To(0.98), solve(0.85), this(0.72), we(0.65)
置信度超过0.95的位置: pos0 (To)
选择: pos0 → 更新为"To"

状态: [To, MASK, MASK, MASK]

Step 2:
┌───────────────────────────────────────┐
│ Logits = model(To, MASK, MASK, MASK)  │
│ 位置1-3仍为MASK，重新预测              │
└───────────────────────────────────────┘
预测: -, solve(0.92), problem(0.88), need(0.82)
置信度超过0.95的: 无（最高0.92 < 0.95）
选择top-1: pos1 → 更新为"solve"

状态: [To, solve, MASK, MASK]

Step 3:
预测: this(0.96), problem(0.89), we(0.78)
置信度超过0.95的: pos2 (this)
选择: pos2 → 更新为"this"

状态: [To, solve, this, MASK]

Step 4:
预测: problem(0.97)
置信度超过0.95的: pos3 (problem)
选择: pos3 → 更新为"problem"

最终Block 6: [To, solve, this, problem]
```

**Block 7的生成过程：**

```
初始状态: [MASK, MASK, MASK, MASK]
已有上下文: prompt + Block 6

Step 1:
预测: ,, 2(0.99), +(0.88), 3(0.75)
选择: pos0 → "," (但这里可能选择pos1的"2")

... 类似迭代 ...

最终Block 7: [,, 2, +, 3]
```

#### Phase 3: 继续生成直到遇到停止词

```
Block 8: equals, 5, ., <|im_end|>
检测到停止词 → 结束生成
```

### 5.4 注意力掩码结构

```
块级下三角掩码 (Block-wise Causal Mask):

     B0  B1  B2  B3  B4  B5  B6  B7
B0 [ 1   0   0   0   0   0   0   0 ]
B1 [ 1   1   0   0   0   0   0   0 ]
B2 [ 1   1   1   0   0   0   0   0 ]
B3 [ 1   1   1   1   0   0   0   0 ]
B4 [ 1   1   1   1   1   0   0   0 ]
B5 [ 1   1   1   1   1   1   0   0 ]
B6 [ 1   1   1   1   1   1   1   0 ]  ← 当前生成块可以看到所有之前的块
B7 [ 1   1   1   1   1   1   1   1 ]

1 = 可以attend
0 = 不能attend

特点：
- 块内：双向注意力（可以看同一块内的其他位置）
- 块间：因果注意力（只能看之前的块，不能看之后的块）
```

### 5.5 KV Cache管理

```
Prefill阶段:
┌──────────────────────────────────────────────────────┐
│ Block 0-5: 存储KV到 past_key_values                  │
│ 每个block的KV都会被缓存，后续生成时复用              │
└──────────────────────────────────────────────────────┘

生成阶段（Block 6）:
┌──────────────────────────────────────────────────────┐
│ 去噪步骤1-3: 不存储KV (store_kv=False)               │
│ 最后一步完成时: 存储整个Block 6的KV (store_kv=True)  │
└──────────────────────────────────────────────────────┘

生成阶段（Block 7）:
┌──────────────────────────────────────────────────────┐
│ 复用Block 0-6的KV Cache                              │
│ 只需要计算Block 7的新KV                               │
└──────────────────────────────────────────────────────┘
```

---

## 6. 关键算法实现

### 6.1 get_num_transfer_tokens()

计算每步需要解码的token数量：

```python
def get_num_transfer_tokens(block_length, steps):
    '''
    计算每个去噪步骤需要转换的token数量

    例如: block_length=32, steps=32
    → base=1, remainder=0
    → 每步转换1个token
    '''
    base = block_length // steps
    remainder = block_length % steps

    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1

    return num_transfer_tokens
```

### 6.2 add_gumbel_noise()

添加Gumbel噪声用于采样：

```python
def add_gumbel_noise(logits, temperature):
    '''
    Gumbel-max采样方法

    当temperature > 0时，添加噪声使采样更具多样性
    当temperature = 0时，直接返回原始logits（贪婪）
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

### 6.3 sample_with_temperature_topk_topp()

结合温度、Top-k和Top-p的采样：

```python
def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    '''
    采样并返回token及其置信度

    返回:
        token: 采样的token ID
        token_prob: 该token在原始概率分布中的概率
    '''
    # 应用温度
    logits = logits / temperature if temperature != 1.0 else logits
    ori_probs = F.softmax(logits, dim=-1)  # 原始概率

    # Top-k过滤
    if top_k > 0:
        logits = top_k_logits(logits, top_k)

    # Top-p过滤
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)  # 采样
    token_prob = torch.gather(ori_probs, -1, token)  # 获取原始概率

    return token, token_prob
```

### 6.4 模型forward调用

```python
# SDARMoeForCausalLM.forward()
def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, store_kv, ...):
    # 调用底层模型
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        store_kv=store_kv,  # SDAR特有参数
        ...
    )

    # 计算logits
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)

    return MoeCausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values, ...)
```

---

## 附录A: 关键文件位置

| 文件 | 完整路径 |
|------|----------|
| 配置文件 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/eval_sdar_hf.py` |
| 推理器 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py` |
| 模型包装 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/opencompass/models/huggingface_bd3.py` |
| 本地模型目录 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/sdar_local_models/` |
| 默认本地模型实现 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe.py` |
| 本地模型变体 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/sdar_local_models/modeling_sdar_moe_modified.py` |
| 本地配置类 | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/sdar_local_models/configuration_sdar_moe.py` |
| 权重目录 | `/data/home/wly/.cache/huggingface/hub/models--JetLM--SDAR-30B-A3B-Chat-b32/snapshots/c351bbc37d240aa6871f167e8f92d694281b0c22/` |
| 论文PDF | `/data/home/wly/dLLM/SDAR_2/evaluation/opencompass/configs/SDAR.pdf` |

## 附录B: SDAR vs AR 对比

| 特性 | AR (自回归) | SDAR (块级扩散) |
|------|-------------|-----------------|
| 解码方式 | 逐token | 块内并行，块间自回归 |
| KV Cache | 每token存储 | 每块存储 |
| 注意力 | 严格因果 | 块内双向，块间因果 |
| 生成速度 | O(N)步 | O(N/B)块 × T步 |
| 推理加速比 | 1× | 最高可达B× |
| 适用场景 | 通用生成 | 需要双向上下文的任务（如科学推理） |

## 附录C: 常见问题

**Q1: block_length和denoising_steps的关系？**
A: 通常设置 `denoising_steps = block_length`，这样每步解码1个token。如果 `denoising_steps < block_length`，每步需要解码多个token。

**Q2: threshold参数如何选择？**
A:
- `threshold = 1.0`: 静态解码，每步固定解码 ⌈B/T⌉ 个token
- `threshold = 0.95`: 动态解码，置信度>0.95的token直接接受
- 更大的模型可以使用更低的threshold（更激进的并行）

**Q3: 为什么使用`store_kv`参数？**
A: 在去噪过程中，中间状态的KV不需要存储（因为会变化）。只有在块完成时才存储KV，以节省内存并保证生成的因果性。

**Q4: 如何处理停止词？**
A: 每个块生成完成后检查是否出现停止词token。如果出现，立即终止生成。

---

*文档生成时间: 2026-03-14*
*基于SDAR论文 (arXiv:2510.06303) 和对应推理代码*
