# SDAR MoE-Offloading Execution Guide

## Purpose

This path evaluates `SDAR-30B-A3B-Chat-b32` with the real SDAR block-diffusion
decoding loop while replacing in-memory MoE experts with the MoE-Offloading
runtime from this project.

The integration keeps these SDAR-specific behaviors unchanged:

- block-level autoregression
- intra-block parallel iterative denoising
- confidence-based token acceptance and remasking
- KV cache commit only after a block is finalized

## Key Files

- `baseline/sdar_builder.py`
  Builds the SDAR model skeleton, loads dense weights to GPU, loads expert
  weights into the CPU offloading cache, and replaces SDAR sparse MoE blocks
  with `SDARSparseMoeWrapper`.
- `baseline/sdar_layers.py`
  SDAR-specific MoE runtime wrapper for routing, cache lookup, swap-buffer
  loads, prefetch, gather/compute/scatter, and NVTX ranges.
- `../opencompass/configs/sdar_local_models/modeling_sdar_moe_offloading.py`
  OpenCompass local modeling entry. `BD3withChatTemplate` loads this module and
  the classmethod `from_pretrained()` delegates model construction to
  `baseline/sdar_builder.py`.
- `tests/test_sdar_offloading.py`
  Standalone latency/memory evaluation entry for SDAR offloading.
- `tests/sdar_offloading_utils.py`
  Helpers for GPU polling, OpenCompass dataset loading, prompt construction,
  memory snapshot export, and result export.

## Runtime Flow

1. `tests/test_sdar_offloading.py` selects a usable GPU from the candidate set.
2. It loads the real OpenCompass dataset config, currently defaulting to:
   `opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799`.
3. It instantiates `BD3withChatTemplate` with:
   - `local_modeling_module=configs.sdar_local_models.modeling_sdar_moe_offloading`
   - SDAR generation kwargs (`mask_id`, `gen_length`, `block_length`,
     `denoising_steps`, `threshold`, ...)
   - offloading kwargs (`enable_gpu_cache`, `cache_policy`,
     `cache_slots_per_layer`, ...)
4. OpenCompass calls the local `SDARMoeForCausalLM.from_pretrained()`.
5. The offloading builder constructs the SDAR model and swaps its MoE layers.
6. Generation still goes through:
   `BD3withChatTemplate.generate() -> block_diffusion_generate() -> SDAR forward()`.

## Example

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
/data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py \
  --candidate-gpus 0,1,2,3 \
  --min-free-memory-gib 40 \
  --num-samples 2 \
  --start-idx 0 \
  --gen-length 128 \
  --max-out-len 128 \
  --track-gpu-memory
```

## Outputs

By default the script writes:

- `profiles/sdar_offloading_start{start}_n{num}_results.json`
- `profiles/sdar_offloading_start{start}_n{num}_gpu_memory.csv`

The JSON includes:

- build time
- per-sample latency
- generated token count
- tokens/s
- processed prediction and reference
- evaluation result
- buffer manager stats
- GPU cache stats

## Verified Baseline

Validated with the SDAR environment:

- Python: `/data/home/wly/.conda/envs/sdar/bin/python`
- Model: `SDAR-30B-A3B-Chat-b32`
- Dataset: OpenCompass GSM8K config above

Observed successful runs:

- `start_idx=0, num_samples=1, gen_length=128`
- `start_idx=0, num_samples=2, gen_length=128`

Both runs produced correct GSM8K answers on the tested samples and exported
latency/memory/cache statistics successfully.

## 中文版

### 目的

这条执行路径用于在保持 SDAR 原始 block-diffusion 解码逻辑不变的前提下，
把 `SDAR-30B-A3B-Chat-b32` 的 MoE 层替换为本项目里的
MoE-Offloading runtime，并评测推理延迟与显存占用。

当前集成刻意保持以下 SDAR 行为不变：

- 块间自回归
- 块内并行迭代去噪
- 基于置信度的 token 接收与 remask
- 只有在一个 block 最终确定后才提交 KV cache

### 关键文件

- `baseline/sdar_builder.py`
  负责构建 SDAR 模型骨架，把 dense 权重加载到 GPU，把 expert 权重加载到
  CPU offloading cache，并把原始 `SDARMoeSparseMoeBlock` 替换成
  `SDARSparseMoeWrapper`。
- `baseline/sdar_layers.py`
  负责 SDAR 场景下的 routing、cache lookup、swap buffer load、prefetch、
  gather/compute/scatter，以及 NVTX 标记。
- `../opencompass/configs/sdar_local_models/modeling_sdar_moe_offloading.py`
  这是 OpenCompass 的本地模型入口。`BD3withChatTemplate` 加载它后，会调用
  其中 `SDARMoeForCausalLM.from_pretrained()`，再转交给
  `baseline/sdar_builder.py` 构建 offloading 版模型。
- `tests/test_sdar_offloading.py`
  独立的 SDAR offloading 延迟/显存评测入口。
- `tests/sdar_offloading_utils.py`
  提供 GPU 轮询、OpenCompass 数据集配置加载、prompt 构造、显存快照导出、
  结果导出等辅助逻辑。

### 运行流程

1. `tests/test_sdar_offloading.py` 从候选 GPU 集合里选择一张可用卡。
2. 它加载真实的 OpenCompass 数据集配置，当前默认是：
   `opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799`。
3. 它实例化 `BD3withChatTemplate`，同时传入：
   - `local_modeling_module=configs.sdar_local_models.modeling_sdar_moe_offloading`
   - SDAR 解码参数，如 `mask_id`、`gen_length`、`block_length`、
     `denoising_steps`、`threshold`
   - offloading 参数，如 `enable_gpu_cache`、`cache_policy`、
     `cache_slots_per_layer`
4. OpenCompass 会调用本地 `SDARMoeForCausalLM.from_pretrained()`。
5. offloading builder 构造 SDAR 模型并替换其 MoE 层。
6. 最终生成路径仍然是：
   `BD3withChatTemplate.generate() -> block_diffusion_generate() -> SDAR forward()`。

### 示例

```bash
cd /data/home/wly/dLLM/SDAR_2/evaluation/MoE-Offloading
/data/home/wly/.conda/envs/sdar/bin/python tests/test_sdar_offloading.py \
  --candidate-gpus 0,1,2,3 \
  --min-free-memory-gib 40 \
  --num-samples 2 \
  --start-idx 0 \
  --gen-length 128 \
  --max-out-len 128 \
  --track-gpu-memory
```

### 输出

默认会写出：

- `profiles/sdar_offloading_start{start}_n{num}_results.json`
- `profiles/sdar_offloading_start{start}_n{num}_gpu_memory.csv`

JSON 里包括：

- 模型构建时间
- 每条样本的生成延迟
- 生成 token 数
- tokens/s
- 处理后的预测与参考答案
- evaluator 结果
- buffer manager 统计
- GPU cache 统计

### 已验证环境

验证环境如下：

- Python: `/data/home/wly/.conda/envs/sdar/bin/python`
- 模型: `SDAR-30B-A3B-Chat-b32`
- 数据集: 上述 OpenCompass GSM8K 配置

已验证通过的运行：

- `start_idx=0, num_samples=1, gen_length=128`
- `start_idx=0, num_samples=2, gen_length=128`

两次运行都在所测试的 GSM8K 样本上得到正确答案，并成功导出了延迟、
显存和 cache 统计结果。
