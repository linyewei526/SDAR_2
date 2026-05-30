# SDAR MoE-Offloading 测评入口与实验目录说明

本文档说明当前 SDAR MoE-Offloading 的两套评估入口、输出目录结构、关键控制参数，以及常用单行命令。

## 一、入口关系

当前有三类相关文件：

- `tests/sdar_offloading_runner.py`
  - 公共评估引擎，不建议直接运行。
  - 负责选 GPU、显存占位、加载模型、加载 OpenCompass 数据集、生成、评测、写结果、写实验配置。

- `tests/test_sdar_offloading.py`
  - 单 benchmark 入口。
  - 适合只测 `gsm8k`、`math`、`humaneval` 或 `sanitized_mbpp` 中某一个。

- `tests/run_sdar_offloading_benchmarks.py`
  - 多 benchmark 总控入口。
  - 默认按顺序运行：
    - `humaneval`
    - `sanitized_mbpp`
    - `gsm8k`
    - `math`
  - 默认在同一个进程里先占住额外 GPU 显存，然后连续跑完 4 个 benchmark，减少其他任务插入干扰。

## 二、输出目录结构

默认不再把结果文件直接散落在 `profiles/` 下。每次运行都会创建一个带时间戳的实验目录。

单 benchmark 默认目录示例：

```text
profiles/20260428_010203_sdar_offloading_single/
```

多 benchmark 默认目录示例：

```text
profiles/20260428_010203_sdar_offloading_suite/
```

单 benchmark 目录通常包含：

```text
experiment_config.json
gsm8k_results.json
gsm8k_summary.json
```

多 benchmark 目录通常包含：

```text
experiment_config.json
all_benchmarks_summary.json
humaneval_results.json
humaneval_summary.json
sanitized_mbpp_results.json
sanitized_mbpp_summary.json
gsm8k_results.json
gsm8k_summary.json
math_results.json
math_summary.json
```

说明：

- `experiment_config.json` 记录本次实验配置，包括模型路径、benchmark、样本范围、生成参数、GPU 筛选参数、GPU cache 参数、显存占位参数、record 模式、输出路径等。
- `*_results.json` 记录完整评测结果，包括 `evaluation`、`aggregate`、逐样本输出、GPU cache stats 等。
- `*_summary.json` 只在 `--record-mode experts|latency|both` 且 `--record-scope` 不是 `none` 时生成，用于专家和时间摘要。
- `all_benchmarks_summary.json` 只在多 benchmark 总控入口生成，用于汇总 4 个 benchmark 的评测结果和效率指标。

如果你想指定实验目录，可以加：

```bash
--output-dir /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/profiles/my_run
```

如果只想给自动时间戳目录加后缀，可以加：

```bash
--run-name cache16_topklru
```

单 benchmark 入口仍保留 `--results-output`、`--record-output` 作为兼容参数；如果显式传入它们，会覆盖默认时间戳目录下的结果路径。新实验建议不要手动传这两个参数，直接使用自动实验目录。

## 三、关键控制参数

### 1. benchmark 和样本数

单 benchmark 入口使用：

```bash
--benchmark gsm8k
--benchmark math
--benchmark humaneval
--benchmark sanitized_mbpp
```

当前 test split 全量样本数：

- `gsm8k`: `1319`
- `math`: `500`
- `humaneval`: `164`
- `sanitized_mbpp`: `257`

多 benchmark 入口默认 `--num-samples 0`，含义是每个 benchmark 自动跑当前 split 剩余全部样本。

### 2. 生成长度

正式评估建议统一使用：

```bash
--gen-length 4096 --max-out-len 4096
```

### 3. GPU 选择和显存占位

正式效率测试建议使用更严格的 GPU 筛选：

```bash
--min-free-memory-gib 60 --max-gpu-utilization 5
```

单 benchmark 入口默认不启用显存占位，需要显式加：

```bash
--reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24
```

多 benchmark 总控入口默认启用显存占位；如果要关闭，显式加：

```bash
--disable-reserve-gpu-memory
```

显存占位的逻辑是：选中 GPU 后占住大部分空闲显存，只留下约 `24GiB` 给当前评估使用，避免其他任务进入同一张 GPU。占位 tensor 不参与计算，进程退出后自动释放。

### 4. GPU cache/offloading 参数

当前可通过命令行控制：

```bash
--cache-policy topk_lru
--cache-slots-per-layer 16
--topk-lru-logit-percentile 90.0
```

常用含义：

- `--cache-policy`: GPU expert cache 替换策略，可选 `static/lru/lfu/topk_lru/tinylfu`。
- `--cache-slots-per-layer`: 每层保留多少个 expert GPU cache slot。
- `--topk-lru-logit-percentile`: `topk_lru` 策略使用的 logit percentile 阈值。

这些参数会写入 `experiment_config.json` 和每个 `*_results.json` 的 `offloading` 字段。

### 5. record 模式

只评估 accuracy/TPS，不记录专家或时间摘要：

```bash
--record-mode none --record-scope none
```

记录专家命中/搬运摘要：

```bash
--record-mode experts --record-scope all
```

记录 layer 内操作时间摘要：

```bash
--record-mode latency --record-scope all
```

同时记录专家和时间摘要：

```bash
--record-mode both --record-scope all
```

如果只想记录当前评测范围的前 `K` 条：

```bash
--record-mode both --record-scope first_k --record-first-k 10
```

## 四、单 benchmark 命令

下面以 `gsm8k` 为例。其他 benchmark 只需要替换 `--benchmark` 和 `--num-samples`。

### 1. 单 benchmark，只记录 TPS/accuracy

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark gsm8k --split test --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode none --record-scope none --run-name gsm8k_speed
```

### 2. 单 benchmark，记录专家指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark gsm8k --split test --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode experts --record-scope all --run-name gsm8k_experts
```

### 3. 单 benchmark，记录时间指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/test_sdar_offloading.py --benchmark gsm8k --split test --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode latency --record-scope all --run-name gsm8k_latency
```

### 4. 单 benchmark，同时记录专家和时间指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/
python tests/test_sdar_offloading.py --benchmark gsm8k --split test --start-idx 0 --num-samples 1319 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-gpu-memory --reserve-gpu-memory-stage pre_build --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode both --record-scope all --run-name gsm8k_both
```

其他 benchmark 的全量 `--num-samples`：

- `math`: `--benchmark math --num-samples 500`
- `humaneval`: `--benchmark humaneval --num-samples 164`
- `sanitized_mbpp`: `--benchmark sanitized_mbpp --num-samples 257`

## 五、多 benchmark 总控命令

多 benchmark 入口默认顺序是：

```text
humaneval -> sanitized_mbpp -> gsm8k -> math
```

默认 `--num-samples 0`，也就是 4 个 benchmark 都跑 test split 全量。

### 1. 多 benchmark，只记录 TPS/accuracy

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/run_sdar_offloading_benchmarks.py --split test --num-samples 0 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode none --record-scope none --run-name suite_speed
```

说明：多 benchmark 入口默认启用显存占位，因此这里不需要写 `--reserve-gpu-memory`。

### 2. 多 benchmark，记录专家指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/run_sdar_offloading_benchmarks.py --split test --num-samples 0 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode experts --record-scope all --run-name suite_experts
```

### 3. 多 benchmark，记录时间指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/python tests/run_sdar_offloading_benchmarks.py --split test --num-samples 0 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode latency --record-scope all --run-name suite_latency
```

### 4. 多 benchmark，同时记录专家和时间指标

```bash
cd /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading && /data_3/wly/miniconda3/envs/sdar/bin/
python tests/run_sdar_offloading_benchmarks.py --split test --num-samples 0 --gen-length 4096 --max-out-len 4096 --candidate-gpus 0,1,2,3 --min-free-memory-gib 60 --max-gpu-utilization 5 --reserve-free-memory-gib 24 --cache-policy topk_lru --cache-slots-per-layer 16 --topk-lru-logit-percentile 90.0 --record-mode both --record-scope all --run-name suite_both
```

## 六、查看结果

运行结束后，终端会打印实验目录，例如：

```text
Experiment directory: /data_3/wly/dLLM-MoE/SDAR_2/evaluation/MoE-Offloading/profiles/20260428_010203_sdar_offloading_suite_suite_speed
```

你主要查看：

- `experiment_config.json`
  - 看本次实验配置是否符合预期。
- `all_benchmarks_summary.json`
  - 多 benchmark 汇总结果。
- `<benchmark>_results.json`
  - 单个 benchmark 的完整 accuracy、TPS、样本输出和 cache stats。
- `<benchmark>_summary.json`
  - record mode 启用时的专家/时间摘要。
