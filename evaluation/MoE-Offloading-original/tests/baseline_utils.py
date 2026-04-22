#!/usr/bin/env python3
"""
Baseline Test Utilities - Common test infrastructure for Qwen3MoE and GPT-OSS baseline tests
以 tests/utils.py 为模板，适配 baseline 纯自回归生成场景
"""

import torch
import os
import sys
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baseline.debug_config import PRINT_GENERATION_RESULT


def get_prompts(benchmark="gsm8k", num_samples=1, start_idx=0):
    """从benchmark目录读取数据 - 与 tests/utils.py 共享逻辑"""
    import pyarrow.parquet as pq

    if benchmark == "gsm8k":
        benchmark_path = "/data/home/tianjianyang/code/adapeagle/benchmark/gsm8k/main/test-00000-of-00001.parquet"
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = start_idx + num_samples
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            prompts.append(row['question'])
        return prompts

    elif benchmark == "humaneval":
        benchmark_path = "/data/home/tianjianyang/code/adapeagle/benchmark/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = start_idx + num_samples
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            prompts.append(row['prompt'])
        return prompts

    elif benchmark == "cnndm":
        benchmark_path = "/data/home/tianjianyang/code/adapeagle/benchmark/CNN-DM.parquet"
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = start_idx + num_samples
        for _, row in df.iloc[start_idx:end_idx].iterrows():
            prompts.append(row['article'])
        return prompts

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


@dataclass
class BaselineModelConfig:
    """
    模型配置类 - 用于区分不同模型的参数
    与 tests/utils.py 的 ModelConfig 对应，但简化为 baseline 场景
    """
    name: str  # "qwen3moe" or "gptoss"
    num_layers: int  # 层数
    num_experts: int  # 专家数
    top_k: int  # 每 token 激活专家数
    builder_func: Callable  # 模型构建函数
    default_model_path: str  # 默认模型路径
    use_sampling: bool = True  # 是否使用 sampling (True) 或 greedy (False)
    default_temperature: float = 0.7
    # MoE 层路径（与 tests/utils.py 保持一致）
    moe_module_path: str = ""
    moe_class_name: str = ""


# 预定义的模型配置 - 与 tests/utils.py 的 QWEN3MOE_CONFIG/GPTOSS_CONFIG 对应
QWEN3MOE_CONFIG = BaselineModelConfig(
    name="qwen3moe",
    num_layers=48,
    num_experts=128,
    top_k=8,
    builder_func=None,  # 将在导入后设置
    default_model_path="/data/home/tianjianyang/models/moe/Qwen3-30B-A3B",
    use_sampling=False,  # Qwen3 使用 greedy decoding
    default_temperature=0.8,
    moe_module_path="baseline.qwen3_layers",
    moe_class_name="Qwen3SimpleMoE",
)

GPTOSS_CONFIG = BaselineModelConfig(
    name="gptoss",
    num_layers=24,
    num_experts=32,
    top_k=4,
    builder_func=None,  # 将在导入后设置
    default_model_path="/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16",
    use_sampling=True,  # GPT-OSS 使用 sampling
    default_temperature=0.7,
    moe_module_path="baseline.gptoss_layers",
    moe_class_name="GptOssSimpleMoE",
)


class BaselineTestBase:
    """
    Baseline 测试基类 - 纯自回归生成
    与 tests/utils.py 的 EagleTestBase 结构对应
    
    使用示例:
        from tests.baseline_utils import BaselineTestBase, QWEN3MOE_CONFIG
        from baseline.qwen3_builder import qwen3_build_model
        QWEN3MOE_CONFIG.builder_func = qwen3_build_model
        
        tester = BaselineTestBase(QWEN3MOE_CONFIG)
        tester.run_test(args)
    """
    
    def __init__(self, model_config: BaselineModelConfig):
        self.config = model_config
        
        # 动态导入 MoE 类（与 tests/utils.py 保持一致）
        if model_config.moe_module_path and model_config.moe_class_name:
            self.moe_module = __import__(model_config.moe_module_path, fromlist=[model_config.moe_class_name])
            self.moe_class = getattr(self.moe_module, model_config.moe_class_name)
        else:
            self.moe_module = None
            self.moe_class = None

    def _reset_for_new_sample(self, target_model):
        """在多样本串行测试前重置运行时状态，避免样本间残留影响。"""
        if self.moe_class is not None:
            for attr, empty in (
                ("router_inputs_for_prefetch", {}),
                ("next_layer_predictions", {}),
                ("moe_activation_log", []),
                ("layer_expert_counts", []),
                ("layer_times", []),
            ):
                if hasattr(self.moe_class, attr):
                    setattr(self.moe_class, attr, empty.copy() if isinstance(empty, dict) else [])
            if hasattr(self.moe_class, "decode_token_counter"):
                self.moe_class.decode_token_counter = 0

        # 显式清理可能存在的模型级 cache 句柄（不同 transformers 版本字段不同）
        for attr in ("past_key_values", "_past_key_values", "cache", "_cache"):
            if hasattr(target_model, attr):
                setattr(target_model, attr, None)
            if hasattr(target_model, "model") and hasattr(target_model.model, attr):
                setattr(target_model.model, attr, None)
    
    def run_test(self, args, build_model_kwargs: Optional[Dict] = None) -> bool:
        """
        运行完整的 Baseline 测试流程
        支持多样本串行处理(bsz=1)，模型只初始化一次
        
        Args:
            args: 命令行参数
            build_model_kwargs: 传递给 builder 函数的额外参数
        """
        from transformers import AutoTokenizer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Building {self.config.name} model with offloading...")
        
        # 构建模型参数
        kwargs = {
            "device": device,
            "state_path": args.base_model_path,
            "enable_gpu_cache": args.enable_gpu_cache,
            "cache_policy": args.cache_policy,
            "topk_lru_logit_percentile": args.topk_lru_logit_percentile,
            "cache_slots_per_layer": args.cache_slots_per_layer,
        }
        
        # 合并额外参数
        if build_model_kwargs:
            kwargs.update(build_model_kwargs)
        
        # 构建 target model (只初始化一次)
        target_model = self.config.builder_func(**kwargs)
        target_model.eval()
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        
        # 准备 prompts - 从 benchmark 加载
        prompts = get_prompts(benchmark=args.benchmark, num_samples=args.num_samples, start_idx=args.start_idx)
        print(f"📋 Loaded {len(prompts)} prompts (num_samples={args.num_samples}, start_idx={args.start_idx})")
        
        # 多样本统计
        all_stats = {
            'total_new_tokens': 0,
            'total_decode_time': 0,
            'per_sample_stats': []
        }
        
        # 串行处理每个 prompt (bsz=1)
        for sample_idx, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"🔄 Processing Sample {sample_idx + 1}/{len(prompts)}")
            print(f"{'='*60}")

            # 显式重置样本级状态（KV/runtime cache）
            self._reset_for_new_sample(target_model)
            
            # 准备 input_ids
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            prefill_length = input_ids.shape[1]
            
            # 生成
            with torch.no_grad():
                # Prefill 阶段
                prefill_start = time.time()
                outputs = target_model(input_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                prefill_time = time.time() - prefill_start
                
                # Decode 阶段
                decode_start_time = time.time()
                
                if PRINT_GENERATION_RESULT:
                    print(f"\n📝 Generation:")
                    print(f"User: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"Assistant: ", end='', flush=True)
                
                current_input_ids = input_ids[:, -1:]
                generated_ids = current_input_ids.clone()
                
                for step in range(args.max_new_tokens):
                    outputs = target_model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    logits = outputs.logits[:, -1, :]
                    
                    # 根据模型配置选择解码策略
                    if self.config.use_sampling:
                        temperature = args.temperature
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    past_key_values = outputs.past_key_values
                    current_input_ids = next_token
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # 实时打印
                    if PRINT_GENERATION_RESULT:
                        token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                        print(token_text, end='', flush=True)
                    
                    # 检查 EOS
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        if PRINT_GENERATION_RESULT:
                            print()
                        break

                    # 与 tests/utils.py 的 max_length 约束保持一致
                    if prefill_length + generated_ids.shape[1] - 1 >= args.max_length:
                        break
                
                decode_end_time = time.time()
            
            # 计算当前样本的统计
            decode_time = decode_end_time - decode_start_time
            new_tokens = generated_ids.shape[1] - 1
            decode_tps = new_tokens / decode_time if decode_time > 0 else 0
            
            # 保存样本统计
            sample_stat = {
                'sample_idx': sample_idx,
                'new_tokens': new_tokens,
                'decode_time': decode_time,
                'decode_tps': decode_tps,
                'prefill_time': prefill_time,
            }
            all_stats['per_sample_stats'].append(sample_stat)
            all_stats['total_new_tokens'] += new_tokens
            all_stats['total_decode_time'] += decode_time
            
            if PRINT_GENERATION_RESULT:
                print(f"\n")
            
            print(f"✅ Sample {sample_idx + 1} completed: {new_tokens} tokens, {decode_tps:.2f} tokens/s")
        
        # 所有样本处理完毕后的汇总统计
        print(f"\n{'='*60}")
        print("📊 OVERALL STATISTICS (All Samples)")
        print(f"{'='*60}")
        print(f"Total samples: {len(prompts)}")
        print(f"Total new tokens: {all_stats['total_new_tokens']}")
        print(f"Total decode time: {all_stats['total_decode_time']:.2f}s")
        
        # 全局平均 TPS
        overall_tps = all_stats['total_new_tokens'] / all_stats['total_decode_time'] if all_stats['total_decode_time'] > 0 else 0
        print(f"⚡ Overall Decode TPS: {overall_tps:.2f} tokens/s")
        
        print(f"{'='*60}\n")
        
        # 打印 GPU Cache 统计（与 tests/utils.py 保持一致）
        self._print_cache_stats(target_model)
        
        # Cleanup
        print("🧹 Cleaning up models...")
        del target_model
        torch.cuda.empty_cache()
        print("✅ Cleanup complete")
        
        return True
    
    def _print_cache_stats(self, target_model):
        """打印 GPU Cache 统计信息 - 与 tests/utils.py 保持一致"""
        if not hasattr(target_model, 'model') or not hasattr(target_model.model, 'layers'):
            return
        
        first_layer = target_model.model.layers[0]
        if not hasattr(first_layer, 'mlp') or not hasattr(first_layer.mlp, 'expert_cache'):
            return
        
        expert_cache = first_layer.mlp.expert_cache
        if hasattr(expert_cache, 'gpu_cache_manager') and expert_cache.gpu_cache_manager is not None:
            expert_cache.gpu_cache_manager.print_cache_stats()


def add_baseline_common_args(parser):
    """添加通用的命令行参数"""
    parser.add_argument("--base-model-path", type=str, help="Path to base model")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "humaneval", "cnndm"])
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for testing")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to test")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--max-length", type=int, default=8192, help="Max total sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--enable-gpu-cache", action="store_true", default=True, help="Enable GPU expert cache")
    parser.add_argument("--cache-policy", type=str, default="lru", choices=["static", "lru", "lfu", "topk_lru", "tinylfu"])
    parser.add_argument("--cache-slots-per-layer", type=int, default=16)
    parser.add_argument("--topk-lru-logit-percentile", type=float, default=90.0)
    return parser
