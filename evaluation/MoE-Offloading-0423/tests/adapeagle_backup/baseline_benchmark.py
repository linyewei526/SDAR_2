#!/usr/bin/env python3
"""
Baseline TPS benchmark test for AR generation with offloading + prefetch
Tests multiple prompts from different benchmarks (GSM8K, HumanEval, CNN/DM)
"""
import torch
import os
import sys
import time
import argparse
import math

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baseline.qwen3_builder import qwen3_build_model
from baseline.gptoss_builder import gptoss_build_model
from transformers import AutoTokenizer


def get_prompts(args):
    """从benchmark目录读取数据"""
    import pyarrow.parquet as pq

    # 基准测试数据路径 - 使用项目根目录相对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.benchmark == "gsm8k":
        benchmark_path = os.path.join(project_root, "benchmark", "gsm8k", "main", "test-00000-of-00001.parquet")
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = args.start_idx + args.num_samples
        for _, row in df.iloc[args.start_idx:end_idx].iterrows():
            prompts.append(row['question'])
        return prompts

    elif args.benchmark == "humaneval":
        benchmark_path = os.path.join(project_root, "benchmark", "openai_humaneval", "openai_humaneval", "test-00000-of-00001.parquet")
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = args.start_idx + args.num_samples
        for _, row in df.iloc[args.start_idx:end_idx].iterrows():
            prompts.append(row['prompt'])
        return prompts

    elif args.benchmark == "cnndm":
        benchmark_path = os.path.join(project_root, "benchmark", "CNN-DM.parquet")
        table = pq.read_table(benchmark_path)
        df = table.to_pandas()
        prompts = []
        end_idx = args.start_idx + args.num_samples
        for _, row in df.iloc[args.start_idx:end_idx].iterrows():
            prompts.append(row['article'])
        return prompts

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


def test_baseline_qwen3moe_benchmark(args):
    """测试 Qwen3-30B-A3B baseline 在指定 benchmark 下的 TPS"""

    device = torch.device("cuda")

    # 加载 prompts
    prompts = get_prompts(args)

    # 构建 offloading 模型
    target_model = qwen3_build_model(
        device=device,
        state_path=args.base_model_path
    )
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    print(f"\n{'='*80}")
    print(f"Testing AR Baseline on {args.num_samples} {args.benchmark} prompts (starting from idx {args.start_idx})")
    print(f"{'='*80}\n")

    total_new_tokens = 0
    total_decode_time = 0
    tps_list = []

    for i, prompt in enumerate(prompts):
        actual_idx = args.start_idx + i
        print(f"[{actual_idx+1}/{args.start_idx+args.num_samples}]")

        # 准备input_ids
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        prefill_length = input_ids.shape[1]

        with torch.no_grad():
            # Prefill阶段
            outputs = target_model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

            # Decode阶段（自回归生成）
            decode_start_time = time.time()

            current_input_ids = input_ids[:, -1:]
            generated_ids = current_input_ids.clone()

            for step in range(args.max_new_tokens):
                # 单步生成
                outputs = target_model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # 更新past_key_values和input_ids
                past_key_values = outputs.past_key_values
                current_input_ids = next_token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 检查EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

            decode_end_time = time.time()

        decode_time = decode_end_time - decode_start_time
        new_tokens = generated_ids.shape[1] - 1
        decode_tps = new_tokens / decode_time if decode_time > 0 else 0

        total_new_tokens += new_tokens
        total_decode_time += decode_time
        tps_list.append(decode_tps)

        print(f"  Generated {new_tokens} tokens in {decode_time:.2f}s")
        print(f"  TPS: {decode_tps:.2f} tokens/s")

    # 计算 mu 和 sigma
    mu = sum(tps_list) / len(tps_list) if tps_list else 0
    sigma = math.sqrt(sum((x - mu) ** 2 for x in tps_list) / len(tps_list)) if len(tps_list) > 1 else 0

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total prompts tested: {args.num_samples}")
    print(f"Total new tokens: {total_new_tokens}")
    print(f"Total decode time: {total_decode_time:.2f}s")
    print(f"TPS mu: {mu:.2f} tokens/s")
    print(f"TPS sigma: {sigma:.2f} tokens/s")
    print(f"{'='*80}\n")

    return True


def test_baseline_gptoss_benchmark(args):
    """测试 GPT-OSS-20B baseline 在指定 benchmark 下的 TPS"""

    device = torch.device("cuda")

    # 加载 prompts
    prompts = get_prompts(args)

    # 构建 offloading 模型
    target_model = gptoss_build_model(
        device=device,
        state_path=args.gptoss_model_path
    )
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.gptoss_model_path)

    print(f"\n{'='*80}")
    print(f"Testing AR Baseline on {args.num_samples} {args.benchmark} prompts (starting from idx {args.start_idx})")
    print(f"{'='*80}\n")

    total_new_tokens = 0
    total_decode_time = 0
    tps_list = []

    for i, prompt in enumerate(prompts):
        actual_idx = args.start_idx + i
        print(f"[{actual_idx+1}/{args.start_idx+args.num_samples}]")

        # 准备input_ids
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        prefill_length = input_ids.shape[1]

        with torch.no_grad():
            # Prefill阶段
            outputs = target_model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

            # Decode阶段（自回归生成）
            decode_start_time = time.time()

            current_input_ids = input_ids[:, -1:]
            generated_ids = current_input_ids.clone()

            for step in range(args.max_new_tokens):
                # 单步生成
                outputs = target_model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                logits = outputs.logits[:, -1, :]

                # Sample with temperature (GPT-OSS默认使用sampling)
                temperature = 0.7
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 更新past_key_values和input_ids
                past_key_values = outputs.past_key_values
                current_input_ids = next_token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 检查EOS
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

            decode_end_time = time.time()

        decode_time = decode_end_time - decode_start_time
        new_tokens = generated_ids.shape[1] - 1
        decode_tps = new_tokens / decode_time if decode_time > 0 else 0

        total_new_tokens += new_tokens
        total_decode_time += decode_time
        tps_list.append(decode_tps)

        print(f"  Generated {new_tokens} tokens in {decode_time:.2f}s")
        print(f"  TPS: {decode_tps:.2f} tokens/s")

    # 计算 mu 和 sigma
    mu = sum(tps_list) / len(tps_list) if tps_list else 0
    sigma = math.sqrt(sum((x - mu) ** 2 for x in tps_list) / len(tps_list)) if len(tps_list) > 1 else 0

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total prompts tested: {args.num_samples}")
    print(f"Total new tokens: {total_new_tokens}")
    print(f"Total decode time: {total_decode_time:.2f}s")
    print(f"TPS mu: {mu:.2f} tokens/s")
    print(f"TPS sigma: {sigma:.2f} tokens/s")
    print(f"{'='*80}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline AR benchmark test")
    parser.add_argument("--model", type=str, default="qwen3moe", choices=["qwen3moe", "gptoss"], help="Model to test")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "humaneval", "cnndm"], help="Benchmark to use")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for testing")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of samples to test")
    parser.add_argument("--base-model-path", type=str, 
                        default="/data/home/tianjianyang/models/moe/Qwen3-30B-A3B", help="Path to base model (for qwen3moe)")
    parser.add_argument("--gptoss-model-path", type=str, 
                        default="/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16", help="Path to GPT-OSS model (for gptoss)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate per prompt")
    args = parser.parse_args()

    if args.model == "qwen3moe":
        test_baseline_qwen3moe_benchmark(args)
    else:  # gptoss
        test_baseline_gptoss_benchmark(args)
