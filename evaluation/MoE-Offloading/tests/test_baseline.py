#!/usr/bin/env python3
"""
Unified Baseline Test - Qwen3-MoE (default) and GPT-OSS
纯自回归生成，测试offloading + prefetch性能
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from tests.baseline_utils import BaselineTestBase, QWEN3MOE_CONFIG, GPTOSS_CONFIG, add_baseline_common_args
except ModuleNotFoundError:
    from baseline_utils import BaselineTestBase, QWEN3MOE_CONFIG, GPTOSS_CONFIG, add_baseline_common_args


def main():
    import argparse
    
    # 第一步：只解析 --model 参数
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", type=str, default="qwen3moe",
                            choices=["qwen3moe", "gpt-oss"])
    model_args, remaining = pre_parser.parse_known_args()
    
    # 第二步：根据 model 设置默认参数和 builder
    if model_args.model == "qwen3moe":
        from baseline.qwen3_builder import qwen3_build_model
        config = QWEN3MOE_CONFIG
        config.builder_func = qwen3_build_model
        defaults = {
            "base_model_path": "/data/home/tianjianyang/models/moe/Qwen3-30B-A3B",
            "temperature": 0.8,
            "enable_gpu_cache": True,
            "cache_policy": "topk_lru",
            "cache_slots_per_layer": 16,
        }
    else:
        from baseline.gptoss_builder import gptoss_build_model
        config = GPTOSS_CONFIG
        config.builder_func = gptoss_build_model
        defaults = {
            "base_model_path": "/data/home/tianjianyang/models/moe/gpt-oss-20b-BF16",
            "temperature": 0.7,
            "enable_gpu_cache": True,
            "cache_policy": "topk_lru",
            "cache_slots_per_layer": 8,
        }
    
    # 第三步：用对应模型的默认参数解析所有参数
    parser = argparse.ArgumentParser(description="Unified Baseline Test", parents=[pre_parser])
    add_baseline_common_args(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    
    print(f"🔧 Testing {config.name} baseline")
    BaselineTestBase(config).run_test(args)


if __name__ == "__main__":
    main()
