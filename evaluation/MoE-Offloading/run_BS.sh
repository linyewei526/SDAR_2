#!/bin/bash

# Unified Baseline Test Script - Run baseline tests for both models across benchmarks

# Qwen3MoE baseline tests
python -u tests/test_baseline.py --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 16 --num-samples 80 --max-new-tokens 1024 >> gsm8k_baseline_qwen3moe_results.txt 2>&1

python -u tests/test_baseline.py --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 16 --num-samples 80 --max-new-tokens 1024 --benchmark humaneval >> humaneval_baseline_qwen3moe_results.txt 2>&1

python -u tests/test_baseline.py --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 16 --num-samples 80 --max-new-tokens 1024 --benchmark cnndm >> cnndm_baseline_qwen3moe_results.txt 2>&1

# GPT-OSS baseline tests
python -u tests/test_baseline.py --model gpt-oss --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 8 --num-samples 80 --max-new-tokens 1024 >> gsm8k_baseline_gpt-oss_results.txt 2>&1

python -u tests/test_baseline.py --model gpt-oss --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 8 --num-samples 80 --max-new-tokens 1024 --benchmark humaneval >> humaneval_baseline_gpt-oss_results.txt 2>&1

python -u tests/test_baseline.py --model gpt-oss --enable-gpu-cache --cache-policy lru --cache-slots-per-layer 8 --num-samples 80 --max-new-tokens 1024 --benchmark cnndm >> cnndm_baseline_gpt-oss_results.txt 2>&1
