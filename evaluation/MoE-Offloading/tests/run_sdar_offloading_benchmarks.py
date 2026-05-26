#!/usr/bin/env python3
"""Run the standard SDAR MoE-Offloading benchmark suite."""

from __future__ import annotations

import argparse

from sdar_offloading_runner import (
    DEFAULT_BENCHMARK_ORDER,
    add_common_sdar_offloading_arguments,
    run_sdar_offloading_benchmark_suite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SDAR MoE-Offloading on HumanEval, MBPP, GSM8K, and MATH500."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARK_ORDER),
        help="Comma-separated benchmark order.",
    )
    add_common_sdar_offloading_arguments(
        parser,
        default_num_samples=0,
        include_benchmark=False,
        include_dataset_args=False,
        include_single_output_args=False,
        default_min_free_memory_gib=60.0,
        default_max_gpu_utilization=5,
        default_reserve_gpu_memory=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_sdar_offloading_benchmark_suite(args)
    print(f"Experiment directory: {summary['run_dir']}")


if __name__ == "__main__":
    main()
