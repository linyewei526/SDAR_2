#!/usr/bin/env python3
"""Evaluate one benchmark with SDAR MoE-Offloading."""

from __future__ import annotations

import argparse

from sdar_offloading_runner import (
    add_common_sdar_offloading_arguments,
    run_sdar_offloading_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SDAR MoE-Offloading single-benchmark evaluation"
    )
    add_common_sdar_offloading_arguments(
        parser,
        default_num_samples=1,
        include_benchmark=True,
        include_dataset_args=True,
        include_single_output_args=True,
        default_min_free_memory_gib=40.0,
        default_max_gpu_utilization=20,
        default_reserve_gpu_memory=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_sdar_offloading_evaluation(
        args,
        run_type="single_benchmark",
        write_config=True,
        cleanup_model=True,
    )
    print(f"Experiment directory: {summary['run_dir']}")


if __name__ == "__main__":
    main()
