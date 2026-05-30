#!/usr/bin/env python3
"""Shared runner for SDAR MoE-Offloading evaluations."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.nvtx_utils import nvtx_range, set_nvtx_enabled
from baseline.sdar_runtime_trace import (
    begin_sample,
    disable_trace,
    enable_trace,
    end_sample,
    export_trace,
)
from sdar_offloading_utils import (
    BENCHMARK_PRESETS,
    OPENCOMPASS_ROOT,
    PROFILES_ROOT,
    capture_cuda_memory_snapshot,
    extract_expert_cache,
    load_dataset_bundle,
    parse_candidate_gpus,
    reserve_cuda_memory,
    resolve_benchmark_preset,
    score_predictions,
    wait_for_available_gpu,
    write_json,
    write_memory_snapshots,
)

if str(OPENCOMPASS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOMPASS_ROOT))

from opencompass.models.huggingface_bd3 import BD3withChatTemplate


DEFAULT_MODEL_PATH = (
    "/data_3/wly/.cache/huggingface/hub/models--JetLM--SDAR-30B-A3B-Chat-b32/"
    "snapshots/c351bbc37d240aa6871f167e8f92d694281b0c22"
)

DEFAULT_BENCHMARK_ORDER = ["humaneval", "sanitized_mbpp", "gsm8k", "math"]


def add_common_sdar_offloading_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_num_samples: int,
    include_benchmark: bool,
    include_dataset_args: bool,
    include_single_output_args: bool,
    default_min_free_memory_gib: float = 40.0,
    default_max_gpu_utilization: int = 20,
    default_reserve_gpu_memory: bool = False,
) -> None:
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--local-modeling-module",
        type=str,
        default="configs.sdar_local_models.modeling_sdar_moe_offloading",
    )
    if include_benchmark:
        parser.add_argument(
            "--benchmark",
            type=str,
            default=None,
            choices=sorted(BENCHMARK_PRESETS),
            help="Benchmark shortcut that fills dataset-module/var-name/index.",
        )
    if include_dataset_args:
        parser.add_argument(
            "--dataset-module",
            type=str,
            default="opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799",
        )
        parser.add_argument("--dataset-var-name", type=str, default="gsm8k_datasets")
        parser.add_argument("--dataset-index", type=int, default=0)

    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=default_num_samples,
        help="Number of samples. Use 0 or a negative value to run the remaining split.",
    )
    parser.add_argument("--max-out-len", type=int, default=None)
    parser.add_argument("--mask-id", type=int, default=151669)
    parser.add_argument("--gen-length", type=int, default=4096)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--enable-gpu-cache", dest="enable_gpu_cache", action="store_true")
    parser.add_argument("--disable-gpu-cache", dest="enable_gpu_cache", action="store_false")
    parser.set_defaults(enable_gpu_cache=True)
    parser.add_argument(
        "--cache-policy",
        type=str,
        default="topk_lru",
        choices=["static", "lru", "lfu", "topk_lru", "tinylfu"],
    )
    parser.add_argument("--cache-slots-per-layer", type=int, default=16)
    parser.add_argument("--topk-lru-logit-percentile", type=float, default=90.0)

    parser.add_argument("--candidate-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--min-free-memory-gib", type=float, default=default_min_free_memory_gib)
    parser.add_argument("--max-gpu-utilization", type=int, default=default_max_gpu_utilization)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=0.0,
        help="0 means wait indefinitely until a GPU becomes available.",
    )

    parser.add_argument(
        "--reserve-gpu-memory",
        dest="reserve_gpu_memory",
        action="store_true",
        help="Reserve most free GPU memory so other jobs are less likely to enter.",
    )
    parser.add_argument(
        "--disable-reserve-gpu-memory",
        dest="reserve_gpu_memory",
        action="store_false",
        help="Disable the GPU memory reservation guard.",
    )
    parser.set_defaults(reserve_gpu_memory=default_reserve_gpu_memory)
    parser.add_argument(
        "--reserve-gpu-memory-stage",
        type=str,
        default="pre_build",
        choices=["pre_build", "post_build"],
    )
    parser.add_argument("--reserve-free-memory-gib", type=float, default=24.0)

    parser.add_argument("--enable-nvtx-ranges", action="store_true")
    parser.add_argument("--track-gpu-memory", action="store_true")
    parser.add_argument("--gpu-memory-output", type=str, default=None)
    if include_single_output_args:
        parser.add_argument("--results-output", type=str, default=None)
        parser.add_argument("--record-output", type=str, default=None)
    parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help="Print per-sample latency, prediction, and reference details.",
    )
    parser.add_argument(
        "--record-mode",
        type=str,
        default="none",
        choices=["none", "experts", "latency", "both"],
    )
    parser.add_argument(
        "--record-scope",
        type=str,
        default="none",
        choices=["none", "all", "first_k"],
    )
    parser.add_argument("--record-first-k", type=int, default=0)
    parser.add_argument(
        "--nsys-capture-range-name",
        type=str,
        default="SDAR_Inference_Capture",
    )
    parser.add_argument(
        "--nsys-use-cuda-profiler-api",
        action="store_true",
        help="Use cudaProfilerStart/Stop around per-sample generation.",
    )

    parser.add_argument("--profiles-root", type=str, default=str(PROFILES_ROOT))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Existing or new run directory. Defaults to profiles/<timestamp>_<run-kind>.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix appended to the auto-created timestamp run directory.",
    )


def make_timestamped_run_dir(
    *,
    profiles_root: Path,
    run_kind: str,
    run_name: Optional[str],
    output_dir: Optional[str],
) -> Path:
    if output_dir:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = f"_{_sanitize_name(run_name)}" if run_name else ""
    base_name = f"{timestamp}_{run_kind}{suffix}"
    profiles_root.mkdir(parents=True, exist_ok=True)

    run_dir = profiles_root / base_name
    counter = 1
    while run_dir.exists():
        run_dir = profiles_root / f"{base_name}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_sdar_offloading_evaluation(
    args: argparse.Namespace,
    *,
    run_dir: Optional[Path] = None,
    run_type: str = "single_benchmark",
    selected_gpu=None,
    memory_reservation_tensors: Optional[List[torch.Tensor]] = None,
    memory_reservation_stats: Optional[Dict[str, Any]] = None,
    write_config: bool = True,
    cleanup_model: bool = True,
) -> Dict[str, Any]:
    args = copy.copy(args)
    _resolve_dataset_args(args)
    _ensure_output_attrs(args)

    if args.record_scope == "first_k" and args.record_first_k <= 0:
        raise ValueError("--record-first-k must be > 0 when --record-scope=first_k")

    if run_dir is None:
        run_dir = make_timestamped_run_dir(
            profiles_root=Path(args.profiles_root),
            run_kind="sdar_offloading_single",
            run_name=args.run_name,
            output_dir=args.output_dir,
        )

    record_experts = args.record_mode in ("experts", "both")
    record_latency = args.record_mode in ("latency", "both")
    recording_enabled = (
        (record_experts or record_latency) and args.record_scope != "none"
    )

    set_nvtx_enabled(args.enable_nvtx_ranges)
    cuda_profiler_runtime = None
    if args.nsys_use_cuda_profiler_api:
        if not torch.cuda.is_available():
            raise RuntimeError("--nsys-use-cuda-profiler-api requires CUDA")
        cuda_profiler_runtime = torch.cuda.cudart()

    benchmark_label = _benchmark_label(args)
    results_output_path = (
        Path(args.results_output)
        if args.results_output
        else run_dir / f"{benchmark_label}_results.json"
    )
    record_output_path = None
    if recording_enabled:
        record_output_path = (
            Path(args.record_output)
            if args.record_output
            else run_dir / f"{benchmark_label}_summary.json"
        )

    if selected_gpu is None:
        candidate_gpus = parse_candidate_gpus(args.candidate_gpus)
        selected_gpu = wait_for_available_gpu(
            candidate_gpus=candidate_gpus,
            min_free_memory_gib=args.min_free_memory_gib,
            max_utilization=args.max_gpu_utilization,
            poll_interval_s=args.poll_interval_s,
            max_wait_minutes=args.max_wait_minutes,
        )
        print(
            f"Using GPU {selected_gpu.index} "
            f"(free={selected_gpu.free_gib:.1f}GiB, util={selected_gpu.utilization}%)"
        )

    torch.cuda.set_device(selected_gpu.index)
    benchmark_start = time.perf_counter()
    memory_snapshots = [] if args.track_gpu_memory else None
    reservation_tensors = memory_reservation_tensors or []
    gpu_memory_reservation = memory_reservation_stats or {
        "enabled": args.reserve_gpu_memory,
        "stage": args.reserve_gpu_memory_stage,
        "target_free_gib": args.reserve_free_memory_gib,
        "reserved_gib": 0.0,
    }

    if (
        memory_reservation_stats is None
        and args.reserve_gpu_memory
        and args.reserve_gpu_memory_stage == "pre_build"
    ):
        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="pre_reserve_gpu_memory",
            relative_time_s=time.perf_counter() - benchmark_start,
        )
        reservation_tensors, gpu_memory_reservation = reserve_cuda_memory(
            args.reserve_free_memory_gib,
            device=selected_gpu.index,
        )
        gpu_memory_reservation["stage"] = args.reserve_gpu_memory_stage
        print(
            "Reserved GPU memory: "
            f"{gpu_memory_reservation.get('reserved_gib', 0.0):.3f}GiB, "
            f"free_after={gpu_memory_reservation.get('free_after_gib', 0.0):.3f}GiB"
        )
        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="post_reserve_gpu_memory",
            relative_time_s=time.perf_counter() - benchmark_start,
        )

    dataset_bundle = load_dataset_bundle(
        dataset_module=args.dataset_module,
        dataset_var_name=args.dataset_var_name,
        dataset_index=args.dataset_index,
    )
    dataset = getattr(dataset_bundle["dataset"], args.split)
    prompt_template = dataset_bundle["prompt_template"]
    pred_postprocessor = dataset_bundle["pred_postprocessor"]
    dataset_postprocessor = dataset_bundle["dataset_postprocessor"]
    evaluator = dataset_bundle["evaluator"]

    num_samples = args.num_samples
    if num_samples is None or num_samples <= 0:
        num_samples = len(dataset) - args.start_idx
    if args.start_idx + num_samples > len(dataset):
        raise ValueError(
            f"Requested samples [{args.start_idx}, {args.start_idx + num_samples}) "
            f"but split `{args.split}` only has {len(dataset)} rows."
        )
    args.num_samples = num_samples

    generation_kwargs = _generation_kwargs(args)

    if write_config:
        _write_experiment_config(
            run_dir,
            {
                "status": "started",
                "run_type": run_type,
                "created_at": _now_string(),
                "command": sys.argv,
                "run_dir": str(run_dir),
                "benchmark_order": [args.benchmark] if args.benchmark else [],
                "args": _to_jsonable(vars(args)),
                "selected_gpu": _to_jsonable(selected_gpu.__dict__),
                "outputs": {
                    "results_output": str(results_output_path),
                    "record_output": str(record_output_path) if record_output_path else None,
                },
            },
        )

    model_wrapper = None
    trace_enabled = False
    try:
        if recording_enabled:
            enable_trace(
                output_path=str(record_output_path),
                metadata={
                    "model_path": args.model_path,
                    "benchmark": args.benchmark,
                    "dataset_module": args.dataset_module,
                    "dataset_var_name": args.dataset_var_name,
                    "dataset_index": args.dataset_index,
                    "split": args.split,
                    "record_mode": args.record_mode,
                    "record_scope": args.record_scope,
                    "record_first_k": args.record_first_k,
                    "generation_kwargs": generation_kwargs,
                },
                record_experts=record_experts,
                record_latency=record_latency,
            )
            trace_enabled = True

        build_start = time.perf_counter()
        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="pre_build",
            relative_time_s=build_start - benchmark_start,
        )
        model_wrapper = BD3withChatTemplate(
            path=args.model_path,
            local_modeling_module=args.local_modeling_module,
            generation_kwargs=generation_kwargs,
            model_kwargs=dict(
                device_map=f"cuda:{selected_gpu.index}",
                torch_dtype=torch.bfloat16,
                enable_gpu_cache=args.enable_gpu_cache,
                cache_policy=args.cache_policy,
                topk_lru_logit_percentile=args.topk_lru_logit_percentile,
                cache_slots_per_layer=args.cache_slots_per_layer,
            ),
        )
        build_time_s = time.perf_counter() - build_start

        if (
            memory_reservation_stats is None
            and args.reserve_gpu_memory
            and args.reserve_gpu_memory_stage == "post_build"
        ):
            capture_cuda_memory_snapshot(
                memory_snapshots,
                stage="pre_reserve_gpu_memory",
                relative_time_s=time.perf_counter() - benchmark_start,
            )
            reservation_tensors, gpu_memory_reservation = reserve_cuda_memory(
                args.reserve_free_memory_gib,
                device=selected_gpu.index,
            )
            gpu_memory_reservation["stage"] = args.reserve_gpu_memory_stage
            print(
                "Reserved GPU memory: "
                f"{gpu_memory_reservation.get('reserved_gib', 0.0):.3f}GiB, "
                f"free_after={gpu_memory_reservation.get('free_after_gib', 0.0):.3f}GiB"
            )
            capture_cuda_memory_snapshot(
                memory_snapshots,
                stage="post_reserve_gpu_memory",
                relative_time_s=time.perf_counter() - benchmark_start,
            )

        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="post_build",
            relative_time_s=time.perf_counter() - benchmark_start,
        )

        summary = _run_samples(
            args=args,
            model_wrapper=model_wrapper,
            dataset_bundle=dataset_bundle,
            dataset=dataset,
            prompt_template=prompt_template,
            pred_postprocessor=pred_postprocessor,
            dataset_postprocessor=dataset_postprocessor,
            evaluator=evaluator,
            num_samples=num_samples,
            recording_enabled=recording_enabled,
            cuda_profiler_runtime=cuda_profiler_runtime,
            benchmark_start=benchmark_start,
            memory_snapshots=memory_snapshots,
            generation_kwargs=generation_kwargs,
            selected_gpu=selected_gpu,
            build_time_s=build_time_s,
            gpu_memory_reservation=gpu_memory_reservation,
            reservation_tensors=reservation_tensors,
        )

        expert_cache = extract_expert_cache(model_wrapper)
        if expert_cache is not None:
            buffer_stats = expert_cache.buffer_manager.get_stats()
            summary["buffer_manager"] = buffer_stats
            print("Buffer manager stats:", buffer_stats)
            if expert_cache.gpu_cache_manager is not None:
                cache_stats = expert_cache.gpu_cache_manager.get_cache_stats()
                summary["gpu_cache"] = cache_stats
                print("GPU cache stats:", cache_stats)

        if summary["evaluation"]:
            print(
                f"[{benchmark_label}] Evaluation:",
                _compact_evaluation_for_console(summary["evaluation"]),
            )
        print(f"[{benchmark_label}] Aggregate:", summary["aggregate"])

        if recording_enabled:
            trace_output = export_trace(str(record_output_path))
            summary["record_summary_output"] = str(trace_output)
            print(f"[{benchmark_label}] Compact summary record saved to: {trace_output}")
            trace_enabled = False
            disable_trace()

        summary["run_dir"] = str(run_dir)
        summary["results_output"] = str(results_output_path)
        write_json(results_output_path, summary)
        print(f"[{benchmark_label}] Results saved to: {results_output_path}")

        if memory_snapshots is not None:
            memory_output_path = (
                Path(args.gpu_memory_output)
                if args.gpu_memory_output
                else run_dir / f"{benchmark_label}_memory.csv"
            )
            write_memory_snapshots(memory_output_path, memory_snapshots)
            summary["gpu_memory_output"] = str(memory_output_path)
            write_json(results_output_path, summary)
            print(f"[{benchmark_label}] Memory snapshots saved to: {memory_output_path}")

        if write_config:
            _write_experiment_config(
                run_dir,
                _single_experiment_config(
                    args=args,
                    run_dir=run_dir,
                    selected_gpu=selected_gpu,
                    summary=summary,
                    results_output_path=results_output_path,
                    record_output_path=record_output_path,
                ),
            )
        return summary
    finally:
        if trace_enabled:
            disable_trace()
        if cleanup_model and model_wrapper is not None:
            del model_wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_sdar_offloading_benchmark_suite(args: argparse.Namespace) -> Dict[str, Any]:
    benchmarks = _parse_benchmark_order(args.benchmarks)
    run_dir = make_timestamped_run_dir(
        profiles_root=Path(args.profiles_root),
        run_kind="sdar_offloading_suite",
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    candidate_gpus = parse_candidate_gpus(args.candidate_gpus)
    selected_gpu = wait_for_available_gpu(
        candidate_gpus=candidate_gpus,
        min_free_memory_gib=args.min_free_memory_gib,
        max_utilization=args.max_gpu_utilization,
        poll_interval_s=args.poll_interval_s,
        max_wait_minutes=args.max_wait_minutes,
    )
    print(
        f"Using GPU {selected_gpu.index} "
        f"(free={selected_gpu.free_gib:.1f}GiB, util={selected_gpu.utilization}%)"
    )
    torch.cuda.set_device(selected_gpu.index)

    reservation_tensors: List[torch.Tensor] = []
    reservation_stats: Dict[str, Any] = {
        "enabled": args.reserve_gpu_memory,
        "stage": args.reserve_gpu_memory_stage,
        "target_free_gib": args.reserve_free_memory_gib,
        "reserved_gib": 0.0,
    }
    if args.reserve_gpu_memory:
        reservation_tensors, reservation_stats = reserve_cuda_memory(
            args.reserve_free_memory_gib,
            device=selected_gpu.index,
        )
        reservation_stats["stage"] = "suite_pre_all_benchmarks"
        reservation_stats["live_tensor_count"] = len(reservation_tensors)
        print(
            "Reserved GPU memory for benchmark suite: "
            f"{reservation_stats.get('reserved_gib', 0.0):.3f}GiB, "
            f"free_after={reservation_stats.get('free_after_gib', 0.0):.3f}GiB"
        )

    suite_config = {
        "status": "started",
        "run_type": "benchmark_suite",
        "created_at": _now_string(),
        "command": sys.argv,
        "run_dir": str(run_dir),
        "benchmark_order": benchmarks,
        "args": _to_jsonable(vars(args)),
        "selected_gpu": _to_jsonable(selected_gpu.__dict__),
        "gpu_memory_reservation": reservation_stats,
        "benchmarks": {},
    }
    _write_experiment_config(run_dir, suite_config)

    benchmark_summaries = {}
    try:
        benchmark_iterator = tqdm(
            benchmarks,
            desc="Benchmark suite",
            unit="benchmark",
            dynamic_ncols=True,
        )
        for benchmark in benchmark_iterator:
            bench_args = copy.copy(args)
            bench_args.benchmark = benchmark
            bench_args.results_output = None
            bench_args.record_output = None
            bench_args.gpu_memory_output = None
            bench_args.output_dir = str(run_dir)
            bench_args._suite_progress = True
            _resolve_dataset_args(bench_args)

            summary = run_sdar_offloading_evaluation(
                bench_args,
                run_dir=run_dir,
                run_type="benchmark_suite_item",
                selected_gpu=selected_gpu,
                memory_reservation_tensors=reservation_tensors,
                memory_reservation_stats=reservation_stats,
                write_config=False,
                cleanup_model=True,
            )
            benchmark_summaries[benchmark] = {
                "evaluation": summary.get("evaluation", {}),
                "aggregate": summary.get("aggregate", {}),
                "results_output": summary.get("results_output"),
                "record_summary_output": summary.get("record_summary_output"),
                "gpu_memory_output": summary.get("gpu_memory_output"),
                "dataset": summary.get("dataset", {}),
                "offloading": summary.get("offloading", {}),
            }
            suite_config["benchmarks"] = benchmark_summaries
            _write_experiment_config(run_dir, suite_config)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    suite_summary = {
        "run_type": "benchmark_suite",
        "created_at": suite_config["created_at"],
        "completed_at": _now_string(),
        "run_dir": str(run_dir),
        "benchmark_order": benchmarks,
        "selected_gpu": _to_jsonable(selected_gpu.__dict__),
        "gpu_memory_reservation": reservation_stats,
        "benchmarks": benchmark_summaries,
    }
    suite_summary_path = run_dir / "all_benchmarks_summary.json"
    write_json(suite_summary_path, suite_summary)

    suite_config["status"] = "completed"
    suite_config["completed_at"] = suite_summary["completed_at"]
    suite_config["suite_summary_output"] = str(suite_summary_path)
    _write_experiment_config(run_dir, suite_config)
    print(f"Suite summary saved to: {suite_summary_path}")
    print(f"Experiment config saved to: {run_dir / 'experiment_config.json'}")
    return suite_summary


def _run_samples(
    *,
    args: argparse.Namespace,
    model_wrapper,
    dataset_bundle: Dict[str, Any],
    dataset,
    prompt_template,
    pred_postprocessor,
    dataset_postprocessor,
    evaluator,
    num_samples: int,
    recording_enabled: bool,
    cuda_profiler_runtime,
    benchmark_start: float,
    memory_snapshots: Optional[List[Dict[str, Any]]],
    generation_kwargs: Dict[str, Any],
    selected_gpu,
    build_time_s: float,
    gpu_memory_reservation: Dict[str, Any],
    reservation_tensors: List[torch.Tensor],
) -> Dict[str, Any]:
    max_out_len = args.max_out_len or args.gen_length
    predictions = []
    references = []
    evaluated_test_set = []
    evaluated_origin_prompts = []
    sample_results = []
    tokenizer = model_wrapper.tokenizer

    progress_desc = f"{_benchmark_label(args)} {args.split}"
    sample_iterator = tqdm(
        range(num_samples),
        desc=progress_desc,
        unit="sample",
        dynamic_ncols=True,
        position=1 if getattr(args, "_suite_progress", False) else 0,
    )
    for local_idx in sample_iterator:
        sample_idx = args.start_idx + local_idx
        entry = dataset[sample_idx]
        prompt = prompt_template.generate_item(entry)
        record_this_sample = recording_enabled and _should_record_sample(local_idx, args)
        if recording_enabled:
            begin_sample(sample_idx, record_this_sample=record_this_sample)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="sample_pre_generate",
            relative_time_s=time.perf_counter() - benchmark_start,
            sample_idx=sample_idx,
        )

        sample_start = time.perf_counter()
        if cuda_profiler_runtime is not None:
            cuda_profiler_runtime.cudaProfilerStart()
        try:
            with nvtx_range(args.nsys_capture_range_name):
                output_text = model_wrapper.generate_from_template(
                    [prompt], max_out_len=max_out_len
                )[0]
        finally:
            if cuda_profiler_runtime is not None:
                cuda_profiler_runtime.cudaProfilerStop()
        sample_latency_s = time.perf_counter() - sample_start
        if recording_enabled:
            end_sample()

        generated_token_count = len(
            tokenizer.encode(output_text, add_special_tokens=False)
        )
        processed_pred = (
            pred_postprocessor(output_text) if pred_postprocessor else output_text
        )
        reference_text = entry[dataset_bundle["dataset_cfg"]["reader_cfg"]["output_column"]]
        processed_ref = (
            dataset_postprocessor(reference_text)
            if dataset_postprocessor
            else reference_text
        )

        predictions.append(processed_pred)
        references.append(processed_ref)
        evaluated_test_set.append(entry)
        evaluated_origin_prompts.append(prompt)

        sample_peak_allocated = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        sample_peak_reserved = (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        )
        capture_cuda_memory_snapshot(
            memory_snapshots,
            stage="sample_post_generate",
            relative_time_s=time.perf_counter() - benchmark_start,
            sample_idx=sample_idx,
            extra_fields={
                "sample_peak_allocated_bytes": sample_peak_allocated,
                "sample_peak_reserved_bytes": sample_peak_reserved,
            },
        )

        sample_result = {
            "sample_idx": sample_idx,
            "question": entry.get("question", ""),
            "raw_prediction": output_text,
            "processed_prediction": processed_pred,
            "reference": processed_ref,
            "latency_s": round(sample_latency_s, 6),
            "generated_token_count": generated_token_count,
            "tokens_per_second": round(generated_token_count / sample_latency_s, 6)
            if sample_latency_s > 0
            else None,
        }
        sample_results.append(sample_result)

        if args.verbose_samples:
            tqdm.write(
                f"[sample {sample_idx}] latency={sample_latency_s:.3f}s "
                f"tokens={generated_token_count} "
                f"tps={sample_result['tokens_per_second']}"
            )
            tqdm.write(f"prediction: {processed_pred}")
            tqdm.write(f"reference : {processed_ref}")

    eval_result = score_predictions(
        evaluator,
        predictions,
        references,
        test_set=evaluated_test_set,
        origin_prompt=evaluated_origin_prompts,
    )

    total_latency_s = sum(item["latency_s"] for item in sample_results)
    total_generated_tokens = sum(item["generated_token_count"] for item in sample_results)
    gpu_memory_reservation = dict(gpu_memory_reservation)
    gpu_memory_reservation["live_tensor_count"] = len(reservation_tensors)

    return {
        "model_path": args.model_path,
        "local_modeling_module": args.local_modeling_module,
        "benchmark": args.benchmark,
        "gpu": {
            "index": selected_gpu.index,
            "free_gib_at_start": round(selected_gpu.free_gib, 3),
            "total_gib": round(selected_gpu.total_gib, 3),
            "utilization_at_start": selected_gpu.utilization,
        },
        "build_time_s": round(build_time_s, 6),
        "generation_kwargs": generation_kwargs,
        "gpu_memory_reservation": gpu_memory_reservation,
        "offloading": {
            "enable_gpu_cache": args.enable_gpu_cache,
            "cache_policy": args.cache_policy,
            "cache_slots_per_layer": args.cache_slots_per_layer,
            "topk_lru_logit_percentile": args.topk_lru_logit_percentile,
        },
        "dataset": {
            "module": args.dataset_module,
            "var_name": args.dataset_var_name,
            "index": args.dataset_index,
            "split": args.split,
            "start_idx": args.start_idx,
            "num_samples": num_samples,
            "benchmark_preset": args.benchmark,
        },
        "recording": {
            "enabled": recording_enabled,
            "mode": args.record_mode,
            "scope": args.record_scope,
            "first_k": args.record_first_k,
        },
        "aggregate": {
            "sample_count": num_samples,
            "total_generation_latency_s": round(total_latency_s, 6),
            "total_generated_tokens": total_generated_tokens,
            "average_latency_s": round(total_latency_s / num_samples, 6)
            if num_samples > 0
            else None,
            "overall_tokens_per_second": round(
                total_generated_tokens / total_latency_s, 6
            )
            if total_latency_s > 0
            else None,
        },
        "evaluation": eval_result,
        "samples": sample_results,
    }


def _single_experiment_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    selected_gpu,
    summary: Dict[str, Any],
    results_output_path: Path,
    record_output_path: Optional[Path],
) -> Dict[str, Any]:
    return {
        "status": "completed",
        "run_type": "single_benchmark",
        "created_at": _now_string(),
        "completed_at": _now_string(),
        "command": sys.argv,
        "run_dir": str(run_dir),
        "benchmark_order": [args.benchmark] if args.benchmark else [],
        "args": _to_jsonable(vars(args)),
        "selected_gpu": _to_jsonable(selected_gpu.__dict__),
        "generation_kwargs": summary.get("generation_kwargs", {}),
        "offloading": summary.get("offloading", {}),
        "recording": summary.get("recording", {}),
        "gpu_memory_reservation": summary.get("gpu_memory_reservation", {}),
        "outputs": {
            "results_output": str(results_output_path),
            "record_output": str(record_output_path) if record_output_path else None,
            "memory_output": summary.get("gpu_memory_output"),
        },
        "result_digest": {
            "evaluation": summary.get("evaluation", {}),
            "aggregate": summary.get("aggregate", {}),
            "dataset": summary.get("dataset", {}),
        },
    }


def _write_experiment_config(run_dir: Path, config: Dict[str, Any]) -> Path:
    return write_json(run_dir / "experiment_config.json", config)


def _resolve_dataset_args(args: argparse.Namespace) -> None:
    benchmark = getattr(args, "benchmark", None)
    benchmark_preset = resolve_benchmark_preset(benchmark)
    if benchmark_preset is not None:
        args.dataset_module = benchmark_preset["dataset_module"]
        args.dataset_var_name = benchmark_preset["dataset_var_name"]
        args.dataset_index = benchmark_preset["dataset_index"]


def _ensure_output_attrs(args: argparse.Namespace) -> None:
    for name in ("results_output", "record_output", "gpu_memory_output"):
        if not hasattr(args, name):
            setattr(args, name, None)
    for name, value in (
        ("profiles_root", str(PROFILES_ROOT)),
        ("output_dir", None),
        ("run_name", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)


def _generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "mask_id": args.mask_id,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "denoising_steps": args.denoising_steps or args.block_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "remasking": args.remasking,
        "threshold": args.threshold,
    }


def _compact_evaluation_for_console(evaluation: Any) -> Any:
    if not isinstance(evaluation, dict):
        return evaluation

    compact = {}
    for key, value in evaluation.items():
        if key == "details":
            if hasattr(value, "__len__"):
                compact["details_count"] = len(value)
            else:
                compact["details"] = "<omitted>"
        else:
            compact[key] = value
    return compact


def _should_record_sample(local_idx: int, args: argparse.Namespace) -> bool:
    if args.record_scope == "all":
        return True
    if args.record_scope == "first_k":
        return local_idx < args.record_first_k
    return False


def _benchmark_label(args: argparse.Namespace) -> str:
    if args.benchmark:
        return _sanitize_name(args.benchmark)
    return _sanitize_name(f"{Path(args.dataset_module).name}_{args.dataset_index}")


def _parse_benchmark_order(raw_benchmarks: str) -> List[str]:
    benchmarks = [item.strip() for item in raw_benchmarks.split(",") if item.strip()]
    unknown = [item for item in benchmarks if item not in BENCHMARK_PRESETS]
    if unknown:
        raise ValueError(
            f"Unknown benchmarks {unknown}. Choices: {sorted(BENCHMARK_PRESETS)}"
        )
    return benchmarks


def _sanitize_name(value: Optional[str]) -> str:
    if not value:
        return ""
    allowed = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def _now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__") and value.__class__.__module__ != "builtins":
        return _to_jsonable(value.__dict__)
    return value
