#!/usr/bin/env python3
"""Evaluate SDAR-30B-A3B block-diffusion decoding under the MoE-Offloading runtime."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

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
    OPENCOMPASS_ROOT,
    capture_cuda_memory_snapshot,
    extract_expert_cache,
    load_dataset_bundle,
    make_default_memory_path,
    make_default_results_path,
    parse_candidate_gpus,
    wait_for_available_gpu,
    write_json,
    write_memory_snapshots,
)


if str(OPENCOMPASS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOMPASS_ROOT))

from opencompass.models.huggingface_bd3 import BD3withChatTemplate


DEFAULT_MODEL_PATH = (
    "/data/home/wly/.cache/huggingface/hub/models--JetLM--SDAR-30B-A3B-Chat-b32/"
    "snapshots/c351bbc37d240aa6871f167e8f92d694281b0c22"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SDAR MoE-Offloading latency and memory evaluation"
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--local-modeling-module",
        type=str,
        default="configs.sdar_local_models.modeling_sdar_moe_offloading",
    )
    parser.add_argument(
        "--dataset-module",
        type=str,
        default="opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799",
    )
    parser.add_argument("--dataset-var-name", type=str, default="gsm8k_datasets")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
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
    parser.add_argument("--min-free-memory-gib", type=float, default=40.0)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=0.0,
        help="0 means wait indefinitely until a GPU becomes available.",
    )
    parser.add_argument("--enable-nvtx-ranges", action="store_true")
    parser.add_argument("--track-gpu-memory", action="store_true")
    parser.add_argument("--gpu-memory-output", type=str, default=None)
    parser.add_argument("--results-output", type=str, default=None)
    parser.add_argument(
        "--record-mode",
        type=str,
        default="none",
        choices=["none", "experts", "latency", "both"],
        help="Compact per-sample summary recording mode.",
    )
    parser.add_argument(
        "--record-scope",
        type=str,
        default="none",
        choices=["none", "all", "first_k"],
        help="Choose which evaluated samples should emit compact summary records.",
    )
    parser.add_argument(
        "--record-first-k",
        type=int,
        default=0,
        help="Used when --record-scope=first_k. Records the first k evaluated samples.",
    )
    parser.add_argument(
        "--record-output",
        type=str,
        default=None,
        help="JSON output path for compact summary records.",
    )
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
    return parser.parse_args()


def should_record_sample(local_idx: int, args) -> bool:
    if args.record_scope == "all":
        return True
    if args.record_scope == "first_k":
        return local_idx < args.record_first_k
    return False


def make_default_record_path(start_idx: int, num_samples: int, record_mode: str) -> Path:
    output_dir = PROJECT_ROOT / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"sdar_record_summary_{record_mode}_start{start_idx}_n{num_samples}.json"


def main():
    args = parse_args()
    if args.record_scope == "first_k" and args.record_first_k <= 0:
        raise ValueError("--record-first-k must be > 0 when --record-scope=first_k")

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

    record_output_path = None
    if recording_enabled:
        record_output_path = (
            Path(args.record_output)
            if args.record_output
            else make_default_record_path(
                args.start_idx, args.num_samples, args.record_mode
            )
        )
        enable_trace(
            output_path=str(record_output_path),
            metadata={
                "model_path": args.model_path,
                "dataset_module": args.dataset_module,
                "dataset_var_name": args.dataset_var_name,
                "dataset_index": args.dataset_index,
                "split": args.split,
                "record_mode": args.record_mode,
                "record_scope": args.record_scope,
                "record_first_k": args.record_first_k,
                "generation_kwargs": {
                    "mask_id": args.mask_id,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "denoising_steps": args.denoising_steps or args.block_length,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "remasking": args.remasking,
                    "threshold": args.threshold,
                },
            },
            record_experts=record_experts,
            record_latency=record_latency,
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
    benchmark_start = time.perf_counter()
    memory_snapshots = [] if args.track_gpu_memory else None

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

    if args.start_idx + args.num_samples > len(dataset):
        raise ValueError(
            f"Requested samples [{args.start_idx}, {args.start_idx + args.num_samples}) "
            f"but split `{args.split}` only has {len(dataset)} rows."
        )

    generation_kwargs = dict(
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps or args.block_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking=args.remasking,
        threshold=args.threshold,
    )

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
    capture_cuda_memory_snapshot(
        memory_snapshots,
        stage="post_build",
        relative_time_s=time.perf_counter() - benchmark_start,
    )

    max_out_len = args.max_out_len or args.gen_length
    predictions = []
    references = []
    sample_results = []
    tokenizer = model_wrapper.tokenizer

    for local_idx in range(args.num_samples):
        sample_idx = args.start_idx + local_idx
        entry = dataset[sample_idx]
        prompt = prompt_template.generate_item(entry)
        record_this_sample = recording_enabled and should_record_sample(local_idx, args)
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
            "tokens_per_second": round(
                generated_token_count / sample_latency_s, 6
            )
            if sample_latency_s > 0
            else None,
        }
        sample_results.append(sample_result)

        print(
            f"[sample {sample_idx}] latency={sample_latency_s:.3f}s "
            f"tokens={generated_token_count} "
            f"tps={sample_result['tokens_per_second']}"
        )
        print(f"prediction: {processed_pred}")
        print(f"reference : {processed_ref}")

    eval_result = evaluator.score(predictions, references) if evaluator else {}

    total_latency_s = sum(item["latency_s"] for item in sample_results)
    total_generated_tokens = sum(item["generated_token_count"] for item in sample_results)
    summary = {
        "model_path": args.model_path,
        "local_modeling_module": args.local_modeling_module,
        "gpu": {
            "index": selected_gpu.index,
            "free_gib_at_start": round(selected_gpu.free_gib, 3),
            "total_gib": round(selected_gpu.total_gib, 3),
            "utilization_at_start": selected_gpu.utilization,
        },
        "build_time_s": round(build_time_s, 6),
        "generation_kwargs": generation_kwargs,
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
            "num_samples": args.num_samples,
        },
        "recording": {
            "enabled": recording_enabled,
            "mode": args.record_mode,
            "scope": args.record_scope,
            "first_k": args.record_first_k,
        },
        "aggregate": {
            "sample_count": args.num_samples,
            "total_generation_latency_s": round(total_latency_s, 6),
            "total_generated_tokens": total_generated_tokens,
            "average_latency_s": round(total_latency_s / args.num_samples, 6),
            "overall_tokens_per_second": round(
                total_generated_tokens / total_latency_s, 6
            )
            if total_latency_s > 0
            else None,
        },
        "evaluation": eval_result,
        "samples": sample_results,
    }

    expert_cache = extract_expert_cache(model_wrapper)
    if expert_cache is not None:
        buffer_stats = expert_cache.buffer_manager.get_stats()
        summary["buffer_manager"] = buffer_stats
        print("Buffer manager stats:", buffer_stats)
        if expert_cache.gpu_cache_manager is not None:
            cache_stats = expert_cache.gpu_cache_manager.get_cache_stats()
            summary["gpu_cache"] = cache_stats
            print("GPU cache stats:", cache_stats)

    if eval_result:
        print("Evaluation:", eval_result)

    if recording_enabled:
        summary["record_summary_output"] = str(record_output_path)

    results_output = (
        Path(args.results_output)
        if args.results_output
        else make_default_results_path("sdar_offloading", args.start_idx, args.num_samples)
    )
    write_json(results_output, summary)
    print(f"Results saved to: {results_output}")

    if memory_snapshots is not None:
        memory_output = (
            Path(args.gpu_memory_output)
            if args.gpu_memory_output
            else make_default_memory_path(
                "sdar_offloading", args.start_idx, args.num_samples
            )
        )
        write_memory_snapshots(memory_output, memory_snapshots)
        print(f"Memory snapshots saved to: {memory_output}")

    if recording_enabled:
        trace_output = export_trace(str(record_output_path))
        print(f"Compact summary record saved to: {trace_output}")
        disable_trace()


if __name__ == "__main__":
    main()
