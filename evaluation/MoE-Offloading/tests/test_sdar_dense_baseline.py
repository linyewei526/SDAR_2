#!/usr/bin/env python3
"""Evaluate pure SDAR dense-GPU decoding and profile per-layer latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.sdar_dense_profile import (
    begin_sample,
    disable_profile,
    enable_profile,
    end_sample,
    export_profile,
)
from sdar_offloading_utils import (
    OPENCOMPASS_ROOT,
    load_dataset_bundle,
    make_default_results_path,
    parse_candidate_gpus,
    wait_for_available_gpu,
    write_json,
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
        description="Pure SDAR dense-baseline latency evaluation"
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--local-modeling-module",
        type=str,
        default="configs.sdar_local_models.modeling_sdar_moe_profiled",
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
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-out-len", type=int, default=128)
    parser.add_argument("--mask-id", type=int, default=151669)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--candidate-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--min-free-memory-gib", type=float, default=70.0)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument("--max-wait-minutes", type=float, default=0.0)
    parser.add_argument("--enable-profile", dest="enable_profile", action="store_true")
    parser.add_argument("--disable-profile", dest="enable_profile", action="store_false")
    parser.set_defaults(enable_profile=True)
    parser.add_argument("--profile-output", type=str, default=None)
    parser.add_argument("--results-output", type=str, default=None)
    return parser.parse_args()


def make_default_profile_path(start_idx: int, num_samples: int) -> Path:
    output_dir = PROJECT_ROOT / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"sdar_dense_baseline_start{start_idx}_n{num_samples}_profile.json"


def main():
    args = parse_args()

    profile_output = None
    if args.enable_profile:
        profile_output = (
            Path(args.profile_output)
            if args.profile_output
            else make_default_profile_path(args.start_idx, args.num_samples)
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

    if args.enable_profile:
        enable_profile(
            output_path=str(profile_output),
            metadata={
                "model_path": args.model_path,
                "local_modeling_module": args.local_modeling_module,
                "dataset_module": args.dataset_module,
                "dataset_var_name": args.dataset_var_name,
                "dataset_index": args.dataset_index,
                "split": args.split,
                "generation_kwargs": generation_kwargs,
            },
        )

    build_start = time.perf_counter()
    model_wrapper = BD3withChatTemplate(
        path=args.model_path,
        local_modeling_module=args.local_modeling_module,
        generation_kwargs=generation_kwargs,
        model_kwargs=dict(
            device_map=f"cuda:{selected_gpu.index}",
            torch_dtype=torch.bfloat16,
        ),
    )
    build_time_s = time.perf_counter() - build_start

    tokenizer = model_wrapper.tokenizer
    predictions = []
    references = []
    sample_results = []

    for local_idx in range(args.num_samples):
        sample_idx = args.start_idx + local_idx
        entry = dataset[sample_idx]
        prompt = prompt_template.generate_item(entry)

        if args.enable_profile:
            begin_sample(
                sample_idx,
                metadata={
                    "question": entry.get("question", ""),
                },
            )

        sample_start = time.perf_counter()
        output_text = model_wrapper.generate_from_template(
            [prompt], max_out_len=args.max_out_len
        )[0]
        sample_latency_s = time.perf_counter() - sample_start
        if args.enable_profile:
            end_sample()

        generated_token_count = len(tokenizer.encode(output_text, add_special_tokens=False))
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
        "dataset": {
            "module": args.dataset_module,
            "var_name": args.dataset_var_name,
            "index": args.dataset_index,
            "split": args.split,
            "start_idx": args.start_idx,
            "num_samples": args.num_samples,
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
    if profile_output is not None:
        summary["profile_output"] = str(profile_output)

    if eval_result:
        print("Evaluation:", eval_result)

    results_output = (
        Path(args.results_output)
        if args.results_output
        else make_default_results_path("sdar_dense_baseline", args.start_idx, args.num_samples)
    )
    write_json(results_output, summary)
    print(f"Results saved to: {results_output}")

    if args.enable_profile:
        profile_path = export_profile(str(profile_output))
        print(f"Dense baseline profile saved to: {profile_path}")
        disable_profile()


if __name__ == "__main__":
    main()
