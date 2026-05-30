#!/usr/bin/env python3
"""Utilities for SDAR MoE-Offloading latency and memory evaluation."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT.parent
OPENCOMPASS_ROOT = EVALUATION_ROOT / "opencompass"
PROFILES_ROOT = PROJECT_ROOT / "profiles"

for path in (PROJECT_ROOT, OPENCOMPASS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "dataset_module": "opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799",
        "dataset_var_name": "gsm8k_datasets",
        "dataset_index": 0,
        "default_split": "test",
        "recommended_gen_length": 128,
        "recommended_max_out_len": 128,
    },
    "math": {
        "dataset_module": "opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5",
        "dataset_var_name": "math_datasets",
        "dataset_index": 0,
        "default_split": "test",
        "recommended_gen_length": 128,
        "recommended_max_out_len": 128,
    },
    "humaneval": {
        "dataset_module": "opencompass.configs.datasets.humaneval.humaneval_gen",
        "dataset_var_name": "humaneval_datasets",
        "dataset_index": 0,
        "default_split": "test",
        "recommended_gen_length": 512,
        "recommended_max_out_len": 512,
    },
    "sanitized_mbpp": {
        "dataset_module": "opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416",
        "dataset_var_name": "sanitized_mbpp_datasets",
        "dataset_index": 0,
        "default_split": "test",
        "recommended_gen_length": 512,
        "recommended_max_out_len": 512,
    },
}


@dataclass
class GPUStatus:
    index: int
    free_gib: float
    total_gib: float
    utilization: int


def parse_candidate_gpus(gpus: str) -> List[int]:
    return [int(item.strip()) for item in gpus.split(",") if item.strip()]


def resolve_benchmark_preset(benchmark: Optional[str]) -> Optional[Dict[str, Any]]:
    if benchmark is None:
        return None
    try:
        return dict(BENCHMARK_PRESETS[benchmark])
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark `{benchmark}`. Choices: {sorted(BENCHMARK_PRESETS)}"
        ) from exc


def query_gpu_statuses(candidate_gpus: Iterable[int]) -> List[GPUStatus]:
    wanted = set(candidate_gpus)
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)

    statuses = []
    for raw_line in output.strip().splitlines():
        gpu_index, free_mib, total_mib, utilization = [
            token.strip() for token in raw_line.split(",")
        ]
        gpu_index_int = int(gpu_index)
        if gpu_index_int not in wanted:
            continue
        statuses.append(
            GPUStatus(
                index=gpu_index_int,
                free_gib=int(free_mib) / 1024.0,
                total_gib=int(total_mib) / 1024.0,
                utilization=int(utilization),
            )
        )
    return sorted(statuses, key=lambda item: item.index)


def wait_for_available_gpu(
    candidate_gpus: Iterable[int],
    min_free_memory_gib: float,
    max_utilization: int,
    poll_interval_s: int,
    max_wait_minutes: float,
) -> GPUStatus:
    start_time = time.time()
    candidate_gpus = list(candidate_gpus)

    while True:
        statuses = query_gpu_statuses(candidate_gpus)
        eligible = [
            gpu
            for gpu in statuses
            if gpu.free_gib >= min_free_memory_gib
            and gpu.utilization <= max_utilization
        ]
        if eligible:
            return max(eligible, key=lambda gpu: (gpu.free_gib, -gpu.utilization))

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        status_text = ", ".join(
            f"GPU{gpu.index}: free={gpu.free_gib:.1f}GiB util={gpu.utilization}%"
            for gpu in statuses
        )
        print(
            f"[{now}] No GPU meets the requirement "
            f"(free>={min_free_memory_gib:.1f}GiB, util<={max_utilization}%). {status_text}"
        )

        if max_wait_minutes > 0:
            waited_minutes = (time.time() - start_time) / 60.0
            if waited_minutes >= max_wait_minutes:
                raise TimeoutError(
                    f"Waited {waited_minutes:.1f} minutes but no eligible GPU became available."
                )

        time.sleep(poll_interval_s)


def reserve_cuda_memory(
    target_free_gib: float,
    *,
    device: Optional[int] = None,
    allocation_margin_gib: float = 0.5,
    chunk_gib: float = 1.0,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    if not torch.cuda.is_available():
        return [], {
            "enabled": False,
            "reason": "cuda_unavailable",
        }
    if target_free_gib <= 0:
        raise ValueError("--reserve-free-memory-gib must be > 0")
    if allocation_margin_gib < 0:
        raise ValueError("allocation_margin_gib must be >= 0")
    if chunk_gib <= 0:
        raise ValueError("chunk_gib must be > 0")

    cuda_device = (
        torch.device(f"cuda:{device}") if device is not None else torch.device("cuda")
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    target_free_bytes = int(target_free_gib * 1024 ** 3)
    margin_bytes = int(allocation_margin_gib * 1024 ** 3)
    bytes_to_reserve = max(0, free_bytes - target_free_bytes - margin_bytes)
    chunk_bytes = int(chunk_gib * 1024 ** 3)

    reservation_tensors = []
    remaining_bytes = bytes_to_reserve
    try:
        while remaining_bytes > 0:
            current_bytes = min(chunk_bytes, remaining_bytes)
            reservation_tensors.append(
                torch.empty(
                    (current_bytes,),
                    dtype=torch.uint8,
                    device=cuda_device,
                )
            )
            remaining_bytes -= current_bytes
    except RuntimeError:
        del reservation_tensors
        torch.cuda.empty_cache()
        raise

    post_free_bytes, _ = torch.cuda.mem_get_info(cuda_device)
    stats = {
        "enabled": True,
        "target_free_gib": target_free_gib,
        "allocation_margin_gib": allocation_margin_gib,
        "chunk_gib": chunk_gib,
        "free_before_gib": round(free_bytes / (1024 ** 3), 6),
        "total_gib": round(total_bytes / (1024 ** 3), 6),
        "requested_reserve_gib": round(bytes_to_reserve / (1024 ** 3), 6),
        "reserved_gib": round(
            sum(tensor.numel() for tensor in reservation_tensors) / (1024 ** 3),
            6,
        ),
        "free_after_gib": round(post_free_bytes / (1024 ** 3), 6),
        "tensor_count": len(reservation_tensors),
    }
    return reservation_tensors, stats


def instantiate_from_config(config: Dict[str, Any]):
    config = dict(config)
    obj_type = config.pop("type")
    return obj_type(**config)


def load_dataset_bundle(dataset_module: str, dataset_var_name: str, dataset_index: int):
    module = importlib.import_module(dataset_module)
    dataset_cfg = getattr(module, dataset_var_name)[dataset_index]

    dataset_kwargs = {
        key: value
        for key, value in dataset_cfg.items()
        if key not in {"infer_cfg", "eval_cfg"}
    }
    dataset = instantiate_from_config(dataset_kwargs)

    prompt_template = instantiate_from_config(
        dataset_cfg["infer_cfg"]["prompt_template"]
    )

    eval_cfg = dataset_cfg.get("eval_cfg", {})
    pred_postprocessor = None
    dataset_postprocessor = None
    evaluator = None

    if "pred_postprocessor" in eval_cfg:
        pred_post_cfg = dict(eval_cfg["pred_postprocessor"])
        pred_postprocessor = pred_post_cfg.pop("type")
        if pred_post_cfg:
            pred_postprocessor = lambda text, func=pred_postprocessor, kwargs=pred_post_cfg: func(text, **kwargs)

    if "dataset_postprocessor" in eval_cfg:
        dataset_post_cfg = dict(eval_cfg["dataset_postprocessor"])
        dataset_postprocessor = dataset_post_cfg.pop("type")
        if dataset_post_cfg:
            dataset_postprocessor = lambda text, func=dataset_postprocessor, kwargs=dataset_post_cfg: func(text, **kwargs)

    if "evaluator" in eval_cfg:
        evaluator = instantiate_from_config(eval_cfg["evaluator"])

    return {
        "dataset_cfg": dataset_cfg,
        "dataset": dataset,
        "prompt_template": prompt_template,
        "pred_postprocessor": pred_postprocessor,
        "dataset_postprocessor": dataset_postprocessor,
        "evaluator": evaluator,
    }


def score_predictions(
    evaluator,
    predictions,
    references,
    *,
    test_set: Optional[List[Dict[str, Any]]] = None,
    origin_prompt: Optional[List[str]] = None,
):
    if evaluator is None:
        return {}

    evaluator_name = evaluator.__class__.__name__
    if evaluator_name == "HumanEvalEvaluator" and test_set is not None:
        return _score_humaneval_subset(evaluator, predictions, references, test_set)

    score_fn = evaluator.score
    try:
        signature = inspect.signature(score_fn)
    except (TypeError, ValueError):
        return score_fn(predictions, references)

    kwargs = {}
    parameters = signature.parameters
    if "test_set" in parameters and test_set is not None:
        kwargs["test_set"] = test_set
    if "origin_prompt" in parameters and origin_prompt is not None:
        kwargs["origin_prompt"] = origin_prompt
    return score_fn(predictions, references, **kwargs)


def _score_humaneval_subset(
    evaluator,
    predictions,
    references,
    test_set: List[Dict[str, Any]],
):
    if len(predictions) != len(references):
        return {"error": "preds and refrs have different length"}

    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    prompts = [item["prompt"] for item in test_set]
    humaneval_preds = []
    for preds, refer in zip(predictions, references):
        if not isinstance(preds, list):
            preds = [preds]
        for pred in preds:
            humaneval_preds.append({"task_id": refer, "completion": pred})

    subset_problems = []
    seen_task_ids = set()
    for item in test_set:
        task_id = item["task_id"]
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        subset_problems.append(
            {
                "task_id": task_id,
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = os.path.join(tmp_dir, "human_eval_subset_samples.jsonl")
        problem_file = os.path.join(tmp_dir, "human_eval_subset_problems.jsonl")
        write_jsonl(sample_file, humaneval_preds)
        write_jsonl(problem_file, subset_problems)
        score = evaluate_functional_correctness(
            sample_file,
            evaluator.k,
            n_workers=4,
            timeout=3.0,
            problem_file=problem_file,
        )

        detail_path = sample_file + "_results.jsonl"
        details = {}
        with open(detail_path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = json.loads(line)
                line["is_correct"] = line["passed"]
                if index < len(prompts):
                    line["prompt"] = prompts[index]
                details[str(index)] = line

    results = {f"humaneval_{k}": score[k] * 100 for k in score}
    results["details"] = details
    return results


def capture_cuda_memory_snapshot(
    snapshots: Optional[List[Dict[str, Any]]],
    stage: str,
    relative_time_s: float,
    sample_idx: Optional[int] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
):
    if snapshots is None or not torch.cuda.is_available():
        return None

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    record = {
        "relative_time_s": round(relative_time_s, 6),
        "sample_idx": "" if sample_idx is None else sample_idx,
        "stage": stage,
        "device_total_bytes": total_bytes,
        "device_free_bytes": free_bytes,
        "device_used_bytes": total_bytes - free_bytes,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if extra_fields:
        record.update(extra_fields)
    snapshots.append(record)
    return record


def write_memory_snapshots(path: Path, snapshots: List[Dict[str, Any]]) -> Path:
    if not snapshots:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    byte_fields = [
        "device_total_bytes",
        "device_free_bytes",
        "device_used_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "torch_max_allocated_bytes",
        "torch_max_reserved_bytes",
        "sample_peak_allocated_bytes",
        "sample_peak_reserved_bytes",
    ]
    gib_fields = [field.replace("_bytes", "_gib") for field in byte_fields]
    fieldnames = ["relative_time_s", "sample_idx", "stage", *byte_fields, *gib_fields]

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in snapshots:
            row = dict(record)
            for byte_field, gib_field in zip(byte_fields, gib_fields):
                value = row.get(byte_field, 0) or 0
                row[gib_field] = round(value / (1024 ** 3), 6)
            writer.writerow(row)
    return path


def extract_expert_cache(model_wrapper):
    model = model_wrapper.model
    for layer in model.model.layers:
        if hasattr(layer.mlp, "expert_cache"):
            return layer.mlp.expert_cache
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def make_default_results_path(prefix: str, start_idx: int, num_samples: int) -> Path:
    return PROFILES_ROOT / f"{prefix}_start{start_idx}_n{num_samples}_results.json"


def make_default_memory_path(prefix: str, start_idx: int, num_samples: int) -> Path:
    return PROFILES_ROOT / f"{prefix}_start{start_idx}_n{num_samples}_gpu_memory.csv"
