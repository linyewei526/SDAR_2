#!/usr/bin/env python3

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


CATEGORY_ORDER = [
    "attention",
    "routing",
    "current_layer_availability_check",
    "current_layer_miss_load",
    "next_layer_prefetch",
    "reorder",
    "gather",
    "expert_compute",
    "scatter",
    "cache_promotion",
]

CATEGORY_LABELS = {
    "attention": "attention",
    "routing": "routing",
    "current_layer_availability_check": "current-layer availability check",
    "current_layer_miss_load": "current-layer miss load",
    "next_layer_prefetch": "next-layer prefetch",
    "reorder": "reorder",
    "gather": "gather",
    "expert_compute": "expert compute",
    "scatter": "scatter",
    "cache_promotion": "cache promotion",
}

NSYS_CATEGORY_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"Attention_Layer\d+$"), "attention"),
    (re.compile(r"Routing_Layer\d+$"), "routing"),
    (
        re.compile(r"Current_Layer_Availability_Check_Layer\d+$"),
        "current_layer_availability_check",
    ),
    (re.compile(r"Current_Layer_Miss_Load_Layer\d+$"), "current_layer_miss_load"),
    (re.compile(r"Next_Layer_Prefetch_Layer\d+$"), "next_layer_prefetch"),
    (re.compile(r"Reorder_Layer\d+$"), "reorder"),
    (re.compile(r"Gather_Layer\d+$"), "gather"),
    (re.compile(r"Expert_Compute_Layer\d+$"), "expert_compute"),
    (re.compile(r"Scatter_Layer\d+$"), "scatter"),
    (re.compile(r"Cache_Promotion_Layer\d+$"), "cache_promotion"),
]

STEP_RE = re.compile(r"Decode_Block(\d+)_Step(\d+)_(Denoise|Finalize)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SDAR MoE-Offloading detailed trace and optional nsys sqlite export."
    )
    parser.add_argument("--trace-json", required=True, help="Path to detailed trace JSON.")
    parser.add_argument("--results-json", required=True, help="Path to results JSON.")
    parser.add_argument(
        "--nsys-sqlite",
        default=None,
        help="Optional path to nsys sqlite export for NVTX cross-check.",
    )
    parser.add_argument(
        "--summary-json-output",
        required=True,
        help="Path to write the summary JSON.",
    )
    parser.add_argument(
        "--report-md-output",
        required=True,
        help="Path to write the Chinese markdown report.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(mean(values))


def safe_rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def round_if_number(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [round_if_number(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: round_if_number(v, digits) for k, v in value.items()}
    return value


def iter_layer_records(step: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    layers = step.get("layers", {})
    if isinstance(layers, dict):
        items = ((int(k), v) for k, v in layers.items())
        return sorted(items, key=lambda x: x[0])
    return list(enumerate(layers))


def iter_decode_steps(sample: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for block in sample.get("decode_blocks", []):
        for step in block.get("steps", []):
            yield block, step


def summarize_timeline_from_trace(sample: Dict[str, Any], step_kind: Optional[str] = None) -> Dict[str, Any]:
    selected_steps: List[Dict[str, Any]] = []
    for _, step in iter_decode_steps(sample):
        if step_kind is not None and step.get("step_kind") != step_kind:
            continue
        selected_steps.append(step)

    summary: Dict[str, Any] = {
        "step_kind_filter": step_kind or "all",
        "step_count": len(selected_steps),
        "average_step_total_ms": mean_or_none(step.get("total_duration_ms", 0.0) for step in selected_steps),
        "categories": {},
    }

    for category in CATEGORY_ORDER:
        first_starts = []
        last_ends = []
        wall_spans = []
        accumulated = []
        shares = []

        for step in selected_steps:
            intervals = [
                item
                for item in step.get("timing_intervals", [])
                if item.get("category") == category
            ]
            if not intervals:
                continue
            first_start = min(item["start_ms"] for item in intervals)
            last_end = max(item["end_ms"] for item in intervals)
            acc_duration = sum(item["duration_ms"] for item in intervals)
            step_total = float(step.get("total_duration_ms", 0.0))

            first_starts.append(first_start)
            last_ends.append(last_end)
            wall_spans.append(last_end - first_start)
            accumulated.append(acc_duration)
            shares.append(safe_rate(acc_duration, step_total) * 100.0)

        summary["categories"][category] = {
            "label": CATEGORY_LABELS[category],
            "step_occurrences": len(first_starts),
            "average_first_start_ms": mean_or_none(first_starts),
            "average_last_end_ms": mean_or_none(last_ends),
            "average_wall_span_ms": mean_or_none(wall_spans),
            "average_accumulated_duration_ms": mean_or_none(accumulated),
            "average_share_of_step_percent": mean_or_none(shares),
        }

    return summary


def summarize_expert_activity(sample: Dict[str, Any]) -> Dict[str, Any]:
    totals = defaultdict(float)
    per_block: List[Dict[str, Any]] = []
    per_layer: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    worst_layer_steps: List[Dict[str, Any]] = []

    for block, step in iter_decode_steps(sample):
        block_totals = defaultdict(float)
        block_totals["decode_block_idx"] = block["decode_block_idx"]
        block_totals["absolute_block_idx"] = block["absolute_block_idx"]
        block_totals["step_idx"] = step["step_idx"]
        block_totals["step_kind"] = step["step_kind"]

        for layer_idx, layer in iter_layer_records(step):
            requested = float(layer.get("requested_expert_count", 0))
            cache_hits = float(layer.get("gpu_cache_hit_count", 0))
            prefetch_hits = float(layer.get("prefetch_hit_count", 0))
            cpu_miss = float(layer.get("cpu_miss_load_count", 0))
            cache_before = float(layer.get("gpu_cache_expert_count_before", 0))
            prefetch_before = float(layer.get("prefetch_available_expert_count_before", 0))

            totals["layer_step_count"] += 1
            totals["requested_expert_count"] += requested
            totals["gpu_cache_hit_count"] += cache_hits
            totals["prefetch_hit_count"] += prefetch_hits
            totals["cpu_miss_load_count"] += cpu_miss
            totals["gpu_cache_expert_count_before_total"] += cache_before
            totals["prefetch_available_expert_count_before_total"] += prefetch_before

            per_layer[layer_idx]["layer_step_count"] += 1
            per_layer[layer_idx]["requested_expert_count"] += requested
            per_layer[layer_idx]["gpu_cache_hit_count"] += cache_hits
            per_layer[layer_idx]["prefetch_hit_count"] += prefetch_hits
            per_layer[layer_idx]["cpu_miss_load_count"] += cpu_miss

            block_totals["layer_step_count"] += 1
            block_totals["requested_expert_count"] += requested
            block_totals["gpu_cache_hit_count"] += cache_hits
            block_totals["prefetch_hit_count"] += prefetch_hits
            block_totals["cpu_miss_load_count"] += cpu_miss

            worst_layer_steps.append(
                {
                    "decode_block_idx": block["decode_block_idx"],
                    "absolute_block_idx": block["absolute_block_idx"],
                    "step_idx": step["step_idx"],
                    "step_kind": step["step_kind"],
                    "layer_idx": layer_idx,
                    "requested_expert_count": int(requested),
                    "gpu_cache_hit_count": int(cache_hits),
                    "prefetch_hit_count": int(prefetch_hits),
                    "cpu_miss_load_count": int(cpu_miss),
                    "cpu_miss_rate": safe_rate(cpu_miss, requested),
                }
            )

        requested = block_totals["requested_expert_count"]
        per_block.append(
            {
                "decode_block_idx": int(block_totals["decode_block_idx"]),
                "absolute_block_idx": int(block_totals["absolute_block_idx"]),
                "step_idx": int(block_totals["step_idx"]),
                "step_kind": block_totals["step_kind"],
                "layer_step_count": int(block_totals["layer_step_count"]),
                "requested_expert_count": int(block_totals["requested_expert_count"]),
                "gpu_cache_hit_count": int(block_totals["gpu_cache_hit_count"]),
                "prefetch_hit_count": int(block_totals["prefetch_hit_count"]),
                "cpu_miss_load_count": int(block_totals["cpu_miss_load_count"]),
                "gpu_cache_direct_hit_rate": safe_rate(block_totals["gpu_cache_hit_count"], requested),
                "combined_hit_rate": safe_rate(
                    block_totals["gpu_cache_hit_count"] + block_totals["prefetch_hit_count"],
                    requested,
                ),
                "cpu_miss_rate": safe_rate(block_totals["cpu_miss_load_count"], requested),
            }
        )

    requested_total = totals["requested_expert_count"]
    layer_stats = []
    for layer_idx, layer_total in sorted(per_layer.items()):
        layer_requested = layer_total["requested_expert_count"]
        layer_stats.append(
            {
                "layer_idx": layer_idx,
                "layer_step_count": int(layer_total["layer_step_count"]),
                "requested_expert_count": int(layer_total["requested_expert_count"]),
                "gpu_cache_hit_count": int(layer_total["gpu_cache_hit_count"]),
                "prefetch_hit_count": int(layer_total["prefetch_hit_count"]),
                "cpu_miss_load_count": int(layer_total["cpu_miss_load_count"]),
                "gpu_cache_direct_hit_rate": safe_rate(layer_total["gpu_cache_hit_count"], layer_requested),
                "combined_hit_rate": safe_rate(
                    layer_total["gpu_cache_hit_count"] + layer_total["prefetch_hit_count"],
                    layer_requested,
                ),
                "cpu_miss_rate": safe_rate(layer_total["cpu_miss_load_count"], layer_requested),
            }
        )

    worst_layer_steps.sort(
        key=lambda item: (
            item["cpu_miss_rate"],
            item["cpu_miss_load_count"],
            item["requested_expert_count"],
        ),
        reverse=True,
    )

    layer_stats_by_miss = sorted(layer_stats, key=lambda item: item["cpu_miss_rate"], reverse=True)
    layer_stats_by_combined = sorted(layer_stats, key=lambda item: item["combined_hit_rate"], reverse=True)

    return {
        "layer_step_count": int(totals["layer_step_count"]),
        "requested_expert_count": int(totals["requested_expert_count"]),
        "gpu_cache_hit_count": int(totals["gpu_cache_hit_count"]),
        "prefetch_hit_count": int(totals["prefetch_hit_count"]),
        "cpu_miss_load_count": int(totals["cpu_miss_load_count"]),
        "gpu_cache_direct_hit_rate": safe_rate(totals["gpu_cache_hit_count"], requested_total),
        "combined_hit_rate": safe_rate(
            totals["gpu_cache_hit_count"] + totals["prefetch_hit_count"],
            requested_total,
        ),
        "cpu_miss_rate": safe_rate(totals["cpu_miss_load_count"], requested_total),
        "average_requested_experts_per_layer_step": safe_rate(
            totals["requested_expert_count"], totals["layer_step_count"]
        ),
        "average_gpu_cache_hits_per_layer_step": safe_rate(
            totals["gpu_cache_hit_count"], totals["layer_step_count"]
        ),
        "average_prefetch_hits_per_layer_step": safe_rate(
            totals["prefetch_hit_count"], totals["layer_step_count"]
        ),
        "average_cpu_miss_loads_per_layer_step": safe_rate(
            totals["cpu_miss_load_count"], totals["layer_step_count"]
        ),
        "average_gpu_cache_resident_experts_before_per_layer_step": safe_rate(
            totals["gpu_cache_expert_count_before_total"], totals["layer_step_count"]
        ),
        "average_prefetch_available_experts_before_per_layer_step": safe_rate(
            totals["prefetch_available_expert_count_before_total"], totals["layer_step_count"]
        ),
        "per_step": per_block,
        "worst_layer_steps_by_cpu_miss_rate": worst_layer_steps[:10],
        "worst_layers_by_cpu_miss_rate": layer_stats_by_miss[:10],
        "best_layers_by_combined_hit_rate": layer_stats_by_combined[:10],
    }


def parse_nsys_category(text: str) -> Optional[str]:
    for pattern, category in NSYS_CATEGORY_PATTERNS:
        if pattern.fullmatch(text):
            return category
    return None


def summarize_timeline_from_nsys(sqlite_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE eventType = 59
          AND (
            text = 'SDAR_Inference_Capture'
            OR text = 'Prefill_Stage'
            OR text LIKE 'Decode_Block%'
            OR text LIKE 'Attention_Layer%'
            OR text LIKE 'Routing_Layer%'
            OR text LIKE 'Current_Layer_Availability_Check_Layer%'
            OR text LIKE 'Current_Layer_Miss_Load_Layer%'
            OR text LIKE 'Next_Layer_Prefetch_Layer%'
            OR text LIKE 'Reorder_Layer%'
            OR text LIKE 'Gather_Layer%'
            OR text LIKE 'Expert_Compute_Layer%'
            OR text LIKE 'Scatter_Layer%'
            OR text LIKE 'Cache_Promotion_Layer%'
          )
        ORDER BY start
        """
    )
    rows = cur.fetchall()
    conn.close()

    capture_total_ms = None
    prefill_total_ms = None
    step_records = []
    op_records = []

    for text, start_ns, end_ns in rows:
        if end_ns is None:
            continue
        duration_ms = (end_ns - start_ns) / 1e6
        if text == "SDAR_Inference_Capture":
            capture_total_ms = duration_ms
            continue
        if text == "Prefill_Stage":
            prefill_total_ms = duration_ms
            continue
        step_match = STEP_RE.fullmatch(text)
        if step_match:
            step_records.append(
                {
                    "decode_block_idx": int(step_match.group(1)),
                    "step_idx": int(step_match.group(2)),
                    "step_kind": step_match.group(3).lower(),
                    "start_ns": int(start_ns),
                    "end_ns": int(end_ns),
                    "total_duration_ms": duration_ms,
                    "operations": defaultdict(list),
                }
            )
            continue
        category = parse_nsys_category(text)
        if category:
            op_records.append(
                {
                    "category": category,
                    "start_ns": int(start_ns),
                    "end_ns": int(end_ns),
                    "duration_ms": duration_ms,
                }
            )

    step_records.sort(key=lambda item: item["start_ns"])
    step_idx = 0
    for op in op_records:
        while step_idx < len(step_records) and op["start_ns"] >= step_records[step_idx]["end_ns"]:
            step_idx += 1
        if step_idx >= len(step_records):
            break
        step = step_records[step_idx]
        if step["start_ns"] <= op["start_ns"] and op["end_ns"] <= step["end_ns"]:
            step["operations"][op["category"]].append(op)

    def summarize_steps(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {
            "step_count": len(steps),
            "average_step_total_ms": mean_or_none(step["total_duration_ms"] for step in steps),
            "categories": {},
        }
        for category in CATEGORY_ORDER:
            starts = []
            ends = []
            spans = []
            durations = []
            shares = []
            for step in steps:
                ops = step["operations"].get(category, [])
                if not ops:
                    continue
                first_start = min((op["start_ns"] - step["start_ns"]) / 1e6 for op in ops)
                last_end = max((op["end_ns"] - step["start_ns"]) / 1e6 for op in ops)
                duration_ms = sum(op["duration_ms"] for op in ops)
                starts.append(first_start)
                ends.append(last_end)
                spans.append(last_end - first_start)
                durations.append(duration_ms)
                shares.append(safe_rate(duration_ms, step["total_duration_ms"]) * 100.0)

            result["categories"][category] = {
                "label": CATEGORY_LABELS[category],
                "step_occurrences": len(starts),
                "average_first_start_ms": mean_or_none(starts),
                "average_last_end_ms": mean_or_none(ends),
                "average_wall_span_ms": mean_or_none(spans),
                "average_accumulated_duration_ms": mean_or_none(durations),
                "average_share_of_step_percent": mean_or_none(shares),
            }
        return result

    all_steps = step_records
    denoise_steps = [step for step in step_records if step["step_kind"] == "denoise"]
    finalize_steps = [step for step in step_records if step["step_kind"] == "finalize"]

    return {
        "capture_total_ms": capture_total_ms,
        "prefill_total_ms": prefill_total_ms,
        "all_steps": summarize_steps(all_steps),
        "denoise_only": summarize_steps(denoise_steps),
        "finalize_only": summarize_steps(finalize_steps),
    }


def build_summary(trace_obj: Dict[str, Any], results_obj: Dict[str, Any], nsys_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sample = trace_obj["samples"][0]
    all_steps = [step for _, step in iter_decode_steps(sample)]
    denoise_steps = [step for step in all_steps if step.get("step_kind") == "denoise"]
    finalize_steps = [step for step in all_steps if step.get("step_kind") == "finalize"]

    summary = {
        "inputs": {
            "model_path": results_obj.get("model_path"),
            "local_modeling_module": results_obj.get("local_modeling_module"),
            "dataset": results_obj.get("dataset", {}),
            "generation_kwargs": results_obj.get("generation_kwargs", {}),
            "offloading": results_obj.get("offloading", {}),
            "gpu": results_obj.get("gpu", {}),
        },
        "sample": {
            "sample_idx": sample.get("sample_idx"),
            "question": results_obj.get("samples", [{}])[0].get("question"),
            "processed_prediction": results_obj.get("samples", [{}])[0].get("processed_prediction"),
            "reference": results_obj.get("samples", [{}])[0].get("reference"),
            "correct": results_obj.get("evaluation", {}).get("details", [{}])[0].get("correct"),
            "generated_token_count": results_obj.get("samples", [{}])[0].get("generated_token_count"),
            "generation_latency_s": results_obj.get("samples", [{}])[0].get("latency_s"),
            "build_time_s": results_obj.get("build_time_s"),
            "prefill_total_ms": sample.get("prefill", {}).get("total_duration_ms"),
            "decode_block_count": len(sample.get("decode_blocks", [])),
            "decode_step_count_all": len(all_steps),
            "decode_step_count_denoise": len(denoise_steps),
            "decode_step_count_finalize": len(finalize_steps),
            "average_decode_step_total_ms_all": mean_or_none(step.get("total_duration_ms", 0.0) for step in all_steps),
            "average_decode_step_total_ms_denoise": mean_or_none(
                step.get("total_duration_ms", 0.0) for step in denoise_steps
            ),
            "average_decode_step_total_ms_finalize": mean_or_none(
                step.get("total_duration_ms", 0.0) for step in finalize_steps
            ),
        },
        "expert_activity": summarize_expert_activity(sample),
        "timeline_trace_cuda_events": {
            "prefill_total_ms": sample.get("prefill", {}).get("total_duration_ms"),
            "all_steps": summarize_timeline_from_trace(sample, step_kind=None),
            "denoise_only": summarize_timeline_from_trace(sample, step_kind="denoise"),
            "finalize_only": summarize_timeline_from_trace(sample, step_kind="finalize"),
        },
        "buffer_manager_results": results_obj.get("buffer_manager", {}),
        "gpu_cache_results": results_obj.get("gpu_cache", {}),
    }

    if nsys_summary is not None:
        summary["timeline_nsys_nvtx_cpu_ranges"] = nsys_summary
        summary["cross_check"] = {
            "trace_prefill_total_ms": sample.get("prefill", {}).get("total_duration_ms"),
            "nsys_prefill_total_ms": nsys_summary.get("prefill_total_ms"),
            "trace_decode_step_count_all": len(all_steps),
            "nsys_decode_step_count_all": nsys_summary.get("all_steps", {}).get("step_count"),
            "trace_average_decode_step_total_ms_all": mean_or_none(
                step.get("total_duration_ms", 0.0) for step in all_steps
            ),
            "nsys_average_decode_step_total_ms_all": nsys_summary.get("all_steps", {}).get(
                "average_step_total_ms"
            ),
            "nsys_capture_total_ms": nsys_summary.get("capture_total_ms"),
        }

    return round_if_number(summary)


def fmt_ms(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.3f} ms"


def fmt_pct(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.2f}%"


def render_markdown(summary: Dict[str, Any]) -> str:
    sample = summary["sample"]
    expert = summary["expert_activity"]
    trace_timeline = summary["timeline_trace_cuda_events"]
    nsys_timeline = summary.get("timeline_nsys_nvtx_cpu_ranges")

    def render_timeline_section(section: Dict[str, Any], title: str) -> List[str]:
        lines = [f"### {title}", "", "| 操作 | 平均开始 | 平均结束 | 平均墙钟跨度 | 平均累计时长 | 平均占 step 比例 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for category in CATEGORY_ORDER:
            item = section["categories"][category]
            lines.append(
                "| "
                + CATEGORY_LABELS[category]
                + f" | {fmt_ms(item['average_first_start_ms'])}"
                + f" | {fmt_ms(item['average_last_end_ms'])}"
                + f" | {fmt_ms(item['average_wall_span_ms'])}"
                + f" | {fmt_ms(item['average_accumulated_duration_ms'])}"
                + f" | {fmt_pct(item['average_share_of_step_percent'])} |"
            )
        lines.append("")
        return lines

    lines: List[str] = []
    lines.append("# SDAR MoE-Offloading 单样本实验分析")
    lines.append("")
    lines.append("## 样本与总体结果")
    lines.append("")
    lines.append(f"- 样本索引: {sample['sample_idx']}")
    lines.append(f"- 生成 token 数: {sample['generated_token_count']}")
    lines.append(f"- 预测结果: `{sample['processed_prediction']}`")
    lines.append(f"- 参考答案: `{sample['reference']}`")
    lines.append(f"- 正确性: `{sample['correct']}`")
    lines.append(f"- 模型构建时间: {sample['build_time_s']:.3f} s")
    lines.append(f"- 单样本生成总延迟: {sample['generation_latency_s']:.3f} s")
    lines.append(f"- prefill 总延迟: {fmt_ms(sample['prefill_total_ms'])}")
    lines.append(f"- decode block 数: {sample['decode_block_count']}")
    lines.append(f"- decode step 总数: {sample['decode_step_count_all']} (denoise={sample['decode_step_count_denoise']}, finalize={sample['decode_step_count_finalize']})")
    lines.append(f"- 平均每个 decode step 总延迟: {fmt_ms(sample['average_decode_step_total_ms_all'])}")
    lines.append(f"- 平均每个 denoise step 总延迟: {fmt_ms(sample['average_decode_step_total_ms_denoise'])}")
    lines.append(f"- 平均每个 finalize step 总延迟: {fmt_ms(sample['average_decode_step_total_ms_finalize'])}")
    lines.append("")
    lines.append("## 专家命中统计")
    lines.append("")
    lines.append(f"- layer-step 总数: {expert['layer_step_count']}")
    lines.append(f"- 请求的去重专家总数: {expert['requested_expert_count']}")
    lines.append(f"- GPU cache 直接命中数: {expert['gpu_cache_hit_count']} ({fmt_pct(expert['gpu_cache_direct_hit_rate'] * 100.0)})")
    lines.append(f"- prefetch 命中数: {expert['prefetch_hit_count']}")
    lines.append(f"- GPU cache + prefetch 总命中率: {fmt_pct(expert['combined_hit_rate'] * 100.0)}")
    lines.append(f"- 仍需 CPU miss load 的专家数: {expert['cpu_miss_load_count']} ({fmt_pct(expert['cpu_miss_rate'] * 100.0)})")
    lines.append(f"- 平均每个 layer-step 请求专家数: {expert['average_requested_experts_per_layer_step']:.3f}")
    lines.append(f"- 平均每个 layer-step 的 GPU cache 直接命中专家数: {expert['average_gpu_cache_hits_per_layer_step']:.3f}")
    lines.append(f"- 平均每个 layer-step 的 prefetch 命中专家数: {expert['average_prefetch_hits_per_layer_step']:.3f}")
    lines.append(f"- 平均每个 layer-step 的 CPU miss load 专家数: {expert['average_cpu_miss_loads_per_layer_step']:.3f}")
    lines.append(f"- 平均每个 layer-step cache 内专家数: {expert['average_gpu_cache_resident_experts_before_per_layer_step']:.3f}")
    lines.append(f"- 平均每个 layer-step prefetch buffer 内可用专家数: {expert['average_prefetch_available_experts_before_per_layer_step']:.3f}")
    lines.append("")
    lines.append("### CPU miss 最严重的 layer-step")
    lines.append("")
    lines.append("| block | abs block | step | kind | layer | requested | cache hit | prefetch hit | cpu miss | miss rate |")
    lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in expert["worst_layer_steps_by_cpu_miss_rate"][:10]:
        lines.append(
            f"| {item['decode_block_idx']} | {item['absolute_block_idx']} | {item['step_idx']} | {item['step_kind']} | {item['layer_idx']} | {item['requested_expert_count']} | {item['gpu_cache_hit_count']} | {item['prefetch_hit_count']} | {item['cpu_miss_load_count']} | {fmt_pct(item['cpu_miss_rate'] * 100.0)} |"
        )
    lines.append("")
    lines.append("### CPU miss 最严重的层")
    lines.append("")
    lines.append("| layer | requested | cache hit | prefetch hit | cpu miss | direct hit | combined hit | cpu miss rate |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in expert["worst_layers_by_cpu_miss_rate"][:10]:
        lines.append(
            f"| {item['layer_idx']} | {item['requested_expert_count']} | {item['gpu_cache_hit_count']} | {item['prefetch_hit_count']} | {item['cpu_miss_load_count']} | {fmt_pct(item['gpu_cache_direct_hit_rate'] * 100.0)} | {fmt_pct(item['combined_hit_rate'] * 100.0)} | {fmt_pct(item['cpu_miss_rate'] * 100.0)} |"
        )
    lines.append("")
    lines.append("## 时间轴")
    lines.append("")
    lines.append("下面的时间轴来自详细 trace 中记录的 CUDA event。对每个 decode step，以该 step 起点记为 0 ms，再把所有 step 的同类操作起止点取平均。")
    lines.append("这里的“平均开始/平均结束”是把一个 step 内 48 层里所有同类区间合并后，取最早开始和最晚结束得到的覆盖窗口；因此它描述的是该类操作在整步时间轴上的分布范围，不是某一个单独 op 的独占连续时长。")
    lines.append("同理，“平均累计时长”是把该类操作在 48 层上的同类区间时长累加后再取平均；由于跨层重复和跨 stream 重叠都存在，各类操作的占比不会加总为 100%。")
    lines.append("")
    lines.extend(render_timeline_section(trace_timeline["all_steps"], "所有 decode step 的平均时间轴"))
    lines.extend(render_timeline_section(trace_timeline["denoise_only"], "仅 denoise step 的平均时间轴"))
    lines.extend(render_timeline_section(trace_timeline["finalize_only"], "仅 finalize step 的平均时间轴"))

    if nsys_timeline is not None:
        lines.append("## nsys 交叉校验")
        lines.append("")
        lines.append(f"- nsys capture 总时长: {fmt_ms(nsys_timeline['capture_total_ms'])}")
        lines.append(f"- nsys prefill 总时长: {fmt_ms(nsys_timeline['prefill_total_ms'])}")
        lines.append(f"- nsys 统计到的 decode step 数: {nsys_timeline['all_steps']['step_count']}")
        lines.append(f"- nsys 平均每个 decode step 总时长: {fmt_ms(nsys_timeline['all_steps']['average_step_total_ms'])}")
        lines.append("")
        lines.append("说明: nsys 这里统计的是 NVTX push/pop 的 CPU 墙钟区间；trace 时间轴统计的是同名 NVTX 范围对应的 CUDA event 时间。对于 `next-layer prefetch` 这种异步 prefetch，CUDA event 更适合用于分析真实串并行关系。")
        lines.append("")

    lines.append("## 串并行关系解读")
    lines.append("")
    lines.append("1. `attention -> routing -> current-layer availability check -> current-layer miss load` 在同一层的主计算链上基本是严格串行的。当前层不先知道有哪些专家缺失，就无法发起 miss load。")
    lines.append("2. `cache promotion` 很短，紧跟在 miss load 之后，本质是把本层刚加载的专家登记进 GPU cache。")
    lines.append("3. `next-layer prefetch` 在当前层 miss load/cache promotion 之后启动，但它跑在单独的 prefetch stream 上，所以会和当前层后半段的 `reorder -> gather -> expert compute -> scatter` 产生明显重叠。")
    lines.append("4. `reorder -> gather -> expert compute -> scatter` 是当前层真正消费专家权重做 MoE 计算的主路径，四者对当前层输出依赖明确，整体上仍然是串行为主。")
    lines.append("5. 到下一层时，如果上一层 prefetch 成功，则下一层会在 `availability check` 后直接命中 swap buffer，减少甚至避免新的 CPU miss load。也就是说，跨层重叠主要来自“当前层 compute”和“下一层专家预取”之间，而不是两层 attention 彼此并行。")
    lines.append("6. 从平均时间轴看，`next-layer prefetch` 的结束时间通常晚于当前层 `scatter` 的结束时间，说明它经常跨过当前层的后半段，进入下一层开始之前的一小段窗口；这正是它能提高 combined hit rate 的来源。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    trace_obj = load_json(args.trace_json)
    results_obj = load_json(args.results_json)
    nsys_summary = None
    if args.nsys_sqlite:
        nsys_summary = summarize_timeline_from_nsys(args.nsys_sqlite)

    summary = build_summary(trace_obj, results_obj, nsys_summary)

    summary_output_path = Path(args.summary_json_output)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_md = render_markdown(summary)
    report_output_path = Path(args.report_md_output)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
