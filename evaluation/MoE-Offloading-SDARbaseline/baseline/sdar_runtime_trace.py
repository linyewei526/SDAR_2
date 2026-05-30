"""Lightweight per-sample summary recording for SDAR MoE-Offloading runs."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch


_TRACE_ENABLED = False
_TRACE_OUTPUT_PATH: Optional[Path] = None
_STATE: Dict[str, Any] = {
    "metadata": {},
    "record_experts": False,
    "record_latency": False,
    "samples": [],
    "current_sample": None,
    "current_context": None,
}

_CATEGORY_PREFIXES = {
    "Attention_": "attention",
    "Routing_": "routing",
    "Current_Layer_Availability_Check_": "current_layer_availability_check",
    "Current_Layer_Miss_Load_": "current_layer_miss_load",
    "Next_Layer_Prefetch_": "next_layer_prefetch",
    "Reorder_": "reorder",
    "Gather_": "gather",
    "Expert_Compute_": "expert_compute",
    "Scatter_": "scatter",
    "Cache_Promotion_": "cache_promotion",
}

_LAYER_SUFFIX_RE = re.compile(r"Layer(\d+)$")


def _make_expert_state() -> Dict[str, Any]:
    return {
        "decode_layer_count": 0,
        "total_active_expert_count": 0,
        "total_gpu_cache_hit_count": 0,
        "total_prefetch_hit_count": 0,
        "total_cpu_miss_load_count": 0,
        "total_prefetch_available_count": 0,
        "total_gpu_cache_replacement_count": 0,
    }


def _make_latency_state() -> Dict[str, Any]:
    return {
        "denoise_layer_count": 0,
        "total_layer_duration_ms": 0.0,
        "operations": {
            category: {
                "occurrence_count": 0,
                "sum_start_ms": 0.0,
                "sum_end_ms": 0.0,
                "sum_wall_span_ms": 0.0,
                "sum_duration_ms": 0.0,
                "sum_share_percent": 0.0,
            }
            for category in _CATEGORY_PREFIXES.values()
        },
    }


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def enable_trace(
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    record_experts: bool = False,
    record_latency: bool = False,
) -> None:
    global _TRACE_ENABLED, _TRACE_OUTPUT_PATH, _STATE

    _TRACE_ENABLED = bool(record_experts or record_latency)
    _TRACE_OUTPUT_PATH = Path(output_path)
    _STATE = {
        "metadata": metadata or {},
        "record_experts": bool(record_experts),
        "record_latency": bool(record_latency),
        "samples": [],
        "current_sample": None,
        "current_context": None,
    }


def disable_trace() -> None:
    global _TRACE_ENABLED, _TRACE_OUTPUT_PATH, _STATE

    _TRACE_ENABLED = False
    _TRACE_OUTPUT_PATH = None
    _STATE = {
        "metadata": {},
        "record_experts": False,
        "record_latency": False,
        "samples": [],
        "current_sample": None,
        "current_context": None,
    }


def is_trace_enabled() -> bool:
    return _TRACE_ENABLED


def is_expert_recording_enabled() -> bool:
    return _TRACE_ENABLED and bool(_STATE.get("record_experts"))


def is_latency_recording_enabled() -> bool:
    return _TRACE_ENABLED and bool(_STATE.get("record_latency"))


def begin_sample(
    sample_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    record_this_sample: bool = True,
) -> None:
    if not _TRACE_ENABLED or not record_this_sample:
        _STATE["current_sample"] = None
        _STATE["current_context"] = None
        return

    _STATE["current_sample"] = {
        "sample_idx": sample_idx,
        "metadata": metadata or {},
        "expert_summary": _make_expert_state() if is_expert_recording_enabled() else None,
        "latency_summary": _make_latency_state() if is_latency_recording_enabled() else None,
        "_latency_steps_raw": [] if is_latency_recording_enabled() else None,
    }
    _STATE["current_context"] = None


def end_sample() -> None:
    if not _TRACE_ENABLED:
        return

    sample = _STATE.get("current_sample")
    if sample is None:
        _STATE["current_context"] = None
        return

    if (
        is_latency_recording_enabled()
        and sample.get("_latency_steps_raw")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
        _finalize_latency_summary(sample)

    _STATE["samples"].append(sample)
    _STATE["current_sample"] = None
    _STATE["current_context"] = None


def _normalize_message(message: str) -> Optional[Dict[str, Any]]:
    for prefix, category in _CATEGORY_PREFIXES.items():
        if not message.startswith(prefix):
            continue
        layer_match = _LAYER_SUFFIX_RE.search(message)
        layer_idx = int(layer_match.group(1)) if layer_match else None
        return {
            "category": category,
            "layer_idx": layer_idx,
            "message": message,
        }
    return None


@contextmanager
def phase_context(
    phase: str,
    *,
    decode_block_idx: Optional[int] = None,
    absolute_block_idx: Optional[int] = None,
    step_idx: Optional[int] = None,
    step_kind: str,
):
    sample = _STATE.get("current_sample")
    if not _TRACE_ENABLED or sample is None:
        yield
        return

    previous_context = _STATE.get("current_context")
    context: Dict[str, Any] = {
        "phase": phase,
        "decode_block_idx": decode_block_idx,
        "absolute_block_idx": absolute_block_idx,
        "step_idx": step_idx,
        "step_kind": step_kind,
    }

    if (
        sample.get("latency_summary") is not None
        and phase == "decode"
        and step_kind == "denoise"
        and torch.cuda.is_available()
    ):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream())
        context["_latency_step"] = {
            "decode_block_idx": decode_block_idx,
            "absolute_block_idx": absolute_block_idx,
            "step_idx": step_idx,
            "step_kind": step_kind,
            "start_event": start_event,
            "timing_intervals": [],
        }

    _STATE["current_context"] = context

    try:
        yield
    finally:
        latency_step = context.get("_latency_step")
        if latency_step is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
            latency_step["end_event"] = end_event
            sample["_latency_steps_raw"].append(latency_step)

        _STATE["current_context"] = previous_context


def record_timing_range(message: str, start_event, end_event) -> None:
    if not is_latency_recording_enabled():
        return

    sample = _STATE.get("current_sample")
    current_context = _STATE.get("current_context")
    if sample is None or current_context is None:
        return

    latency_step = current_context.get("_latency_step")
    if latency_step is None:
        return

    normalized = _normalize_message(message)
    if normalized is None:
        return

    latency_step["timing_intervals"].append(
        {
            **normalized,
            "_start_event": start_event,
            "_end_event": end_event,
        }
    )


def record_layer_activity(
    layer_idx: int,
    active_expert_count: int,
    load_trace: Optional[Dict[str, Any]],
) -> None:
    if not is_expert_recording_enabled():
        return

    sample = _STATE.get("current_sample")
    current_context = _STATE.get("current_context")
    if sample is None or current_context is None or current_context.get("phase") != "decode":
        return

    expert_summary = sample.get("expert_summary")
    if expert_summary is None:
        return

    load_trace = load_trace or {}
    requested_expert_count = int(
        load_trace.get("requested_expert_count", active_expert_count)
    )
    expert_summary["decode_layer_count"] += 1
    expert_summary["total_active_expert_count"] += requested_expert_count
    expert_summary["total_gpu_cache_hit_count"] += int(
        load_trace.get("gpu_cache_hit_count", 0)
    )
    expert_summary["total_prefetch_hit_count"] += int(
        load_trace.get("prefetch_hit_count", 0)
    )
    expert_summary["total_cpu_miss_load_count"] += int(
        load_trace.get("cpu_miss_load_count", 0)
    )
    expert_summary["total_prefetch_available_count"] += int(
        load_trace.get("prefetch_available_expert_count_before", 0)
    )
    expert_summary["total_gpu_cache_replacement_count"] += int(
        load_trace.get("gpu_cache_replacement_count", 0)
    )


def _finalize_latency_summary(sample: Dict[str, Any]) -> None:
    latency_summary = sample.get("latency_summary")
    raw_steps = sample.get("_latency_steps_raw") or []
    if latency_summary is None or not raw_steps:
        sample["_latency_steps_raw"] = []
        return

    for step in raw_steps:
        step_start_event = step["start_event"]
        intervals_by_layer: Dict[int, list] = {}

        for raw_interval in step["timing_intervals"]:
            layer_idx = raw_interval.get("layer_idx")
            if layer_idx is None:
                continue

            interval_start_event = raw_interval["_start_event"]
            interval_end_event = raw_interval["_end_event"]
            start_ms = step_start_event.elapsed_time(interval_start_event)
            end_ms = step_start_event.elapsed_time(interval_end_event)
            duration_ms = interval_start_event.elapsed_time(interval_end_event)

            intervals_by_layer.setdefault(layer_idx, []).append(
                {
                    "category": raw_interval["category"],
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "duration_ms": float(duration_ms),
                }
            )

        for layer_intervals in intervals_by_layer.values():
            layer_start_ms = min(item["start_ms"] for item in layer_intervals)
            layer_end_ms = max(item["end_ms"] for item in layer_intervals)
            layer_total_ms = layer_end_ms - layer_start_ms
            if layer_total_ms <= 0:
                continue

            latency_summary["denoise_layer_count"] += 1
            latency_summary["total_layer_duration_ms"] += layer_total_ms

            category_to_intervals: Dict[str, list] = {}
            for item in layer_intervals:
                category_to_intervals.setdefault(item["category"], []).append(item)

            for category, category_state in latency_summary["operations"].items():
                category_intervals = category_to_intervals.get(category)
                if not category_intervals:
                    continue

                first_start_ms = (
                    min(item["start_ms"] for item in category_intervals) - layer_start_ms
                )
                last_end_ms = (
                    max(item["end_ms"] for item in category_intervals) - layer_start_ms
                )
                total_duration_ms = sum(
                    item["duration_ms"] for item in category_intervals
                )

                category_state["occurrence_count"] += 1
                category_state["sum_start_ms"] += first_start_ms
                category_state["sum_end_ms"] += last_end_ms
                category_state["sum_wall_span_ms"] += last_end_ms - first_start_ms
                category_state["sum_duration_ms"] += total_duration_ms
                category_state["sum_share_percent"] += (
                    _safe_div(total_duration_ms, layer_total_ms) * 100.0
                )

    sample["_latency_steps_raw"] = []


def _serialize_expert_summary(expert_summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if expert_summary is None:
        return None

    decode_layer_count = expert_summary["decode_layer_count"]
    total_active = expert_summary["total_active_expert_count"]
    total_gpu_cache_hits = expert_summary["total_gpu_cache_hit_count"]
    total_prefetch_hits = expert_summary["total_prefetch_hit_count"]
    total_cpu_miss_loads = expert_summary["total_cpu_miss_load_count"]
    total_prefetch_available = expert_summary["total_prefetch_available_count"]
    total_gpu_cache_replacements = expert_summary["total_gpu_cache_replacement_count"]

    return {
        "recorded_decode_layer_count": decode_layer_count,
        "average_active_unique_experts_per_decode_layer": _round(
            _safe_div(total_active, decode_layer_count)
        ),
        "average_gpu_cache_hits_per_decode_layer": _round(
            _safe_div(total_gpu_cache_hits, decode_layer_count)
        ),
        "average_prefetch_hits_per_decode_layer": _round(
            _safe_div(total_prefetch_hits, decode_layer_count)
        ),
        "average_cpu_miss_loads_per_decode_layer": _round(
            _safe_div(total_cpu_miss_loads, decode_layer_count)
        ),
        "average_prefetch_available_experts_before_miss_load_per_decode_layer": _round(
            _safe_div(total_prefetch_available, decode_layer_count)
        ),
        "average_gpu_cache_replacements_per_decode_layer": _round(
            _safe_div(total_gpu_cache_replacements, decode_layer_count)
        ),
        "gpu_cache_hit_ratio": _round(_safe_div(total_gpu_cache_hits, total_active)),
        "prefetch_hit_ratio": _round(_safe_div(total_prefetch_hits, total_active)),
        "cpu_miss_load_ratio": _round(_safe_div(total_cpu_miss_loads, total_active)),
    }


def _serialize_latency_summary(
    latency_summary: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if latency_summary is None:
        return None

    denoise_layer_count = latency_summary["denoise_layer_count"]
    operations = {}
    for category, category_state in latency_summary["operations"].items():
        occurrence_count = category_state["occurrence_count"]
        operations[category] = {
            "occurrence_layer_count": occurrence_count,
            "occurrence_ratio": _round(_safe_div(occurrence_count, denoise_layer_count)),
            "average_start_ms": _round(
                _safe_div(category_state["sum_start_ms"], occurrence_count)
            ),
            "average_end_ms": _round(
                _safe_div(category_state["sum_end_ms"], occurrence_count)
            ),
            "average_wall_span_ms": _round(
                _safe_div(category_state["sum_wall_span_ms"], occurrence_count)
            ),
            "average_duration_ms": _round(
                _safe_div(category_state["sum_duration_ms"], occurrence_count)
            ),
            "average_share_of_layer_percent": _round(
                _safe_div(category_state["sum_share_percent"], occurrence_count)
            ),
        }

    return {
        "recorded_denoise_layer_count": denoise_layer_count,
        "average_decode_layer_total_ms": _round(
            _safe_div(latency_summary["total_layer_duration_ms"], denoise_layer_count)
        ),
        "operations": operations,
    }


def _serialize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"sample_idx": sample["sample_idx"]}
    if sample.get("metadata"):
        payload["metadata"] = sample["metadata"]

    expert_summary = _serialize_expert_summary(sample.get("expert_summary"))
    if expert_summary is not None:
        payload["expert_summary"] = expert_summary

    latency_summary = _serialize_latency_summary(sample.get("latency_summary"))
    if latency_summary is not None:
        payload["latency_summary"] = latency_summary

    return payload


def _aggregate_expert_summary(samples) -> Optional[Dict[str, Any]]:
    total_layers = 0
    total_active = 0
    total_gpu_cache_hits = 0
    total_prefetch_hits = 0
    total_cpu_miss_loads = 0
    total_prefetch_available = 0
    total_gpu_cache_replacements = 0

    for sample in samples:
        expert_summary = sample.get("expert_summary")
        if expert_summary is None:
            continue
        total_layers += expert_summary["decode_layer_count"]
        total_active += expert_summary["total_active_expert_count"]
        total_gpu_cache_hits += expert_summary["total_gpu_cache_hit_count"]
        total_prefetch_hits += expert_summary["total_prefetch_hit_count"]
        total_cpu_miss_loads += expert_summary["total_cpu_miss_load_count"]
        total_prefetch_available += expert_summary["total_prefetch_available_count"]
        total_gpu_cache_replacements += expert_summary[
            "total_gpu_cache_replacement_count"
        ]

    if total_layers == 0:
        return None

    return {
        "recorded_decode_layer_count": total_layers,
        "average_active_unique_experts_per_decode_layer": _round(
            _safe_div(total_active, total_layers)
        ),
        "average_gpu_cache_hits_per_decode_layer": _round(
            _safe_div(total_gpu_cache_hits, total_layers)
        ),
        "average_prefetch_hits_per_decode_layer": _round(
            _safe_div(total_prefetch_hits, total_layers)
        ),
        "average_cpu_miss_loads_per_decode_layer": _round(
            _safe_div(total_cpu_miss_loads, total_layers)
        ),
        "average_prefetch_available_experts_before_miss_load_per_decode_layer": _round(
            _safe_div(total_prefetch_available, total_layers)
        ),
        "average_gpu_cache_replacements_per_decode_layer": _round(
            _safe_div(total_gpu_cache_replacements, total_layers)
        ),
        "gpu_cache_hit_ratio": _round(_safe_div(total_gpu_cache_hits, total_active)),
        "prefetch_hit_ratio": _round(_safe_div(total_prefetch_hits, total_active)),
        "cpu_miss_load_ratio": _round(_safe_div(total_cpu_miss_loads, total_active)),
    }


def _aggregate_latency_summary(samples) -> Optional[Dict[str, Any]]:
    total_layers = 0
    total_layer_duration_ms = 0.0
    aggregated_ops = _make_latency_state()["operations"]

    for sample in samples:
        latency_summary = sample.get("latency_summary")
        if latency_summary is None:
            continue
        total_layers += latency_summary["denoise_layer_count"]
        total_layer_duration_ms += latency_summary["total_layer_duration_ms"]

        for category, category_state in latency_summary["operations"].items():
            aggregate_state = aggregated_ops[category]
            aggregate_state["occurrence_count"] += category_state["occurrence_count"]
            aggregate_state["sum_start_ms"] += category_state["sum_start_ms"]
            aggregate_state["sum_end_ms"] += category_state["sum_end_ms"]
            aggregate_state["sum_wall_span_ms"] += category_state["sum_wall_span_ms"]
            aggregate_state["sum_duration_ms"] += category_state["sum_duration_ms"]
            aggregate_state["sum_share_percent"] += category_state["sum_share_percent"]

    if total_layers == 0:
        return None

    operations = {}
    for category, category_state in aggregated_ops.items():
        occurrence_count = category_state["occurrence_count"]
        operations[category] = {
            "occurrence_layer_count": occurrence_count,
            "occurrence_ratio": _round(_safe_div(occurrence_count, total_layers)),
            "average_start_ms": _round(
                _safe_div(category_state["sum_start_ms"], occurrence_count)
            ),
            "average_end_ms": _round(
                _safe_div(category_state["sum_end_ms"], occurrence_count)
            ),
            "average_wall_span_ms": _round(
                _safe_div(category_state["sum_wall_span_ms"], occurrence_count)
            ),
            "average_duration_ms": _round(
                _safe_div(category_state["sum_duration_ms"], occurrence_count)
            ),
            "average_share_of_layer_percent": _round(
                _safe_div(category_state["sum_share_percent"], occurrence_count)
            ),
        }

    return {
        "recorded_denoise_layer_count": total_layers,
        "average_decode_layer_total_ms": _round(
            _safe_div(total_layer_duration_ms, total_layers)
        ),
        "operations": operations,
    }


def export_trace(output_path: Optional[str] = None) -> Optional[Path]:
    if not _TRACE_ENABLED:
        return None

    target_path = Path(output_path) if output_path else _TRACE_OUTPUT_PATH
    if target_path is None:
        raise ValueError("Trace output path is not set.")

    samples = _STATE.get("samples", [])
    payload: Dict[str, Any] = {
        "metadata": _STATE.get("metadata", {}),
        "recording": {
            "record_experts": bool(_STATE.get("record_experts")),
            "record_latency": bool(_STATE.get("record_latency")),
            "recorded_sample_count": len(samples),
        },
        "samples": [_serialize_sample(sample) for sample in samples],
        "aggregate": {},
    }

    aggregate_expert = _aggregate_expert_summary(samples)
    if aggregate_expert is not None:
        payload["aggregate"]["expert_summary"] = aggregate_expert

    aggregate_latency = _aggregate_latency_summary(samples)
    if aggregate_latency is not None:
        payload["aggregate"]["latency_summary"] = aggregate_latency

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return target_path
