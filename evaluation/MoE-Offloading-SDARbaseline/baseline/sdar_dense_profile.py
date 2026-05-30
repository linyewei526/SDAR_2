"""Runtime profiling helpers for pure SDAR dense-GPU baseline runs."""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch


_ENABLED = False
_OUTPUT_PATH: Optional[Path] = None
_STATE: Dict[str, Any] = {
    "metadata": {},
    "samples": [],
    "current_sample": None,
    "current_layer": None,
}

_CATEGORIES = [
    "attention",
    "routing",
    "moe_dispatch",
    "moe_expert_compute_hbm_fetch",
    "moe_scatter",
]


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def enable_profile(output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    global _ENABLED, _OUTPUT_PATH, _STATE
    _ENABLED = True
    _OUTPUT_PATH = Path(output_path)
    _STATE = {
        "metadata": metadata or {},
        "samples": [],
        "current_sample": None,
        "current_layer": None,
    }


def disable_profile() -> None:
    global _ENABLED, _OUTPUT_PATH, _STATE
    _ENABLED = False
    _OUTPUT_PATH = None
    _STATE = {
        "metadata": {},
        "samples": [],
        "current_sample": None,
        "current_layer": None,
    }


def is_profile_enabled() -> bool:
    return _ENABLED


def begin_sample(sample_idx: int, metadata: Optional[Dict[str, Any]] = None) -> None:
    if not _ENABLED:
        return

    _STATE["current_sample"] = {
        "sample_idx": sample_idx,
        "metadata": metadata or {},
        "_layers": [],
    }
    _STATE["current_layer"] = None


def end_sample() -> None:
    if not _ENABLED:
        return

    sample = _STATE.get("current_sample")
    if sample is None:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sample["summary"] = _summarize_sample(sample)
    sample.pop("_layers", None)
    _STATE["samples"].append(sample)
    _STATE["current_sample"] = None
    _STATE["current_layer"] = None


@contextmanager
def layer_context(layer_idx: int, enabled: bool):
    sample = _STATE.get("current_sample")
    if not (_ENABLED and enabled and sample is not None and torch.cuda.is_available()):
        yield
        return

    previous = _STATE.get("current_layer")
    layer_record = {
        "layer_idx": layer_idx,
        "_start_event": torch.cuda.Event(enable_timing=True),
        "_end_event": torch.cuda.Event(enable_timing=True),
        "intervals": [],
    }
    layer_record["_start_event"].record(torch.cuda.current_stream())
    _STATE["current_layer"] = layer_record

    try:
        yield
    finally:
        layer_record["_end_event"].record(torch.cuda.current_stream())
        sample["_layers"].append(layer_record)
        _STATE["current_layer"] = previous


@contextmanager
def profile_range(category: str):
    current_layer = _STATE.get("current_layer")
    if not (
        _ENABLED
        and current_layer is not None
        and torch.cuda.is_available()
        and category in _CATEGORIES
    ):
        yield
        return

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream())
    try:
        yield
    finally:
        end_event.record(torch.cuda.current_stream())
        current_layer["intervals"].append(
            {
                "category": category,
                "_start_event": start_event,
                "_end_event": end_event,
            }
        )


def maybe_profile_range(category: str):
    if not _ENABLED:
        return nullcontext()
    return profile_range(category)


def _make_operation_state() -> Dict[str, Dict[str, float]]:
    return {
        category: {
            "occurrence_count": 0,
            "sum_start_ms": 0.0,
            "sum_end_ms": 0.0,
            "sum_wall_span_ms": 0.0,
            "sum_duration_ms": 0.0,
            "sum_share_percent": 0.0,
        }
        for category in _CATEGORIES
    }


def _summarize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    layers = sample.get("_layers", [])
    layer_count = 0
    total_layer_ms = 0.0
    operation_states = _make_operation_state()
    other_sum_ms = 0.0
    other_share_sum = 0.0

    for layer in layers:
        layer_start = layer["_start_event"]
        layer_end = layer["_end_event"]
        layer_total_ms = layer_start.elapsed_time(layer_end)
        if layer_total_ms <= 0:
            continue

        layer_count += 1
        total_layer_ms += layer_total_ms
        summed_known_duration_ms = 0.0

        category_to_intervals: Dict[str, list] = {}
        for interval in layer.get("intervals", []):
            category = interval["category"]
            start_ms = layer_start.elapsed_time(interval["_start_event"])
            end_ms = layer_start.elapsed_time(interval["_end_event"])
            duration_ms = interval["_start_event"].elapsed_time(interval["_end_event"])
            category_to_intervals.setdefault(category, []).append(
                {
                    "start_ms": float(start_ms),
                    "end_ms": float(end_ms),
                    "duration_ms": float(duration_ms),
                }
            )

        for category, intervals in category_to_intervals.items():
            first_start_ms = min(item["start_ms"] for item in intervals)
            last_end_ms = max(item["end_ms"] for item in intervals)
            total_duration_ms = sum(item["duration_ms"] for item in intervals)
            state = operation_states[category]
            state["occurrence_count"] += 1
            state["sum_start_ms"] += first_start_ms
            state["sum_end_ms"] += last_end_ms
            state["sum_wall_span_ms"] += last_end_ms - first_start_ms
            state["sum_duration_ms"] += total_duration_ms
            state["sum_share_percent"] += _safe_div(total_duration_ms, layer_total_ms) * 100.0
            summed_known_duration_ms += total_duration_ms

        other_ms = max(0.0, layer_total_ms - summed_known_duration_ms)
        other_sum_ms += other_ms
        other_share_sum += _safe_div(other_ms, layer_total_ms) * 100.0

    operations = {}
    for category, state in operation_states.items():
        count = state["occurrence_count"]
        operations[category] = {
            "occurrence_layer_count": count,
            "occurrence_ratio": _round(_safe_div(count, layer_count)),
            "average_start_ms": _round(_safe_div(state["sum_start_ms"], count)),
            "average_end_ms": _round(_safe_div(state["sum_end_ms"], count)),
            "average_wall_span_ms": _round(_safe_div(state["sum_wall_span_ms"], count)),
            "average_duration_ms": _round(_safe_div(state["sum_duration_ms"], count)),
            "average_share_of_layer_percent": _round(
                _safe_div(state["sum_share_percent"], count)
            ),
        }

    operations["other_layer_overhead"] = {
        "occurrence_layer_count": layer_count,
        "occurrence_ratio": _round(_safe_div(layer_count, layer_count)),
        "average_start_ms": None,
        "average_end_ms": None,
        "average_wall_span_ms": None,
        "average_duration_ms": _round(_safe_div(other_sum_ms, layer_count)),
        "average_share_of_layer_percent": _round(_safe_div(other_share_sum, layer_count)),
    }

    return {
        "recorded_denoise_layer_count": layer_count,
        "average_decode_layer_total_ms": _round(_safe_div(total_layer_ms, layer_count)),
        "operations": operations,
    }


def _aggregate_samples(samples) -> Optional[Dict[str, Any]]:
    if not samples:
        return None

    total_layers = 0
    total_layer_ms = 0.0
    op_state = _make_operation_state()
    other_occurrence = 0
    other_sum_ms = 0.0
    other_sum_share = 0.0

    for sample in samples:
        summary = sample.get("summary")
        if not summary:
            continue
        total_layers += int(summary["recorded_denoise_layer_count"])
        total_layer_ms += float(summary["average_decode_layer_total_ms"] or 0.0) * int(
            summary["recorded_denoise_layer_count"]
        )

    if total_layers == 0:
        return None

    for sample in samples:
        raw_layers = sample.get("_layers")
        if raw_layers:
            raise RuntimeError("Unexpected raw layer data present during aggregation.")

    # Re-aggregate from per-sample summaries weighted by occurrence count.
    for sample in samples:
        summary = sample.get("summary")
        if not summary:
            continue
        for category, category_summary in summary["operations"].items():
            if category == "other_layer_overhead":
                count = int(category_summary["occurrence_layer_count"])
                other_occurrence += count
                other_sum_ms += float(category_summary["average_duration_ms"] or 0.0) * count
                other_sum_share += float(
                    category_summary["average_share_of_layer_percent"] or 0.0
                ) * count
                continue

            state = op_state[category]
            count = int(category_summary["occurrence_layer_count"])
            state["occurrence_count"] += count
            state["sum_start_ms"] += float(category_summary["average_start_ms"] or 0.0) * count
            state["sum_end_ms"] += float(category_summary["average_end_ms"] or 0.0) * count
            state["sum_wall_span_ms"] += float(
                category_summary["average_wall_span_ms"] or 0.0
            ) * count
            state["sum_duration_ms"] += float(
                category_summary["average_duration_ms"] or 0.0
            ) * count
            state["sum_share_percent"] += float(
                category_summary["average_share_of_layer_percent"] or 0.0
            ) * count

    operations = {}
    for category, state in op_state.items():
        count = state["occurrence_count"]
        operations[category] = {
            "occurrence_layer_count": count,
            "occurrence_ratio": _round(_safe_div(count, total_layers)),
            "average_start_ms": _round(_safe_div(state["sum_start_ms"], count)),
            "average_end_ms": _round(_safe_div(state["sum_end_ms"], count)),
            "average_wall_span_ms": _round(_safe_div(state["sum_wall_span_ms"], count)),
            "average_duration_ms": _round(_safe_div(state["sum_duration_ms"], count)),
            "average_share_of_layer_percent": _round(
                _safe_div(state["sum_share_percent"], count)
            ),
        }

    operations["other_layer_overhead"] = {
        "occurrence_layer_count": other_occurrence,
        "occurrence_ratio": _round(_safe_div(other_occurrence, total_layers)),
        "average_start_ms": None,
        "average_end_ms": None,
        "average_wall_span_ms": None,
        "average_duration_ms": _round(_safe_div(other_sum_ms, other_occurrence)),
        "average_share_of_layer_percent": _round(
            _safe_div(other_sum_share, other_occurrence)
        ),
    }

    return {
        "recorded_denoise_layer_count": total_layers,
        "average_decode_layer_total_ms": _round(_safe_div(total_layer_ms, total_layers)),
        "operations": operations,
    }


def export_profile(output_path: Optional[str] = None) -> Optional[Path]:
    if not _ENABLED:
        return None

    target_path = Path(output_path) if output_path else _OUTPUT_PATH
    if target_path is None:
        raise ValueError("Profile output path is not set.")

    payload = {
        "metadata": _STATE.get("metadata", {}),
        "samples": [],
        "aggregate": {},
    }
    for sample in _STATE.get("samples", []):
        payload["samples"].append(
            {
                "sample_idx": sample["sample_idx"],
                "metadata": sample.get("metadata", {}),
                "latency_summary": sample.get("summary"),
            }
        )

    aggregate_summary = _aggregate_samples(_STATE.get("samples", []))
    if aggregate_summary is not None:
        payload["aggregate"]["latency_summary"] = aggregate_summary

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target_path
