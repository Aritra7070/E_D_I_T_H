from __future__ import annotations

from models.base import DetectorResult, calibrate_score, clamp, status_from_score


def finalize_result(content_type: str, result: DetectorResult) -> dict[str, object]:
    base_score = calibrate_score(result.fake_score)
    final_score = base_score
    details = dict(result.details)
    status = str(details.get("status_override") or status_from_score(final_score))
    confidence = clamp((result.confidence * 0.8) + (abs(final_score - 0.5) * 0.5), 0.4, 0.99)
    details["component_scores"] = {key: round(value, 4) for key, value in result.component_scores.items()}
    details["decision_engine"] = {
        "base_module_score": round(base_score, 4),
        "thresholds": {"authentic_below": 0.34, "manipulated_above": 0.68},
        "top_signals": [name for name, _ in sorted(result.component_scores.items(), key=lambda item: item[1], reverse=True)[:3]],
    }
    details["score_band"] = status

    return {
        "type": content_type,
        "signal_score": round(final_score, 4),
        "confidence": round(confidence, 4),
        "status": status,
        "explanation": result.explanation,
        "evidence": list(details.get("reason_labels", [])),
        "details": details,
    }
