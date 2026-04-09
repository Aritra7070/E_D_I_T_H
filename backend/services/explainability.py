from __future__ import annotations


def attach_explainability_metadata(content_type: str, payload: dict[str, object]) -> dict[str, object]:
    details = dict(payload.get("details", {}))
    reason_labels = details.get("reason_labels", [])

    if content_type == "text":
        details["explainability"] = {
            "mode": "claim-evidence",
            "summary": "The text pipeline extracted the core claim and compared it against related source evidence.",
        }
    elif content_type == "image":
        details["explainability"] = {
            "mode": "pattern-inspection",
            "summary": "The image analysis highlights blur, noise, and edge consistency patterns that affected authenticity confidence.",
        }
    elif content_type == "video":
        details["explainability"] = {
            "mode": "frame-markers",
            "summary": "Sampled frames were scored for authenticity signals and compared for consistency.",
        }
    elif content_type == "audio":
        details["explainability"] = {
            "mode": "frequency-anomalies",
            "summary": "The spectrogram highlights audio regularity patterns that can indicate editing or synthesis.",
        }

    details["key_reasons"] = reason_labels
    payload["details"] = details
    return payload
