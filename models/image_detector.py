from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from models.base import DetectorResult, array_to_data_url, calibrate_score, clamp, normalize


class ImageDetector:
    def __init__(self) -> None:
        self.model_backend = "heuristic-fallback"
        self._model = None
        self._model_loaded = False
        self._load_error: str | None = None

    def _load_model(self):
        """Lazy-load CNN model. Falls back to heuristics-only mode on failure."""
        if self._model_loaded:
            return self._model

        self._model_loaded = True
        try:
            from huggingface_hub import hf_hub_download
            import torch

            model_path = hf_hub_download("Medsa/ai-image-authenticity-detector", "detector_scripted.pt")
            self._model = torch.jit.load(model_path, map_location="cpu")
            self._model.eval()
            self.model_backend = "cnn:Medsa/ai-image-authenticity-detector"
            self._load_error = None
            print("[ImageDetector] CNN model loaded from HuggingFace Hub")
        except Exception as exc:
            self._model = None
            self.model_backend = "heuristic-fallback"
            self._load_error = str(exc)
            print(f"[ImageDetector] CNN model unavailable ({exc}) - using heuristics only")
        return self._model

    def _cnn_inference(self, image: Image.Image, heuristic_score: float) -> tuple[float, str]:
        model = self._load_model()
        if model is None:
            return calibrate_score(heuristic_score), "fallback"

        import torch

        resized = image.convert("RGB").resize((32, 32))
        array = np.asarray(resized).astype("float32") / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)

        with torch.no_grad():
            logit, _ = model(tensor)

        raw_score = float(torch.sigmoid(logit).item())
        if raw_score < 0.25:
            return calibrate_score(max((heuristic_score * 0.9) + 0.08, raw_score)), "active"
        return calibrate_score(raw_score), "active"

    def _score_components(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("float32")

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian_variance = float(np.var(laplacian))
        blur_signal = clamp((1200.0 - laplacian_variance) / 1200.0)

        denoised = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        residual = gray - denoised
        noise_variance = float(np.var(residual))
        low_noise_signal = clamp((140.0 - noise_variance) / 140.0)

        edge_map = cv2.Canny(gray.astype("uint8"), 80, 180).astype("float32")
        edge_sharpness = float(np.mean(edge_map))
        edge_softness_signal = clamp((8.0 - edge_sharpness) / 8.0)

        return rgb, bgr, {
            "blur_signal": blur_signal,
            "low_noise_signal": low_noise_signal,
            "edge_softness_signal": edge_softness_signal,
            "laplacian_variance": laplacian_variance,
            "noise_variance": noise_variance,
            "edge_sharpness": edge_sharpness,
        }

    def score_rgb_array(self, rgb_array: np.ndarray) -> dict[str, object]:
        image = Image.fromarray(np.clip(rgb_array, 0, 255).astype("uint8"))
        _, _, signals = self._score_components(image)
        heuristic_score = calibrate_score(
            (signals["blur_signal"] * 0.4) + (signals["low_noise_signal"] * 0.35) + (signals["edge_softness_signal"] * 0.25)
        )
        model_score, cnn_status = self._cnn_inference(image, heuristic_score)
        final_score = calibrate_score((model_score * 0.5) + (heuristic_score * 0.5))
        return {
            "fake_score": final_score,
            "confidence": clamp(0.64 + min(abs(final_score - 0.5), 0.3) + (0.05 if cnn_status == "active" else 0.0)),
            "reason_labels": self._reason_labels(signals),
            "component_scores": {
                "model_score": model_score,
                "blur_signal": signals["blur_signal"],
                "low_noise_signal": signals["low_noise_signal"],
                "edge_softness_signal": signals["edge_softness_signal"],
            },
        }

    def _reason_labels(self, signals: dict[str, float]) -> list[str]:
        reasons = []
        if signals["blur_signal"] > 0.45:
            reasons.append("High smoothness detected")
        if signals["low_noise_signal"] > 0.45:
            reasons.append("Low noise variation")
        if signals["edge_softness_signal"] > 0.4:
            reasons.append("Soft edge sharpness detected")
        if not reasons:
            reasons.append("Natural image texture detected")
        unique_reasons = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                unique_reasons.append(reason)
                seen.add(reason)
        return unique_reasons

    def _build_explanation(self, score: float, reason_labels: list[str]) -> str:
        triggered_conditions = "; ".join(reason_labels)
        if score > 0.75:
            return f"Triggered conditions: {triggered_conditions}. These signals raise manipulation concern."
        if score < 0.35:
            return f"Triggered conditions: {triggered_conditions}. The image retains stronger authenticity cues overall."
        return f"Triggered conditions: {triggered_conditions}. The evidence is mixed, so the image remains suspicious."

    def analyze(self, file_path: str) -> DetectorResult:
        source_image = Image.open(file_path).convert("RGB")
        rgb, bgr, signals = self._score_components(source_image)

        heuristic_score = calibrate_score(
            (signals["blur_signal"] * 0.4) + (signals["low_noise_signal"] * 0.35) + (signals["edge_softness_signal"] * 0.25)
        )
        model_score, cnn_status = self._cnn_inference(source_image, heuristic_score)
        fake_score = calibrate_score((model_score * 0.5) + (heuristic_score * 0.5))

        if signals["blur_signal"] > 0.65 and signals["low_noise_signal"] > 0.45 and signals["edge_softness_signal"] > 0.45:
            fake_score = max(fake_score, 0.78)
        if signals["blur_signal"] < 0.25 and signals["low_noise_signal"] < 0.2 and signals["edge_softness_signal"] < 0.25:
            fake_score = min(fake_score, 0.22)

        reason_labels = self._reason_labels(signals)
        explanation = self._build_explanation(fake_score, reason_labels)

        edge_map = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 80, 180).astype("float32")
        blur_map = normalize(np.abs(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
        heatmap = normalize((blur_map * 0.65) + ((1.0 - normalize(edge_map)) * 0.35))
        heatmap_uint8 = (heatmap * 255).astype("uint8")
        colorized = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(bgr, 0.72, colorized, 0.28, 0.0)

        confidence = clamp(0.66 + min(abs(fake_score - 0.5), 0.3) + (0.05 if cnn_status == "active" else 0.0))

        return DetectorResult(
            fake_score=fake_score,
            confidence=confidence,
            explanation=explanation,
            details={
                "model_backend": self.model_backend,
                "cnn_status": cnn_status,
                "cnn_load_error": self._load_error,
                "heatmap_image": array_to_data_url(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
                "image_size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
                "reason_labels": reason_labels[:4],
            },
            component_scores={
                "model_score": model_score,
                "blur_signal": signals["blur_signal"],
                "low_noise_signal": signals["low_noise_signal"],
                "edge_softness_signal": signals["edge_softness_signal"],
            },
        )
