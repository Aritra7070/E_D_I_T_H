from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from models.base import DetectorResult, array_to_data_url, calibrate_score, clamp

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[VideoDetector] Using device: {DEVICE}")

# Module-level model cache
_video_extractor = None
_video_model = None
_model_loaded = False
_load_error = None


def _load_face_cascade():
    """Load Haar cascade; return None if unavailable (graceful skip)."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("[VideoDetector] haarcascade_frontalface_default.xml not found - skipping face crop")
        return None
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[VideoDetector] Failed to load haarcascade - skipping face crop")
        return None
    return cascade


def _load_model() -> None:
    """Lazy load the deepfake detection model."""
    global _video_extractor, _video_model, _model_loaded, _load_error
    
    if _model_loaded or _load_error:
        return
    
    try:
        print("[VideoDetector] Loading prithivMLmods deepfake detector...")
        _video_extractor = AutoFeatureExtractor.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model",
            trust_remote_code=False
        )
        _video_model = AutoModelForImageClassification.from_pretrained(
            "prithivMLmods/Deep-Fake-Detector-Model",
            trust_remote_code=False
        )
        _video_model = _video_model.to(DEVICE)
        _video_model.eval()
        _model_loaded = True
        print(f"[VideoDetector] Model loaded successfully on device: {DEVICE}")
    except Exception as e:
        _load_error = str(e)
        print(f"[VideoDetector] Model load failed, will use heuristics: {type(e).__name__}: {e}")


def extract_frames(video_path: str, n_frames: int = 8) -> list[np.ndarray]:
    """Extract n_frames evenly spaced frames from video as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot read video file: {video_path}")
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError("Video has no frames or cannot determine frame count")
        
        # Evenly spaced indices
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        return frames
    
    finally:
        cap.release()


def detect_scene_changes(frames: list[np.ndarray], threshold: float = 0.25) -> list[int]:
    """Detect frame indices where scene changes occur (cuts)."""
    scene_changes = []
    
    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)
        
        # Mean absolute difference normalized by 255
        diff = np.mean(np.abs(frame1 - frame2)) / 255.0
        
        if diff > threshold:
            scene_changes.append(i + 1)
    
    return scene_changes


def get_face_crop(frame_rgb: np.ndarray, face_cascade) -> np.ndarray:
    """Detect and crop the largest face in frame, or return full frame if no face found."""
    if face_cascade is None:
        return frame_rgb
    
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
    
    if len(faces) == 0:
        return frame_rgb
    
    # Get largest face by area
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    return frame_rgb[y : y + h, x : x + w]


def score_frame_with_model(frame_rgb: np.ndarray, extractor, model) -> float:
    """Score a frame using the ML model."""
    if model is None or extractor is None:
        return score_frame_heuristic(frame_rgb)
    
    try:
        # Resize frame to standard size for model
        frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        frame_pil = Image.fromarray(frame_resized.astype("uint8"))
        
        # Feature extraction
        inputs = extractor(frame_pil, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        # Class indices: typically 0=real, 1=fake (verify with model output_labels if available)
        fake_prob = float(probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item())
        
        return calibrate_score(fake_prob)
    
    except Exception as e:
        print(f"[VideoDetector] Model inference failed: {type(e).__name__}: {e}")
        return score_frame_heuristic(frame_rgb)


def score_frame_heuristic(frame_rgb: np.ndarray) -> float:
    """Heuristic-based per-frame authenticity score using blur and edge detection."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Laplacian variance (blur detection) - higher = sharper
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = float(np.var(laplacian))
    # Very blurry frames (var < 100) -> high fake score
    blur_signal = clamp(1.0 - (laplacian_var / 400.0))
    
    # Edge detection for consistency
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32)
    edge_density = float(np.mean(edges) / 255.0)
    # Very few edges (processed) -> high fake score
    edge_signal = clamp(1.0 - (edge_density * 3.0))
    
    # Sobel X and Y for motion artifacts
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    motion_intensity = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    # Very low motion intensity -> high fake score
    motion_signal = clamp(1.0 - (motion_intensity / 50.0))
    
    heuristic_score = (blur_signal * 0.4) + (edge_signal * 0.35) + (motion_signal * 0.25)
    return calibrate_score(heuristic_score)


def temporal_consistency_score(frames: list[np.ndarray]) -> float:
    """Compute temporal consistency score. High score = jittery/suspicious."""
    if len(frames) < 2:
        return 0.0
    
    differences = []
    
    for i in range(len(frames) - 1):
        frame1 = cv2.resize(frames[i], (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        frame2 = cv2.resize(frames[i + 1], (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        
        diff = np.mean(np.abs(frame1 - frame2)) / 255.0
        differences.append(diff)
    
    if not differences:
        return 0.0
    
    std_dev = float(np.std(differences))
    return clamp((std_dev - 0.05) / 0.15)


def build_frame_preview(frame_rgb: np.ndarray, suspicious: bool) -> str:
    preview = cv2.resize(frame_rgb, (320, 180), interpolation=cv2.INTER_AREA)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    if suspicious:
        cv2.rectangle(preview_bgr, (4, 4), (preview_bgr.shape[1] - 4, preview_bgr.shape[0] - 4), (36, 39, 211), 3)
    return array_to_data_url(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB))


class VideoDetector:
    """Detects deepfakes in video using frame-level ML model + temporal analysis."""
    
    def __init__(self, n_frames: int = 8) -> None:
        self.n_frames = n_frames
        self.extractor = None
        self.model = None
        self.model_loaded = False
        self.face_cascade = _load_face_cascade()
        self._load_model()
    
    def _load_model(self) -> None:
        global _video_extractor, _video_model, _model_loaded
        _load_model()
        if _model_loaded:
            self.extractor = _video_extractor
            self.model = _video_model
            self.model_loaded = True
    
    def analyze(self, file_path: str) -> DetectorResult:
        """Analyze video for deepfake or manipulation indicators."""
        print(f"[VideoDetector] Analyzing video: {file_path}")
        
        # Extract frames
        frames = extract_frames(file_path, n_frames=self.n_frames)
        print(f"[VideoDetector] Extracted {len(frames)} frames (configured: {self.n_frames})")
        
        # Detect scene changes
        scene_changes = detect_scene_changes(frames)
        print(f"[VideoDetector] Detected {len(scene_changes)} scene changes")
        
        # Score each frame
        frame_scores = []
        
        for i, frame_rgb in enumerate(frames):
            face_crop = get_face_crop(frame_rgb, self.face_cascade)
            score = score_frame_with_model(face_crop, self.extractor, self.model)
            frame_scores.append(score)
            print(f"[VideoDetector] Frame {i}: {score:.4f}")
        
        # Average frame score
        avg_frame_score = float(np.mean(frame_scores))
        print(f"[VideoDetector] Average frame score: {avg_frame_score:.4f}")
        
        # Temporal consistency
        temporal_score = temporal_consistency_score(frames)
        print(f"[VideoDetector] Temporal score: {temporal_score:.4f}")
        
        # Final composite score (increase model weight if ML model is loaded)
        if self.model_loaded:
            final_score = calibrate_score(
                (avg_frame_score * 0.85) + (temporal_score * 0.15)
            )
        else:
            final_score = calibrate_score(
                (avg_frame_score * 0.70) + (temporal_score * 0.30)
            )
        
        print(f"[VideoDetector] Final score: {final_score:.4f} (model_loaded={self.model_loaded})")
        
        suspicious_frame_indices = [index for index, score in enumerate(frame_scores) if score >= 0.68]
        frame_previews = [
            {"index": index, "image": build_frame_preview(frame, index in suspicious_frame_indices)}
            for index, frame in enumerate(frames)
        ]
        suspicious_frames = [
            {"frame_index": index, "frame_preview": frame_previews[index]["image"], "score": round(frame_scores[index], 4)}
            for index in suspicious_frame_indices
        ]

        # Build reason labels
        reason_labels = []
        if avg_frame_score > 0.65:
            reason_labels.append("Frame-level manipulation detected")
        if temporal_score > 0.5:
            reason_labels.append("Temporal inconsistency detected")
        if scene_changes:
            reason_labels.append(f"{len(scene_changes)} suspicious scene transition(s) found")
        if avg_frame_score < 0.3 and temporal_score < 0.2:
            reason_labels.append("Video appears temporally consistent")
        
        if not reason_labels:
            reason_labels.append("Mixed temporal and frame-level signals")
        
        # Component scores
        component_scores = {
            "avg_frame_score": avg_frame_score,
            "temporal_consistency": temporal_score,
            "scene_changes_count": float(len(scene_changes)),
        }
        
        # Confidence
        confidence = clamp(0.55 + abs(final_score - 0.5) * 0.8, 0.4, 0.95)
        
        # Explanation
        if final_score >= 0.68:
            explanation = (
                "Frame analysis detected strong manipulation signals. "
                "Temporal patterns show inconsistencies typical of deepfakes."
            )
        elif final_score < 0.34:
            explanation = (
                "Frames show natural visual characteristics and smooth temporal flow. "
                "Video appears consistent with authentic recording."
            )
        else:
            explanation = (
                "Frame-level and temporal analysis show mixed evidence. "
                "Some signals suggest potential editing, but evidence is inconclusive."
            )
        
        model_backend = (
            "prithivMLmods/Deep-Fake-Detector-Model" 
            if self.model_loaded 
            else "heuristic-fallback"
        )
        
        return DetectorResult(
            fake_score=final_score,
            confidence=confidence,
            explanation=explanation,
            details={
                "model_backend": model_backend,
                "frames_analyzed": len(frames),
                "scene_changes": scene_changes,
                "sampled_frame_indices": list(range(len(frames))),
                "suspicious_frame_indices": suspicious_frame_indices,
                "frame_previews": frame_previews,
                "suspicious_frames": suspicious_frames,
                "status_override": None,
                "reason_labels": reason_labels,
            },
            component_scores=component_scores,
        )
