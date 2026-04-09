from __future__ import annotations

import mimetypes
from pathlib import Path

from models.audio_detector import AudioDetector
from models.image_detector import ImageDetector
from models.text_detector import TextDetector
from models.video_detector import VideoDetector


TEXT_EXTENSIONS = {".txt", ".md", ".rtf", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}


text_detector = TextDetector()
image_detector = ImageDetector()
video_detector = VideoDetector(n_frames=20)  # Change this number to analyze more/fewer frames
audio_detector = AudioDetector()


def detect_content_type(filename: str | None = None, mime_type: str | None = None, text: str | None = None) -> str:
    if text and text.strip():
        return "text"

    suffix = Path(filename or "").suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"

    guessed_mime, _ = mimetypes.guess_type(filename or "")
    mime_type = mime_type or guessed_mime or ""
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("text/"):
        return "text"

    return "unknown"


def route_to_model(content_type: str, *, text: str | None = None, file_path: str | None = None):
    if content_type == "text":
        if text and text.strip():
            return text_detector.analyze(text)
        if not file_path:
            raise ValueError("Text analysis requires inline text or a text file.")
        return text_detector.analyze(Path(file_path).read_text(encoding="utf-8", errors="ignore"))

    if not file_path:
        raise ValueError(f"{content_type.title()} analysis requires an uploaded file.")

    if content_type == "image":
        return image_detector.analyze(file_path)
    if content_type == "video":
        return video_detector.analyze(file_path)
    if content_type == "audio":
        return audio_detector.analyze(file_path)

    raise ValueError(f"Unsupported content type: {content_type}")
