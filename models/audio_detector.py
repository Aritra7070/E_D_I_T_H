from __future__ import annotations

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    pipeline,
)

from models.base import DetectorResult, calibrate_score, clamp, pil_image_to_data_url
from models.text_detector import TextDetector

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AudioDetector] Using device: {DEVICE}")

# Module-level model cache
_voice_extractor = None
_voice_model = None
_whisper_pipe = None
_voice_model_loaded = False
_whisper_loaded = False
_voice_load_error = None
_whisper_load_error = None


def _load_voice_model() -> None:
    """Lazy load the voice deepfake detection model."""
    global _voice_extractor, _voice_model, _voice_model_loaded, _voice_load_error
    
    if _voice_model_loaded or _voice_load_error:
        return
    
    try:
        print("[AudioDetector] Loading voice deepfake model...")
        _voice_extractor = AutoFeatureExtractor.from_pretrained(
            "mo-thecreator/deepfake-audio-detection-model",
            trust_remote_code=False
        )
        _voice_model = AutoModelForAudioClassification.from_pretrained(
            "mo-thecreator/deepfake-audio-detection-model",
            trust_remote_code=False
        )
        _voice_model = _voice_model.to(DEVICE)
        _voice_model.eval()
        _voice_model_loaded = True
        print("[AudioDetector] Voice model loaded successfully on device:", DEVICE)
    except Exception as e:
        _voice_load_error = str(e)
        print(f"[AudioDetector] Voice model load failed (will use heuristic): {type(e).__name__}: {e}")


def _load_whisper() -> None:
    """Lazy load the Whisper model for transcription."""
    global _whisper_pipe, _whisper_loaded, _whisper_load_error
    
    if _whisper_loaded or _whisper_load_error:
        return
    
    try:
        print("[AudioDetector] Loading Whisper model...")
        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=DEVICE,
            batch_size=1,
            chunk_length_s=30
        )
        _whisper_loaded = True
        print("[AudioDetector] Whisper loaded successfully on device:", DEVICE)
    except Exception as e:
        _whisper_load_error = str(e)
        print(f"[AudioDetector] Whisper load failed (will skip transcription): {type(e).__name__}: {e}")


def load_audio(audio_path: str, target_sr: int = 22050):
    """
    Load audio with broad format support.
    Compressed formats are transcoded to a temporary WAV file via pydub+ffmpeg.
    WAV and FLAC-style formats are loaded directly by soundfile when available.
    """
    import tempfile
    from pathlib import Path

    direct_formats = {".wav", ".flac", ".aiff", ".aif"}
    compressed_formats = {".m4a", ".mp3", ".aac", ".ogg", ".opus", ".wma", ".mp4"}

    ext = Path(audio_path).suffix.lower()

    if ext in direct_formats:
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform, sr

    if ext in compressed_formats:
        try:
            from pydub import AudioSegment

            ffmpeg_override = os.environ.get("EDITH_FFMPEG")
            if ffmpeg_override:
                AudioSegment.converter = ffmpeg_override

            print(f"[AudioDetector] Transcoding {ext} -> WAV via pydub")
            audio_segment = AudioSegment.from_file(audio_path)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_segment.export(tmp_path, format="wav")
            try:
                waveform, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
            finally:
                os.unlink(tmp_path)
            return waveform, sr
        except ImportError as exc:
            raise RuntimeError(
                "pydub is not installed. Run: pip install pydub\nFFmpeg must also be on your system PATH."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Audio conversion failed for {ext} format: {exc}\n"
                "Ensure FFmpeg is installed and on your system PATH.\n"
                "Download: https://ffmpeg.org/download.html"
            ) from exc

    try:
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform, sr
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load audio file with extension '{ext}'. Supported formats: WAV, FLAC, M4A, MP3, AAC, OGG, OPUS. Original error: {exc}"
        ) from exc


def build_spectrogram_image(waveform: np.ndarray, sr: int) -> str | None:
    try:
        import io
        from PIL import Image

        fig, ax = plt.subplots(figsize=(8, 3))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        plot = librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set(title="Audio Spectrogram")
        fig.colorbar(plot, ax=ax, format="%+2.0f dB")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return pil_image_to_data_url(Image.open(buffer).convert("RGB"))
    except Exception as exc:
        print(f"[AudioDetector] Spectrogram generation failed: {exc}")
        return None


def score_voice_authenticity_heuristic(waveform: np.ndarray, sr: int) -> float:
    """Heuristic fallback for voice authenticity using spectral features."""
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        
        centroid_cv = float(np.std(spectral_centroid) / max(np.mean(spectral_centroid), 1.0))
        zcr_cv = float(np.std(zcr) / max(np.mean(zcr), 1e-4))
        mfcc_variation = float(np.mean(np.std(mfcc, axis=1)))
        
        centroid_signal = clamp(1.0 - (centroid_cv / 0.24))
        zcr_signal = clamp(1.0 - (zcr_cv / 0.55))
        mfcc_signal = clamp(1.0 - (mfcc_variation / 15.0))
        
        return calibrate_score(
            centroid_signal * 0.4 + zcr_signal * 0.35 + mfcc_signal * 0.25
        )
    except Exception as e:
        print(f"[AudioDetector] Heuristic voice scoring failed: {e}")
        return 0.5


class AudioDetector:
    """Detects synthetic/AI-generated audio using voice authenticity + claim verification."""
    
    def __init__(self, use_ml_models: bool = True) -> None:
        self.use_ml_models = use_ml_models
        self.voice_extractor = None
        self.voice_model = None
        self.whisper_pipe = None
        self.voice_model_loaded = False
        self.whisper_loaded = False
        
        if use_ml_models:
            self._load_models()
        else:
            print("[AudioDetector] ML models disabled - using heuristics only")
    
    def _load_models(self) -> None:
        """Load both voice model and Whisper."""
        _load_voice_model()
        _load_whisper()
        
        global _voice_extractor, _voice_model, _voice_model_loaded, _whisper_pipe, _whisper_loaded
        
        if _voice_model_loaded:
            self.voice_extractor = _voice_extractor
            self.voice_model = _voice_model
            self.voice_model_loaded = True
        
        if _whisper_loaded:
            self.whisper_pipe = _whisper_pipe
            self.whisper_loaded = True
    
    def score_voice_authenticity(self, waveform: np.ndarray, sr: int) -> tuple[float, str]:
        """Score voice authenticity using ML model or heuristic fallback."""
        if not self.voice_model_loaded or self.voice_model is None:
            return score_voice_authenticity_heuristic(waveform, sr), "heuristic"
        
        try:
            # Resample to 16000 Hz if needed
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Normalize waveform to [-1, 1]
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            # Feature extraction
            inputs = self.voice_extractor(
                waveform, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.voice_model(**inputs)
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Assume class 1 is fake, class 0 is real
            fake_prob = float(probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item())
            
            return calibrate_score(fake_prob), "model"
        
        except Exception as e:
            print(f"[AudioDetector] Voice model inference failed: {type(e).__name__}: {e}")
            return score_voice_authenticity_heuristic(waveform, sr), "heuristic"
    
    def transcribe_audio(self, audio_path: str) -> str | None:
        """Transcribe audio using Whisper."""
        if not self.whisper_loaded or self.whisper_pipe is None:
            print("[AudioDetector] Whisper not available, skipping transcription")
            return None
        
        try:
            print(f"[AudioDetector] Transcribing audio: {audio_path}")
            result = self.whisper_pipe(audio_path, generate_kwargs={"language": "en"})
            transcript = result.get("text", "").strip() if result else None
            
            if transcript:
                print(f"[AudioDetector] Transcription successful, length: {len(transcript)} chars")
                return transcript
            else:
                print("[AudioDetector] Transcription returned empty result")
                return None
                
        except Exception as e:
            print(f"[AudioDetector] Transcription failed: {type(e).__name__}: {e}")
            return None
    
    def verify_transcript_claim(self, transcript: str) -> tuple[float, list[str]]:
        """Verify transcript claims using TextDetector (DDGS search)."""
        if not transcript or len(transcript) < 10:
            return 0.5, ["Transcript too short to verify"]
        
        try:
            text_detector = TextDetector()
            result = text_detector.analyze(transcript)
            
            # Extract score and reason labels from the DetectorResult
            claim_score = result.fake_score
            claim_reasons = result.details.get("reason_labels", [])[:2]
            
            return claim_score, claim_reasons
        
        except Exception as e:
            print(f"[AudioDetector] Claim verification failed: {e}")
            return 0.5, ["Claim verification unavailable"]
    
    def analyze(self, file_path: str) -> DetectorResult:
        """Analyze audio for voice authenticity and claim verification."""
        print(f"[AudioDetector] Starting analysis for: {file_path}")
        
        try:
            # Load audio
            try:
                waveform, sr = load_audio(file_path, target_sr=16000)
                print(f"[AudioDetector] Audio loaded: {len(waveform)} samples @ {sr}Hz")
            except RuntimeError as exc:
                return DetectorResult(
                    fake_score=0.5,
                    confidence=0.4,
                    explanation=f"Audio could not be loaded: {str(exc)}",
                    details={
                        "error": str(exc),
                        "model_backend": "unavailable",
                        "reason_labels": ["Audio format not supported or FFmpeg not available"],
                    },
                    component_scores={},
                )
            
            if waveform.size == 0:
                raise ValueError("Audio file is empty or cannot be loaded")
            
            # Score voice authenticity
            print("[AudioDetector] Scoring voice authenticity...")
            voice_score, voice_method = self.score_voice_authenticity(waveform, sr)
            print(f"[AudioDetector] Voice score: {voice_score:.4f} (method: {voice_method})")
            
            # Transcribe audio
            print("[AudioDetector] Transcribing audio...")
            transcript = self.transcribe_audio(file_path)
            if transcript:
                print(f"[AudioDetector] Transcript: {transcript[:100]}...")
            else:
                print("[AudioDetector] No transcript generated")
            
            # Verify transcript claims if available
            claim_score = None
            claim_reasons = []
            
            if transcript:
                print("[AudioDetector] Verifying transcript claims...")
                claim_score, claim_reasons = self.verify_transcript_claim(transcript)
                print(f"[AudioDetector] Claim score: {claim_score:.4f}")
                final_score = calibrate_score(voice_score * 0.5 + claim_score * 0.5)
            else:
                final_score = calibrate_score(voice_score)
                claim_reasons = ["Speech-to-text transcription unavailable"]
            
            print(f"[AudioDetector] Final score: {final_score:.4f}")
        
            # Build component scores
            component_scores = {
                "voice_authenticity_score": voice_score,
                "voice_detection_method": voice_method,
            }
            if claim_score is not None:
                component_scores["claim_verification_score"] = claim_score
            
            # Build reason labels
            reason_labels = []
            
            if voice_score > 0.65:
                reason_labels.append("Voice patterns suggest AI-generated speech")
            if voice_score < 0.35:
                reason_labels.append("Voice patterns appear natural and human")
            
            if transcript:
                truncated = transcript[:120] if len(transcript) > 120 else transcript
                reason_labels.append(f'Transcript: "{truncated}..."')
            
            reason_labels.extend(claim_reasons)
            
            if claim_score is not None:
                if claim_score > 0.6 and transcript:
                    reason_labels.append("Spoken claims not corroborated by credible sources")
                elif claim_score < 0.35 and transcript:
                    reason_labels.append("Spoken claims appear consistent with credible sources")
            
            # Build explanation
            if final_score >= 0.68:
                explanation = (
                    "Voice analysis detects characteristics consistent with AI-generated or heavily processed audio. "
                    + (f"Spoken claims: not corroborated." if claim_score and claim_score > 0.6 else "")
                )
            elif final_score < 0.34:
                explanation = (
                    "Voice patterns and spectral features are consistent with natural human speech. "
                    + (f"Spoken claims appear credible." if claim_score and claim_score < 0.35 else "")
                )
            else:
                explanation = (
                    "Audio shows mixed signals. Some features resemble synthetic speech, but evidence is uncertain."
                )
            
            # Confidence
            confidence = clamp(0.5 + abs(final_score - 0.5) * 0.9, 0.4, 0.95)
            
            model_backend = f"voice:{voice_method} + whisper-small + DDGS"
            
            # Build details dictionary for frontend
            details = {
                "model_backend": model_backend,
                "status_override": None,
                "reason_labels": reason_labels[:6],
                "transcript": transcript,
                "spectrogram_image": build_spectrogram_image(waveform, sr),
                "voice_authenticity_score": voice_score,
                "voice_detection_method": voice_method,
            }
            
            if claim_score is not None:
                details["claim_verification_score"] = claim_score
            
            print("[AudioDetector] Analysis complete")
            return DetectorResult(
                fake_score=final_score,
                confidence=confidence,
                explanation=explanation,
                details=details,
                component_scores=component_scores,
            )
        
        except Exception as e:
            print(f"[AudioDetector] CRITICAL ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
