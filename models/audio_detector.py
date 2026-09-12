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


def load_audio(audio_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int, str | None]:
    """
    Load audio with broad format support.
    
    Native formats (WAV, FLAC, AIFF) are loaded directly via librosa.
    Compressed formats (M4A, MP3, AAC, OGG, OPUS) require FFmpeg + pydub for transcoding.
    
    Returns (waveform, sample_rate, temp_wav_path_or_none).
    If a temporary WAV was created (for M4A/MP3 etc), its path is returned so caller can reuse it.
    Caller is responsible for cleaning up temp files.
    """
    import tempfile
    from pathlib import Path

    direct_formats = {".wav", ".flac", ".aiff", ".aif"}
    compressed_formats = {".m4a", ".mp3", ".aac", ".ogg", ".opus", ".wma", ".mp4"}

    ext = Path(audio_path).suffix.lower()

    # Try native formats first (no trans-coding needed)
    if ext in direct_formats:
        try:
            waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return waveform, sr, None  # No temp file created
        except Exception as e:
            raise RuntimeError(f"Failed to load {ext} file: {e}") from e

    # Handle compressed formats (require FFmpeg + pydub)
    if ext in compressed_formats:
        # Check if pydub is available
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "pydub not installed (required for M4A/MP3). "
                "Install: pip install pydub\n"
                "FFmpeg is also required: https://ffmpeg.org/download.html"
            )
        
        # Set up FFmpeg paths if provided via environment variables
        ffmpeg_path = os.environ.get("FFMPEG_PATH")
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            AudioSegment.ffmpeg = ffmpeg_path
            print(f"[AudioDetector] Using FFmpeg from FFMPEG_PATH: {ffmpeg_path}")
        else:
            if ffmpeg_path:
                print(f"[AudioDetector] WARNING: FFMPEG_PATH set but file not found: {ffmpeg_path}")
        
        edith_ffmpeg = os.environ.get("EDITH_FFMPEG")
        if edith_ffmpeg and os.path.isfile(edith_ffmpeg):
            AudioSegment.converter = edith_ffmpeg
            print(f"[AudioDetector] Using FFmpeg from EDITH_FFMPEG: {edith_ffmpeg}")
        else:
            if edith_ffmpeg:
                print(f"[AudioDetector] WARNING: EDITH_FFMPEG set but file not found: {edith_ffmpeg}")
        
        # Try to transcode using pydub (which will auto-detect FFmpeg on PATH)
        try:
            print(f"[AudioDetector] Attempting to transcode {ext} -> WAV via pydub+FFmpeg")
            audio_segment = AudioSegment.from_file(audio_path)
            
            # Export to temporary WAV file (CALLER will clean up)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            
            audio_segment.export(tmp_path, format="wav")
            waveform, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
            print(f"[AudioDetector] Successfully transcoded {ext}: {len(waveform)} samples @ {sr}Hz")
            return waveform, sr, tmp_path  # Return path for Whisper to reuse
        
        except FileNotFoundError as e:
            # This typically means FFmpeg is not installed/found
            install_msg = (
                f"FFmpeg not found on system PATH. Required for {ext} format transcoding.\n\n"
                "Installation instructions:\n"
                "  Windows: Download FFmpeg from https://ffmpeg.org/download.html\n"
                "           Extract to C:\\ffmpeg or set FFMPEG_PATH environment variable\n"
                "  macOS:   brew install ffmpeg\n"
                "  Linux:   sudo apt-get install ffmpeg  (Ubuntu/Debian)\n\n"
                "Then restart your terminal/IDE and try again.\n"
                f"Original error: {e}"
            )
            raise RuntimeError(install_msg) from e
        
        except RuntimeError as e:
            # Re-raise RuntimeError as-is (pydub's own errors)
            raise
        
        except Exception as e:
            error_msg = str(e)
            # Try to detect if it's an FFmpeg-related error
            if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower():
                ffmpeg_msg = (
                    f"FFmpeg error encountered during {ext} transcoding:\n{error_msg}\n\n"
                    "Troubleshooting:\n"
                    "1. Verify FFmpeg is installed: ffmpeg -version (in terminal)\n"
                    "2. Set FFMPEG_PATH environment variable if FFmpeg is not on PATH\n"
                    "3. Reinstall FFmpeg from https://ffmpeg.org/download.html"
                )
                raise RuntimeError(ffmpeg_msg) from e
            else:
                raise RuntimeError(
                    f"Audio transcoding failed for {ext}: {error_msg}"
                ) from e

    # Unsupported format
    raise RuntimeError(
        f"Unsupported audio format: {ext}\n"
        "Supported: WAV, FLAC, AIFF (native) or M4A, MP3, AAC, OGG, OPUS, WMA, MP4 (with FFmpeg)"
    )


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
    """
    Heuristic fallback for voice authenticity using spectral features.
    Handles edge cases like very short audio or silent segments.
    """
    try:
        # Guard 1: Check minimum duration (500ms = sr/2 samples @ 22050Hz)
        min_samples = max(sr // 2, 1000)  # At least 500ms or 1000 samples
        if len(waveform) < min_samples:
            print(f"[AudioDetector] Audio too short ({len(waveform)} samples, need {min_samples}). Returning neutral score.")
            return 0.5
        
        # Guard 2: Check for silence (RMS energy too low)
        rms_energy = float(np.sqrt(np.mean(waveform ** 2)))
        if rms_energy < 1e-6:
            print(f"[AudioDetector] Audio appears silent (RMS={rms_energy:.2e}). Returning neutral score.")
            return 0.5
        
        # Guard 3: Normalize waveform to prevent numerical issues
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Extract spectral features with error handling
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            if len(spectral_centroid) == 0 or np.any(np.isnan(spectral_centroid)):
                raise ValueError("Spectral centroid produced NaN values")
        except Exception as e:
            print(f"[AudioDetector] Spectral centroid extraction failed: {e}")
            return 0.5
        
        try:
            zcr = librosa.feature.zero_crossing_rate(waveform)[0]
            if len(zcr) == 0 or np.any(np.isnan(zcr)):
                raise ValueError("ZCR produced NaN values")
        except Exception as e:
            print(f"[AudioDetector] Zero-crossing rate extraction failed: {e}")
            return 0.5
        
        try:
            mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            if mfcc.size == 0 or np.any(np.isnan(mfcc)):
                raise ValueError("MFCC produced NaN values")
        except Exception as e:
            print(f"[AudioDetector] MFCC extraction failed: {e}")
            return 0.5
        
        # Compute coefficients of variation, guarding against division errors
        centroid_mean = np.mean(spectral_centroid)
        centroid_cv = float(np.std(spectral_centroid) / max(centroid_mean, 1.0))
        
        zcr_mean = np.mean(zcr)
        zcr_cv = float(np.std(zcr) / max(zcr_mean, 1e-4))
        
        mfcc_variation = float(np.mean(np.std(mfcc, axis=1)))
        
        # Guard 4: Check for NaN or Inf in computed values
        if not (np.isfinite(centroid_cv) and np.isfinite(zcr_cv) and np.isfinite(mfcc_variation)):
            print("[AudioDetector] Feature computation produced invalid values. Returning neutral score.")
            return 0.5
        
        # Compute signals with clipping
        centroid_signal = clamp(1.0 - (centroid_cv / 0.24))
        zcr_signal = clamp(1.0 - (zcr_cv / 0.55))
        mfcc_signal = clamp(1.0 - (mfcc_variation / 15.0))
        
        heuristic_score = centroid_signal * 0.4 + zcr_signal * 0.35 + mfcc_signal * 0.25
        return calibrate_score(heuristic_score)
    
    except Exception as e:
        print(f"[AudioDetector] Heuristic voice scoring failed unexpectedly: {type(e).__name__}: {e}")
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
    
    def verify_transcript_claim(self, transcript: str) -> DetectorResult | None:
        """
        Run the transcript through the full TextDetector algorithm.
        Returns the complete TextDetector result (same as if user pasted text).
        """
        if not transcript or len(transcript) < 10:
            return None
        
        try:
            text_detector = TextDetector()
            result = text_detector.analyze(transcript)
            print(f"[AudioDetector] Text analysis complete - Status: {result.details.get('status_override', 'Unknown')}")
            return result
        
        except Exception as e:
            print(f"[AudioDetector] Text analysis failed: {e}")
            return None
    
    
    def analyze(self, file_path: str) -> DetectorResult:
        """
        Analyze audio for voice authenticity and claim verification.
        Returns clean DetectorResult even on error (never raises exception).
        """
        print(f"[AudioDetector] Starting analysis for: {file_path}")
        
        try:
            return self._analyze_internal(file_path)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[AudioDetector] Analysis failed with error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Return clean error response (never 500)
            return DetectorResult(
                fake_score=0.5,
                confidence=0.35,
                explanation=f"Audio analysis encountered an error: {str(e)}",
                details={
                    "error": error_msg,
                    "model_backend": "error",
                    "status_override": "Unverified",
                    "reason_labels": [
                        "Audio processing failed",
                        "Check FFmpeg installation: https://ffmpeg.org/download.html",
                    ],
                },
                component_scores={},
            )
    
    def _analyze_internal(self, file_path: str) -> DetectorResult:
        """Internal analysis logic (may raise exceptions; wrapped by analyze())."""
        import os
        
        temp_wav_path = None  # Track temp file for cleanup
        
        try:
            # Load audio (may create a temporary WAV for compressed formats)
            try:
                waveform, sr, temp_wav_path = load_audio(file_path, target_sr=16000)
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
            
            # Transcribe audio using the converted WAV if available, otherwise original file
            transcription_path = temp_wav_path if temp_wav_path else file_path
            print("[AudioDetector] Transcribing audio...")
            transcript = self.transcribe_audio(transcription_path)
            if transcript:
                print(f"[AudioDetector] Transcript: {transcript[:100]}...")
            else:
                print("[AudioDetector] No transcript generated")
            
            # Run transcript through TextDetector (same algorithm as text input)
            text_result = None
            if transcript:
                print("[AudioDetector] Running transcript through text analysis...")
                text_result = self.verify_transcript_claim(transcript)
            
            # Prepare final result
            if text_result:
                # Use TextDetector results as base, enhanced with voice analysis
                text_status = text_result.details.get("status_override", "Suspicious")
                text_score = text_result.fake_score
                text_confidence = text_result.confidence
                text_explanation = text_result.explanation
                text_evidence = text_result.details.get("reason_labels", [])
                
                # Combine voice + text scores (60% text credibility + 40% voice authenticity)
                combined_score = calibrate_score(text_score * 0.6 + voice_score * 0.4)
                combined_confidence = clamp((text_confidence * 0.7 + text_confidence * 0.3), 0.4, 0.95)
                
                # Build combined explanation
                combined_explanation = (
                    f"Audio contains: {transcript[:80]}... "
                    f"[Voice: {'natural' if voice_score < 0.35 else 'AI-like' if voice_score > 0.65 else 'mixed'}] "
                    f"{text_explanation}"
                )
                
                # Build combined reason labels
                reason_labels = list(text_evidence)  # Use all text evidence
                if voice_score > 0.65:
                    reason_labels.insert(0, "Voice patterns suggest AI generation")
                if voice_score < 0.35:
                    reason_labels.insert(0, "Voice patterns appear natural")
                
                final_status = text_status  # Use text status, not voice status
                final_score = combined_score
                final_confidence = combined_confidence
                final_explanation = combined_explanation
                
                component_scores = {
                    "voice_authenticity_score": voice_score,
                    "text_credibility_score": text_score,
                }
            else:
                # Fallback: only voice analysis (no transcript)
                final_score = calibrate_score(voice_score)
                final_confidence = clamp(0.5 + abs(voice_score - 0.5) * 0.9, 0.4, 0.95)
                final_explanation = (
                    f"Audio analysis: Voice patterns "
                    f"{'suggest AI-generated speech' if voice_score > 0.65 else 'appear natural' if voice_score < 0.35 else 'show mixed signals'}. "
                    f"Unable to verify spoken claims without transcription."
                )
                reason_labels = ["Transcription unavailable - voice analysis only"]
                final_status = "Unverified"
                component_scores = {
                    "voice_authenticity_score": voice_score,
                }
            
            print(f"[AudioDetector] Final score: {final_score:.4f}, Status: {final_status}")
            print(f"[AudioDetector] Analysis complete")
            
            model_backend = f"voice:{voice_method} + whisper-small + TextDetector (DDGS)"

            # Build details dictionary for frontend
            details = {
                "model_backend": model_backend,
                "status_override": final_status,
                "reason_labels": reason_labels[:6],
                "transcript": transcript,
                "spectrogram_image": build_spectrogram_image(waveform, sr),
                "voice_authenticity_score": voice_score,
                "voice_detection_method": voice_method,
            }
            if text_result:
                details["claim_verification_score"] = text_result.fake_score
                details["claim_verification_status"] = text_result.details.get("status_override", "Suspicious")
            
            return DetectorResult(
                fake_score=final_score,
                confidence=final_confidence,
                explanation=final_explanation,
                details=details,
                component_scores=component_scores,
            )
        
        finally:
            # Clean up temporary WAV file if it was created
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                    print(f"[AudioDetector] Cleaned up temp file: {temp_wav_path}")
                except Exception as e:
                    print(f"[AudioDetector] Failed to clean up temp file: {e}")
