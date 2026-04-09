# EDITH

EDITH is a multi-modal authenticity analysis MVP. It accepts text, images, audio, and video, detects the content type automatically, routes the input to the matching analysis module, and returns a calibrated concern score, confidence score, status, explanation, and modality-specific evidence.

## Stack

- Backend: FastAPI
- Frontend: React + Vite
- Detection modules: live-search text fact checking, CNN-assisted image analysis, frame-averaged video analysis, and audio heuristics
- Explainability: text claim extraction plus source evidence, image heatmap overlay, suspicious video frames, audio spectrogram anomalies

## Project Structure

```text
backend/
models/
frontend/
samples/
```

## Backend Setup

Use a 64-bit Python 3.11 or 3.12 environment for the backend. The current numerical stack used by OpenCV, librosa, and matplotlib is not dependable on 32-bit Python 3.14 because prebuilt wheels are generally unavailable there.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

Optional Hugging Face cache acceleration:

```bash
pip install -r backend/requirements-optional.txt
```

Run the API from the project root:

```bash
uvicorn backend.main:app --reload
```

## Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

The frontend defaults to `http://localhost:8000`.

## API

### `POST /analyze`

Accepts either:

- `multipart/form-data` with `file`
- `multipart/form-data` with `text`
- `application/json` with `{ "text": "..." }`

Response shape:

```json
{
  "type": "text",
  "signal_score": 0.72,
  "confidence": 0.84,
  "status": "Suspicious",
  "explanation": "EDITH found mixed corroboration and some manipulative language cues.",
  "evidence": [
    "Only one trusted source was relevant",
    "Clickbait language detected"
  ],
  "details": {
    "claim": "..."
  }
}
```

## Notes

- The backend is intentionally modular. `detect_content_type()` and `route_to_model()` are isolated in `backend/services/content_router.py`.
- The text detector now extracts a core claim, queries DuckDuckGo search for the top 5 live results, and combines match quality with clickbait and emotional-language heuristics.
- The image detector uses a lightweight CNN path plus blur, noise variance, and edge consistency heuristics to decide whether the image shows authentic or manipulated signals.
- The video detector extracts 5 frames, scores each frame with the image detector, and averages those frame scores while preserving suspicious frame indices.
- Score bands are calibrated as: `score >= 0.68 -> Manipulated`, `score >= 0.34 -> Suspicious`, otherwise `Authentic Signals`.
- The news module uses explicit status overrides: `Authentic Signals`, `Suspicious`, or `Unverified`.
