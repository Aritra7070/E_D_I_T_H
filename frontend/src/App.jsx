import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const CONTENT_TYPE_CONFIG = {
  text: {
    label: "Text",
    accept: ".txt,.md",
  },
  image: {
    label: "Image",
    accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp",
  },
  video: {
    label: "Video",
    accept: "video/*,.mp4,.mov,.avi,.mkv,.webm",
  },
  audio: {
    label: "Audio",
    accept: "audio/*,.m4a,.mp3,.aac,.ogg,.wav,.flac,.opus,.wma",
  },
};

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function statusTone(status) {
  if (status === "Manipulated") return "manipulated";
  if (status === "Unverified") return "unverified";
  if (status === "Suspicious") return "suspicious";
  return "authentic";
}

function renderHighlightedText(text, highlights) {
  if (!text) return null;
  const sorted = [...(highlights ?? [])].sort((a, b) => a.start - b.start);
  const nodes = [];
  let cursor = 0;

  sorted.forEach((highlight, index) => {
    if (highlight.start > cursor) {
      nodes.push(<span key={`plain-${index}-${cursor}`}>{text.slice(cursor, highlight.start)}</span>);
    }

    nodes.push(
      <mark key={`mark-${index}`} title={highlight.reason}>
        {text.slice(highlight.start, highlight.end)}
      </mark>,
    );
    cursor = highlight.end;
  });

  if (cursor < text.length) {
    nodes.push(<span key={`tail-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return <p className="highlighted-copy">{nodes}</p>;
}

function DetailMetric({ label, value, tone = "neutral" }) {
  return (
    <div className={`detail-metric detail-metric--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvidenceSection({ evidence, tone }) {
  if (!evidence?.length) return null;
  return (
    <section className={`result-section explanation-box explanation-box--${tone}`}>
      <div className="section-heading">
        <h3>Evidence</h3>
        <span>{evidence.length} findings</span>
      </div>
      <div className="chip-row">
        {evidence.map((item) => (
          <span key={item} className={`chip chip--${tone}`}>
            {item}
          </span>
        ))}
      </div>
    </section>
  );
}

function MediaPanel({ result }) {
  const details = result?.details ?? {};
  const type = result?.type;

  if (type === "text") {
    return (
      <>
        <section className="result-section">
          <div className="section-heading">
            <h3>Key Claim</h3>
            <span>Extracted from input</span>
          </div>
          <p className="claim-copy">{details.claim || "No claim extracted."}</p>
        </section>

        <section className="result-section">
          <div className="section-heading">
            <h3>Language Cues</h3>
            <span>{details.highlights?.length ?? 0} highlighted terms</span>
          </div>
          {renderHighlightedText(details.text, details.highlights)}
        </section>

        <section className="result-section">
          <div className="section-heading">
            <h3>Related Sources</h3>
            <span>{details.related_articles?.length ?? 0} matches</span>
          </div>
          <div className="source-list">
            {(details.related_articles ?? []).map((article) => (
              <article key={article.id ?? article.href ?? article.title} className="source-card">
                <div className="source-head">
                  <strong>{article.title}</strong>
                  <span>{article.source}</span>
                </div>
                <p>{article.summary}</p>
                <div className="chip-row">
                  <span className="chip chip--muted">trust: {Math.round((article.trust ?? 0) * 100)}%</span>
                  <span className="chip chip--muted">match: {Math.round((article.match_score ?? 0) * 100)}%</span>
                </div>
              </article>
            ))}
            {(!details.related_articles || details.related_articles.length === 0) && (
              <div className="empty-inline">No live search results were returned for this claim.</div>
            )}
          </div>
        </section>
      </>
    );
  }

  if (type === "image" && details.heatmap_image) {
    return (
      <section className="result-section viz-panel">
        <div className="section-heading">
          <h3>Authenticity Heatmap</h3>
          <span>Pattern inspection</span>
        </div>
        <img className="media-preview" src={details.heatmap_image} alt="Image analysis heatmap" />
        <p className="viz-caption">Redder areas reflect suspicious blur or texture smoothing.</p>
      </section>
    );
  }

  if (type === "video") {
    return (
      <>
        <section className="result-section">
          <div className="section-heading">
            <h3>Frame Evidence</h3>
            <span>{details.sampled_frame_indices?.length ?? 0} sampled frames</span>
          </div>
          <div className="chip-row">
            {(details.suspicious_frame_indices ?? []).map((frameIndex) => (
              <span key={frameIndex} className="chip">
                Frame {frameIndex}
              </span>
            ))}
            {(!details.suspicious_frame_indices || details.suspicious_frame_indices.length === 0) && (
              <span className="chip chip--muted">No strong frame anomalies</span>
            )}
          </div>
          <div className="preview-grid">
            {(details.frame_previews ?? []).map((frame) => (
              <figure key={frame.index} className="preview-card">
                <img src={frame.image} alt={`Frame ${frame.index}`} />
                <figcaption>Frame {frame.index}</figcaption>
              </figure>
            ))}
          </div>
        </section>

        {details.suspicious_frames?.length > 0 && (
          <section className="result-section viz-panel">
            <div className="section-heading">
              <h3>Suspicious Frames</h3>
              <span>{details.suspicious_frames.length} flagged</span>
            </div>
            <div className="viz-grid">
              {details.suspicious_frames.map((frame) => (
                <img key={frame.frame_index} src={frame.frame_preview} alt={`Frame ${frame.frame_index}`} className="viz-thumb" />
              ))}
            </div>
          </section>
        )}
      </>
    );
  }

  if (type === "audio") {
    return (
      <>
        {details.spectrogram_image && (
          <section className="result-section viz-panel">
            <div className="section-heading">
              <h3>Frequency Anomaly View</h3>
              <span>Spectrogram</span>
            </div>
            <img className="media-preview" src={details.spectrogram_image} alt="Audio spectrogram" />
            <p className="viz-caption">Uniform vertical striping can indicate synthetic periodicity.</p>
          </section>
        )}

        {details.transcript && (
          <section className="result-section">
            <div className="section-heading">
              <h3>Speech Transcript</h3>
              <span>Extracted by Whisper</span>
            </div>
            <div className="transcript-box">
              <p>{details.transcript}</p>
            </div>
          </section>
        )}

        {details.voice_authenticity_score !== undefined && details.claim_verification_score !== undefined && (
          <section className="result-section">
            <div className="section-heading">
              <h3>Dual Signal Analysis</h3>
              <span>Voice authenticity + Claim verification</span>
            </div>
            <div className="dual-signal-row">
              <div className="signal-card">
                <strong>Voice Authenticity</strong>
                <div className="meter-track">
                  <div
                    className={`meter-fill meter-fill--${
                      details.voice_authenticity_score >= 0.68 ? "manipulated" :
                      details.voice_authenticity_score >= 0.34 ? "suspicious" :
                      "authentic"
                    }`}
                    style={{ width: percent(details.voice_authenticity_score) }}
                  />
                </div>
                <span className="signal-value">{percent(details.voice_authenticity_score)}</span>
                <span className="signal-method">{details.voice_detection_method || "model"}</span>
              </div>

              <div className="signal-card">
                <strong>Claim Verification</strong>
                <div className="meter-track">
                  <div
                    className={`meter-fill meter-fill--${
                      details.claim_verification_score >= 0.68 ? "manipulated" :
                      details.claim_verification_score >= 0.34 ? "suspicious" :
                      "authentic"
                    }`}
                    style={{ width: percent(details.claim_verification_score) }}
                  />
                </div>
                <span className="signal-value">{percent(details.claim_verification_score)}</span>
                <span className="signal-method">via text detection</span>
              </div>
            </div>
          </section>
        )}
      </>
    );
  }

  return null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState("text");
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function submitForAnalysis(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (activeTab === "text" && !textInput.trim()) {
      setError("Paste text before submitting.");
      return;
    }

    if (activeTab !== "text" && !selectedFile) {
      setError("Choose a file before submitting.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      if (activeTab === "text") {
        formData.append("text", textInput);
      } else if (selectedFile) {
        formData.append("file", selectedFile);
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        setError(payload.explanation || payload.detail || `Server error ${response.status}`);
        setResult(payload);
        return;
      }

      setResult(payload);
    } catch (submissionError) {
      setError(`Network error: ${submissionError.message}. Is the backend running on port 8000?`);
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(file) {
    if (!file) return;
    setSelectedFile(file);
    setError("");
  }

  const tone = result ? statusTone(result.status) : "authentic";

  return (
    <div className="app-shell">
      <div className="ambient ambient--left" />
      <div className="ambient ambient--right" />

      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">EDITH</p>
          <h1>Authenticity Analysis System</h1>
          <p className="hero-text">
            EDITH analyzes text, images, audio, and video for authenticity signals. It prioritizes
            evidence, source consistency, and human-readable explanations over binary content labels.
          </p>
        </div>
        <div className="hero-card">
          <DetailMetric label="Statuses" value="Authentic Signals, Suspicious, Unverified, Manipulated" />
          <DetailMetric label="Text mode" value="Live search + evidence ranking" />
          <DetailMetric label="Search" value="Top 5 DuckDuckGo results" />
        </div>
      </header>

      <main className="workspace">
        <section className="panel input-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Analyze Content</p>
              <h2>Submission Console</h2>
            </div>
            <div className="tab-group">
              {Object.entries(CONTENT_TYPE_CONFIG).map(([tab, config]) => (
                <button
                  key={tab}
                  type="button"
                  className={activeTab === tab ? "tab-btn tab-btn--active" : "tab-btn"}
                  onClick={() => {
                    setActiveTab(tab);
                    setSelectedFile(null);
                    setTextInput("");
                    setError("");
                  }}
                >
                  {config.label}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={submitForAnalysis} className="submission-form">
            {activeTab === "text" ? (
              <div className="text-entry">
                <textarea
                  value={textInput}
                  onChange={(event) => setTextInput(event.target.value)}
                  placeholder="Paste a news claim, article excerpt, or social post."
                />
              </div>
            ) : (
              <label
                className={isDragging ? "dropzone dropzone--active" : "dropzone"}
                onDragEnter={() => setIsDragging(true)}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  handleFileChange(event.dataTransfer.files?.[0]);
                }}
              >
                <input
                  type="file"
                  accept={CONTENT_TYPE_CONFIG[activeTab]?.accept || ""}
                  onChange={(event) => handleFileChange(event.target.files?.[0])}
                />
                <span className="dropzone-title">Drag and drop {CONTENT_TYPE_CONFIG[activeTab]?.label.toLowerCase()} here</span>
                <span className="dropzone-subtitle">or click to browse local files</span>
                {selectedFile && (
                  <strong className="dropzone-meta">
                    {selectedFile.name} | {Math.max(1, selectedFile.size / 1024).toFixed(1)} KB
                  </strong>
                )}
              </label>
            )}

            {error && <div className="error-banner">{error}</div>}

            <div className="submit-row">
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Analyzing..." : "Run EDITH"}
              </button>
              <p className="submit-note">Text claims are checked against live web results at analysis time.</p>
            </div>
          </form>
        </section>

        <section className="panel result-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Results</p>
              <h2>Authenticity Dashboard</h2>
            </div>
          </div>

          {loading && (
            <div className="loading-state">
              <div className="spinner" />
              <div>
                <h3>Analyzing content...</h3>
                <p>This may take a few seconds.</p>
              </div>
            </div>
          )}

          {!loading && !result && (
            <div className="empty-state">
              <h3>No analysis yet</h3>
              <p>Submit content to see authenticity status, evidence, and modality-specific explanation.</p>
            </div>
          )}

          {!loading && result && (
            <div className="result-stack">
              <section className="summary-strip">
                <div className={`verdict-card verdict-card--${tone}`}>
                  <span className="eyebrow">Status</span>
                  <strong>{result.status}</strong>
                  <p>{(result.type ?? "unknown").toUpperCase()} analysis</p>
                </div>
                <DetailMetric label="Concern Score" value={percent(result.signal_score)} tone={tone} />
                <DetailMetric label="Confidence" value={percent(result.confidence)} tone={tone} />
              </section>

              <section className={`result-section explanation-box explanation-box--${tone}`}>
                <div className="section-heading">
                  <h3>Why EDITH Said This</h3>
                  <span>{result.details?.explainability?.mode ?? "summary"}</span>
                </div>
                <p className="explanation-copy">{result.explanation}</p>
              </section>

              <EvidenceSection evidence={result.evidence} tone={tone} />

              <section className="result-section">
                <div className="section-heading">
                  <h3>Confidence</h3>
                  <span>{percent(result.confidence)}</span>
                </div>
                <div className="meter-track">
                  <div className={`meter-fill meter-fill--${tone}`} style={{ width: percent(result.confidence) }} />
                </div>
              </section>

              <section className="result-section">
                <div className="section-heading">
                  <h3>Concern Score</h3>
                  <span>{percent(result.signal_score)}</span>
                </div>
                <div className="meter-track">
                  <div className={`meter-fill meter-fill--${tone}`} style={{ width: percent(result.signal_score) }} />
                </div>
              </section>

              <MediaPanel result={result} />

              <section className="result-section">
                <div className="section-heading">
                  <h3>Signal Breakdown</h3>
                  <span>{result.status}</span>
                </div>
                <div className="metrics-grid">
                  {Object.entries(result.details?.component_scores ?? {}).map(([key, value]) => (
                    <DetailMetric
                      key={key}
                      label={key.replaceAll("_", " ")}
                      value={percent(value)}
                      tone={value >= 0.68 ? "manipulated" : value >= 0.34 ? "suspicious" : "authentic"}
                    />
                  ))}
                </div>
              </section>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
