import { motion } from "framer-motion";

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function statusAppearance(status) {
  if (status === "Authentic Signals" || status === "Authentic") {
    return {
      labelClass: "text-emerald-400",
      dotClass: "bg-emerald-400",
      barClass: "from-emerald-400 to-emerald-300",
      glowClass: "shadow-[0_0_30px_rgba(74,222,128,0.18)]",
    };
  }

  if (status === "Manipulated") {
    return {
      labelClass: "text-[#ff4d4d]",
      dotClass: "bg-[#ff4d4d]",
      barClass: "from-[#ff2d2d] to-[#ff5a5a]",
      glowClass: "shadow-[0_0_30px_rgba(255,45,45,0.2)]",
    };
  }

  return {
    labelClass: "text-[#ffcc00]",
    dotClass: "bg-[#ffcc00]",
    barClass: "from-[#ffcc00] to-[#ffd84d]",
    glowClass: "shadow-[0_0_28px_rgba(255,204,0,0.14)]",
  };
}

function normalizeSources(result) {
  if (Array.isArray(result?.sources) && result.sources.length) return result.sources;

  if (Array.isArray(result?.details?.related_articles)) {
    return result.details.related_articles.map((source) => ({
      title: source.title,
      domain: source.domain || source.source,
      trust: source.trust,
    }));
  }

  return [];
}

function normalizeFrames(result) {
  if (Array.isArray(result?.frames) && result.frames.length) return result.frames;

  if (Array.isArray(result?.details?.frame_previews)) {
    return result.details.frame_previews.map((frame) => ({
      index: frame.index,
      image: frame.image,
    }));
  }

  return [];
}

function WaveformBars() {
  const bars = [36, 62, 40, 76, 28, 58, 82, 34, 68, 44, 74, 30];
  return (
    <div className="mt-4 flex h-28 items-end gap-2 rounded-2xl border border-white/10 bg-black/35 px-4 py-4">
      {bars.map((height, index) => (
        <span
          key={`${height}-${index}`}
          className="flex-1 rounded-full bg-gradient-to-t from-[#ff2d2d] to-[#ff7a7a]"
          style={{ height: `${height}%` }}
        />
      ))}
    </div>
  );
}

function TypeSpecificEvidence({ result }) {
  const type = result?.type;
  const evidence = result?.evidence ?? [];
  const sources = normalizeSources(result);
  const frames = normalizeFrames(result);

  if (type === "text") {
    return (
      <div className="rounded-[1.6rem] border border-white/10 bg-black/28 p-5">
        <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Source Verification</p>
        <div className="mt-4 space-y-3">
          {sources.length ? (
            sources.map((source, index) => (
              <div key={`${source.title}-${index}`} className="flex items-center justify-between gap-4 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3">
                <div>
                  <p className="text-sm text-white">{source.title || source.domain || "Source"}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.22em] text-zinc-500">{source.domain || "Unknown domain"}</p>
                </div>
                <div className="text-sm text-zinc-300">
                  {typeof source.trust === "number" ? `${Math.round(source.trust * 100)}%` : "Trust n/a"}
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-zinc-500">No source metadata was returned for this text analysis.</p>
          )}
        </div>
      </div>
    );
  }

  if (type === "image") {
    return (
      <div className="rounded-[1.6rem] border border-white/10 bg-black/28 p-5">
        <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Image Analysis</p>
        <div className="mt-4 flex flex-wrap gap-3">
          {(evidence.length ? evidence : ["Edge Artifacts", "AI Texture Pattern", "Abnormal Smoothness"]).map((item) => (
            <span
              key={item}
              className="rounded-full border border-[#ff3a3a]/20 bg-[#ff2d2d]/10 px-4 py-2 text-sm text-zinc-200"
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (type === "video") {
    return (
      <div className="rounded-[1.6rem] border border-white/10 bg-black/28 p-5">
        <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Frame Analysis Preview</p>
        <div className="mt-4 flex flex-wrap gap-3 text-sm text-zinc-300">
          {(evidence.length ? evidence : ["Frame inconsistency", "Temporal flicker"]).map((item) => (
            <span key={item} className="rounded-full border border-[#ff3a3a]/20 bg-[#ff2d2d]/10 px-4 py-2">
              {item}
            </span>
          ))}
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {frames.length ? (
            frames.slice(0, 5).map((frame, index) => (
              <div key={`${frame.index ?? index}`} className="overflow-hidden rounded-2xl border border-white/10 bg-black/35">
                <img src={frame.image} alt={`Frame ${frame.index ?? index + 1}`} className="h-28 w-full object-cover" />
                <div className="px-3 py-2 text-xs uppercase tracking-[0.18em] text-zinc-400">
                  Frame {frame.index ?? index + 1}
                </div>
              </div>
            ))
          ) : (
            <div className="sm:col-span-3 text-sm text-zinc-500">No frame thumbnails were returned.</div>
          )}
        </div>
      </div>
    );
  }

  if (type === "audio") {
    return (
      <div className="rounded-[1.6rem] border border-white/10 bg-black/28 p-5">
        <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Audio Analysis</p>
        <div className="mt-4 flex flex-wrap gap-3 text-sm text-zinc-300">
          {(evidence.length ? evidence : ["Frequency anomalies", "Spectrogram pattern"]).map((item) => (
            <span key={item} className="rounded-full border border-[#ff3a3a]/20 bg-[#ff2d2d]/10 px-4 py-2">
              {item}
            </span>
          ))}
        </div>
        <WaveformBars />
      </div>
    );
  }

  return null;
}

export default function ResultPanel({ result, loading, error }) {
  const appearance = statusAppearance(result?.status);
  const reasons =
    Array.isArray(result?.evidence) && result.evidence.length
      ? result.evidence
      : result?.explanation
        ? [result.explanation]
        : [];

  return (
    <div className="panel-shell min-h-[520px]">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Output</p>
          <h3 className="panel-title">Analysis Result</h3>
        </div>
      </div>

      {loading && (
        <div className="panel-placeholder">
          <div className="h-11 w-11 animate-spin rounded-full border-2 border-white/10 border-t-[#ff2d2d]" />
          <p className="mt-5 text-lg text-white">Running explainable analysis</p>
          <p className="mt-2 text-sm text-zinc-500">Rendering confidence and evidence traces.</p>
        </div>
      )}

      {!loading && !result && !error && (
        <div className="panel-placeholder">
          <p className="text-lg text-white">Awaiting analysis</p>
          <p className="mt-2 max-w-sm text-sm text-zinc-500">
            Submit content to reveal status, confidence, evidence tags, and modality-specific explanation.
          </p>
        </div>
      )}

      {!loading && error && !result && (
        <div className="rounded-2xl border border-[#ff3a3a]/30 bg-[#1a0d0d] px-5 py-4 text-sm text-[#ff8c8c]">
          {error}
        </div>
      )}

      {!loading && result && (
        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-5">
            <div className={`rounded-[1.75rem] border border-white/10 bg-black/40 p-6 ${appearance.glowClass}`}>
              <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Status</p>
              <h4 className={`mt-3 font-orbitron text-3xl tracking-[0.18em] ${appearance.labelClass}`}>
                {result.status}
              </h4>
            </div>

            <div className="rounded-[1.75rem] border border-white/10 bg-black/32 p-6">
              <div className="mb-5">
                <div className="mb-2 flex items-center justify-between text-sm text-zinc-400">
                  <span>Confidence</span>
                  <span>{percent(result.confidence)}</span>
                </div>
                <div className="h-3 rounded-full bg-white/8">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: percent(result.confidence) }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className={`h-full rounded-full bg-gradient-to-r ${appearance.barClass}`}
                  />
                </div>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between text-sm text-zinc-400">
                  <span>Signal Score</span>
                  <span>{percent(result.signal_score)}</span>
                </div>
                <div className="h-3 rounded-full bg-white/8">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: percent(result.signal_score ?? result.confidence) }}
                    transition={{ duration: 0.9, ease: "easeOut", delay: 0.1 }}
                    className={`h-full rounded-full bg-gradient-to-r ${appearance.barClass}`}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-5">
            <TypeSpecificEvidence result={result} />

            <div className="rounded-[1.75rem] border border-white/10 bg-black/32 p-6">
              <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Evidence / Explanation</p>
              <ul className="mt-4 space-y-3">
                {reasons.map((reason) => (
                  <li key={reason} className="flex items-start gap-3 text-sm text-zinc-300">
                    <span className={`mt-1.5 h-2.5 w-2.5 rounded-full ${appearance.dotClass}`} />
                    <span>{reason}</span>
                  </li>
                ))}
                {!reasons.length && <li className="text-sm text-zinc-500">No evidence details were returned.</li>}
              </ul>
            </div>

            <div className="rounded-[1.75rem] border border-white/10 bg-black/32 p-6">
              <p className="text-xs uppercase tracking-[0.28em] text-zinc-500">Why EDITH Said This</p>
              <p className="mt-4 text-sm leading-7 text-zinc-300">
                {result.explanation ||
                  "This content shows signal patterns that the EDITH pipeline associates with authenticity risk, confidence weighting, and explainable evidence traces."}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
