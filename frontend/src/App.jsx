import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { AudioLines, FileText, ImageIcon, ShieldAlert, Video } from "lucide-react";
import HeroSection from "./components/HeroSection";
import ResultPanel from "./components/ResultPanel";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const INPUT_TYPES = [
  { id: "text", label: "Text", icon: FileText, accept: ".txt,.md,.json,.rtf" },
  { id: "image", label: "Image", icon: ImageIcon, accept: "image/*,.png,.jpg,.jpeg,.bmp,.webp" },
  { id: "video", label: "Video", icon: Video, accept: "video/*,.mp4,.mov,.avi,.mkv,.webm" },
  { id: "audio", label: "Audio", icon: AudioLines, accept: "audio/*,.wav,.mp3,.ogg,.m4a,.flac" },
];

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.25 },
  transition: { duration: 0.55, ease: "easeOut" },
};

function InputTypeCard({ item, active, onClick }) {
  const Icon = item.icon;

  return (
    <motion.button
      type="button"
      onClick={() => onClick(item.id)}
      whileHover={{ scale: 1.03 }}
      transition={{ duration: 0.2 }}
      className={`group relative overflow-hidden rounded-2xl border px-5 py-5 text-left transition-all duration-300 ${
        active
          ? "border-[#ff3a3a]/70 bg-white/8 shadow-[0_0_30px_rgba(255,45,45,0.18)]"
          : "border-white/10 bg-white/[0.04] hover:border-[#ff3a3a]/40 hover:shadow-[0_0_24px_rgba(255,45,45,0.12)]"
      }`}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,45,45,0.16),transparent_60%)] opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      <div className="relative flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#ff3a3a]/25 bg-black/40 text-[#ff5a5a]">
          <Icon size={22} strokeWidth={1.8} />
        </div>
        <div>
          <p className="font-orbitron text-sm uppercase tracking-[0.24em] text-white">{item.label}</p>
          <p className="mt-1 text-sm text-zinc-400">Premium multi-modal intake</p>
        </div>
      </div>
    </motion.button>
  );
}

function InputPreview({ activeType, textInput, selectedFile, previewUrl }) {
  return (
    <div className="panel-shell min-h-[520px]">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Preview</p>
          <h3 className="panel-title">Input Preview</h3>
        </div>
      </div>

      {!textInput.trim() && !selectedFile && (
        <div className="panel-placeholder">
          <ShieldAlert className="mb-4 text-[#ff4d4d]" size={28} strokeWidth={1.8} />
          <p className="text-lg text-white">No input loaded</p>
          <p className="mt-2 max-w-sm text-sm text-zinc-500">
            Select a modality, then drop a file or enter text to prepare the analysis payload.
          </p>
        </div>
      )}

      {activeType === "text" && textInput.trim() && (
        <div className="rounded-2xl border border-white/10 bg-black/35 p-5">
          <p className="mb-3 text-xs uppercase tracking-[0.24em] text-zinc-500">Text Input</p>
          <p className="max-h-[380px] overflow-auto whitespace-pre-wrap text-[1.02rem] leading-7 text-zinc-200">
            {textInput}
          </p>
        </div>
      )}

      {selectedFile && activeType === "image" && previewUrl && (
        <div className="space-y-4">
          <img
            src={previewUrl}
            alt="Input preview"
            className="h-[380px] w-full rounded-2xl border border-white/10 object-cover"
          />
          <div className="rounded-2xl border border-white/10 bg-black/35 px-4 py-3 text-sm text-zinc-300">
            {selectedFile.name}
          </div>
        </div>
      )}

      {selectedFile && activeType === "video" && previewUrl && (
        <div className="space-y-4">
          <video
            src={previewUrl}
            controls
            className="h-[380px] w-full rounded-2xl border border-white/10 bg-black object-cover"
          />
          <div className="rounded-2xl border border-white/10 bg-black/35 px-4 py-3 text-sm text-zinc-300">
            {selectedFile.name}
          </div>
        </div>
      )}

      {selectedFile && activeType === "audio" && previewUrl && (
        <div className="space-y-4">
          <div className="flex h-[380px] items-center justify-center rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_center,rgba(255,45,45,0.18),transparent_60%)]">
            <div className="text-center">
              <AudioLines className="mx-auto text-[#ff5a5a]" size={42} strokeWidth={1.8} />
              <p className="mt-4 text-lg text-white">{selectedFile.name}</p>
              <p className="mt-2 text-sm text-zinc-500">Audio waveform preview</p>
            </div>
          </div>
          <audio controls className="w-full opacity-85">
            <source src={previewUrl} />
          </audio>
        </div>
      )}

      {selectedFile && !previewUrl && activeType !== "text" && (
        <div className="panel-placeholder">
          <p className="text-lg text-white">{selectedFile.name}</p>
          <p className="mt-2 text-sm text-zinc-500">File loaded and ready for analysis.</p>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [activeType, setActiveType] = useState("text");
  const [textInput, setTextInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const inputSectionRef = useRef(null);

  const activeConfig = useMemo(
    () => INPUT_TYPES.find((item) => item.id === activeType) ?? INPUT_TYPES[0],
    [activeType]
  );

  useEffect(() => {
    if (!selectedFile || activeType === "text") {
      setPreviewUrl("");
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [selectedFile, activeType]);

  function resetForType(type) {
    setActiveType(type);
    setSelectedFile(null);
    setTextInput("");
    setResult(null);
    setError("");
  }

  function handleFileChange(file) {
    if (!file) return;
    setSelectedFile(file);
    setError("");
    setResult(null);
  }

  async function submitForAnalysis(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (activeType === "text" && !textInput.trim()) {
      setError("Enter text before running the analysis.");
      return;
    }

    if (activeType !== "text" && !selectedFile) {
      setError("Select a file before running the analysis.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      if (activeType === "text") {
        formData.append("text", textInput);
      } else {
        formData.append("file", selectedFile);
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || payload.explanation || `Request failed with status ${response.status}`);
      }

      setResult(payload);
    } catch (submissionError) {
      setError(submissionError.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      <div className="hero-grid pointer-events-none fixed inset-0 opacity-50" />
      <div className="hero-glow pointer-events-none fixed inset-0" />

      <main className="relative z-10">
        <HeroSection
          onStartDetection={() =>
            inputSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
          }
        />

        <div ref={inputSectionRef} className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-6 pb-24 lg:px-10">
          <motion.section {...fadeUp} className="space-y-5">
            <div className="space-y-2 text-center lg:text-left">
              <p className="eyebrow">Mode Selection</p>
              <h2 className="section-title">Select Input Type</h2>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {INPUT_TYPES.map((item) => (
                <InputTypeCard key={item.id} item={item} active={item.id === activeType} onClick={resetForType} />
              ))}
            </div>
          </motion.section>

          <motion.section {...fadeUp} className="space-y-6">
            <div className="section-shell">
              <div className="space-y-2">
                <p className="eyebrow">Upload Interface</p>
                <h2 className="section-title">Drop your file or enter text</h2>
              </div>

              <form onSubmit={submitForAnalysis} className="space-y-6">
                {activeType === "text" ? (
                  <textarea
                    value={textInput}
                    onChange={(event) => setTextInput(event.target.value)}
                    placeholder="Paste the content you want EDITH to analyze."
                    className="min-h-[220px] w-full rounded-[1.75rem] border border-white/10 bg-black/45 px-6 py-5 text-lg leading-8 text-zinc-100 outline-none transition placeholder:text-zinc-500 focus:border-[#ff3a3a]/55 focus:shadow-[0_0_24px_rgba(255,45,45,0.18)]"
                  />
                ) : (
                  <label
                    className={`flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-[1.75rem] border-2 border-dashed bg-black/40 px-6 py-8 text-center transition-all duration-300 ${
                      isDragging
                        ? "border-[#ff4d4d] shadow-[0_0_32px_rgba(255,45,45,0.2)]"
                        : "border-white/15 hover:border-[#ff4d4d]/65 hover:shadow-[0_0_26px_rgba(255,45,45,0.14)]"
                    }`}
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
                      accept={activeConfig.accept}
                      className="hidden"
                      onChange={(event) => handleFileChange(event.target.files?.[0])}
                    />
                    <p className="font-orbitron text-lg uppercase tracking-[0.18em] text-white">
                      Drop your file or enter text
                    </p>
                    <p className="mt-3 text-sm text-zinc-400">Drag and drop here or click to browse.</p>
                    {selectedFile && (
                      <div className="mt-6 rounded-full border border-[#ff3a3a]/30 bg-white/5 px-5 py-2 text-sm text-zinc-200">
                        {selectedFile.name}
                      </div>
                    )}
                  </label>
                )}

                {error && (
                  <div className="rounded-2xl border border-[#ff3a3a]/30 bg-[#1a0d0d] px-5 py-4 text-sm text-[#ff8c8c]">
                    {error}
                  </div>
                )}

                <div className="flex justify-center">
                  <motion.button
                    type="submit"
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.98 }}
                    className="premium-red-button rounded-full px-10 py-4 font-orbitron text-sm uppercase tracking-[0.22em] text-white transition"
                  >
                    <span>{loading ? "Analyzing..." : "Analyze Now"}</span>
                  </motion.button>
                </div>
              </form>
            </div>
          </motion.section>

          <motion.section {...fadeUp} className="grid gap-6 xl:grid-cols-2">
            <InputPreview
              activeType={activeType}
              textInput={textInput}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
            />
            <ResultPanel result={result} loading={loading} error={error} />
          </motion.section>
        </div>
      </main>
    </div>
  );
}
