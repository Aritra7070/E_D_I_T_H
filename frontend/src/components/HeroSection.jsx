import { useRef } from "react";
import { motion } from "framer-motion";
import HeroCanvas from "./HeroCanvas";

export default function HeroSection({ onStartDetection }) {
  const containerRef = useRef(null);

  return (
    <section ref={containerRef} className="hero-scroll-shell">
      <div className="hero-sticky-layer">
        <HeroCanvas containerRef={containerRef} />
        <div className="hero-overlay" />
        <div className="hero-vignette" />

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="absolute inset-0 z-20 flex items-center justify-center px-6 text-center"
        >
          <div className="max-w-4xl">
            <p className="eyebrow mx-auto">Enhanced Threat Interface</p>
            <h1 className="mt-5 font-orbitron text-6xl tracking-[0.28em] text-white sm:text-7xl lg:text-[7.4rem]">
              EDITH
            </h1>
            <p className="mt-6 font-orbitron text-lg uppercase tracking-[0.18em] text-zinc-200 sm:text-xl">
              AI Multi-Modal Detection System
            </p>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-zinc-300/80">
              Analyze text, image, video, and audio inputs through a cinematic explainable AI workflow.
            </p>

            <motion.button
              type="button"
              onClick={onStartDetection}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.98 }}
              animate={{
                boxShadow: [
                  "0 0 0 rgba(255,45,45,0)",
                  "0 0 28px rgba(255,45,45,0.26)",
                  "0 0 0 rgba(255,45,45,0)",
                ],
              }}
              transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
              className="premium-red-button mt-12 inline-flex items-center justify-center rounded-full px-8 py-4 font-orbitron text-sm uppercase tracking-[0.22em] text-white"
            >
              <span>Start Detection</span>
            </motion.button>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
