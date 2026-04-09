import { useEffect, useRef, useState } from "react";
import { useMotionValueEvent, useScroll } from "framer-motion";

const FRAME_COUNT = 176;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${src}`));
    image.src = src;
  });
}

function loadFirstAvailable(sources) {
  return new Promise((resolve, reject) => {
    let index = 0;

    function tryNext() {
      if (index >= sources.length) {
        reject(new Error(`Failed to load frame from: ${sources.join(", ")}`));
        return;
      }

      loadImage(sources[index])
        .then(resolve)
        .catch(() => {
          index += 1;
          tryNext();
        });
    }

    tryNext();
  });
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export default function HeroCanvas({ containerRef }) {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const imagesRef = useRef([]);
  const currentFrameRef = useRef(-1);
  const animationFrameRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    contextRef.current = canvas.getContext("2d", { alpha: false });

    function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      drawFrame(currentFrameRef.current < 0 ? 0 : currentFrameRef.current);
    }

    async function preloadFrames() {
      try {
        const frameRequests = Array.from({ length: FRAME_COUNT }, (_, index) => {
          const paddedFour = String(index + 1).padStart(4, "0");
          const paddedThree = String(index + 1).padStart(3, "0");

          return loadFirstAvailable([
            `/assets/frames/frame_${paddedFour}.jpg`,
            `/assets/frames/ezgif-frame-${paddedThree}.jpg`,
          ]);
        });

        imagesRef.current = await Promise.all(frameRequests);
        setReady(true);
        setLoadFailed(false);
        resizeCanvas();
      } catch (error) {
        setLoadFailed(true);
        console.error("Frame preload failed", error);
      }
    }

    function drawFrame(index) {
      const canvasElement = canvasRef.current;
      const context = contextRef.current;
      const image = imagesRef.current[index];

      if (!canvasElement || !context || !image) return;

      const canvasRatio = canvasElement.width / canvasElement.height;
      const imageRatio = image.width / image.height;

      let drawWidth;
      let drawHeight;
      let offsetX;
      let offsetY;

      if (imageRatio > canvasRatio) {
        drawHeight = canvasElement.height;
        drawWidth = image.width * (canvasElement.height / image.height);
        offsetX = (canvasElement.width - drawWidth) / 2;
        offsetY = 0;
      } else {
        drawWidth = canvasElement.width;
        drawHeight = image.height * (canvasElement.width / image.width);
        offsetX = 0;
        offsetY = (canvasElement.height - drawHeight) / 2;
      }

      context.clearRect(0, 0, canvasElement.width, canvasElement.height);
      context.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
      currentFrameRef.current = index;
    }

    function requestDraw(index) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = requestAnimationFrame(() => {
        drawFrame(index);
      });
    }

    preloadFrames();
    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    if (!ready || imagesRef.current.length !== FRAME_COUNT) return;

    const nextFrame = clamp(
      Math.floor(latest * (FRAME_COUNT - 1)),
      0,
      FRAME_COUNT - 1
    );

    if (nextFrame !== currentFrameRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = requestAnimationFrame(() => {
        const canvas = canvasRef.current;
        const context = contextRef.current;
        const image = imagesRef.current[nextFrame];
        if (!canvas || !context || !image) return;

        const canvasRatio = canvas.width / canvas.height;
        const imageRatio = image.width / image.height;

        let drawWidth;
        let drawHeight;
        let offsetX;
        let offsetY;

        if (imageRatio > canvasRatio) {
          drawHeight = canvas.height;
          drawWidth = image.width * (canvas.height / image.height);
          offsetX = (canvas.width - drawWidth) / 2;
          offsetY = 0;
        } else {
          drawWidth = canvas.width;
          drawHeight = image.height * (canvas.width / image.width);
          offsetX = 0;
          offsetY = (canvas.height - drawHeight) / 2;
        }

        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
        currentFrameRef.current = nextFrame;
      });
    }
  });

  return (
    <>
      <canvas ref={canvasRef} className="hero-canvas" aria-hidden="true" />
      {!ready && (
        <div className="hero-loading-layer">
          <div className="hero-loading-ring" />
          <p>{loadFailed ? "EDITH visual stream unavailable" : "Initializing EDITH visual stream"}</p>
        </div>
      )}
    </>
  );
}
