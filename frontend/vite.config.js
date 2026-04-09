import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const rootDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root: rootDir,
  cacheDir: ".vite",
  plugins: [tailwindcss(), react()],
  optimizeDeps: {
    disabled: "dev"
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true
  }
});
