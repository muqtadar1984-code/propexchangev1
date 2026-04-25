import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// LOCAL DEV:  Vite proxies /api/* → FastAPI on :8502 (started by Streamlit)
// PRODUCTION: Set VITE_API_URL env var in Vercel → direct fetch to Railway URL
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main:       resolve(__dirname, "index.html"),
        individual: resolve(__dirname, "individual.html"),
        personal:   resolve(__dirname, "personal.html"),
      },
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8502",
        changeOrigin: true,
      },
    },
  },
});
