import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "../static"),
    emptyOutDir: true,
    sourcemap: false,
    assetsDir: "",
    rollupOptions: {
      input: resolve(__dirname, "src/main.jsx"),
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "chunk-[name].js",
        assetFileNames: (assetInfo) => {
          if (String(assetInfo.name || "").endsWith(".css")) {
            return "app.css";
          }
          return "asset-[name][extname]";
        },
      },
    },
  },
});
