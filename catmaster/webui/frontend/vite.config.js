import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

function muiSystemEsm() {
  const packageRoot = resolve(__dirname, "node_modules/@mui/system/esm");
  return {
    name: "catmaster-mui-system-esm",
    enforce: "pre",
    resolveId(source) {
      if (source !== "@mui/system" && !source.startsWith("@mui/system/")) return null;
      const subpath = source === "@mui/system" ? "index" : source.slice("@mui/system/".length);
      const candidates = [
        resolve(packageRoot, `${subpath}.js`),
        resolve(packageRoot, subpath, "index.js"),
      ];
      return candidates.find((candidate) => existsSync(candidate)) || null;
    },
  };
}

export default defineConfig({
  base: "/static/",
  // Ketcher's Redux reducer records its current editor state on `global`.
  // Browsers expose the same global object as `globalThis`, not Node's
  // `global`, so make that published browser assumption explicit.
  define: {
    global: "globalThis",
  },
  plugins: [
    // MUI 5 publishes several @mui/system subpaths as CommonJS-only entry
    // files even though matching ESM files are present. Vite 8/Rolldown can
    // otherwise wrap their default exports twice, which makes Ketcher fail at
    // runtime inside createStyled(). Resolve those subpaths to the publisher's
    // own ESM build; this is deliberately scoped to @mui/system.
    muiSystemEsm(),
    react(),
    svelte(),
  ],
  worker: {
    plugins: () => [svelte()],
  },
  build: {
    outDir: resolve(__dirname, "../static"),
    emptyOutDir: true,
    sourcemap: false,
    assetsDir: "",
    commonjsOptions: {
      // Ketcher ships mixed ESM/CommonJS Indigo wrappers. Transforming only
      // files with obvious CJS syntax leaves default imports callable at build
      // time but broken in the browser.
      transformMixedEsModules: true,
    },
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
