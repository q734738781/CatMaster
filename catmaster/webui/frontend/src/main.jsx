import React from "react";
import { createRoot } from "react-dom/client";

import CatMasterWorkspace from "./v2/CatMasterWorkspace";
import "./styles.css";
import "katex/dist/katex.min.css";

const root = document.getElementById("app");

if (!root) {
  throw new Error("Missing #app mount");
}

const boot = window.CATMASTER_BOOT || {
  view: document.body.dataset.catmasterView || "workspace",
};

createRoot(root).render(
  <React.StrictMode>
    <CatMasterWorkspace boot={boot} />
  </React.StrictMode>,
);
