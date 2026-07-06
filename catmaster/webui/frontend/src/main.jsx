import React from "react";
import { createRoot } from "react-dom/client";

import CatMasterWorkspace from "./v2/CatMasterWorkspace";
import "./styles.css";

const root = document.getElementById("app");

if (!root) {
  throw new Error("Missing #app mount");
}

createRoot(root).render(
  <React.StrictMode>
    <CatMasterWorkspace boot={window.CATMASTER_BOOT || { view: "workspace" }} />
  </React.StrictMode>,
);
