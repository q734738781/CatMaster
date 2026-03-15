import React from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

const root = document.getElementById("app");

if (!root) {
  throw new Error("Missing #app mount");
}

createRoot(root).render(
  <React.StrictMode>
    <App boot={window.CATMASTER_BOOT || { view: "home" }} />
  </React.StrictMode>,
);
