import test from "node:test";
import assert from "node:assert/strict";

import {
  apiErrorPresentation,
  displayValue,
  formatBytes,
  isInternalStoragePath,
  plainText,
  presentError,
  redactErrorText,
  userFacingFileTitle,
} from "./presentation.js";

test("422 responses become field-specific, actionable messages", () => {
  const result = apiErrorPresentation(
    422,
    JSON.stringify({
      detail: [
        { loc: ["body", "project_name"], msg: "Field required", type: "missing" },
        { loc: ["body", "files", 1], msg: "Must be a PDF", type: "value_error" },
      ],
    }),
    "application/json",
  );

  assert.match(result.message, /Project name: Field required/);
  assert.match(result.message, /Files · item 2: Must be a PDF/);
  assert.match(result.message, /try again/);
  assert.match(result.technicalDetails, /HTTP 422/);
  assert.doesNotMatch(result.message, /"loc"/);
});

test("HTML error pages never leak markup into the ordinary error message", () => {
  const result = apiErrorPresentation(
    502,
    "<!doctype html><html><body><h1>Bad gateway</h1></body></html>",
    "text/html",
  );

  assert.equal(
    result.message,
    "CatMaster could not complete this request because the server had a problem. Try again.",
  );
  assert.doesNotMatch(result.message, /html|h1|gateway/i);
  assert.match(result.technicalDetails, /HTML error page/);
});

test("object values are projected into readable scalar text", () => {
  assert.equal(displayValue({ path: "files/result.csv", checksum: "hidden" }), "files/result.csv");
  assert.equal(displayValue({ count: 4, ready: true }), "Count: 4 · Ready: true");
  assert.equal(plainText("<strong>Upload failed</strong>"), "Upload failed");
});

test("byte formatting uses human file-size units", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(5 * 1024 * 1024), "5.0 MB");
  assert.equal(formatBytes(2 * 1024 * 1024 * 1024), "2.0 GB");
});

test("presentError hides serialized payloads from the main message", () => {
  const result = presentError(JSON.stringify({ detail: { message: "Workspace changed" } }));
  assert.equal(result.message, "Workspace changed");
  assert.doesNotMatch(result.message, /detail/);
});

test("errors redact private run storage and absolute filesystem paths", () => {
  const source = "Failed under /workspace/metadata/runs/private/run-12 and /home/researcher/project/secret.json";
  const redacted = redactErrorText(source);
  assert.doesNotMatch(redacted, /metadata|runs|private|researcher|secret\.json/i);
  assert.match(redacted, /restricted data/);
  assert.match(redacted, /internal path/);

  const result = presentError(new Error(source));
  assert.doesNotMatch(result.message, /metadata|runs|private\/run|researcher|secret\.json/i);
});

test("opaque object errors fall back to an actionable user message", () => {
  const result = presentError(new Error("[object Object]"));
  assert.equal(result.message, "CatMaster could not complete this action. Try again.");
});

test("managed storage paths receive a public title without exposing receipt IDs", () => {
  const path = "files/.deepagents/dpdispatcher/receipts/dp_20260706_150711_68742b5e.json";
  assert.equal(isInternalStoragePath(path), true);
  assert.equal(isInternalStoragePath("files/results/o2_relaxed.xyz"), false);
  assert.equal(userFacingFileTitle("dp_20260706_150711_68742b5e.json", path), "Execution receipt");
  assert.equal(userFacingFileTitle("O2 relaxation report", path), "O2 relaxation report");
  assert.equal(userFacingFileTitle("", "files/results/o2_relaxed.xyz"), "o2_relaxed.xyz");
});
