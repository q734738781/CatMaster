import assert from "node:assert/strict";
import test from "node:test";

import {
  workspaceInlineImageUrl,
  workspaceMarkdownUrlTransform,
  workspacePathFromSandboxHref,
  workspacePathFromInlineImageUrl,
} from "./workspaceLinks.js";

test("workspace sandbox links resolve to files-root-relative paths", () => {
  assert.equal(
    workspacePathFromSandboxHref("sandbox:/files/reports/CO2%20FTS.md"),
    "files/reports/CO2 FTS.md",
  );
  assert.equal(
    workspacePathFromSandboxHref("sandbox:/reports/result.md"),
    "files/reports/result.md",
  );
});

test("workspace sandbox links reject traversal and unrelated schemes", () => {
  assert.equal(workspacePathFromSandboxHref("sandbox:/files/../metadata/secret"), "");
  assert.equal(workspacePathFromSandboxHref("sandbox:/files/%2E%2E/secret"), "");
  assert.equal(workspacePathFromSandboxHref("javascript:alert(1)"), "");
  assert.equal(workspacePathFromSandboxHref("https://example.com/report"), "");
});

test("markdown URL transform preserves only valid workspace sandbox anchors", () => {
  const anchor = { tagName: "a" };
  assert.equal(
    workspaceMarkdownUrlTransform("sandbox:/files/reports/result.md", "href", anchor),
    "sandbox:/files/reports/result.md",
  );
  assert.equal(
    workspaceMarkdownUrlTransform("javascript:alert(1)", "href", anchor),
    "",
  );
  assert.equal(
    workspaceMarkdownUrlTransform("https://example.com", "href", anchor),
    "https://example.com",
  );
});

test("workspace Markdown images resolve through the current thread", () => {
  const image = { tagName: "img" };
  const resolved = workspaceMarkdownUrlTransform(
    "sandbox:/files/figures/strain%20curve.png",
    "src",
    image,
    "thread:demo.1",
  );

  assert.equal(
    resolved,
    "/api/threads/thread%3Ademo.1/files/image?path=files%2Ffigures%2Fstrain%20curve.png",
  );
  assert.equal(
    workspacePathFromInlineImageUrl(resolved, "thread:demo.1"),
    "files/figures/strain curve.png",
  );
  assert.equal(
    workspaceInlineImageUrl("sandbox:/files/figures/strain%20curve.png", ""),
    "",
  );
});

test("workspace Markdown image transform rejects unsafe or unscoped paths", () => {
  const image = { tagName: "img" };
  assert.equal(
    workspaceMarkdownUrlTransform("sandbox:/files/../metadata/secret.png", "src", image, "thread_1"),
    "",
  );
  assert.equal(
    workspaceMarkdownUrlTransform("sandbox:/files/plot.png", "src", image),
    "",
  );
  assert.equal(
    workspacePathFromInlineImageUrl(
      "/api/threads/thread_2/files/image?path=files%2Fplot.png",
      "thread_1",
    ),
    "",
  );
});
