// PDF fetch + disk-streaming helpers for the SJTU literature downloader.
//
// All functions take proxy + target explicitly so they work regardless of
// which script calls them. Bytes are fetched inside the page's authenticated
// context via fetch(), then transferred to Node in base64 chunks and written
// to disk. This is the same approach as the original code but:
//  - deduplicated (fetchToFile / fetchAnyToFile share fetchToBuffer + streamToDisk)
//  - window variable is randomized + deleted after use (avoids multi-tab collisions)
//  - maxBytes guard prevents OOM on huge files
//  - requirePdf flag controls %PDF head validation

import fs from "node:fs";
import path from "node:path";
import { evalJs } from "./cdp-utils.mjs";
import { STATUS } from "./status-codes.mjs";

const DEFAULT_MAX_BYTES = 200 * 1024 * 1024; // 200 MB guard
const DEFAULT_CHUNK = 1048576; // 1 MB per base64 round-trip

/**
 * Check if a byte array starts with the %PDF signature.
 */
export function isPdfHead(bytes) {
  if (!bytes || bytes.length < 5) return false;
  const head = String.fromCharCode(...bytes.slice(0, 5));
  return head === "%PDF-";
}

/**
 * Fetch a URL inside the target tab's authenticated context.
 * Returns { ok, status, size, head, contentType, url } or { ok:false, err }.
 *
 * The bytes are stored in a randomized window variable to avoid collisions
 * when multiple tabs download concurrently.
 */
export async function fetchToBuffer(
  proxy,
  target,
  url,
  { requirePdf = true, maxBytes = DEFAULT_MAX_BYTES } = {}
) {
  // Random window var name so concurrent tabs don't clobber each other.
  const varName = `__sjtuPdf_${Math.random().toString(36).slice(2, 10)}`;
  const js = `(async()=>{try{
    const r=await fetch(${JSON.stringify(url)},{credentials:'include'});
    const ab=await r.arrayBuffer();
    const b=new Uint8Array(ab);
    if(b.length>${maxBytes}){return JSON.stringify({ok:false,err:'pdf_too_large',size:b.length});}
    window['${varName}']=b;
    return JSON.stringify({ok:r.ok,status:r.status,size:b.length,head:Array.from(b.slice(0,8)),contentType:r.headers.get('content-type')||'',url:location.href});
  }catch(e){return JSON.stringify({ok:false,err:String(e).slice(0,200)})}})()`;
  const raw = await evalJs(proxy, target, js, 120000);
  const meta = JSON.parse(raw || "{}");

  if (!meta.ok || !meta.size) {
    return { ok: false, err: meta.err || "empty response", varName };
  }
  if (meta.err === "pdf_too_large") {
    return { ok: false, err: STATUS.PDF_TOO_LARGE, size: meta.size, varName };
  }
  if (requirePdf) {
    const headBytes = meta.head || [];
    if (!isPdfHead(headBytes)) {
      // Clean up the window var before returning.
      await evalJs(proxy, target, `delete window['${varName}']`).catch(() => {});
      return { ok: false, err: "not a PDF (head mismatch)", head: meta.head, varName };
    }
  }
  return {
    ok: true,
    status: meta.status,
    size: meta.size,
    head: meta.head,
    contentType: meta.contentType,
    url: meta.url,
    varName,
  };
}

/**
 * Stream bytes from a window variable to disk in base64 chunks.
 * Deletes the window variable when done (or on error).
 */
export async function streamToDisk(
  proxy,
  target,
  varName,
  size,
  outPath,
  chunkSize = DEFAULT_CHUNK,
  onProgress
) {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const ws = fs.createWriteStream(outPath);
  try {
    for (let s = 0; s < size; s += chunkSize) {
      const e = Math.min(s + chunkSize, size);
      const b64 = await evalJs(
        proxy,
        target,
        `(()=>{const b=window['${varName}'].slice(${s},${e});let x='';for(let i=0;i<b.length;i+=0x8000){x+=String.fromCharCode.apply(null,b.subarray(i,i+0x8000));}return btoa(x);})()`,
        60000
      );
      ws.write(Buffer.from(b64, "base64"));
      if (onProgress) onProgress(e, size);
    }
    await new Promise((r) => ws.end(r));
  } finally {
    // Always clean up the window var, even on error.
    await evalJs(proxy, target, `delete window['${varName}']`).catch(() => {});
  }
  return { file: outPath, bytes: size };
}

/**
 * Fetch a URL (requiring %PDF) and stream to disk.
 * Returns { ok:true, file, bytes } or { ok:false, err }.
 */
export async function fetchToFile(proxy, target, url, outPath, { onProgress, maxBytes } = {}) {
  const meta = await fetchToBuffer(proxy, target, url, { requirePdf: true, maxBytes });
  if (!meta.ok) return { ok: false, err: meta.err };
  const res = await streamToDisk(
    proxy,
    target,
    meta.varName,
    meta.size,
    outPath,
    DEFAULT_CHUNK,
    onProgress
  );
  return { ok: true, ...res };
}

/**
 * Like fetchToFile but accepts any binary (SI can be jpg/xlsx/docx — not PDF).
 * Returns { ok:true, bytes } or { ok:false, err }.
 */
export async function fetchAnyToFile(proxy, target, url, outPath, { onProgress, maxBytes } = {}) {
  const meta = await fetchToBuffer(proxy, target, url, { requirePdf: false, maxBytes });
  if (!meta.ok) return { ok: false, err: meta.err };
  const res = await streamToDisk(
    proxy,
    target,
    meta.varName,
    meta.size,
    outPath,
    DEFAULT_CHUNK,
    onProgress
  );
  return { ok: true, bytes: res.bytes };
}
