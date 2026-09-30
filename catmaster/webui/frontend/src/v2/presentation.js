const HTML_DOCUMENT_RE = /<(?:!doctype|html|head|body)\b/i;
const HTML_TAG_RE = /<\/?[a-z][^>]*>/gi;
const PRIVATE_RUN_PATH_RE = /(?:[\\/])?(?:workspace[\\/]+)?metadata[\\/]+runs[\\/]+private(?:[\\/][^\s"'<>)]*)?/gi;
const ABSOLUTE_INTERNAL_PATH_RE = /(?:\/(?:home|root|users|workspace|mnt|srv|tmp)\/|[a-z]:\\(?:users|workspace)\\)[^\s"'<>)]*/gi;

function decodeCommonEntities(value) {
  return String(value || "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, "\"")
    .replace(/&#39;/gi, "'");
}

export function plainText(value) {
  if (value === null || value === undefined) return "";
  const source = String(value);
  if (!HTML_TAG_RE.test(source)) return decodeCommonEntities(source).trim();
  HTML_TAG_RE.lastIndex = 0;
  return decodeCommonEntities(source.replace(HTML_TAG_RE, " ").replace(/\s+/g, " ")).trim();
}

export function redactErrorText(value) {
  return String(value || "")
    .replace(PRIVATE_RUN_PATH_RE, "[restricted data]")
    .replace(ABSOLUTE_INTERNAL_PATH_RE, "[internal path]");
}

export function humanizeKey(value) {
  const text = String(value || "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim();
  return text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : "Value";
}

export function isInternalStoragePath(value) {
  const path = String(value || "").replace(/\\/g, "/").replace(/^\/+/, "").toLowerCase();
  return (
    path === "metadata"
    || path.startsWith("metadata/")
    || path === ".runtime"
    || path.startsWith(".runtime/")
    || /(^|\/)\.(?:deepagents|catmaster)(?:\/|$)/.test(path)
  );
}

export function userFacingFileTitle(title, path, fallback = "Result") {
  const rawTitle = displayValue(title, "");
  const basename = String(path || "").replace(/\\/g, "/").split("/").filter(Boolean).pop() || "";
  if (!isInternalStoragePath(path)) return rawTitle || basename || fallback;
  if (rawTitle && rawTitle !== basename && !/^dp_[0-9_]+[a-z0-9_-]*\.json$/i.test(rawTitle)) {
    return rawTitle;
  }
  if (/^dp_[0-9_]+[a-z0-9_-]*\.json$/i.test(rawTitle || basename)) return "Execution receipt";
  return fallback;
}

export function displayValue(value, fallback = "Not available") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string") return plainText(value) || fallback;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const values = value.map((item) => displayValue(item, "")).filter(Boolean);
    return values.length ? values.join(", ") : fallback;
  }
  if (typeof value === "object") {
    const preferred = ["label", "title", "name", "message", "summary", "value", "path"];
    for (const key of preferred) {
      const candidate = value[key];
      if (["string", "number", "boolean"].includes(typeof candidate) && String(candidate).trim()) {
        return displayValue(candidate, fallback);
      }
    }
    const scalarRows = Object.entries(value)
      .filter(([, item]) => ["string", "number", "boolean"].includes(typeof item) && String(item).trim())
      .map(([key, item]) => `${humanizeKey(key)}: ${displayValue(item, "")}`);
    return scalarRows.length ? scalarRows.join(" · ") : "Structured details";
  }
  return fallback;
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown size";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / (1024 ** 2)).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  return `${(bytes / (1024 ** 3)).toFixed(bytes < 10 * 1024 ** 3 ? 1 : 0)} GB`;
}

function safeJson(text) {
  try {
    return JSON.parse(String(text || ""));
  } catch {
    return null;
  }
}

function validationLocation(location) {
  const rows = Array.isArray(location) ? location : [];
  const meaningful = rows.filter((part, index) => (
    !(index === 0 && ["body", "query", "path", "header"].includes(String(part).toLowerCase()))
  ));
  if (!meaningful.length) return "Request";
  return meaningful.map((part) => (
    typeof part === "number" ? `item ${part + 1}` : humanizeKey(part)
  )).join(" · ");
}

function validationIssues(payload) {
  const detail = payload?.detail;
  if (!Array.isArray(detail)) return [];
  return detail.map((issue) => ({
    field: redactErrorText(validationLocation(issue?.loc)),
    message: redactErrorText(displayValue(issue?.msg || issue?.message, "This value is not valid")),
  }));
}

function statusMessage(status) {
  if (status === 400) return "CatMaster could not use that request. Review the values and try again.";
  if (status === 401) return "Your session has expired. Sign in again and retry.";
  if (status === 403) return "You do not have permission to perform this action.";
  if (status === 404) return "This item is no longer available. Refresh the workspace and try again.";
  if (status === 409) return "This item changed elsewhere. Refresh it before trying again.";
  if (status === 413) return "This upload is too large. Choose a smaller file and try again.";
  if (status === 422) return "Some request values need attention. Review them and try again.";
  if (status === 429) return "CatMaster is receiving too many requests. Wait a moment and try again.";
  if (status >= 500) return "CatMaster could not complete this request because the server had a problem. Try again.";
  return "CatMaster could not complete this request. Review the details and try again.";
}

function serverMessage(payload) {
  const detail = payload?.detail;
  if (typeof detail === "string") return redactErrorText(plainText(detail));
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    return redactErrorText(displayValue(detail.message || detail.summary || detail.error, ""));
  }
  return redactErrorText(displayValue(payload?.message || payload?.error, ""));
}

function sizeHint(payload) {
  const detail = payload?.detail && typeof payload.detail === "object" ? payload.detail : payload;
  const limit = Number(
    detail?.max_bytes
    || detail?.limit_bytes
    || detail?.maximum_bytes
    || detail?.max_size,
  );
  return Number.isFinite(limit) && limit > 0 ? ` The current limit is ${formatBytes(limit)}.` : "";
}

export function apiErrorPresentation(status, text, contentType = "") {
  const code = Number(status || 0);
  const source = String(text || "");
  const payload = safeJson(source);
  const issues = validationIssues(payload);
  const isHtml = String(contentType).toLowerCase().includes("text/html") || HTML_DOCUMENT_RE.test(source);
  const technical = [`HTTP ${code || "request failure"}`];

  if (issues.length) {
    for (const issue of issues) technical.push(`${issue.field}: ${issue.message}`);
    return {
      message: `Some request values need attention. ${issues.map((issue) => `${issue.field}: ${issue.message}.`).join(" ")} Review them and try again.`,
      technicalDetails: technical.join("\n"),
      details: payload?.detail,
    };
  }

  if (isHtml) {
    technical.push("The server returned an HTML error page instead of application data.");
    return {
      message: statusMessage(code),
      technicalDetails: technical.join("\n"),
      details: {},
    };
  }

  const detailMessage = serverMessage(payload);
  if (detailMessage) technical.push(detailMessage);
  const base = detailMessage || statusMessage(code);
  return {
    message: `${base}${code === 413 ? sizeHint(payload) : ""}`,
    technicalDetails: technical.join("\n"),
    details: payload?.detail && typeof payload.detail === "object" ? payload.detail : payload || {},
  };
}

export function presentError(error, fallback = "CatMaster could not complete this action. Try again.") {
  if (!error) return { message: "", technicalDetails: "" };
  if (typeof error === "string") {
    const source = error.trim();
    if (!source) return { message: fallback, technicalDetails: "" };
    if (HTML_DOCUMENT_RE.test(source)) {
      return { message: fallback, technicalDetails: "The server returned an HTML error page." };
    }
    const parsed = safeJson(source);
    if (parsed) return apiErrorPresentation(0, source, "application/json");
    const safe = redactErrorText(plainText(source));
    return { message: safe && safe !== "[object Object]" ? safe : fallback, technicalDetails: "" };
  }
  const message = redactErrorText(displayValue(error.userMessage || error.message, fallback));
  return {
    message: message && message !== "[object Object]" ? message : fallback,
    technicalDetails: redactErrorText(String(error.technicalDetails || "").trim()),
  };
}

export function makeApiError(status, text, contentType = "") {
  const presentation = apiErrorPresentation(status, text, contentType);
  const error = new Error(presentation.message);
  error.status = Number(status || 0);
  error.details = presentation.details;
  error.technicalDetails = presentation.technicalDetails;
  return error;
}
