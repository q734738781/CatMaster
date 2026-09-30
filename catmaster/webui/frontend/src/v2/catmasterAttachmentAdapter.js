const CLIENT_ATTACHMENT_INLINE_LIMIT_BYTES = 64 * 1024 * 1024;

function fileKind(file) {
  const type = String(file?.type || "").toLowerCase();
  const name = String(file?.name || "").toLowerCase();
  if (type.startsWith("image/")) return "image";
  if (type.startsWith("text/")) return "text";
  if (type === "application/pdf" || name.endsWith(".pdf")) return "document";
  if (type.startsWith("audio/")) return "audio";
  if (type.startsWith("video/")) return "video";
  return "file";
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

async function fileDataUrl(file) {
  if (typeof FileReader !== "undefined") {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  }
  const buffer = await file.arrayBuffer();
  const encoded = bytesToBase64(new Uint8Array(buffer));
  return `data:${file.type || "application/octet-stream"};base64,${encoded}`;
}

async function fileText(file) {
  if (typeof FileReader !== "undefined") {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = (error) => reject(error);
      reader.readAsText(file);
    });
  }
  return file.text();
}

export class CatMasterAttachmentAdapter {
  accept = "*";

  async add({ file }) {
    return {
      id: `${file.name}-${file.size}-${file.lastModified || 0}`,
      type: fileKind(file),
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: {
        type: "requires-action",
        reason: "composer-send",
      },
    };
  }

  async send(attachment) {
    const file = attachment.file;
    if (!file) {
      return {
        ...attachment,
        status: { type: "complete" },
        content: [],
      };
    }
    if (file.size > CLIENT_ATTACHMENT_INLINE_LIMIT_BYTES) {
      throw new Error(`Attachment ${file.name} exceeds ${CLIENT_ATTACHMENT_INLINE_LIMIT_BYTES} bytes.`);
    }
    const mimeType = file.type || attachment.contentType || "application/octet-stream";
    const kind = fileKind(file);
    const data = await fileDataUrl(file);
    const part = {
      type: kind === "image" ? "image" : "file",
      filename: file.name,
      name: file.name,
      mimeType,
      contentType: mimeType,
      data,
      sizeBytes: file.size,
    };
    if (kind === "image") {
      part.image = data;
    }
    if (mimeType.startsWith("text/")) {
      part.text = await fileText(file);
    }
    return {
      ...attachment,
      type: kind,
      contentType: mimeType,
      status: { type: "complete" },
      content: [part],
    };
  }

  async remove() {}
}
