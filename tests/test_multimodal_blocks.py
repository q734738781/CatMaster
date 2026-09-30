from __future__ import annotations

import base64
from types import SimpleNamespace

from catmaster.runtime.multimodal_blocks import (
    ModelMultimodalCapability,
    PreparedAttachment,
    build_turn_content,
    file_to_content_block,
    infer_attachment_kind,
)


def test_multimodal_capability_defaults_and_overrides() -> None:
    openrouter = ModelMultimodalCapability.from_llm_config(
        SimpleNamespace(provider="openrouter", multimodal={"audio": "true", "current_turn_inline_limit_bytes": "1234"})
    )
    unknown = ModelMultimodalCapability.from_llm_config(SimpleNamespace(provider="unknown"))

    assert openrouter.supports_kind("image") is True
    assert openrouter.supports_kind("pdf") is True
    assert openrouter.supports_kind("audio") is True
    assert openrouter.current_turn_inline_limit_bytes == 1234
    assert unknown.supports_kind("image") is False
    assert unknown.supports_kind("text") is True


def test_attachment_kind_detection_and_file_blocks(tmp_path) -> None:
    image = tmp_path / "figure.png"
    pdf = tmp_path / "paper.pdf"
    image.write_bytes(b"fake-image")
    pdf.write_bytes(b"%PDF-1.4")

    image_block = file_to_content_block(image, mime_type="image/png", kind=infer_attachment_kind(image.name, "image/png"))
    pdf_block = file_to_content_block(pdf, mime_type="application/pdf", kind=infer_attachment_kind(pdf.name, "application/pdf"))

    assert image_block == {
        "type": "image",
        "base64": base64.b64encode(b"fake-image").decode("ascii"),
        "mime_type": "image/png",
    }
    assert pdf_block == {
        "type": "file",
        "base64": base64.b64encode(b"%PDF-1.4").decode("ascii"),
        "mime_type": "application/pdf",
        "filename": "paper.pdf",
    }
    assert infer_attachment_kind("report.docx") == "document"
    assert infer_attachment_kind("results.xlsx") == "document"
    assert infer_attachment_kind("slides.pptx") == "document"
    assert infer_attachment_kind("legacy.ppt") == "unsupported"


def test_build_turn_content_uses_blocks_but_sidecar_has_no_base64() -> None:
    attachment = PreparedAttachment(
        artifact_id="art_1",
        workspace_path="files/attachments/thread/001_figure.png",
        filename="figure.png",
        mime_type="image/png",
        size_bytes=9,
        kind="image",
        current_turn_block={"type": "image", "base64": "ZmFrZS1pbWFnZQ==", "mime_type": "image/png"},
    )

    content = build_turn_content("inspect", [attachment])
    sidecar = attachment.sidecar()

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["base64"] == "ZmFrZS1pbWFnZQ=="
    assert "base64" not in sidecar
    assert sidecar["sent_to_model"] is True
    assert sidecar["sent_as"] == "image"
