from __future__ import annotations

from catmaster.llm.factory import _sanitize_openrouter_message_dicts


def _assert_openrouter_tool_message_accepts(message: dict) -> None:
    from openrouter.components.chattoolmessage import ChatToolMessage

    ChatToolMessage.model_validate(message)


def test_openrouter_sanitizer_preserves_non_tool_file_messages() -> None:
    sanitized = _sanitize_openrouter_message_dicts(
        [
            {"role": "system", "content": "You are a reviewer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Review this PDF."},
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": "application/pdf",
                        "data": "ZmFrZQ==",
                        "filename": "draft.pdf",
                    },
                ],
            },
        ]
    )

    assert sanitized[0]["role"] == "system"
    assert sanitized[0]["content"] == "You are a reviewer."
    assert sanitized[1]["role"] == "user"
    assert sanitized[1]["content"][0]["type"] == "text"
    assert sanitized[1]["content"][1]["type"] == "file"


def test_openrouter_sanitizer_converts_tool_image_blocks() -> None:
    sanitized = _sanitize_openrouter_message_dicts(
        [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {
                        "type": "image",
                        "id": "img_123",
                        "base64": "ZmFrZQ==",
                        "mime_type": "image/jpeg",
                    }
                ],
            },
        ]
    )

    message = sanitized[0]

    assert message["role"] == "tool"
    assert message["content"][0]["type"] == "image_url"
    assert message["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZQ=="
    _assert_openrouter_tool_message_accepts(message)


def test_openrouter_sanitizer_preserves_tool_multimodal_blocks() -> None:
    sanitized = _sanitize_openrouter_message_dicts(
        [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {"type": "text", "text": "image generated"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                        "mime_type": "image/png",
                    },
                    {
                        "type": "file",
                        "filename": "figure.pdf",
                        "mime_type": "application/pdf",
                        "base64": "JVBERi0=",
                    },
                ],
            },
        ]
    )

    content = sanitized[0]["content"]

    assert content[0] == {"type": "text", "text": "image generated"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,ZmFrZQ=="
    assert content[2]["type"] == "file"
    assert content[2]["file"]["filename"] == "figure.pdf"
    assert content[2]["file"]["file_data"] == "data:application/pdf;base64,JVBERi0="
    _assert_openrouter_tool_message_accepts(sanitized[0])
