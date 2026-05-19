from __future__ import annotations

from catmaster.llm.factory import _sanitize_openrouter_message_dicts


def test_native_openrouter_file_wrapper_preserves_role_for_file_messages() -> None:
    from langchain_openrouter import chat_models as chat_models_mod

    wrapped = chat_models_mod._wrap_messages_for_sdk(
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

    serialized = [item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item for item in wrapped]

    assert serialized[0]["role"] == "system"
    assert serialized[0]["content"] == "You are a reviewer."
    assert serialized[1]["role"] == "user"
    assert serialized[1]["content"][0]["type"] == "text"
    assert serialized[1]["content"][1]["type"] == "file"


def test_openrouter_sanitizer_textualizes_replayed_tool_image_blocks() -> None:
    sanitized = _sanitize_openrouter_message_dicts(
        [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [
                    {
                        "type": "image",
                        "id": "img_123",
                        "mime_type": "image/jpeg",
                    }
                ],
            },
        ]
    )

    message = sanitized[0]

    assert message["role"] == "tool"
    assert message["content"][0]["type"] == "text"
    assert "image block omitted" in message["content"][0]["text"]
    assert "img_123" in message["content"][0]["text"]


def test_openrouter_sanitizer_textualizes_any_non_text_tool_block() -> None:
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
                    },
                ],
            },
        ]
    )

    content = sanitized[0]["content"]

    assert content[0] == {"type": "text", "text": "image generated"}
    assert content[1]["type"] == "text"
    assert "image_url block omitted" in content[1]["text"]
    assert content[2]["type"] == "text"
    assert "figure.pdf" in content[2]["text"]
