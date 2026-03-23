from __future__ import annotations

from catmaster.llm.factory import _patch_langchain_openrouter_file_wrapper


def test_openrouter_file_wrapper_preserves_role_for_file_messages() -> None:
    from langchain_openrouter import chat_models as chat_models_mod

    _patch_langchain_openrouter_file_wrapper()

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
