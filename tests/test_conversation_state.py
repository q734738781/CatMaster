from __future__ import annotations

from catmaster.runtime.conversation_state import ConversationState, message_item


def test_conversation_state_appends_items() -> None:
    state = ConversationState()
    state.append_input_message("user", "hello")
    state.append_model_output_items([
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    ])
    state.append_function_call_output("call-1", {"ok": True})

    assert state.input_items[0] == message_item("user", "hello")
    assert state.input_items[1]["type"] == "message"
    assert state.input_items[2]["type"] == "function_call_output"
    assert state.input_items[2]["call_id"] == "call-1"
