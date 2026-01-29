from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


def message_item(role: str, text: str) -> dict:
    return {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    }


@dataclass
class ConversationState:
    input_items: list[dict] = field(default_factory=list)

    def append_input_message(self, role: str, text: str) -> None:
        self.input_items.append(message_item(role, text))

    def append_model_output_items(self, output_items: list[dict]) -> None:
        self.input_items.extend(output_items)

    def append_function_call_output(self, call_id: str, output: Any) -> None:
        """Append a tool result using the standard Responses API item.

        For interoperability with OpenAI/LC/LangGraph tool-calling ecosystems,
        `output` should be a string. If we receive dict/list results from local
        tools we JSON-encode them.
        """
        if output is None:
            output_text = ""
        elif isinstance(output, str):
            output_text = output
        elif isinstance(output, (dict, list)):
            output_text = json.dumps(output, ensure_ascii=False)
        else:
            output_text = str(output)

        self.input_items.append({
            "type": "function_call_output",
            "id": f"fco_{uuid.uuid4().hex}",
            "call_id": call_id,
            "output": output_text,
        })
