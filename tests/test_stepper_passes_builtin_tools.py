from __future__ import annotations

from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.llm.types import TurnResult
from catmaster.runtime.tool_backend import ToolBackend


class RecordingDriver:
    def __init__(self) -> None:
        self.last_tools = None

    def create_turn(self, *, input_items, tools=None, **kwargs):
        self.last_tools = tools
        return TurnResult(output_text="", tool_calls=[], output_items_raw=[])


class DummyBackend(ToolBackend):
    def list_function_tools(self):
        return []

    def call(self, name, arguments_json, *, toolcall_key, call_id=None):
        raise AssertionError("call should not be invoked")


def test_stepper_passes_builtin_tools() -> None:
    driver = RecordingDriver()
    backend = DummyBackend()
    stepper = ToolCallingTaskStepper(driver=driver, backend=backend)

    stepper.run(
        task_id="task-1",
        task_goal="Search",
        context_pack={},
        function_tools=[],
        builtin_tools=[{"type": "web_search"}],
    )

    assert driver.last_tools is not None
    assert any(tool.get("type") == "web_search" for tool in driver.last_tools)
