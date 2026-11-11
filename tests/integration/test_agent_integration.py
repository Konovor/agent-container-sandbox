"""Agent-level smoke tests: a deep agent drives the sandbox end to end.

A scripted chat model stands in for a real LLM, so these tests need no API key
and no network. They check the path the README describes: create_deep_agent ->
tool call -> sandbox.execute() -> ToolMessage.
"""

from __future__ import annotations

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent_container_sandbox import ManagedContainerSandbox


class _ScriptedModel(FakeMessagesListChatModel):
    """Replays fixed responses; ignores the tools the agent binds."""

    def bind_tools(self, tools, **kwargs):
        return self


def _run_scripted_command(backend, command: str) -> str:
    model = _ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": command},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(model=model, backend=backend)
    result = agent.invoke({"messages": [{"role": "user", "content": "Run the command."}]})
    outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(outputs) == 1
    return str(outputs[0].content)


def test_agent_executes_a_command_in_the_sandbox(sandbox):
    assert "42" in _run_scripted_command(sandbox, "printf 42")


def test_readme_quick_start_pattern_works_with_an_agent(client, make_spec):
    with ManagedContainerSandbox(client, make_spec()) as sandbox:
        assert "hello" in _run_scripted_command(sandbox, "echo hello")
