"""A minimal, explicit tool-calling agent loop.

What makes a node an *agent* rather than a prompt: the model decides which
tools to call, in what order, with what arguments, and when it has enough
information to answer. This loop is that decision cycle, written out plainly
instead of hidden behind a prebuilt abstraction:

    1. send the conversation to the model with tools bound
    2. if the reply contains tool calls, execute them and append the results
    3. repeat until the model answers directly (or the round budget runs out)
    4. parse the final reply as JSON
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser

from tracing.langfuse_config import get_callbacks

_json_parser = JsonOutputParser()

_JSON_REMINDER = (
    "Your answer must be a single JSON object with the exact keys specified in "
    "the system prompt. Return it now, with no other text."
)


def _parse_object(content) -> dict | None:
    """The reply as a JSON object, or None if it is malformed or not an object."""
    try:
        parsed = _json_parser.parse(content if isinstance(content, str) else str(content))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _final_answer(llm, messages: list, content, config: dict) -> dict:
    """Parse a final reply; on a bad one, ask once more, then give up with {}.

    Smaller models occasionally wrap JSON in prose or truncate it. One retry
    recovers most of those; failing soft with {} lets each node fall back to
    its own default instead of crashing the whole run.
    """
    parsed = _parse_object(content)
    if parsed is not None:
        return parsed
    messages.append(HumanMessage(content=_JSON_REMINDER))
    retry = llm.invoke(messages, config=config)
    return _parse_object(retry.content) or {}


def run_tool_agent(
    llm,
    tools: list,
    system_prompt: str,
    user_prompt: str,
    max_rounds: int = 6,
    agent_name: str = "agent",
) -> dict:
    """Run the tool loop and return the agent's final answer as a dict."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    config = {"callbacks": get_callbacks()}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        if not response.tool_calls:
            return _final_answer(llm, messages, response.content, config)

        for call in response.tool_calls:
            print(f"  [{agent_name}] tool call: {call['name']}({call['args']})")
            tool = tool_map.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Round budget exhausted: demand a final answer with tools disabled so the
    # model cannot keep exploring forever.
    messages.append(
        HumanMessage(
            content="Tool budget exhausted. Answer now with your final JSON only, "
            "using the information gathered so far."
        )
    )
    response = llm.invoke(messages, config=config)
    return _final_answer(llm, messages, response.content, config)
