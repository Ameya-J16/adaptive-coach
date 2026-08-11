"""List the models the configured provider actually offers your API key.

Run with: python -m scripts.list_models

Provider catalogues change (Groq deprecated the Llama 3.x ids in June 2026),
so this asks the provider rather than trusting a hardcoded list. Also does a
one-call smoke test of tool calling, which is the capability the four agents
depend on and the one most likely to be missing on a smaller model.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import PROVIDERS, active_provider, get_llm, model_for
import os


def list_models() -> None:
    provider = active_provider()
    spec = PROVIDERS[provider]
    base = spec["base_url"] or "https://api.openai.com/v1"
    key = os.getenv(spec["key_env"]) if spec["key_env"] else "not-needed"

    if spec["key_env"] and not key:
        sys.exit(f"{spec['key_env']} is not set — add it to .env first.")

    req = urllib.request.Request(
        f"{base}/models", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        sys.exit(f"{provider} /models returned {e.code}: {e.read().decode()[:200]}")

    ids = sorted(m["id"] for m in data.get("data", []))
    print(f"{provider} exposes {len(ids)} models to this key:\n")
    for i in ids:
        marker = "  <- configured default" if i == model_for() else ""
        print(f"  {i}{marker}")


def smoke_test_tool_calling() -> None:
    """The agents live or die on tool calling — check it before a full run."""
    from langchain_core.tools import tool

    @tool
    def get_weekly_volume(week: str) -> str:
        """Return total training volume in kg for a given week."""
        return "8200 kg"

    print(f"\nTool-calling smoke test on {model_for()}...")
    llm = get_llm(temperature=0.0).bind_tools([get_weekly_volume])
    resp = llm.invoke(
        "What was my training volume for the week of 2026-08-03? Use the tool."
    )
    if resp.tool_calls:
        print(f"  PASS — model requested: {resp.tool_calls[0]['name']}"
              f"({resp.tool_calls[0]['args']})")
    else:
        print("  FAIL — model answered without calling the tool. "
              "The agents will not work correctly on this model.")
        print(f"  reply: {str(resp.content)[:160]}")


if __name__ == "__main__":
    list_models()
    smoke_test_tool_calling()
