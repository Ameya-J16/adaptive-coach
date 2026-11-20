"""One place that decides which LLM the agents talk to.

Every provider here speaks the OpenAI chat-completions protocol, so a single
`ChatOpenAI` client covers all of them — switching backends is a change of
base URL and API key, not a change of library. That keeps the open-weight
providers dependency-free and makes the swap reversible.

Configure with environment variables:

    LLM_PROVIDER=groq          # groq | openai | ollama | together | openrouter
    GROQ_API_KEY=gsk_...       # provider's own key variable

Per-node model overrides (optional) let the cheap deterministic-ish nodes run
a smaller model than the ones doing real reasoning:

    LLM_MODEL=openai/gpt-oss-120b        # default for every agent
    LLM_MODEL_CRITIC=openai/gpt-oss-120b # override one node
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class ProviderError(RuntimeError):
    """Raised when the selected provider is missing its API key."""


# base_url=None means "the OpenAI default". Every other entry is an
# OpenAI-compatible endpoint serving open-weight models.
PROVIDERS = {
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "open_weights": False,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # 120B open-weight model with solid tool-calling — the agents need it.
        "default_model": "openai/gpt-oss-120b",
        "open_weights": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "open_weights": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "open_weights": True,
    },
    "ollama": {
        # Local models. No key needed; the OpenAI-compatible shim wants a
        # non-empty string anyway.
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "default_model": "llama3.1:8b",
        "open_weights": True,
    },
}

DEFAULT_PROVIDER = "openai"


def active_provider() -> str:
    name = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ProviderError(
            f"Unknown LLM_PROVIDER {name!r}. Choose one of: {', '.join(sorted(PROVIDERS))}"
        )
    return name


def model_for(node: str | None = None) -> str:
    """Model id for a node: LLM_MODEL_<NODE>, else LLM_MODEL, else provider default."""
    if node:
        override = os.getenv(f"LLM_MODEL_{node.upper()}")
        if override:
            return override
    return os.getenv("LLM_MODEL") or PROVIDERS[active_provider()]["default_model"]


# OpenAI reasoning models (the GPT-5 family and the o-series) reject a
# non-default temperature, so it is left out for them. LLM_REASONING_EFFORT
# (none | minimal | low | medium | high) tunes how long they think; unset keeps
# the provider default.
#
# Chat Completions refuses function tools combined with a reasoning effort on
# newer models ("use /v1/responses or set reasoning_effort to 'none'"), and
# every agent here calls tools. So a real effort level switches the client to
# the Responses API, which supports both; "none" stays on Chat Completions.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    name = model.lower().rsplit("/", 1)[-1]  # tolerate "openai/gpt-5..." ids
    return name.startswith(REASONING_MODEL_PREFIXES)


def get_llm(
    node: str | None = None,
    temperature: float = 0.2,
    timeout: int = 120,
    max_retries: int = 2,
) -> ChatOpenAI:
    """Build the chat client for one agent node."""
    provider = active_provider()
    spec = PROVIDERS[provider]

    key_env = spec["key_env"]
    if key_env:
        api_key = os.getenv(key_env)
        if not api_key:
            raise ProviderError(
                f"LLM_PROVIDER={provider} needs {key_env} set in your environment or .env"
            )
    else:
        api_key = "not-needed"  # local Ollama ignores it

    model = model_for(node)
    kwargs = {
        "model": model,
        "timeout": timeout,
        "max_retries": max_retries,
        "api_key": api_key,
    }
    if provider == "openai" and is_reasoning_model(model):
        effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
        if effort == "none":
            kwargs["reasoning_effort"] = "none"
        elif effort:
            kwargs["use_responses_api"] = True
            kwargs["reasoning"] = {"effort": effort}
    else:
        kwargs["temperature"] = temperature
    if spec["base_url"]:
        kwargs["base_url"] = spec["base_url"]
    return ChatOpenAI(**kwargs)


def describe() -> str:
    """One-line summary of the active backend, for logs and the UI."""
    provider = active_provider()
    spec = PROVIDERS[provider]
    weights = "open-weight" if spec["open_weights"] else "proprietary"
    # Per-agent overrides (LLM_MODEL_CRITIC=...) change what actually ran, so
    # eval reports and the UI must show them, not just the default model.
    prefix = "LLM_MODEL_"
    overrides = sorted(
        f"{k[len(prefix):].lower()}: {v}" for k, v in os.environ.items()
        if k.startswith(prefix) and v and v != model_for()
    )
    extra = f" ({', '.join(overrides)})" if overrides else ""
    return f"{provider} · {model_for()}{extra} · {weights}"
