import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

_client = None
_client_failed = False


def _has_credentials() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_client():
    """Return a cached Langfuse client, or None if unavailable.

    Cached deliberately: the pipeline calls this once per node (6+ times per run,
    more when the critic loop fires) and constructing a client each time sets up
    an OTel tracer provider and exporter every call.
    """
    global _client, _client_failed

    if _client is not None or _client_failed:
        return _client

    if not _has_credentials():
        _client_failed = True
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return _client
    except Exception as e:
        print(f"[langfuse] client init failed, tracing disabled: {e}")
        _client_failed = True
        return None


def get_langfuse_callback():
    """Return a Langchain CallbackHandler for Langfuse, or None if unavailable."""
    if get_client() is None:
        return None

    try:
        # langfuse >= 3 moved this out of langfuse.callback; the handler reads
        # config from the client initialised above rather than taking keys.
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as e:
        print(f"[langfuse] callback handler unavailable, tracing disabled: {e}")
        return None


def get_callbacks() -> list:
    """Return list of active callbacks for LangChain calls."""
    cb = get_langfuse_callback()
    return [cb] if cb is not None else []


@contextmanager
def trace_node(name: str) -> Generator:
    """Wrap node execution in a named Langfuse span. No-ops without credentials."""
    client = get_client()
    if client is None:
        yield
        return

    try:
        with client.start_as_current_observation(name=name) as span:
            yield span
    except Exception as e:
        print(f"[langfuse] span '{name}' failed, continuing untraced: {e}")
        yield


def flush() -> None:
    """Flush pending traces. Call once at the end of a pipeline run."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
