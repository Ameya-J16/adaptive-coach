import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv

load_dotenv()


def get_langfuse_callback():
    """Return a LangchainCallbackHandler for Langfuse tracing. Returns None if credentials are missing."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
    except ImportError:
        return None


def get_callbacks() -> list:
    """Return list of active callbacks for LangChain calls."""
    cb = get_langfuse_callback()
    return [cb] if cb is not None else []


@contextmanager
def trace_node(name: str) -> Generator:
    """Context manager that wraps node execution with a named Langfuse span."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        yield
        return

    try:
        from langfuse import Langfuse
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        trace = client.trace(name=name)
        span = trace.span(name=name)
        try:
            yield span
        finally:
            span.end()
            client.flush()
    except Exception:
        yield
