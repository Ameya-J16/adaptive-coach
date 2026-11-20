import os
from contextlib import ExitStack, contextmanager
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


@contextmanager
def trace_run(name: str, user_id: str | None = None, tags: list[str] | None = None,
              input: dict | None = None) -> Generator[dict, None, None]:
    """Group one whole pipeline run into a single Langfuse trace.

    Without this, every node's span starts its own trace, so one plan shows up
    as seven unrelated entries. Wrapping the run in a root span makes the node
    spans and LLM generations its children. Yields a dict whose "trace_url" is
    the link to this run in the Langfuse UI (None when tracing is off) and
    whose "span" can receive the run's output.
    """
    info: dict = {"trace_url": None, "span": None}
    client = get_client()
    if client is None:
        yield info
        return

    with ExitStack() as stack:
        try:
            info["span"] = stack.enter_context(
                client.start_as_current_observation(name=name, as_type="agent", input=input)
            )
            try:
                from langfuse import propagate_attributes
                stack.enter_context(
                    propagate_attributes(user_id=user_id, tags=tags, trace_name=name)
                )
            except Exception as e:  # older SDKs: trace still works, just unlabelled
                print(f"[langfuse] could not set trace attributes: {e}")
        except Exception as e:
            print(f"[langfuse] run trace '{name}' unavailable, continuing: {e}")
        try:
            # Needs the project id from the Langfuse API; a miss only loses the link.
            info["trace_url"] = client.get_trace_url(trace_id=client.get_current_trace_id())
        except Exception as e:
            print(f"[langfuse] trace link unavailable: {e}")
        yield info


def flush() -> None:
    """Flush pending traces. Call once at the end of a pipeline run."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
