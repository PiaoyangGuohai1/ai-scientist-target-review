"""LLM provider factory.

Two providers, selected by ``LLM_PROVIDER``:

- ``openai``    — any OpenAI-compatible endpoint (default). ``OPENAI_BASE_URL``
                  lets us point at a local model server (e.g. OMLX at
                  ``http://127.0.0.1:8080/v1``) or a cloud API without code
                  changes. Tool-calling goes through the standard
                  ``tool_calls`` field.
- ``anthropic`` — Claude via ``langchain_anthropic``.

``get_chat_model()`` is the only public surface; every agent node calls it.
No "offline" mode: tests that need deterministic draft content should
monkeypatch ``get_chat_model`` (see ``tests/conftest.py::patched_writer``).

Proxy behaviour:
    For localhost endpoints (OMLX / Ollama / etc.) the HTTP client is
    constructed with ``trust_env=False`` so a machine-level proxy does
    not try to tunnel to 127.0.0.1. For everything else we honour the
    user's ``HTTPS_PROXY`` / ``HTTP_PROXY`` env vars but deliberately
    ignore ``ALL_PROXY`` — ``ALL_PROXY=socks5://...`` is a common default
    (Mihomo / Clash / shadowsocks) and httpx needs the optional
    ``httpx[socks]`` extra to route through it, which would otherwise
    explode with a cryptic ``socksio`` ImportError mid-run. Users who
    specifically need SOCKS can install the extra and unset this branch.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

import httpx

Provider = Literal["openai", "anthropic"]

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in _LOCAL_HOSTS


def _select_http_proxy() -> str | None:
    """Pick an HTTP/HTTPS proxy URL from the environment.

    Explicitly skips ``ALL_PROXY`` / ``all_proxy`` because those are
    typically SOCKS on developer laptops, and httpx cannot route SOCKS
    without the ``httpx[socks]`` extra installed.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.getenv(var)
        if val and not val.lower().startswith("socks"):
            return val
    return None


def get_provider() -> Provider:
    p = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if p not in ("openai", "anthropic"):
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={p!r}. Use 'openai' or 'anthropic'."
        )
    return p  # type: ignore[return-value]


def get_chat_model(temperature: float = 0.0):
    """Return a ``langchain_core.language_models.BaseChatModel``.

    Raises:
        RuntimeError: when the selected provider's API key is missing.
    """
    provider = get_provider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "For the bundled OMLX demo, copy .env.example to .env."
            )
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("OPENAI_BASE_URL") or None

        if _is_local_url(base_url):
            # Local endpoint — skip all proxies.
            http_client = httpx.Client(
                headers={"User-Agent": "ai-scientist/0.1"},
                timeout=httpx.Timeout(120.0, connect=10.0),
                trust_env=False,
            )
        else:
            # Cloud endpoint — honour HTTP/HTTPS proxy env vars, ignore SOCKS.
            proxy = _select_http_proxy()
            http_client = httpx.Client(
                headers={"User-Agent": "ai-scientist/0.1"},
                timeout=httpx.Timeout(120.0, connect=10.0),
                trust_env=False,
                proxy=proxy,
            )
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    # anthropic
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        temperature=temperature,
    )
