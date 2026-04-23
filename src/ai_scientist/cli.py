"""CLI entry point.

Usage:
    ai-scientist assess "评估一下 TP53 作为心血管疾病靶点的潜在风险和可行性"

Options:
    --provider      Override LLM_PROVIDER for a single run (openai / anthropic).
    --log-dir       Override LOG_DIR.
    --save          Write the final report to <path>.md.
    --max-revisions Override MAX_REVISIONS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()  # read .env on import

# ---- Proxy normalisation --------------------------------------------------
# langchain-openai spins up both sync and async httpx clients internally, and
# the async one picks up env proxies regardless of what we pass. So we fix up
# the process environment up-front:
#
#   * If ``OPENAI_BASE_URL`` points at localhost (OMLX / Ollama), clear every
#     proxy var — we must not tunnel to 127.0.0.1.
#   * Otherwise, clear only SOCKS-flavoured proxies (ALL_PROXY / all_proxy).
#     httpx cannot route SOCKS without the optional ``httpx[socks]`` extra
#     and failing mid-run with a cryptic ``socksio`` ImportError is worse
#     than silently preferring the sibling HTTP_PROXY / HTTPS_PROXY vars
#     that Clash / Mihomo also expose.
_base_url = os.getenv("OPENAI_BASE_URL") or ""
_is_local = False
if _base_url:
    try:
        _host = (urlparse(_base_url).hostname or "").lower()
        _is_local = _host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    except Exception:
        _is_local = False

if _is_local:
    _vars_to_clear = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                      "HTTPS_PROXY", "https_proxy")
else:
    _vars_to_clear = ("ALL_PROXY", "all_proxy")
for _var in _vars_to_clear:
    os.environ.pop(_var, None)

from .graph import build_graph, initial_state  # noqa: E402
from .logging_utils import RunLogger, console  # noqa: E402

app = typer.Typer(add_completion=False, help="AIDD target triage agent.")


@app.command()
def assess(
    query: str = typer.Argument(..., help="User query, e.g. '评估 TP53 作为心血管靶点...'"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override LLM_PROVIDER (openai / anthropic)."
    ),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save final report to this path."),
    max_revisions: Optional[int] = typer.Option(None, "--max-revisions"),
) -> None:
    """Run the full Scientist → Writer → Reviewer loop on a single query."""

    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if log_dir:
        os.environ["LOG_DIR"] = str(log_dir)
    if max_revisions is not None:
        os.environ["MAX_REVISIONS"] = str(max_revisions)

    with RunLogger() as logger:
        logger.banner("AI Scientist — target triage")
        logger.kv(
            "Run config",
            {
                "query": query,
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "model": os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") or "(default)",
                "base_url": os.getenv("OPENAI_BASE_URL", "(provider default)"),
                "max_revisions": int(os.getenv("MAX_REVISIONS") or 2),
                "stagnation_threshold": int(os.getenv("STAGNATION_THRESHOLD") or 2),
                "log_file": str(logger.path),
            },
        )

        graph = build_graph()
        state = initial_state(query)

        final_state: dict = {}
        for step in graph.stream(state, stream_mode="updates"):
            for node, update in step.items():
                logger.node_update(node, update)
                final_state.update(update)

        logger.banner("Done")
        console.print(
            f"verdict=[bold]{final_state.get('verdict')}[/]  "
            f"revisions={final_state.get('revision_count')}  "
            f"open_issues={len(final_state.get('issues') or [])}"
        )

        if save:
            save.parent.mkdir(parents=True, exist_ok=True)
            save.write_text(final_state.get("final_report") or "", encoding="utf-8")
            console.print(f"[green]Saved report to {save}[/]")


if __name__ == "__main__":
    app()
