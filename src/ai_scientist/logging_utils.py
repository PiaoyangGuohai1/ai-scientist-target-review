"""Structured event logging — JSONL trace + Rich console rendering.

Every node entry / exit emits one line to ``logs/run-<ts>.jsonl`` so that
the README-required "complete log" can be produced without screenshots.

The console side uses Rich to make the Scientist thought / tool call /
Reviewer issue stream human-readable during development.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.table import Table

console = Console()


class RunLogger:
    """Writes JSONL trace and mirrors key events to the console."""

    def __init__(self, log_dir: str | Path | None = None, run_id: str | None = None):
        log_dir = Path(log_dir or os.getenv("LOG_DIR") or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = log_dir / f"run-{self.run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------
    def emit(self, event: str, payload: dict[str, Any]) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # High-level helpers, used from cli.py while streaming the graph
    # ------------------------------------------------------------------
    def node_update(self, node: str, update: dict[str, Any]) -> None:
        summary = _summarise_update(node, update)
        self.emit("node_update", {"node": node, "summary": summary})

        if node == "scientist":
            msgs = update.get("messages") or []
            last = msgs[-1] if msgs else None
            if isinstance(last, AIMessage):
                tc = getattr(last, "tool_calls", None) or []
                if tc:
                    table = Table(title=f"[bold cyan]Scientist → tool_call(s)[/]", show_header=True)
                    table.add_column("tool")
                    table.add_column("args")
                    for call in tc:
                        table.add_row(call.get("name", "?"), json.dumps(call.get("args") or {}, ensure_ascii=False))
                    content = getattr(last, "content", "") or ""
                    if content:
                        console.print(Panel.fit(content, title="Scientist thought", border_style="cyan"))
                    console.print(table)
                else:
                    content = getattr(last, "content", "") or ""
                    console.print(Panel(content or "(finalize)", title="Scientist → brief", border_style="cyan"))

        elif node == "tools":
            msgs = update.get("messages") or []
            for m in msgs:
                if isinstance(m, ToolMessage):
                    preview = (m.content or "")[:400]
                    if len(m.content or "") > 400:
                        preview += " …"
                    from rich.text import Text

                    console.print(
                        Panel(
                            Text(preview),
                            title=f"Tool result: {m.name}",
                            border_style="green",
                        )
                    )

        elif node == "writer":
            draft = update.get("draft_report") or ""
            rev = update.get("revision_count")
            from rich.text import Text

            console.print(
                Panel(
                    Text(draft[:1200] + (" …" if len(draft) > 1200 else "")),
                    title=f"Writer draft (revision_count={rev})",
                    border_style="magenta",
                )
            )

        elif node == "reviewer":
            issues = update.get("issues") or []
            claims = update.get("claims") or []
            console.print(
                Rule(f"[bold yellow]Reviewer  claims={len(claims)}  issues={len(issues)}[/]")
            )
            if issues:
                table = Table(show_header=True, header_style="bold yellow")
                table.add_column("id")
                table.add_column("type")
                table.add_column("sev")
                table.add_column("claim (truncated)", overflow="fold")
                table.add_column("evidence_note", overflow="fold")
                for i in issues:
                    table.add_row(
                        i.get("id", ""),
                        i.get("type", ""),
                        i.get("severity", ""),
                        (i.get("claim_text") or "")[:120],
                        (i.get("evidence_note") or "")[:120],
                    )
                console.print(table)
            else:
                console.print("[green]Reviewer found no issues.[/]")

        elif node == "verdict":
            console.print(
                Panel(
                    f"verdict = [bold]{update.get('verdict')}[/]\nreason = {update.get('reject_reason')}",
                    title="Verdict",
                    border_style="blue",
                )
            )

        elif node == "finalize":
            fr = update.get("final_report") or ""
            from rich.text import Text

            console.print(Panel(Text(fr[:2000]), title="Final report", border_style="bold green"))

    def banner(self, text: str) -> None:
        console.print(Rule(f"[bold]{text}[/]"))

    def kv(self, title: str, data: dict[str, Any]) -> None:
        console.print(Panel(Pretty(data), title=title))


def _summarise_update(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSONL log readable — drop long messages, keep the signal."""
    out: dict[str, Any] = {}
    for k, v in update.items():
        if k == "messages":
            m_sum = []
            for m in v or []:
                role = type(m).__name__
                if isinstance(m, AIMessage):
                    m_sum.append(
                        {
                            "role": role,
                            "content": (m.content or "")[:500],
                            "tool_calls": [
                                {"name": tc.get("name"), "args": tc.get("args")}
                                for tc in (getattr(m, "tool_calls", None) or [])
                            ],
                        }
                    )
                elif isinstance(m, ToolMessage):
                    m_sum.append(
                        {
                            "role": role,
                            "name": m.name,
                            "content": (m.content or "")[:500],
                        }
                    )
                elif isinstance(m, (SystemMessage, HumanMessage)):
                    m_sum.append({"role": role, "content": (m.content or "")[:200]})
                else:
                    m_sum.append({"role": role, "repr": repr(m)[:200]})
            out["messages"] = m_sum
        elif k == "tool_outputs":
            out["tool_outputs_keys"] = sorted((v or {}).keys())
        elif k == "draft_report":
            out["draft_report_len"] = len(v or "")
        elif k == "final_report":
            out["final_report_len"] = len(v or "")
        elif k in {"claims", "issues", "tool_calls_log"}:
            out[f"{k}_len"] = len(v or [])
            if k == "issues":
                out["issues"] = v
        else:
            out[k] = v
    return out
