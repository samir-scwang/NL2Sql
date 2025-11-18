"""
FastAPI entrypoint exposing the NL2SQL agent as a HTTP API.

This module keeps all heavy LangChain work in a background thread so the ASGI
event loop stays responsive, and optionally streams out rich debug metadata
(tool logs, LangGraph updates) for front-ends that need to visualize them.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from Nl2Sql.agent import agent


app = FastAPI(
    title="NL2SQL Agent API",
    version="0.1.0",
    description="Thin FastAPI wrapper that exposes the LangChain + DeepSeek NL2SQL agent.",
)

# Allow the web front-end (which could be served from another host/port) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryPayload(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language question that should be answered via SQL.",
    )
    user_name: str = Field(
        default="api_user",
        description="Optional identifier for the caller (useful for auditing/routing).",
    )
    include_debug: bool = Field(
        default=False,
        description="Return tool logs and LangGraph updates for UI debugging.",
    )


class AgentQueryResponse(BaseModel):
    question: str
    final_answer: Optional[str] = None
    todos: Optional[Any] = None
    tool_logs: Optional[List[str]] = None
    updates: Optional[List[Any]] = None
    final_state: Optional[Dict[str, Any]] = None


@app.get("/healthz", tags=["system"])
def health_check() -> Dict[str, str]:
    """Simple probe endpoint for readiness checks."""
    return {"status": "ok"}


@app.post(
    "/nl2sql/query",
    response_model=AgentQueryResponse,
    response_class=JSONResponse,
    tags=["nl2sql"],
)
async def run_nl2sql_query(payload: QueryPayload) -> AgentQueryResponse:
    """
    Run the NL2SQL agent with the provided natural-language question.
    Heavy synchronous work is delegated to a worker thread to keep FastAPI responsive.
    """

    try:
        return await run_in_threadpool(_execute_agent_call, payload)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _execute_agent_call(payload: QueryPayload) -> AgentQueryResponse:
    """Execute the LangChain agent and collect optional debug metadata."""

    final_answer: Optional[str] = None
    todos: Optional[Any] = None
    final_state: Optional[Dict[str, Any]] = None

    tool_logs: Optional[List[str]] = [] if payload.include_debug else None
    updates: Optional[List[Any]] = [] if payload.include_debug else None

    for stream_mode, chunk in agent.stream(
        {
            "messages": [{"role": "user", "content": payload.question}],
            "user_name": payload.user_name,
        },
        stream_mode=["updates", "custom", "values"],
    ):
        if stream_mode == "custom":
            if tool_logs is not None:
                tool_logs.append(str(chunk))
            continue

        if stream_mode == "updates":
            final_answer = final_answer or _extract_final_answer(chunk)
            todos = _extract_todos(chunk) or todos
            if updates is not None:
                updates.append(_to_serializable(chunk))
            continue

        if stream_mode == "values":
            final_state = _to_serializable(chunk)
            final_answer = final_answer or _extract_final_answer(chunk)
            todos = _extract_todos(chunk) or todos

    return AgentQueryResponse(
        question=payload.question,
        final_answer=final_answer,
        todos=todos,
        tool_logs=tool_logs,
        updates=updates,
        final_state=final_state,
    )


def _extract_final_answer(chunk: Any) -> Optional[str]:
    """Try to extract the most recent AI response from a LangGraph state chunk."""
    if isinstance(chunk, dict) and "model" in chunk:
        model_part = chunk.get("model")
        if isinstance(model_part, dict):
            messages = model_part.get("messages") or []
            if isinstance(messages, list) and messages:
                last_msg = messages[-1]
                # When the message is still the LangChain object
                if isinstance(last_msg, AIMessage) and isinstance(last_msg.content, str):
                    return last_msg.content
                if isinstance(last_msg, BaseMessage):
                    content = getattr(last_msg, "content", None)
                    if isinstance(content, str):
                        return content
                # When updates already contain dicts produced by LangGraph JSON serialization
                if isinstance(last_msg, dict):
                    content = last_msg.get("content")
                    if isinstance(content, str):
                        return content
    return None


def _extract_todos(chunk: Any) -> Optional[Any]:
    if isinstance(chunk, dict) and "todos" in chunk:
        return _to_serializable(chunk.get("todos"))
    return None


def _to_serializable(data: Any) -> Any:
    """Convert LangChain structures to JSON-serializable primitives/dicts."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, BaseMessage):
            return {
                "type": obj.type,
                "content": getattr(obj, "content", None),
                "name": getattr(obj, "name", None),
                "additional_kwargs": getattr(obj, "additional_kwargs", {}),
            }
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    try:
        serialized = json.dumps(data, ensure_ascii=False, default=_default)
        return json.loads(serialized)
    except TypeError:
        # Fallback to a string representation to avoid dropping the data entirely.
        return {"value": str(data)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Nl2Sql.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
