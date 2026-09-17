"""Playground: run one prompt across 2–4 models and compare."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.db.models.catalog import Model
from app.db.models.chat import PlaygroundComparison
from app.db.models.enums import RunProvenance
from app.schemas.chat import ChatMessage
from app.schemas.common import Ack, Page
from app.schemas.playground import (
    PlaygroundComparisonRead,
    PlaygroundEntry,
    PlaygroundRunRequest,
    PlaygroundRunResponse,
    PlaygroundVoteRequest,
)
from app.services.inference.registry import registry as inference_registry
from app.services.logbook import logbook

router = APIRouter(prefix="/playground", tags=["playground"])


@router.post("/run", response_model=PlaygroundRunResponse)
async def run_playground(
    payload: PlaygroundRunRequest, session: SessionDep
) -> PlaygroundRunResponse:
    models = [get_or_404(session, Model, model_id, "Model") for model_id in payload.model_ids]

    messages: list[ChatMessage] = []
    if payload.system_prompt:
        messages.append(ChatMessage(role="system", content=payload.system_prompt))
    messages.append(ChatMessage(role="user", content=payload.prompt))

    async def run_one(model: Model) -> PlaygroundEntry:
        try:
            adapter = await inference_registry.get_adapter(model)
            result = await adapter.complete(messages, payload.params)
            return PlaygroundEntry(
                model_id=model.id,
                model_name=model.display_name or model.name,
                content=result.text,
                latency_ms=round(result.latency_ms, 2),
                time_to_first_token_ms=round(result.time_to_first_token_ms, 2),
                tokens_per_sec=round(result.tokens_per_sec, 2),
                completion_tokens=result.completion_tokens,
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.total_tokens,
                memory_mb=result.memory_mb or model.vram_estimate_mb,
                finish_reason=result.finish_reason,
                provenance=result.provenance,
                engine=result.engine,
            )
        except Exception as exc:  # noqa: BLE001
            logbook.error(f"Playground run failed for {model.name}: {exc}", source="playground")
            return PlaygroundEntry(
                model_id=model.id,
                model_name=model.display_name or model.name,
                content="",
                latency_ms=0.0,
                time_to_first_token_ms=0.0,
                tokens_per_sec=0.0,
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                error=f"{type(exc).__name__}: {exc}",
            )

    entries = await asyncio.gather(*(run_one(model) for model in models))

    comparison_id = None
    if payload.persist:
        record = PlaygroundComparison(
            prompt=payload.prompt,
            system_prompt=payload.system_prompt,
            params=payload.params.model_dump(mode="json"),
            entries=[entry.model_dump(mode="json") for entry in entries],
            provenance=RunProvenance.SIMULATED
            if all(e.provenance == "simulated" for e in entries)
            else RunProvenance.MEASURED,
            is_demo=all(model.is_demo for model in models),
        )
        session.add(record)
        session.flush()
        comparison_id = record.id

    return PlaygroundRunResponse(
        id=comparison_id,
        prompt=payload.prompt,
        entries=list(entries),
        created_at=datetime.now(timezone.utc),
    )


@router.get("/comparisons", response_model=Page[PlaygroundComparisonRead])
def list_comparisons(
    session: SessionDep, pagination: PaginationDep
) -> Page[PlaygroundComparisonRead]:
    statement = select(PlaygroundComparison).order_by(PlaygroundComparison.created_at.desc())
    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[PlaygroundComparisonRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/comparisons/{comparison_id}", response_model=PlaygroundComparisonRead)
def get_comparison(comparison_id: str, session: SessionDep) -> PlaygroundComparisonRead:
    record = get_or_404(session, PlaygroundComparison, comparison_id, "Comparison")
    return PlaygroundComparisonRead.model_validate(record)


@router.post("/comparisons/{comparison_id}/vote", response_model=PlaygroundComparisonRead)
def vote(
    comparison_id: str, payload: PlaygroundVoteRequest, session: SessionDep
) -> PlaygroundComparisonRead:
    record = get_or_404(session, PlaygroundComparison, comparison_id, "Comparison")
    votes = dict(record.votes or {})
    votes[payload.winner_model_id] = votes.get(payload.winner_model_id, 0) + 1
    record.votes = votes
    record.winner_model_id = payload.winner_model_id
    session.flush()
    return PlaygroundComparisonRead.model_validate(record)


@router.delete("/comparisons/{comparison_id}", response_model=Ack)
def delete_comparison(comparison_id: str, session: SessionDep) -> Ack:
    record = get_or_404(session, PlaygroundComparison, comparison_id, "Comparison")
    session.delete(record)
    return Ack(ok=True, message="Comparison deleted", id=comparison_id)
