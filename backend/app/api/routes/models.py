"""Model catalog endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import PaginationDep, SessionDep, get_or_404, paginate
from app.config import settings
from app.core.errors import ConflictError, ValidationError
from app.core.events import Topics, bus
from app.core.paths import (
    ALLOWED_MODEL_SUFFIXES,
    dir_size_bytes,
    resolve_within,
    validate_hf_repo_id,
)
from app.db.models.catalog import Model
from app.db.models.enums import ModelFormat, ModelSource, ModelStatus, Precision
from app.db.models.training import Checkpoint
from app.schemas.common import Ack, Page
from app.schemas.models import (
    CheckpointRead,
    ModelCloneRequest,
    ModelDetail,
    ModelExportRequest,
    ModelExportResult,
    ModelImportRequest,
    ModelLoadRequest,
    ModelRead,
    ModelUpdate,
)
from app.services.inference.registry import registry as inference_registry
from app.services.logbook import logbook

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=Page[ModelRead])
def list_models(
    session: SessionDep,
    pagination: PaginationDep,
    status: ModelStatus | None = None,
    source: ModelSource | None = None,
    search: str | None = None,
    include_demo: bool = True,
) -> Page[ModelRead]:
    statement = select(Model).order_by(Model.created_at.desc())
    if status:
        statement = statement.where(Model.status == status)
    if source:
        statement = statement.where(Model.source == source)
    if search:
        statement = statement.where(Model.name.ilike(f"%{search}%"))
    if not include_demo:
        statement = statement.where(Model.is_demo.is_(False))

    rows, total = paginate(session, statement, pagination)
    return Page(
        items=[ModelRead.model_validate(row) for row in rows],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.post("/import", response_model=ModelDetail, status_code=201)
def import_model(payload: ModelImportRequest, session: SessionDep) -> ModelDetail:
    """Register a model from Hugging Face, a local directory, or a custom path.

    Registration records metadata only — weights are downloaded/loaded lazily by
    the inference engine, so this never blocks on a multi-GB transfer.
    """
    name = payload.name
    local_path: str | None = None
    repo_id: str | None = None
    size_bytes: int | None = None
    config: dict[str, Any] = {}
    tokenizer: dict[str, Any] = {}

    if payload.source == "huggingface":
        if not payload.repo_id:
            raise ValidationError("repo_id is required when importing from Hugging Face")
        repo_id = validate_hf_repo_id(payload.repo_id)
        name = name or repo_id.split("/")[-1]
    else:
        if not payload.path:
            raise ValidationError("path is required when importing a local model")
        resolved = resolve_within(settings.models_dir, payload.path)
        if not resolved.exists():
            raise ValidationError(
                f"Path not found inside the data root: {payload.path}",
                details={"root": str(settings.models_dir)},
            )
        if resolved.is_file() and resolved.suffix.lower() not in ALLOWED_MODEL_SUFFIXES:
            raise ValidationError(
                f"Unsupported model file type {resolved.suffix!r}",
                details={"allowed": sorted(ALLOWED_MODEL_SUFFIXES)},
            )
        local_path = str(resolved)
        size_bytes = dir_size_bytes(resolved)
        name = name or resolved.stem
        config, tokenizer = _read_local_metadata(resolved)

    existing = session.scalar(select(Model).where(Model.name == name))
    if existing:
        raise ConflictError(f"A model named {name!r} already exists")

    fmt = payload.format
    if local_path and Path(local_path).suffix.lower() == ".gguf":
        fmt = ModelFormat.GGUF

    model = Model(
        name=name or "unnamed-model",
        display_name=name,
        description=payload.description,
        source=ModelSource(payload.source),
        format=fmt,
        status=ModelStatus.READY,
        repo_id=repo_id,
        revision=payload.revision,
        local_path=local_path,
        architecture=config.get("architectures", [None])[0]
        if config.get("architectures")
        else None,
        parameters=_estimate_parameters(config, size_bytes),
        context_length=payload.context_length
        or config.get("max_position_embeddings")
        or config.get("n_positions"),
        precision=payload.precision or Precision.BF16,
        size_bytes=size_bytes,
        vram_estimate_mb=int(size_bytes / 1024**2 * 1.15) if size_bytes else None,
        tokenizer=tokenizer,
        config=config,
        tags=payload.tags,
        is_demo=False,
    )
    session.add(model)
    session.flush()

    logbook.info(
        f"Registered model {model.name} ({model.source}/{model.format})",
        source="models",
        context={"model_id": model.id, "repo_id": repo_id, "path": local_path},
    )
    bus.publish(Topics.MODEL_STATUS, {"model_id": model.id, "status": str(model.status)})
    return ModelDetail.model_validate(model)


@router.get("/{model_id}", response_model=ModelDetail)
def get_model(model_id: str, session: SessionDep) -> ModelDetail:
    model = get_or_404(session, Model, model_id, "Model")
    return ModelDetail.model_validate(model)


@router.patch("/{model_id}", response_model=ModelDetail)
def update_model(model_id: str, payload: ModelUpdate, session: SessionDep) -> ModelDetail:
    model = get_or_404(session, Model, model_id, "Model")
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    notes = data.pop("notes", None)
    for field, value in data.items():
        setattr(model, field, value)
    if notes is not None:
        model.config = {**(model.config or {}), "notes": notes}
    session.flush()
    return ModelDetail.model_validate(model)


@router.delete("/{model_id}", response_model=Ack)
def delete_model(model_id: str, session: SessionDep) -> Ack:
    model = get_or_404(session, Model, model_id, "Model")
    if model.status == ModelStatus.TRAINING:
        raise ConflictError("Cannot delete a model while it is training — stop the job first.")
    name = model.name
    session.delete(model)
    logbook.warn(f"Deleted model {name}", source="models", context={"model_id": model_id})
    return Ack(ok=True, message=f"Deleted {name}", id=model_id)


@router.post("/{model_id}/clone", response_model=ModelDetail, status_code=201)
def clone_model(model_id: str, payload: ModelCloneRequest, session: SessionDep) -> ModelDetail:
    """Clone a model's *configuration* — the weights are referenced, not copied."""
    source = get_or_404(session, Model, model_id, "Model")
    if session.scalar(select(Model).where(Model.name == payload.name)):
        raise ConflictError(f"A model named {payload.name!r} already exists")

    clone = Model(
        name=payload.name,
        display_name=payload.name,
        description=payload.description or f"Clone of {source.name}",
        source=source.source,
        format=source.format,
        status=ModelStatus.READY,
        repo_id=source.repo_id,
        revision=source.revision,
        local_path=source.local_path,
        architecture=source.architecture,
        parameters=source.parameters,
        context_length=source.context_length,
        precision=source.precision,
        quantization=source.quantization,
        size_bytes=source.size_bytes,
        vram_estimate_mb=source.vram_estimate_mb,
        tokenizer=dict(source.tokenizer or {}),
        config=dict(source.config or {}),
        metrics=dict(source.metrics or {}),
        tags=[*(source.tags or []), "clone"],
        parent_model_id=source.id,
        license=source.license,
        is_demo=source.is_demo,
    )
    session.add(clone)
    session.flush()

    if payload.copy_checkpoints:
        for checkpoint in source.checkpoints:
            session.add(
                Checkpoint(
                    model_id=clone.id,
                    job_id=checkpoint.job_id,
                    experiment_id=checkpoint.experiment_id,
                    step=checkpoint.step,
                    epoch=checkpoint.epoch,
                    path=checkpoint.path,
                    size_bytes=checkpoint.size_bytes,
                    train_loss=checkpoint.train_loss,
                    val_loss=checkpoint.val_loss,
                    benchmark_score=checkpoint.benchmark_score,
                    is_best=checkpoint.is_best,
                    provenance=checkpoint.provenance,
                    is_demo=checkpoint.is_demo,
                )
            )
        session.flush()

    logbook.info(f"Cloned {source.name} → {clone.name}", source="models")
    return ModelDetail.model_validate(clone)


@router.get("/{model_id}/checkpoints", response_model=list[CheckpointRead])
def list_model_checkpoints(model_id: str, session: SessionDep) -> list[CheckpointRead]:
    get_or_404(session, Model, model_id, "Model")
    rows = session.scalars(
        select(Checkpoint).where(Checkpoint.model_id == model_id).order_by(Checkpoint.step.desc())
    ).all()
    return [CheckpointRead.model_validate(row) for row in rows]


@router.post("/{model_id}/load", response_model=ModelDetail)
async def load_model(model_id: str, payload: ModelLoadRequest, session: SessionDep) -> ModelDetail:
    model = get_or_404(session, Model, model_id, "Model")
    engine = None if payload.engine == "auto" else payload.engine
    adapter = await inference_registry.get_adapter(model, engine=engine)
    model.status = ModelStatus.LOADED
    model.loaded_at = datetime.now(timezone.utc)
    model.error = None
    session.flush()

    info = adapter.info()
    logbook.info(
        f"Loaded {model.name} via {info.engine} ({info.provenance})",
        source="models",
        context={"model_id": model.id, "engine": info.engine},
    )
    bus.publish(
        Topics.MODEL_STATUS, {"model_id": model.id, "status": "loaded", "engine": info.engine}
    )
    return ModelDetail.model_validate(model)


@router.post("/{model_id}/unload", response_model=ModelDetail)
async def unload_model(model_id: str, session: SessionDep) -> ModelDetail:
    model = get_or_404(session, Model, model_id, "Model")
    await inference_registry.unload(model_id)
    model.status = ModelStatus.STOPPED
    session.flush()
    bus.publish(Topics.MODEL_STATUS, {"model_id": model.id, "status": "stopped"})
    return ModelDetail.model_validate(model)


@router.post("/{model_id}/export", response_model=ModelExportResult)
def export_model(
    model_id: str, payload: ModelExportRequest, session: SessionDep
) -> ModelExportResult:
    """Write an export manifest into the exports directory.

    Weight conversion (safetensors ↔ GGUF) requires the optional ML stack; when
    it is unavailable the manifest records exactly what would be converted
    instead of silently producing an empty artefact.
    """
    model = get_or_404(session, Model, model_id, "Model")
    target = resolve_within(
        settings.exports_dir, payload.destination or f"{model.name}-{payload.format}"
    )
    target.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": {
            "id": model.id,
            "name": model.name,
            "architecture": model.architecture,
            "parameters_millions": model.parameters,
            "precision": str(model.precision),
            "context_length": model.context_length,
            "source": str(model.source),
            "repo_id": model.repo_id,
            "local_path": model.local_path,
        },
        "export": {
            "format": payload.format,
            "quantization": payload.quantization,
            "include_tokenizer": payload.include_tokenizer,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "weights_written": False,
        "note": (
            "Manifest only. Weight conversion needs the optional ML stack "
            "(and, for GGUF, llama.cpp's converter); no weights were written."
        ),
    }
    (target / "foxtrot-export.json").write_text(json.dumps(manifest, indent=2))
    logbook.info(f"Exported manifest for {model.name} → {target}", source="models")

    return ModelExportResult(
        ok=True,
        format=payload.format,
        destination=str(target),
        manifest=manifest,
        message=f"Export manifest written to {target}",
    )


@router.get("/{model_id}/engine")
async def model_engine_info(model_id: str, session: SessionDep, engine: str | None = Query(None)):
    model = get_or_404(session, Model, model_id, "Model")
    adapter = await inference_registry.get_adapter(model, engine=engine)
    info = adapter.info()
    return {
        "engine": info.engine,
        "available": info.available,
        "provenance": info.provenance,
        "device": info.device,
        "detail": info.detail,
        "token_count_method": adapter.token_count_method(),
    }


def _read_local_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read config.json / tokenizer_config.json when present."""
    directory = path if path.is_dir() else path.parent
    config: dict[str, Any] = {}
    tokenizer: dict[str, Any] = {}
    try:
        config_file = directory / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
        tokenizer_file = directory / "tokenizer_config.json"
        if tokenizer_file.exists():
            raw = json.loads(tokenizer_file.read_text(encoding="utf-8"))
            tokenizer = {
                "type": raw.get("tokenizer_class"),
                "vocab_size": config.get("vocab_size"),
                "bos_token": _token_value(raw.get("bos_token")),
                "eos_token": _token_value(raw.get("eos_token")),
                "pad_token": _token_value(raw.get("pad_token")),
                "unk_token": _token_value(raw.get("unk_token")),
                "chat_template": bool(raw.get("chat_template")),
            }
    except Exception:  # noqa: BLE001 - metadata is best-effort
        pass
    return config, tokenizer


def _token_value(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("content")
    return value if isinstance(value, str) else None


def _estimate_parameters(config: dict[str, Any], size_bytes: int | None) -> int | None:
    """Parameter count in millions, from config where possible."""
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    vocab = config.get("vocab_size")
    if hidden and layers:
        intermediate = config.get("intermediate_size", hidden * 4)
        per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
        embeddings = 2 * (vocab or 32000) * hidden
        return round((per_layer * layers + embeddings) / 1e6)
    if size_bytes:
        return round(size_bytes / 2 / 1e6)  # assume ~2 bytes/param (fp16/bf16)
    return None
