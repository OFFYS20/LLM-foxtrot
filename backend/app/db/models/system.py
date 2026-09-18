"""Logs, hardware snapshots and key/value settings."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, id_column, utcnow
from app.db.models.enums import LogLevel


class LogEntry(Base):
    __tablename__ = "log_entries"
    __table_args__ = (Index("ix_log_entries_ts_source", "ts", "source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    level: Mapped[LogLevel] = mapped_column(String(12), default=LogLevel.INFO, index=True)
    source: Mapped[str] = mapped_column(String(48), default="system", index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    job_id: Mapped[str | None] = mapped_column(String(40), default=None, index=True)
    context: Mapped[dict[str, Any]] = mapped_column(default=dict)


class HardwareSample(Base):
    __tablename__ = "hardware_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Setting(Base, TimestampMixin):
    __tablename__ = "settings"

    id: Mapped[str] = id_column("set")
    key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    value: Mapped[dict[str, Any]] = mapped_column(default=dict)
