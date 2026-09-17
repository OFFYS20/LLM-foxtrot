"use client";

import { AlertTriangle, CheckCircle2, CircleDot, FlaskConical, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { JobStatus, ModelStatus, Provenance } from "@/lib/types";

/**
 * Provenance is the most important badge in the product: it says whether a
 * number was measured on this machine or produced by the simulator. It is
 * never inferred — it always comes from the API payload.
 */
export function ProvenanceBadge({
  provenance,
  className,
  compact = false,
}: {
  provenance: Provenance | string | undefined;
  className?: string;
  compact?: boolean;
}) {
  if (provenance === "measured") {
    return (
      <Badge variant="good" className={className} title="Produced by a real engine on this machine">
        <CheckCircle2 className="h-3 w-3" />
        {compact ? "real" : "measured"}
      </Badge>
    );
  }
  if (provenance === "imported") {
    return (
      <Badge variant="outline" className={className} title="Imported from an external result file">
        imported
      </Badge>
    );
  }
  return (
    <Badge
      variant="warning"
      className={className}
      title="Simulated demo data — not produced by a real model run"
    >
      <FlaskConical className="h-3 w-3" />
      {compact ? "demo" : "demo data"}
    </Badge>
  );
}

const JOB_STATUS_STYLES: Record<JobStatus, { variant: Parameters<typeof Badge>[0]["variant"]; dot: string }> = {
  queued: { variant: "muted", dot: "bg-ink-muted" },
  running: { variant: "accent", dot: "bg-accent animate-pulse-dot" },
  paused: { variant: "warning", dot: "bg-warning" },
  stopping: { variant: "warning", dot: "bg-warning animate-pulse-dot" },
  completed: { variant: "good", dot: "bg-good" },
  stopped: { variant: "muted", dot: "bg-ink-muted" },
  failed: { variant: "critical", dot: "bg-critical" },
};

export function JobStatusBadge({ status, className }: { status: JobStatus; className?: string }) {
  const style = JOB_STATUS_STYLES[status] ?? JOB_STATUS_STYLES.queued;
  return (
    <Badge variant={style.variant} className={className}>
      <span className={cn("h-1.5 w-1.5 rounded-full", style.dot)} />
      {status}
    </Badge>
  );
}

const MODEL_STATUS_STYLES: Record<ModelStatus, { variant: Parameters<typeof Badge>[0]["variant"]; dot: string }> = {
  ready: { variant: "default", dot: "bg-ink-muted" },
  loaded: { variant: "good", dot: "bg-good" },
  training: { variant: "accent", dot: "bg-accent animate-pulse-dot" },
  stopped: { variant: "muted", dot: "bg-ink-muted" },
  importing: { variant: "warning", dot: "bg-warning animate-pulse-dot" },
  error: { variant: "critical", dot: "bg-critical" },
};

export function ModelStatusBadge({ status, className }: { status: ModelStatus; className?: string }) {
  const style = MODEL_STATUS_STYLES[status] ?? MODEL_STATUS_STYLES.ready;
  return (
    <Badge variant={style.variant} className={className}>
      <span className={cn("h-1.5 w-1.5 rounded-full", style.dot)} />
      {status}
    </Badge>
  );
}

export function CorrectnessBadge({ correct }: { correct: boolean }) {
  return correct ? (
    <Badge variant="good">
      <CheckCircle2 className="h-3 w-3" />
      correct
    </Badge>
  ) : (
    <Badge variant="critical">
      <XCircle className="h-3 w-3" />
      incorrect
    </Badge>
  );
}

/** "Official split" vs "bundled sample" — never let a sample look official. */
export function DataSourceBadge({
  official,
  dataSource,
}: {
  official?: boolean;
  dataSource?: string;
}) {
  if (official) {
    return (
      <Badge variant="good" title="Run against the full local split">
        official split
      </Badge>
    );
  }
  return (
    <Badge
      variant="warning"
      title="Run against a small bundled sample — not the published benchmark split"
    >
      <AlertTriangle className="h-3 w-3" />
      {dataSource === "user_dataset" ? "user dataset" : "sample · not official"}
    </Badge>
  );
}

export function LiveDot({ active, label }: { active: boolean; label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-2xs uppercase tracking-wider text-ink-muted">
      <CircleDot
        className={cn("h-3 w-3", active ? "text-good animate-pulse-dot" : "text-ink-muted")}
      />
      {label ?? (active ? "live" : "idle")}
    </span>
  );
}
