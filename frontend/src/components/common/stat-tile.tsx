"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

interface StatTileProps {
  label: string;
  value: React.ReactNode;
  unit?: string;
  hint?: React.ReactNode;
  tone?: "default" | "good" | "warning" | "critical" | "accent";
  icon?: React.ReactNode;
  sparkline?: React.ReactNode;
  className?: string;
}

/**
 * A value tile, not a chart: big number, small label, optional sparkline.
 * Values use tabular figures so columns of tiles line up.
 */
export function StatTile({
  label,
  value,
  unit,
  hint,
  tone = "default",
  icon,
  sparkline,
  className,
}: StatTileProps) {
  const toneClass = {
    default: "text-ink",
    good: "text-good",
    warning: "text-warning",
    critical: "text-critical",
    accent: "text-accent",
  }[tone];

  return (
    <div className={cn("rounded-lg border border-border bg-panel px-3.5 py-3", className)}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-2xs font-medium uppercase tracking-[0.12em] text-ink-muted">{label}</span>
        {icon ? <span className="text-ink-muted">{icon}</span> : null}
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className={cn("mono text-xl font-semibold leading-none", toneClass)}>{value}</span>
        {unit ? <span className="text-xs text-ink-muted">{unit}</span> : null}
      </div>
      {sparkline ? <div className="mt-2 h-8">{sparkline}</div> : null}
      {hint ? <div className="mt-1.5 text-2xs text-ink-muted">{hint}</div> : null}
    </div>
  );
}

export function MetricRow({
  label,
  value,
  mono = true,
  className,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("flex items-baseline justify-between gap-4 py-1.5", className)}>
      <span className="text-xs text-ink-muted">{label}</span>
      <span className={cn("text-xs text-ink-secondary", mono && "mono")}>{value}</span>
    </div>
  );
}
