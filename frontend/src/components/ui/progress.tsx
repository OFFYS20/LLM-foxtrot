"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  tone?: "accent" | "good" | "warning" | "critical";
  size?: "sm" | "default";
}

/** Deliberately plain: a 4px track with a 2px rounded fill, no gradient. */
export function Progress({
  value,
  max = 100,
  tone = "accent",
  size = "default",
  className,
  ...props
}: ProgressProps) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const toneClass = {
    accent: "bg-accent",
    good: "bg-good",
    warning: "bg-warning",
    critical: "bg-critical",
  }[tone];

  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn(
        "w-full overflow-hidden rounded-full bg-elevated",
        size === "sm" ? "h-1" : "h-1.5",
        className,
      )}
      {...props}
    >
      <div className={cn("h-full rounded-full transition-[width] duration-500", toneClass)} style={{ width: `${pct}%` }} />
    </div>
  );
}
