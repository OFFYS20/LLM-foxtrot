"use client";

import { Terminal } from "lucide-react";
import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatTime } from "@/lib/format";
import { cn } from "@/lib/utils";

export interface ConsoleLine {
  message: string;
  level: string;
  ts: string;
}

const LEVEL_CLASS: Record<string, string> = {
  debug: "text-ink-muted",
  info: "text-ink-secondary",
  warn: "text-warning",
  error: "text-critical",
};

/** Terminal-style log panel: monospace, autoscrolling, level-coloured. */
export function TrainingConsole({
  lines,
  title = "Training console",
  height = 320,
  className,
}: {
  lines: ConsoleLine[];
  title?: string;
  height?: number;
  className?: string;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [follow, setFollow] = React.useState(true);

  React.useEffect(() => {
    if (!follow || !containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [lines, follow]);

  return (
    <div className={cn("overflow-hidden rounded-lg border border-border bg-base", className)}>
      <div className="flex items-center justify-between border-b border-border bg-panel px-3 py-2">
        <div className="flex items-center gap-2">
          <Terminal className="h-3.5 w-3.5 text-ink-muted" />
          <span className="text-2xs font-semibold uppercase tracking-[0.14em] text-ink-secondary">
            {title}
          </span>
          <Badge variant="muted">{lines.length} lines</Badge>
        </div>
        <Button
          size="xs"
          variant={follow ? "secondary" : "ghost"}
          onClick={() => setFollow((value) => !value)}
        >
          {follow ? "Following" : "Paused"}
        </Button>
      </div>
      <div
        ref={containerRef}
        style={{ height }}
        className="scrollbar-thin overflow-y-auto px-3 py-2"
        onScroll={(event) => {
          const element = event.currentTarget;
          const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 40;
          if (!atBottom && follow) setFollow(false);
        }}
      >
        {lines.length === 0 ? (
          <p className="mono text-2xs text-ink-muted">
            waiting for output…
          </p>
        ) : (
          lines.map((line, index) => (
            <div key={index} className="mono flex gap-2 py-px text-2xs leading-relaxed">
              <span className="shrink-0 text-ink-muted/70">{formatTime(line.ts)}</span>
              <span className={cn("whitespace-pre-wrap break-words", LEVEL_CLASS[line.level] ?? "text-ink-secondary")}>
                {line.message}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
