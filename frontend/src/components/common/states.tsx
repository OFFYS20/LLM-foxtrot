"use client";

import { AlertTriangle, Inbox, Loader2, ServerCrash } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function EmptyState({
  title,
  description,
  action,
  icon,
  className,
}: {
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid-texture flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-panel/40 px-6 py-12 text-center",
        className,
      )}
    >
      <div className="mb-3 rounded-full border border-border bg-panel p-3 text-ink-muted">
        {icon ?? <Inbox className="h-5 w-5" />}
      </div>
      <p className="text-sm font-medium text-ink">{title}</p>
      {description ? <p className="mt-1 max-w-md text-xs text-ink-muted">{description}</p> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  className,
}: {
  error: unknown;
  onRetry?: () => void;
  className?: string;
}) {
  const message = error instanceof Error ? error.message : String(error ?? "Unknown error");
  const isNetwork = message.toLowerCase().includes("cannot reach");

  return (
    <div
      className={cn(
        "flex flex-col items-start gap-3 rounded-lg border border-critical/40 bg-critical/5 px-4 py-4",
        className,
      )}
    >
      <div className="flex items-start gap-2.5">
        <ServerCrash className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
        <div>
          <p className="text-sm font-medium text-ink">
            {isNetwork ? "Backend unreachable" : "Request failed"}
          </p>
          <p className="mono mt-1 text-xs text-ink-secondary">{message}</p>
          {isNetwork ? (
            <p className="mt-2 text-xs text-ink-muted">
              Start it with <code className="mono text-ink-secondary">uvicorn app.main:app --port 8000</code> from{" "}
              <code className="mono text-ink-secondary">backend/</code>, or set{" "}
              <code className="mono text-ink-secondary">NEXT_PUBLIC_DATA_PROVIDER=mock</code> to browse the UI
              on client-side fixtures.
            </p>
          ) : null}
        </div>
      </div>
      {onRetry ? (
        <Button size="sm" variant="secondary" onClick={onRetry}>
          Retry
        </Button>
      ) : null}
    </div>
  );
}

export function LoadingState({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton key={index} className="h-12 w-full" />
      ))}
    </div>
  );
}

export function InlineSpinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-muted">
      <Loader2 className="h-3.5 w-3.5 animate-spin" />
      {label}
    </span>
  );
}

export function WarningNote({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-ink-secondary",
        className,
      )}
    >
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />
      <div>{children}</div>
    </div>
  );
}
