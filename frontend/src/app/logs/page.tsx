"use client";

import { Pause, Play, Search, Trash2 } from "lucide-react";
import * as React from "react";

import { PageHeader } from "@/components/common/page-header";
import { ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { TrainingConsole } from "@/components/training/console";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { useLogSources, useLogs } from "@/lib/hooks/queries";
import { useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { formatDateTime } from "@/lib/format";
import type { LogEntry, LogLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

const LEVELS: (LogLevel | "all")[] = ["all", "debug", "info", "warn", "error"];

const LEVEL_BADGE: Record<string, Parameters<typeof Badge>[0]["variant"]> = {
  debug: "muted",
  info: "default",
  warn: "warning",
  error: "critical",
};

export default function LogsPage() {
  const [level, setLevel] = React.useState<LogLevel | "all">("all");
  const [source, setSource] = React.useState("all");
  const [search, setSearch] = React.useState("");
  const [follow, setFollow] = React.useState(true);
  const [streamed, setStreamed] = React.useState<LogEntry[]>([]);

  const { data: sources } = useLogSources();
  const { data, isLoading, error, refetch } = useLogs(
    {
      limit: 300,
      level: level === "all" ? undefined : level,
      source: source === "all" ? undefined : source,
      search: search || undefined,
    },
    follow ? 6000 : 0,
  );

  useRealtimeTopic<LogEntry>(
    "log.entry",
    (event) => {
      if (!follow) return;
      setStreamed((current) => [...current.slice(-300), event.data]);
    },
    follow,
  );

  const entries = React.useMemo(() => {
    const base = [...(data?.items ?? [])].reverse();
    const merged = [...base, ...streamed];
    return merged.filter((entry) => {
      if (level !== "all" && entry.level !== level) return false;
      if (source !== "all" && entry.source !== source) return false;
      if (search && !entry.message.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [data, level, search, source, streamed]);

  const consoleLines = entries.map((entry) => ({
    message: `[${entry.source}] ${entry.message}`,
    level: String(entry.level),
    ts: entry.ts,
  }));

  return (
    <Workspace
      inspectorTitle="Filters"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Level</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-1.5 py-3">
              {LEVELS.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setLevel(item)}
                  className={cn(
                    "rounded border px-2 py-1 text-2xs uppercase tracking-wider transition-colors",
                    level === item
                      ? "border-accent/60 bg-accent/10 text-ink"
                      : "border-border text-ink-muted hover:text-ink-secondary",
                  )}
                >
                  {item}
                </button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Source</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <Select value={source} onValueChange={setSource}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All sources</SelectItem>
                  {(sources ?? []).map((item) => (
                    <SelectItem key={item} value={item}>
                      {item}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          <Button
            size="sm"
            variant="danger"
            className="w-full"
            onClick={async () => {
              if (confirm("Clear all persisted log entries?")) {
                await api.logs.clear();
                setStreamed([]);
                await refetch();
              }
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear log history
          </Button>
        </>
      }
    >
      <PageHeader
        title="Logs"
        description="Everything the platform records: training steps, benchmark runs, imports, engine and hardware notices."
        actions={
          <Button size="sm" variant={follow ? "secondary" : "default"} onClick={() => setFollow((value) => !value)}>
            {follow ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {follow ? "Pause stream" : "Resume stream"}
          </Button>
        }
      />

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-muted" />
          <Input
            placeholder="Filter messages…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="h-8 w-[280px] pl-8 text-xs"
          />
        </div>
        <span className="mono ml-auto text-2xs text-ink-muted">{entries.length} entries shown</span>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <LoadingState rows={6} />
      ) : (
        <>
          <TrainingConsole lines={consoleLines} title="Live stream" height={320} />

          <Card>
            <CardHeader>
              <CardTitle>Structured entries</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="scrollbar-thin max-h-[420px] divide-y divide-border overflow-y-auto">
                {[...entries].reverse().map((entry, index) => (
                  <div key={`${entry.id}-${index}`} className="flex items-start gap-3 px-4 py-2">
                    <span className="mono w-[120px] shrink-0 text-2xs text-ink-muted">
                      {formatDateTime(entry.ts)}
                    </span>
                    <Badge variant={LEVEL_BADGE[String(entry.level)] ?? "default"}>{entry.level}</Badge>
                    <span className="mono w-[90px] shrink-0 text-2xs text-ink-muted">{entry.source}</span>
                    <span className="mono flex-1 whitespace-pre-wrap text-2xs text-ink-secondary">
                      {entry.message}
                    </span>
                  </div>
                ))}
                {entries.length === 0 ? (
                  <p className="p-4 text-xs text-ink-muted">No log entries match the current filters.</p>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </Workspace>
  );
}
