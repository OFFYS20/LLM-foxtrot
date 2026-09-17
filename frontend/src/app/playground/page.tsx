"use client";

import { PanelsTopLeft, Play, ThumbsUp } from "lucide-react";
import * as React from "react";

import { CHART_COLORS } from "@/components/charts/chart-kit";
import { ProvenanceBadge } from "@/components/common/badges";
import { CopyButton } from "@/components/common/copy-button";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { EmptyState, ErrorState, InlineSpinner } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useModels, usePlaygroundHistory, usePlaygroundMutations } from "@/lib/hooks/queries";
import { useWorkspace } from "@/lib/store";
import { formatFloat, formatNumber, formatRelative, truncate } from "@/lib/format";
import type { PlaygroundEntry, PlaygroundRun } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function PlaygroundPage() {
  const { data: models } = useModels({ limit: 200 });
  const selectedIds = useWorkspace((state) => state.playgroundModelIds);
  const setSelectedIds = useWorkspace((state) => state.setPlaygroundModels);
  const params = useWorkspace((state) => state.chatParams);
  const setParams = useWorkspace((state) => state.setChatParams);

  const [prompt, setPrompt] = React.useState("");
  const [systemPrompt, setSystemPrompt] = React.useState("");
  const [result, setResult] = React.useState<PlaygroundRun | null>(null);

  const { run, vote } = usePlaygroundMutations();
  const { data: history } = usePlaygroundHistory();

  const items = React.useMemo(() => models?.items ?? [], [models]);

  React.useEffect(() => {
    if (selectedIds.length === 0 && items.length >= 2) {
      setSelectedIds([items[0].id, items[1].id]);
    }
  }, [items, selectedIds.length, setSelectedIds]);

  const toggleModel = (id: string) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter((item) => item !== id));
    } else if (selectedIds.length < 4) {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const submit = () => {
    if (!prompt.trim() || selectedIds.length === 0) return;
    run.mutate(
      { model_ids: selectedIds, prompt, system_prompt: systemPrompt, params },
      { onSuccess: (data) => setResult(data) },
    );
  };

  return (
    <Workspace
      inspectorTitle="Comparison history"
      inspector={
        <>
          {(history?.items ?? []).length === 0 ? (
            <p className="text-xs text-ink-muted">
              Saved comparisons appear here — including which response you voted for.
            </p>
          ) : (
            (history?.items ?? []).map((comparison) => (
              <Card key={comparison.id}>
                <CardContent className="space-y-2 p-3">
                  <p className="line-clamp-2 text-xs text-ink-secondary">{comparison.prompt}</p>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {comparison.entries.map((entry) => (
                      <Badge
                        key={entry.model_id}
                        variant={comparison.winner_model_id === entry.model_id ? "good" : "muted"}
                      >
                        {truncate(entry.model_name, 18)}
                      </Badge>
                    ))}
                  </div>
                  <div className="mono text-2xs text-ink-muted">{formatRelative(comparison.created_at)}</div>
                </CardContent>
              </Card>
            ))
          )}
        </>
      }
    >
      <PageHeader
        title="Playground"
        description="Send one prompt to 2–4 models and compare responses, latency, throughput and memory side by side."
      />

      <Card>
        <CardHeader>
          <CardTitle>Models ({selectedIds.length}/4)</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {items.map((model) => {
            const index = selectedIds.indexOf(model.id);
            const selected = index >= 0;
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => toggleModel(model.id)}
                className={cn(
                  "flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs transition-colors",
                  selected
                    ? "border-accent/60 bg-accent/10 text-ink"
                    : "border-border bg-elevated/40 text-ink-muted hover:text-ink-secondary",
                )}
              >
                {selected ? (
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                  />
                ) : null}
                {model.display_name ?? model.name}
                {model.is_demo ? <span className="text-2xs text-warning">demo</span> : null}
              </button>
            );
          })}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Prompt</CardTitle>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Label className="whitespace-nowrap">Temp</Label>
              <div className="w-24">
                <Slider
                  value={[params.temperature]}
                  min={0}
                  max={2}
                  step={0.05}
                  onValueChange={([value]) => setParams({ temperature: value })}
                />
              </div>
              <span className="mono text-2xs text-ink-secondary">{params.temperature.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Label className="whitespace-nowrap">Max tokens</Label>
              <Input
                type="number"
                value={params.max_tokens}
                onChange={(event) => setParams({ max_tokens: Number(event.target.value) })}
                className="h-7 w-20 text-xs"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            placeholder="System prompt (optional)"
            className="h-16 text-xs"
          />
          <Textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="One prompt, sent to every selected model…"
            className="h-28 text-sm"
          />
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={submit} disabled={run.isPending || !prompt.trim() || selectedIds.length === 0}>
              <Play className="h-3.5 w-3.5" />
              {run.isPending ? "Running…" : `Run on ${selectedIds.length} model${selectedIds.length === 1 ? "" : "s"}`}
            </Button>
            {run.isPending ? <InlineSpinner label="generating in parallel" /> : null}
          </div>
          {run.error ? <ErrorState error={run.error} /> : null}
        </CardContent>
      </Card>

      {result ? (
        <>
          <SectionTitle>Responses</SectionTitle>
          <div
            className={cn(
              "grid gap-3",
              result.entries.length === 1
                ? "grid-cols-1"
                : result.entries.length === 2
                  ? "lg:grid-cols-2"
                  : result.entries.length === 3
                    ? "lg:grid-cols-3"
                    : "lg:grid-cols-2 2xl:grid-cols-4",
            )}
          >
            {result.entries.map((entry, index) => (
              <ResponseCard
                key={entry.model_id}
                entry={entry}
                color={CHART_COLORS[index % CHART_COLORS.length]}
                winner={result.id ? undefined : undefined}
                onVote={
                  result.id
                    ? () => vote.mutate({ comparisonId: result.id as string, winner: entry.model_id })
                    : undefined
                }
              />
            ))}
          </div>

          <SectionTitle>Measurements</SectionTitle>
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Latency</TableHead>
                  <TableHead className="text-right">TTFT</TableHead>
                  <TableHead className="text-right">Tokens/s</TableHead>
                  <TableHead className="text-right">Completion tokens</TableHead>
                  <TableHead className="text-right">Total tokens</TableHead>
                  <TableHead className="text-right">Memory</TableHead>
                  <TableHead>Finish</TableHead>
                  <TableHead>Source</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.entries.map((entry) => (
                  <TableRow key={entry.model_id}>
                    <TableCell className="text-xs text-ink">{entry.model_name}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(entry.latency_ms, 0)} ms</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(entry.time_to_first_token_ms, 0)} ms</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(entry.tokens_per_sec, 1)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatNumber(entry.completion_tokens)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatNumber(entry.total_tokens)}</TableCell>
                    <TableCell className="mono text-right text-xs">
                      {entry.memory_mb ? `${(entry.memory_mb / 1024).toFixed(1)} GB` : "—"}
                    </TableCell>
                    <TableCell className="mono text-xs">{entry.finish_reason}</TableCell>
                    <TableCell><ProvenanceBadge provenance={entry.provenance} compact /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </>
      ) : (
        <EmptyState
          icon={<PanelsTopLeft className="h-5 w-5" />}
          title="No comparison yet"
          description="Pick up to four models, write a prompt, and run them in parallel."
        />
      )}
    </Workspace>
  );
}

function ResponseCard({
  entry,
  color,
  winner,
  onVote,
}: {
  entry: PlaygroundEntry;
  color: string;
  winner?: boolean;
  onVote?: () => void;
}) {
  return (
    <Card className={cn(winner && "border-good/50")}>
      <CardHeader>
        <div className="flex min-w-0 items-center gap-2">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
          <CardTitle className="truncate">{entry.model_name}</CardTitle>
        </div>
        <div className="flex items-center gap-1">
          <ProvenanceBadge provenance={entry.provenance} compact />
          <CopyButton value={entry.content} size="xs" />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {entry.error ? (
          <p className="mono text-xs text-critical">{entry.error}</p>
        ) : (
          <div className="scrollbar-thin max-h-[320px] overflow-y-auto whitespace-pre-wrap text-sm text-ink-secondary">
            {entry.content}
          </div>
        )}
        <div className="mono flex flex-wrap gap-x-3 gap-y-1 border-t border-border pt-2 text-2xs text-ink-muted">
          <span>{formatFloat(entry.latency_ms, 0)} ms</span>
          <span>{formatFloat(entry.tokens_per_sec, 1)} tok/s</span>
          <span>{formatNumber(entry.completion_tokens)} tokens</span>
          {entry.memory_mb ? <span>{(entry.memory_mb / 1024).toFixed(1)} GB</span> : null}
        </div>
        {onVote ? (
          <Button size="xs" variant="secondary" onClick={onVote} className="w-full">
            <ThumbsUp className="h-3 w-3" />
            Vote as better
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}
