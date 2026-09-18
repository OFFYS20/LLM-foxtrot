"use client";

import { Download, GitBranch, MessageSquare, Play, Target, Trash2 } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useCheckpointMutations, useCheckpoints, useModels } from "@/lib/hooks/queries";
import {
  formatBytes,
  formatDateTime,
  formatFloat,
  formatNumber,
  formatRelative,
} from "@/lib/format";
import type { Checkpoint } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function CheckpointsPage() {
  const [modelFilter, setModelFilter] = React.useState<string>("all");
  const [bestOnly, setBestOnly] = React.useState(false);
  const [selected, setSelected] = React.useState<Checkpoint | null>(null);

  const { data: models } = useModels({ limit: 200 });
  const { data, isLoading, error, refetch } = useCheckpoints({
    limit: 200,
    model_id: modelFilter === "all" ? undefined : modelFilter,
    best_only: bestOnly || undefined,
  });
  const { load, evaluate, exportCheckpoint, remove } = useCheckpointMutations();

  const checkpoints = data?.items ?? [];
  const active = selected ?? checkpoints[0] ?? null;
  const modelName = (id: string) =>
    models?.items.find((model) => model.id === id)?.display_name ?? id;

  return (
    <Workspace
      inspectorTitle="Checkpoint"
      inspector={
        active ? (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Step {formatNumber(active.step)}</CardTitle>
                <ProvenanceBadge provenance={active.provenance} compact />
              </CardHeader>
              <CardContent className="py-3">
                <div className="divide-y divide-border">
                  <MetricRow label="Model" value={modelName(active.model_id)} />
                  <MetricRow label="Epoch" value={formatFloat(active.epoch, 2)} />
                  <MetricRow label="Train loss" value={formatFloat(active.train_loss, 4)} />
                  <MetricRow label="Val loss" value={formatFloat(active.val_loss, 4)} />
                  <MetricRow label="Benchmark" value={formatFloat(active.benchmark_score, 2)} />
                  <MetricRow label="Size" value={formatBytes(active.size_bytes)} />
                  <MetricRow label="Created" value={formatDateTime(active.created_at)} />
                  <MetricRow label="Best" value={active.is_best ? "yes" : "no"} />
                </div>
                {active.path ? (
                  <p className="mono mt-2 break-all text-2xs text-ink-muted">{active.path}</p>
                ) : null}
              </CardContent>
            </Card>

            <div className="space-y-2">
              <Button
                size="sm"
                className="w-full"
                onClick={() => load.mutate(active.id)}
                disabled={load.isPending}
              >
                <Play className="h-3.5 w-3.5" />
                Load as model
              </Button>
              <Button
                size="sm"
                variant="secondary"
                className="w-full"
                onClick={() => evaluate.mutate({ id: active.id, suite: "mmlu" })}
                disabled={evaluate.isPending}
              >
                <Target className="h-3.5 w-3.5" />
                Evaluate (MMLU)
              </Button>
              <Button size="sm" variant="secondary" className="w-full" asChild>
                <Link href={`/chat?model=${active.model_id}`}>
                  <MessageSquare className="h-3.5 w-3.5" />
                  Chat
                </Link>
              </Button>
              <Button
                size="sm"
                variant="secondary"
                className="w-full"
                onClick={() => exportCheckpoint.mutate(active.id)}
                disabled={exportCheckpoint.isPending}
              >
                <Download className="h-3.5 w-3.5" />
                Export
              </Button>
              <Button
                size="sm"
                variant="danger"
                className="w-full"
                onClick={() => {
                  if (confirm(`Delete checkpoint at step ${active.step}?`)) {
                    remove.mutate(active.id);
                    setSelected(null);
                  }
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </Button>
            </div>

            {load.data ? (
              <p className="text-2xs text-good">
                Loaded as model{" "}
                <Link href={`/models/${load.data.id}`} className="underline">
                  {load.data.name}
                </Link>
              </p>
            ) : null}
            {evaluate.data ? (
              <p className="text-2xs text-accent">
                Benchmark queued —{" "}
                <Link href={`/benchmarks/${evaluate.data.id}`} className="underline">
                  open run
                </Link>
              </p>
            ) : null}
          </>
        ) : (
          <p className="text-xs text-ink-muted">Select a checkpoint to inspect it.</p>
        )
      }
    >
      <PageHeader
        title="Checkpoints"
        description="Every checkpoint written by a training run, with the metrics recorded at that step."
      />

      <div className="flex flex-wrap items-center gap-3">
        <Select value={modelFilter} onValueChange={setModelFilter}>
          <SelectTrigger className="h-8 w-[220px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All models</SelectItem>
            {(models?.items ?? []).map((model) => (
              <SelectItem key={model.id} value={model.id}>
                {model.display_name ?? model.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2">
          <Switch checked={bestOnly} onCheckedChange={setBestOnly} id="best-only" />
          <Label htmlFor="best-only" className="normal-case tracking-normal">
            Best only
          </Label>
        </div>
        <span className="mono ml-auto text-2xs text-ink-muted">{checkpoints.length} checkpoints</span>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <LoadingState rows={4} />
      ) : checkpoints.length === 0 ? (
        <EmptyState
          icon={<GitBranch className="h-5 w-5" />}
          title="No checkpoints"
          description="Checkpoints are written during training at the interval set in the run configuration."
          action={<Button size="sm" asChild><Link href="/training">Configure a run</Link></Button>}
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label="Total" value={formatNumber(checkpoints.length)} />
            <StatTile
              label="Best val loss"
              value={formatFloat(
                Math.min(...checkpoints.map((c) => c.val_loss ?? Number.POSITIVE_INFINITY)),
                4,
              )}
              tone="good"
            />
            <StatTile
              label="Disk used"
              value={formatBytes(checkpoints.reduce((sum, c) => sum + (c.size_bytes ?? 0), 0))}
            />
            <StatTile label="Marked best" value={formatNumber(checkpoints.filter((c) => c.is_best).length)} />
          </div>

          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-right">Step</TableHead>
                  <TableHead className="text-right">Epoch</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Train loss</TableHead>
                  <TableHead className="text-right">Val loss</TableHead>
                  <TableHead className="text-right">Benchmark</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Flags</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {checkpoints.map((checkpoint) => (
                  <TableRow
                    key={checkpoint.id}
                    className={cn("cursor-pointer", active?.id === checkpoint.id && "bg-elevated/60")}
                    onClick={() => setSelected(checkpoint)}
                  >
                    <TableCell className="mono text-right text-xs text-ink">
                      {formatNumber(checkpoint.step)}
                    </TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.epoch, 2)}</TableCell>
                    <TableCell className="text-xs text-ink-secondary">{modelName(checkpoint.model_id)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.train_loss, 4)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.val_loss, 4)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.benchmark_score, 1)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatBytes(checkpoint.size_bytes)}</TableCell>
                    <TableCell className="text-2xs text-ink-muted">{formatRelative(checkpoint.created_at)}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {checkpoint.is_best ? <Badge variant="good">best</Badge> : null}
                        <ProvenanceBadge provenance={checkpoint.provenance} compact />
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </>
      )}
    </Workspace>
  );
}
