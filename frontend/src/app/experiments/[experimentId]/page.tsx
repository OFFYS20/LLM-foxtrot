"use client";

import { ArrowLeft, Copy, Save } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import * as React from "react";

import { CHART_COLORS, ChartFrame, TimeSeriesChart } from "@/components/charts/chart-kit";
import { JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useCheckpoints,
  useExperiment,
  useExperimentMutations,
  useTrainingMetrics,
} from "@/lib/hooks/queries";
import {
  formatBytes,
  formatDateTime,
  formatDuration,
  formatFloat,
  formatNumber,
  formatScientific,
} from "@/lib/format";

export default function ExperimentDetailPage() {
  const params = useParams<{ experimentId: string }>();
  const experimentId = params?.experimentId ?? null;

  const { data: experiment, isLoading, error, refetch } = useExperiment(experimentId);
  const { data: metrics } = useTrainingMetrics(experiment?.job_id ?? null);
  const { data: checkpoints } = useCheckpoints(
    experimentId ? { experiment_id: experimentId, limit: 100 } : undefined,
  );
  const { update, duplicate } = useExperimentMutations();

  const [notes, setNotes] = React.useState("");
  React.useEffect(() => {
    if (experiment) setNotes(experiment.notes ?? "");
  }, [experiment]);

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !experiment) return <div className="p-4"><LoadingState rows={4} /></div>;

  const hyper = experiment.hyperparameters ?? {};
  const chartData = (metrics ?? []).map((point) => ({
    step: point.step,
    loss: point.loss ?? null,
    val_loss: point.val_loss ?? null,
  }));

  return (
    <Workspace
      inspectorTitle="Experiment"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Method" value={String(hyper.method ?? experiment.method)} />
                <MetricRow label="Epochs" value={formatFloat(hyper.epochs, 1)} />
                <MetricRow label="Batch size" value={formatNumber(hyper.batch_size)} />
                <MetricRow label="Grad accumulation" value={formatNumber(hyper.gradient_accumulation_steps)} />
                <MetricRow label="Learning rate" value={formatScientific(hyper.learning_rate)} />
                <MetricRow label="Scheduler" value={String(hyper.lr_scheduler ?? "—")} />
                <MetricRow label="Optimizer" value={String(hyper.optimizer ?? "—")} />
                <MetricRow label="Precision" value={String(hyper.precision ?? "—").toUpperCase()} />
                <MetricRow label="Max seq len" value={formatNumber(hyper.max_sequence_length)} />
                <MetricRow label="Seed" value={formatNumber(hyper.seed)} />
                {hyper.lora ? (
                  <>
                    <MetricRow label="LoRA rank" value={formatNumber(hyper.lora.rank)} />
                    <MetricRow label="LoRA alpha" value={formatNumber(hyper.lora.alpha)} />
                    <MetricRow label="LoRA dropout" value={formatFloat(hyper.lora.dropout, 3)} />
                  </>
                ) : null}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Notes</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 py-3">
              <Textarea
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                className="h-28 text-xs"
                placeholder="What did you learn from this run?"
              />
              <Button
                size="xs"
                variant="secondary"
                className="w-full"
                onClick={() => update.mutate({ id: experiment.id, payload: { notes } })}
                disabled={update.isPending}
              >
                <Save className="h-3 w-3" />
                Save notes
              </Button>
            </CardContent>
          </Card>

          <Button
            size="sm"
            variant="secondary"
            className="w-full"
            onClick={() => duplicate.mutate({ id: experiment.id, name: `${experiment.name} (copy)` })}
            disabled={duplicate.isPending}
          >
            <Copy className="h-3.5 w-3.5" />
            Duplicate configuration
          </Button>
          {experiment.job_id ? (
            <Button size="sm" variant="secondary" className="w-full" asChild>
              <Link href={`/training/${experiment.job_id}`}>Open training run</Link>
            </Button>
          ) : null}
        </>
      }
    >
      <div>
        <Link href="/experiments" className="mb-2 inline-flex items-center gap-1 text-2xs text-ink-muted hover:text-ink">
          <ArrowLeft className="h-3 w-3" />
          Experiments
        </Link>
        <PageHeader
          title={experiment.name}
          description={`${experiment.model_name ?? "—"} · ${experiment.dataset_name ?? "—"}`}
          actions={
            <>
              <JobStatusBadge status={experiment.status} />
              <ProvenanceBadge provenance={experiment.provenance} />
              <Badge variant="outline">{experiment.method}</Badge>
            </>
          }
        />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatTile label="Final train loss" value={formatFloat(experiment.final_train_loss, 4)} tone="accent" />
        <StatTile label="Final val loss" value={formatFloat(experiment.final_val_loss, 4)} />
        <StatTile label="Duration" value={formatDuration(experiment.duration_seconds)} />
        <StatTile label="Checkpoints" value={formatNumber(experiment.checkpoint_count)} />
        <StatTile label="Started" value={formatDateTime(experiment.started_at)} />
        <StatTile label="Ended" value={formatDateTime(experiment.ended_at)} />
      </div>

      {chartData.length ? (
        <ChartFrame
          title="Loss curve"
          provenance={experiment.provenance}
          series={[
            { key: "loss", label: "train", color: CHART_COLORS[0], area: true },
            { key: "val_loss", label: "validation", color: CHART_COLORS[1] },
          ]}
          height={240}
        >
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[
              { key: "loss", label: "train", color: CHART_COLORS[0], area: true },
              { key: "val_loss", label: "validation", color: CHART_COLORS[1] },
            ]}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Checkpoints</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {!checkpoints?.items.length ? (
            <p className="p-4 text-xs text-ink-muted">No checkpoints recorded for this experiment.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-right">Step</TableHead>
                  <TableHead className="text-right">Epoch</TableHead>
                  <TableHead className="text-right">Train loss</TableHead>
                  <TableHead className="text-right">Val loss</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead>Flags</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {checkpoints.items.map((checkpoint) => (
                  <TableRow key={checkpoint.id}>
                    <TableCell className="mono text-right text-xs">{formatNumber(checkpoint.step)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.epoch, 2)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.train_loss, 4)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.val_loss, 4)}</TableCell>
                    <TableCell className="mono text-right text-xs">{formatBytes(checkpoint.size_bytes)}</TableCell>
                    <TableCell>
                      {checkpoint.is_best ? <Badge variant="good">best</Badge> : null}
                      {checkpoint.id === experiment.best_checkpoint_id ? (
                        <Badge variant="accent">experiment best</Badge>
                      ) : null}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Benchmark results for this model</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {!experiment.benchmark_runs.length ? (
            <p className="p-4 text-xs text-ink-muted">
              No benchmark runs recorded for this model yet.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Suite</TableHead>
                  <TableHead className="text-right">Score</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead>Data</TableHead>
                  <TableHead>Run</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiment.benchmark_runs.map((run) => (
                  <TableRow key={run.id}>
                    <TableCell className="text-xs text-ink">{run.suite_label}</TableCell>
                    <TableCell className="mono text-right text-xs">
                      {run.overall_score ? `${(run.overall_score * 100).toFixed(1)}%` : "—"}
                    </TableCell>
                    <TableCell className="mono text-right text-xs">
                      {run.accuracy ? `${(run.accuracy * 100).toFixed(1)}%` : "—"}
                    </TableCell>
                    <TableCell>
                      <Badge variant={run.official_split ? "good" : "warning"}>
                        {run.official_split ? "official split" : "sample"}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Link href={`/benchmarks/${run.id}`} className="text-2xs text-accent hover:underline">
                        open →
                      </Link>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </Workspace>
  );
}
