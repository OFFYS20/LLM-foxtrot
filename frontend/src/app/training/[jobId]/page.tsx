"use client";

import { ArrowLeft, Pause, Play, Save, Square } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import * as React from "react";

import { CHART_COLORS, ChartFrame, TimeSeriesChart } from "@/components/charts/chart-kit";
import { JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { TrainingConsole } from "@/components/training/console";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useTrainingLogs, useTrainingMutations } from "@/lib/hooks/queries";
import { useLiveTrainingJob, useTrainingConsole } from "@/lib/hooks/use-live-training";
import {
  formatCompact,
  formatDateTime,
  formatEta,
  formatFloat,
  formatNumber,
  formatPercent,
  formatScientific,
} from "@/lib/format";

export default function TrainingJobPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params?.jobId ?? null;

  const { job, metrics, isLoading, error, refetch } = useLiveTrainingJob(jobId);
  const { data: logData } = useTrainingLogs(jobId);
  const seedLines = React.useMemo(
    () =>
      (logData?.entries ?? []).map((entry) => ({
        message: entry.message,
        level: String(entry.level),
        ts: entry.ts,
      })),
    [logData],
  );
  const lines = useTrainingConsole(jobId, seedLines);
  const { pause, resume, stop, checkpoint } = useTrainingMutations();

  const chartData = React.useMemo(
    () =>
      metrics.map((point) => ({
        step: point.step,
        loss: point.loss ?? null,
        val_loss: point.val_loss ?? null,
        learning_rate: point.learning_rate ?? null,
        grad_norm: point.grad_norm ?? null,
        tokens_per_sec: point.tokens_per_sec ?? null,
        gpu_utilization: point.gpu_utilization ?? null,
        vram_gb: point.vram_used_mb ? point.vram_used_mb / 1024 : null,
      })),
    [metrics],
  );

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={refetch} /></div>;
  if (isLoading || !job) return <div className="p-4"><LoadingState rows={5} /></div>;

  const progress = job.total_steps ? (job.current_step / job.total_steps) * 100 : 0;
  const isActive = job.status === "running" || job.status === "paused";
  const config = job.config;

  return (
    <Workspace
      inspectorTitle="Run configuration"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Job</CardTitle>
              <ProvenanceBadge provenance={job.provenance} compact />
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Job id" value={job.id} />
                <MetricRow label="Backend" value={job.backend} />
                <MetricRow label="Model" value={job.model_name ?? job.model_id} />
                <MetricRow label="Dataset" value={job.dataset_name ?? job.dataset_id} />
                <MetricRow label="Started" value={formatDateTime(job.started_at)} />
                <MetricRow label="Ended" value={formatDateTime(job.ended_at)} />
                <MetricRow
                  label="Experiment"
                  value={
                    job.experiment_id ? (
                      <Link href={`/experiments/${job.experiment_id}`} className="text-accent hover:underline">
                        open
                      </Link>
                    ) : (
                      "—"
                    )
                  }
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Method" value={config.method} />
                <MetricRow label="Epochs" value={formatFloat(config.epochs, 1)} />
                <MetricRow label="Batch size" value={formatNumber(config.batch_size)} />
                <MetricRow label="Grad accumulation" value={formatNumber(config.gradient_accumulation_steps)} />
                <MetricRow
                  label="Effective batch"
                  value={formatNumber(config.batch_size * config.gradient_accumulation_steps)}
                />
                <MetricRow label="Learning rate" value={formatScientific(config.learning_rate)} />
                <MetricRow label="Warmup" value={formatNumber(config.warmup_steps)} />
                <MetricRow label="Scheduler" value={config.lr_scheduler} />
                <MetricRow label="Optimizer" value={config.optimizer} />
                <MetricRow label="Precision" value={config.precision?.toUpperCase()} />
                <MetricRow label="Max seq len" value={formatNumber(config.max_sequence_length)} />
                <MetricRow label="Seed" value={formatNumber(config.seed)} />
                {config.method === "lora" || config.method === "qlora" ? (
                  <>
                    <MetricRow label="LoRA rank" value={formatNumber(config.lora?.rank)} />
                    <MetricRow label="LoRA alpha" value={formatNumber(config.lora?.alpha)} />
                    <MetricRow label="LoRA dropout" value={formatFloat(config.lora?.dropout, 3)} />
                    <MetricRow label="Target modules" value={(config.lora?.target_modules ?? []).join(", ")} />
                  </>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </>
      }
    >
      <div>
        <Link href="/training" className="mb-2 inline-flex items-center gap-1 text-2xs text-ink-muted hover:text-ink">
          <ArrowLeft className="h-3 w-3" />
          Training
        </Link>
        <PageHeader
          title={job.name}
          description={`${job.model_name ?? job.model_id} · ${job.dataset_name ?? job.dataset_id}`}
          actions={
            <>
              <JobStatusBadge status={job.status} />
              <ProvenanceBadge provenance={job.provenance} />
              <Badge variant="outline">{job.method}</Badge>
            </>
          }
        />
      </div>

      {job.error ? (
        <div className="rounded-md border border-critical/40 bg-critical/5 px-3 py-2">
          <p className="mono text-xs text-critical">{job.error}</p>
        </div>
      ) : null}

      {job.warnings?.length ? (
        <div className="space-y-2">
          {job.warnings.map((warning) => (
            <WarningNote key={warning}>{warning}</WarningNote>
          ))}
        </div>
      ) : null}

      <Card>
        <CardContent className="space-y-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="mono text-xs text-ink-secondary">
              step {formatNumber(job.current_step)} / {formatNumber(job.total_steps)} ·{" "}
              epoch {formatFloat(job.current_epoch, 2)} / {formatFloat(job.total_epochs, 1)}
            </div>
            <div className="flex items-center gap-2">
              {job.status === "paused" ? (
                <Button size="sm" onClick={() => resume.mutate(job.id)} disabled={resume.isPending}>
                  <Play className="h-3.5 w-3.5" />
                  Resume
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => pause.mutate(job.id)}
                  disabled={pause.isPending || job.status !== "running"}
                >
                  <Pause className="h-3.5 w-3.5" />
                  Pause
                </Button>
              )}
              <Button
                size="sm"
                variant="secondary"
                onClick={() => checkpoint.mutate(job.id)}
                disabled={checkpoint.isPending || !isActive}
              >
                <Save className="h-3.5 w-3.5" />
                Save checkpoint
              </Button>
              <Button
                size="sm"
                variant="danger"
                onClick={() => stop.mutate(job.id)}
                disabled={stop.isPending || !isActive}
              >
                <Square className="h-3.5 w-3.5" />
                Stop
              </Button>
            </div>
          </div>
          <Progress value={progress} tone={job.status === "failed" ? "critical" : "accent"} />
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <StatTile label="Loss" value={formatFloat(job.loss, 4)} tone="accent" />
        <StatTile
          label="Validation loss"
          value={formatFloat(job.val_loss, 4)}
          hint={job.best_val_loss ? `best ${formatFloat(job.best_val_loss, 4)}` : undefined}
        />
        <StatTile label="Learning rate" value={formatScientific(job.learning_rate)} />
        <StatTile label="Grad norm" value={formatFloat(job.grad_norm, 3)} />
        <StatTile label="Tokens/sec" value={formatNumber(job.tokens_per_sec)} />
        <StatTile label="Samples/sec" value={formatFloat(job.samples_per_sec, 2)} />
        <StatTile label="Tokens processed" value={formatCompact(job.tokens_processed)} />
        <StatTile label="GPU utilisation" value={formatPercent(job.gpu_utilization ?? 0, 1)} />
        <StatTile
          label="VRAM"
          value={job.vram_used_mb ? (job.vram_used_mb / 1024).toFixed(1) : "—"}
          unit="GB"
        />
        <StatTile label="ETA" value={formatEta(job.eta_seconds)} />
        <StatTile label="Epoch" value={formatFloat(job.current_epoch, 2)} />
        <StatTile label="Step" value={formatNumber(job.current_step)} />
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <ChartFrame
          title="Loss"
          provenance={job.provenance}
          series={[
            { key: "loss", label: "train", color: CHART_COLORS[0], area: true },
            { key: "val_loss", label: "validation", color: CHART_COLORS[1] },
          ]}
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

        <ChartFrame
          title="Learning rate"
          subtitle={`${config.lr_scheduler} · warmup ${config.warmup_steps}`}
          provenance={job.provenance}
        >
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[
              {
                key: "learning_rate",
                label: "lr",
                color: CHART_COLORS[6],
                format: (value) => value.toExponential(2),
              },
            ]}
            yTickFormatter={(value) => value.toExponential(0)}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>

        <ChartFrame title="Gradient norm" provenance={job.provenance}>
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[{ key: "grad_norm", label: "grad norm", color: CHART_COLORS[4] }]}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>

        <ChartFrame title="Throughput" subtitle="tokens per second" provenance={job.provenance}>
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[{ key: "tokens_per_sec", label: "tokens/s", color: CHART_COLORS[2], area: true }]}
            yTickFormatter={(value) => formatCompact(value)}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>

        <ChartFrame title="GPU utilisation" provenance={job.provenance}>
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[{ key: "gpu_utilization", label: "GPU", color: CHART_COLORS[3], area: true, unit: "%" }]}
            yDomain={[0, 100]}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>

        <ChartFrame title="VRAM usage" provenance={job.provenance}>
          <TimeSeriesChart
            data={chartData}
            xKey="step"
            series={[{ key: "vram_gb", label: "VRAM", color: CHART_COLORS[5], area: true, unit: " GB" }]}
            tooltipLabelFormatter={(value) => `step ${value}`}
          />
        </ChartFrame>
      </div>

      <TrainingConsole lines={lines} height={360} />
    </Workspace>
  );
}
