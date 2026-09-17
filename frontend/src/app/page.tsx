"use client";

import {
  Activity,
  ArrowUpRight,
  Boxes,
  Cpu,
  Gauge,
  HardDrive,
  MemoryStick,
  Target,
  Timer,
  Zap,
} from "lucide-react";
import Link from "next/link";
import * as React from "react";

import {
  ChartFrame,
  CHART_COLORS,
  Sparkline,
  TimeSeriesChart,
  UsageMeter,
} from "@/components/charts/chart-kit";
import { JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { EmptyState, ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Workspace } from "@/components/layout/app-shell";
import { isMockProvider } from "@/lib/api";
import {
  useActiveJobs,
  useBenchmarkRuns,
  useExperiments,
  useHardware,
  useHardwareHistory,
  useModels,
  useSystemInfo,
} from "@/lib/hooks/queries";
import { useLiveTrainingJob } from "@/lib/hooks/use-live-training";
import { useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { useWorkspace } from "@/lib/store";
import {
  formatCompact,
  formatDuration,
  formatEta,
  formatFloat,
  formatNumber,
  formatParameters,
  formatPercent,
  formatRelative,
  formatScientific,
} from "@/lib/format";
import type { HardwareSnapshot } from "@/lib/types";

export default function DashboardPage() {
  const { data: system, error: systemError, refetch: refetchSystem } = useSystemInfo();
  const { data: models } = useModels({ limit: 100 });
  const { data: activeJobs, isLoading: jobsLoading } = useActiveJobs();
  const { data: experiments } = useExperiments({ limit: 6 });
  const { data: benchmarks } = useBenchmarkRuns({ limit: 6, status: "completed" });
  const { data: polledHardware } = useHardware(isMockProvider ? 4000 : 20_000);
  const { data: history } = useHardwareHistory(120, isMockProvider ? 8000 : 20_000);

  const activeModelId = useWorkspace((state) => state.activeModelId);
  const activeModel = models?.items.find((model) => model.id === activeModelId);

  const [liveHardware, setLiveHardware] = React.useState<HardwareSnapshot | null>(null);
  useRealtimeTopic<HardwareSnapshot>("hardware.sample", (event) => setLiveHardware(event.data));
  const hardware = liveHardware ?? polledHardware;

  const runningJob =
    activeJobs?.find((job) => job.status === "running") ?? activeJobs?.[0] ?? null;
  const { job, metrics } = useLiveTrainingJob(runningJob?.id ?? null);

  const chartData = React.useMemo(
    () =>
      metrics.map((point) => ({
        step: point.step,
        loss: point.loss ?? null,
        val_loss: point.val_loss ?? null,
        learning_rate: point.learning_rate ?? null,
        tokens_per_sec: point.tokens_per_sec ?? null,
        gpu_utilization: point.gpu_utilization ?? null,
        vram_gb: point.vram_used_mb ? point.vram_used_mb / 1024 : null,
      })),
    [metrics],
  );

  const hardwareSeries = React.useMemo(
    () =>
      (history?.points ?? []).map((point, index) => ({
        index,
        ts: point.ts,
        gpu: point.gpu_utilization[0] ?? 0,
        vram_gb: (point.gpu_memory_used_mb[0] ?? 0) / 1024,
        cpu: point.cpu_percent,
        ram: point.ram_used_gb,
      })),
    [history],
  );

  const gpu = hardware?.gpus?.[0];
  const progress = job && job.total_steps ? (job.current_step / job.total_steps) * 100 : 0;

  if (systemError) {
    return (
      <div className="p-4">
        <ErrorState error={systemError} onRetry={() => refetchSystem()} />
      </div>
    );
  }

  return (
    <Workspace
      inspectorTitle="Workspace"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Active model</CardTitle>
              {activeModel ? <ProvenanceBadge provenance={activeModel.is_demo ? "simulated" : "measured"} compact /> : null}
            </CardHeader>
            <CardContent className="py-3">
              {activeModel ? (
                <>
                  <Link
                    href={`/models/${activeModel.id}`}
                    className="text-sm font-medium text-ink hover:text-accent"
                  >
                    {activeModel.display_name ?? activeModel.name}
                  </Link>
                  <div className="mono mt-0.5 text-2xs text-ink-muted">{activeModel.name}</div>
                  <div className="mt-3 divide-y divide-border">
                    <MetricRow label="Architecture" value={activeModel.architecture ?? "—"} />
                    <MetricRow label="Parameters" value={formatParameters(activeModel.parameters)} />
                    <MetricRow label="Context" value={formatNumber(activeModel.context_length)} />
                    <MetricRow label="Precision" value={activeModel.precision.toUpperCase()} />
                    <MetricRow label="Format" value={activeModel.format} />
                    <MetricRow label="Status" value={activeModel.status} />
                  </div>
                </>
              ) : (
                <p className="text-xs text-ink-muted">
                  No model selected — pick one from the top bar or import one on the Models page.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>System</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Compute" value={system?.compute_mode?.toUpperCase() ?? "—"} />
                <MetricRow label="Inference engine" value={system?.inference_engine ?? "—"} />
                <MetricRow label="Telemetry" value={system?.hardware_provider ?? "—"} />
                <MetricRow label="Demo mode" value={system?.demo_mode ? "on" : "off"} />
                <MetricRow label="Version" value={system?.version ?? "—"} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Capabilities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1.5 py-3">
              {(system?.capabilities ?? []).map((capability) => (
                <div key={capability.name} className="flex items-start justify-between gap-2">
                  <span className="mono text-2xs text-ink-muted">{capability.name}</span>
                  <Badge variant={capability.available ? "good" : "muted"}>
                    {capability.available ? "available" : "off"}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      }
    >
      <PageHeader
        title="Dashboard"
        description="Live state of the workspace: the active run, its optimisation curves, and the machine underneath it."
        actions={
          <>
            <Button asChild variant="secondary" size="sm">
              <Link href="/training">
                <Activity className="h-3.5 w-3.5" />
                Training
              </Link>
            </Button>
            <Button asChild size="sm">
              <Link href="/training?new=1">Start a run</Link>
            </Button>
          </>
        }
      />

      {system && !system.gpu_available ? (
        <WarningNote>
          No GPU detected — the platform is operating in <strong>CPU mode</strong>. Training and
          inference will be dramatically slower; demo backends are used where enabled.
        </WarningNote>
      ) : null}

      {(system?.demo_mode || isMockProvider) && (
        <WarningNote>
          Demo mode is active. Records marked <ProvenanceBadge provenance="simulated" compact /> are
          simulated so every screen is usable before real weights exist — they are never presented as
          measured results.
        </WarningNote>
      )}

      {/* ---------------------------------------------------------- run tiles */}
      <section>
        <SectionTitle
          right={
            job ? (
              <Link href={`/training/${job.id}`} className="text-2xs text-accent hover:underline">
                Open run →
              </Link>
            ) : null
          }
        >
          Current training job
        </SectionTitle>

        {jobsLoading ? (
          <LoadingState rows={2} />
        ) : job ? (
          <div className="space-y-3">
            <Card>
              <CardHeader>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <Link href={`/training/${job.id}`} className="truncate text-sm font-medium text-ink hover:text-accent">
                      {job.name}
                    </Link>
                    <JobStatusBadge status={job.status} />
                    <ProvenanceBadge provenance={job.provenance} compact />
                    <Badge variant="outline">{job.method}</Badge>
                    <Badge variant="muted">backend: {job.backend}</Badge>
                  </div>
                  <div className="mono mt-1 text-2xs text-ink-muted">
                    {job.model_name ?? job.model_id} · {job.dataset_name ?? job.dataset_id}
                  </div>
                </div>
                <div className="text-right">
                  <div className="mono text-sm text-ink">
                    {formatNumber(job.current_step)} / {formatNumber(job.total_steps)}
                  </div>
                  <div className="text-2xs text-ink-muted">steps</div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <Progress value={progress} tone={job.status === "failed" ? "critical" : "accent"} />
                <div className="flex flex-wrap items-center justify-between gap-3 text-2xs text-ink-muted">
                  <span>
                    epoch <span className="mono text-ink-secondary">{formatFloat(job.current_epoch, 2)}</span> /{" "}
                    {formatFloat(job.total_epochs, 1)}
                  </span>
                  <span>
                    ETA <span className="mono text-ink-secondary">{formatEta(job.eta_seconds)}</span>
                  </span>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
              <StatTile
                label="Training loss"
                value={formatFloat(job.loss, 4)}
                tone="accent"
                sparkline={<Sparkline data={chartData.slice(-40)} dataKey="loss" color={CHART_COLORS[0]} />}
              />
              <StatTile
                label="Validation loss"
                value={formatFloat(job.val_loss, 4)}
                hint={job.best_val_loss ? `best ${formatFloat(job.best_val_loss, 4)}` : undefined}
                sparkline={<Sparkline data={chartData.slice(-40)} dataKey="val_loss" color={CHART_COLORS[1]} />}
              />
              <StatTile label="Learning rate" value={formatScientific(job.learning_rate)} icon={<Gauge className="h-3.5 w-3.5" />} />
              <StatTile
                label="Tokens processed"
                value={formatCompact(job.tokens_processed)}
                icon={<Zap className="h-3.5 w-3.5" />}
              />
              <StatTile
                label="Tokens / sec"
                value={formatNumber(job.tokens_per_sec)}
                hint={job.samples_per_sec ? `${formatFloat(job.samples_per_sec, 2)} samples/s` : undefined}
              />
              <StatTile
                label="Grad norm"
                value={formatFloat(job.grad_norm, 3)}
                icon={<Timer className="h-3.5 w-3.5" />}
              />
            </div>
          </div>
        ) : (
          <EmptyState
            icon={<Activity className="h-5 w-5" />}
            title="No active training job"
            description="Configure a run on the Training page — pick a model, a dataset and a method, then watch the console stream here."
            action={
              <Button asChild size="sm">
                <Link href="/training">Configure a run</Link>
              </Button>
            }
          />
        )}
      </section>

      {/* ------------------------------------------------------------ charts */}
      <section>
        <SectionTitle>Optimisation</SectionTitle>
        <div className="grid gap-3 lg:grid-cols-2">
          <ChartFrame
            title="Training & validation loss"
            subtitle="lower is better · same scale, one axis"
            provenance={job?.provenance}
            series={[
              { key: "loss", label: "train", color: CHART_COLORS[0], area: true },
              { key: "val_loss", label: "validation", color: CHART_COLORS[1] },
            ]}
          >
            {chartData.length ? (
              <TimeSeriesChart
                data={chartData}
                xKey="step"
                series={[
                  { key: "loss", label: "train", color: CHART_COLORS[0], area: true },
                  { key: "val_loss", label: "validation", color: CHART_COLORS[1] },
                ]}
                tooltipLabelFormatter={(value) => `step ${value}`}
              />
            ) : (
              <NoSeries />
            )}
          </ChartFrame>

          <ChartFrame
            title="Learning rate"
            subtitle={job ? `${job.config?.lr_scheduler ?? "—"} schedule` : undefined}
            provenance={job?.provenance}
            series={[{ key: "learning_rate", label: "lr", color: CHART_COLORS[6] }]}
          >
            {chartData.length ? (
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
            ) : (
              <NoSeries />
            )}
          </ChartFrame>

          <ChartFrame
            title="Throughput"
            subtitle="tokens per second"
            provenance={job?.provenance}
            series={[{ key: "tokens_per_sec", label: "tokens/s", color: CHART_COLORS[2] }]}
          >
            {chartData.length ? (
              <TimeSeriesChart
                data={chartData}
                xKey="step"
                series={[
                  { key: "tokens_per_sec", label: "tokens/s", color: CHART_COLORS[2], area: true },
                ]}
                yTickFormatter={(value) => formatCompact(value)}
                tooltipLabelFormatter={(value) => `step ${value}`}
              />
            ) : (
              <NoSeries />
            )}
          </ChartFrame>

          <ChartFrame
            title="GPU utilisation & VRAM"
            subtitle={
              hardware?.provenance === "simulated" ? "simulated telemetry" : "measured via NVML"
            }
            provenance={hardware?.provenance}
            series={[
              { key: "gpu", label: "GPU %", color: CHART_COLORS[3] },
              { key: "vram_gb", label: "VRAM GB", color: CHART_COLORS[4] },
            ]}
          >
            {hardwareSeries.length ? (
              <TimeSeriesChart
                data={hardwareSeries}
                xKey="index"
                series={[
                  { key: "gpu", label: "GPU %", color: CHART_COLORS[3], area: true, unit: "%" },
                  { key: "vram_gb", label: "VRAM", color: CHART_COLORS[4], unit: " GB" },
                ]}
                xTickFormatter={() => ""}
                tooltipLabelFormatter={() => "sample"}
              />
            ) : (
              <NoSeries />
            )}
          </ChartFrame>
        </div>
      </section>

      {/* ---------------------------------------------------------- hardware */}
      <section>
        <SectionTitle
          right={
            <Link href="/hardware" className="text-2xs text-accent hover:underline">
              Hardware →
            </Link>
          }
        >
          Machine
        </SectionTitle>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatTile
            label="GPU utilisation"
            value={formatPercent(gpu?.utilization ?? 0, 0)}
            hint={gpu?.name}
            icon={<Zap className="h-3.5 w-3.5" />}
            tone={(gpu?.utilization ?? 0) > 95 ? "warning" : "default"}
          />
          <StatTile
            label="VRAM"
            value={gpu ? `${(gpu.memory_used_mb / 1024).toFixed(1)}` : "—"}
            unit={gpu ? `/ ${(gpu.memory_total_mb / 1024).toFixed(0)} GB` : undefined}
            icon={<HardDrive className="h-3.5 w-3.5" />}
            sparkline={
              gpu ? (
                <UsageMeter used={gpu.memory_used_mb / 1024} total={gpu.memory_total_mb / 1024} />
              ) : undefined
            }
          />
          <StatTile
            label="CPU"
            value={formatPercent(hardware?.cpu_percent ?? 0, 0)}
            hint={hardware?.cpu_model ?? undefined}
            icon={<Cpu className="h-3.5 w-3.5" />}
          />
          <StatTile
            label="System RAM"
            value={hardware ? hardware.ram_used_gb.toFixed(1) : "—"}
            unit={hardware ? `/ ${hardware.ram_total_gb.toFixed(0)} GB` : undefined}
            icon={<MemoryStick className="h-3.5 w-3.5" />}
            sparkline={
              hardware ? <UsageMeter used={hardware.ram_used_gb} total={hardware.ram_total_gb} /> : undefined
            }
          />
        </div>
      </section>

      {/* ------------------------------------------- experiments + benchmarks */}
      <section className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Recent experiments</CardTitle>
            <Link href="/experiments" className="text-2xs text-accent hover:underline">
              All <ArrowUpRight className="inline h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent className="p-0">
            {experiments?.items.length ? (
              <ul className="divide-y divide-border">
                {experiments.items.map((experiment) => (
                  <li key={experiment.id}>
                    <Link
                      href={`/experiments/${experiment.id}`}
                      className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-elevated/60"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="truncate text-xs text-ink">{experiment.name}</span>
                          <JobStatusBadge status={experiment.status} />
                        </div>
                        <div className="mono mt-0.5 text-2xs text-ink-muted">
                          {experiment.method} · {formatRelative(experiment.created_at)}
                          {experiment.duration_seconds
                            ? ` · ${formatDuration(experiment.duration_seconds)}`
                            : ""}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="mono text-xs text-ink-secondary">
                          {formatFloat(experiment.final_val_loss, 3)}
                        </div>
                        <div className="text-2xs text-ink-muted">val loss</div>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="p-4">
                <p className="text-xs text-ink-muted">
                  No experiments yet — every training run records one automatically.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Latest benchmark scores</CardTitle>
            <Link href="/benchmarks" className="text-2xs text-accent hover:underline">
              All <ArrowUpRight className="inline h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent className="p-0">
            {benchmarks?.items.length ? (
              <ul className="divide-y divide-border">
                {benchmarks.items.map((run) => (
                  <li key={run.id}>
                    <Link
                      href={`/benchmarks/${run.id}`}
                      className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-elevated/60"
                    >
                      <Target className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="truncate text-xs text-ink">{run.suite_label}</span>
                          {!run.config?.official_split ? (
                            <Badge variant="warning">sample</Badge>
                          ) : null}
                        </div>
                        <div className="mono mt-0.5 truncate text-2xs text-ink-muted">
                          {models?.items.find((model) => model.id === run.model_id)?.display_name ??
                            run.model_id}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="mono text-xs text-ink">
                          {run.overall_score !== null && run.overall_score !== undefined
                            ? `${(run.overall_score * 100).toFixed(1)}%`
                            : "—"}
                        </div>
                        <div className="text-2xs text-ink-muted">{run.config?.num_examples ?? 0} items</div>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="p-4">
                <p className="text-xs text-ink-muted">
                  No benchmark runs yet. Results appear here only after a suite actually runs.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section>
        <SectionTitle
          right={
            <Link href="/models" className="text-2xs text-accent hover:underline">
              Models →
            </Link>
          }
        >
          Model catalog
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {(models?.items ?? []).slice(0, 6).map((model) => (
            <Link key={model.id} href={`/models/${model.id}`}>
              <Card className="transition-colors hover:border-border-strong">
                <CardContent className="space-y-2 p-3.5">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium text-ink">
                        {model.display_name ?? model.name}
                      </div>
                      <div className="mono truncate text-2xs text-ink-muted">{model.architecture ?? model.format}</div>
                    </div>
                    <Boxes className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge variant="outline">{formatParameters(model.parameters)}</Badge>
                    <Badge variant="muted">{model.precision.toUpperCase()}</Badge>
                    <Badge variant="muted">{formatCompact(model.context_length ?? 0)} ctx</Badge>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </section>
    </Workspace>
  );
}

function NoSeries() {
  return (
    <div className="flex h-full items-center justify-center">
      <p className="text-2xs text-ink-muted">No data yet — start a run to populate this chart.</p>
    </div>
  );
}
