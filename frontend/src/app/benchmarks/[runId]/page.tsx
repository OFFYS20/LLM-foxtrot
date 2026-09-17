"use client";

import { ArrowLeft, Filter } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import * as React from "react";

import { BarSeriesChart, CHART_COLORS, ChartFrame } from "@/components/charts/chart-kit";
import {
  CorrectnessBadge,
  DataSourceBadge,
  JobStatusBadge,
  ProvenanceBadge,
} from "@/components/common/badges";
import { CopyButton } from "@/components/common/copy-button";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useBenchmarkRun } from "@/lib/hooks/queries";
import { BENCHMARK_CATEGORY_LABELS } from "@/lib/constants";
import { formatDuration, formatFloat, formatNumber, truncate } from "@/lib/format";
import type { BenchmarkItem } from "@/lib/types";

export default function BenchmarkResultPage() {
  const params = useParams<{ runId: string }>();
  const runId = params?.runId ?? null;
  const [onlyIncorrect, setOnlyIncorrect] = React.useState(false);
  const [inspecting, setInspecting] = React.useState<BenchmarkItem | null>(null);

  const { data: run, isLoading, error, refetch } = useBenchmarkRun(runId, onlyIncorrect, 4000);
  const isActive = run?.status === "running" || run?.status === "queued";

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !run) return <div className="p-4"><LoadingState rows={5} /></div>;

  const categoryData = Object.entries(run.category_scores ?? {}).map(([category, score]) => ({
    category: BENCHMARK_CATEGORY_LABELS[category] ?? category,
    score: Number((score * 100).toFixed(2)),
  }));

  const progressPct = run.total_items ? (run.completed_items / run.total_items) * 100 : 0;

  return (
    <Workspace
      inspectorTitle="Run metadata"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Suite" value={run.suite_label} />
                <MetricRow label="Category" value={BENCHMARK_CATEGORY_LABELS[run.category] ?? run.category} />
                <MetricRow label="Examples" value={formatNumber(run.config?.num_examples)} />
                <MetricRow label="Few-shot" value={formatNumber(run.config?.few_shot)} />
                <MetricRow label="Temperature" value={formatFloat(run.config?.temperature, 2)} />
                <MetricRow label="Max tokens" value={formatNumber(run.config?.max_tokens)} />
                <MetricRow label="Seed" value={formatNumber(run.config?.seed)} />
                <MetricRow label="Data source" value={run.config?.data_source ?? "—"} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Provenance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 py-3">
              <ProvenanceBadge provenance={run.provenance} />
              <DataSourceBadge official={run.config?.official_split} dataSource={run.config?.data_source} />
              {run.config?.suite_notes ? (
                <p className="text-2xs text-ink-muted">{run.config.suite_notes}</p>
              ) : null}
            </CardContent>
          </Card>

          <Button variant="secondary" size="sm" className="w-full" asChild>
            <Link href={`/models/${run.model_id}`}>Open model</Link>
          </Button>
        </>
      }
    >
      <div>
        <Link href="/benchmarks" className="mb-2 inline-flex items-center gap-1 text-2xs text-ink-muted hover:text-ink">
          <ArrowLeft className="h-3 w-3" />
          Benchmarks
        </Link>
        <PageHeader
          title={run.name}
          description={`${run.suite_label} · ${run.model_name ?? run.model_id}`}
          actions={
            <>
              <JobStatusBadge status={run.status} />
              <ProvenanceBadge provenance={run.provenance} />
            </>
          }
        />
      </div>

      {run.error ? (
        <div className="rounded-md border border-critical/40 bg-critical/5 px-3 py-2">
          <p className="mono text-xs text-critical">{run.error}</p>
        </div>
      ) : null}

      {!run.config?.official_split ? (
        <WarningNote>
          This run used <strong>{run.config?.data_source ?? "a non-official source"}</strong>. The score
          below describes this item pool only — it is not an official {run.suite_label} result and is
          not comparable to published leaderboard numbers.
        </WarningNote>
      ) : null}

      {isActive ? (
        <Card>
          <CardContent className="space-y-2 py-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-ink-secondary">Running…</span>
              <span className="mono text-ink-muted">
                {run.completed_items}/{run.total_items}
              </span>
            </div>
            <Progress value={progressPct} />
          </CardContent>
        </Card>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatTile
          label="Overall score"
          value={run.overall_score !== null && run.overall_score !== undefined ? `${(run.overall_score * 100).toFixed(1)}%` : "—"}
          tone="accent"
        />
        <StatTile
          label="Accuracy"
          value={run.accuracy !== null && run.accuracy !== undefined ? `${(run.accuracy * 100).toFixed(1)}%` : "—"}
        />
        <StatTile
          label="Pass@1"
          value={run.pass_at_1 !== null && run.pass_at_1 !== undefined ? `${(run.pass_at_1 * 100).toFixed(1)}%` : "n/a"}
          hint={run.pass_at_1 === null ? "requires sandboxed execution" : undefined}
        />
        <StatTile label="Avg latency" value={formatFloat(run.avg_latency_ms, 0)} unit="ms" />
        <StatTile label="Avg tokens/s" value={formatFloat(run.avg_tokens_per_sec, 1)} />
        <StatTile label="Total runtime" value={formatDuration(run.runtime_seconds)} />
      </div>

      {categoryData.length > 0 ? (
        <ChartFrame
          title="Score by category"
          subtitle="percentage of items graded correct"
          provenance={run.provenance}
          height={Math.max(160, categoryData.length * 34 + 40)}
        >
          <BarSeriesChart
            data={categoryData}
            xKey="category"
            layout="vertical"
            series={[{ key: "score", label: "score", color: CHART_COLORS[0], unit: "%" }]}
            valueDomain={[0, 100]}
            yTickFormatter={(value) => `${value}%`}
          />
        </ChartFrame>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Items ({run.items.length})</CardTitle>
          <Button
            size="xs"
            variant={onlyIncorrect ? "default" : "secondary"}
            onClick={() => setOnlyIncorrect((value) => !value)}
          >
            <Filter className="h-3 w-3" />
            {onlyIncorrect ? "Showing incorrect only" : "Filter incorrect"}
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10 text-right">#</TableHead>
                <TableHead>Question</TableHead>
                <TableHead>Expected</TableHead>
                <TableHead>Model response</TableHead>
                <TableHead>Result</TableHead>
                <TableHead className="text-right">Score</TableHead>
                <TableHead className="text-right">Latency</TableHead>
                <TableHead className="text-right">Tokens</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {run.items.map((item) => (
                <TableRow
                  key={item.index}
                  className="cursor-pointer"
                  onClick={() => setInspecting(item)}
                >
                  <TableCell className="mono text-right text-2xs text-ink-muted">{item.index + 1}</TableCell>
                  <TableCell className="max-w-[280px] text-xs text-ink-secondary">
                    {truncate(item.question, 120)}
                  </TableCell>
                  <TableCell className="mono max-w-[140px] text-2xs text-ink-muted">
                    {truncate(item.expected, 60)}
                  </TableCell>
                  <TableCell className="mono max-w-[240px] text-2xs text-ink-secondary">
                    {truncate(item.response, 100)}
                  </TableCell>
                  <TableCell><CorrectnessBadge correct={item.correct} /></TableCell>
                  <TableCell className="mono text-right text-xs">{formatFloat(item.score, 2)}</TableCell>
                  <TableCell className="mono text-right text-xs">{formatFloat(item.latency_ms, 0)}</TableCell>
                  <TableCell className="mono text-right text-xs">{formatNumber(item.tokens_used)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {run.items.length === 0 ? (
            <p className="p-4 text-xs text-ink-muted">
              {onlyIncorrect ? "No incorrect items — every graded item passed." : "No items recorded yet."}
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Dialog open={Boolean(inspecting)} onOpenChange={(open) => !open && setInspecting(null)}>
        <DialogContent wide>
          <DialogHeader>
            <div className="flex items-center gap-2">
              <DialogTitle>Item #{(inspecting?.index ?? 0) + 1}</DialogTitle>
              {inspecting ? <CorrectnessBadge correct={inspecting.correct} /> : null}
              {inspecting ? <Badge variant="muted">score {formatFloat(inspecting.score, 2)}</Badge> : null}
            </div>
          </DialogHeader>
          <div className="scrollbar-thin max-h-[70vh] space-y-4 overflow-y-auto p-4">
            <section>
              <div className="mb-1 flex items-center justify-between">
                <h4 className="text-2xs uppercase tracking-[0.14em] text-ink-muted">Full prompt</h4>
                <CopyButton value={inspecting?.prompt ?? ""} size="xs" />
              </div>
              <pre className="mono whitespace-pre-wrap rounded-md border border-border bg-base p-3 text-2xs text-ink-secondary">
                {inspecting?.prompt}
              </pre>
            </section>
            <section>
              <h4 className="mb-1 text-2xs uppercase tracking-[0.14em] text-ink-muted">Expected answer</h4>
              <pre className="mono whitespace-pre-wrap rounded-md border border-border bg-base p-3 text-2xs text-good">
                {inspecting?.expected}
              </pre>
            </section>
            <section>
              <div className="mb-1 flex items-center justify-between">
                <h4 className="text-2xs uppercase tracking-[0.14em] text-ink-muted">Generated output</h4>
                <CopyButton value={inspecting?.raw_output ?? ""} size="xs" />
              </div>
              <pre className="mono whitespace-pre-wrap rounded-md border border-border bg-base p-3 text-2xs text-ink-secondary">
                {inspecting?.raw_output || inspecting?.response}
              </pre>
            </section>
            <section className="grid grid-cols-3 gap-3">
              <StatTile label="Latency" value={formatFloat(inspecting?.latency_ms, 0)} unit="ms" />
              <StatTile label="Tokens used" value={formatNumber(inspecting?.tokens_used)} />
              <StatTile label="Category" value={inspecting?.category ?? "—"} />
            </section>
          </div>
        </DialogContent>
      </Dialog>
    </Workspace>
  );
}
