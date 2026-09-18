"use client";

import { Gauge, Scale } from "lucide-react";
import * as React from "react";

import {
  BarSeriesChart,
  CHART_COLORS,
  ChartFrame,
  RadarProfileChart,
} from "@/components/charts/chart-kit";
import { DataSourceBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { EmptyState, ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useCompareModels, useLeaderboard, useModels } from "@/lib/hooks/queries";
import {
  formatBytes,
  formatCompact,
  formatFloat,
  formatNumber,
  formatParameters,
} from "@/lib/format";
import type { ComparisonRow } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function EvaluationsPage() {
  const { data: models } = useModels({ limit: 200 });
  const { data: leaderboard, isLoading, error, refetch } = useLeaderboard();
  const compare = useCompareModels();

  const [selected, setSelected] = React.useState<string[]>([]);

  React.useEffect(() => {
    if (selected.length === 0 && (models?.items.length ?? 0) >= 2) {
      setSelected(models!.items.slice(0, 3).map((model) => model.id));
    }
  }, [models, selected.length]);

  const toggle = (id: string) => {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : current.length < 8
          ? [...current, id]
          : current,
    );
  };

  const runComparison = () => {
    if (selected.length >= 2) compare.mutate({ modelIds: selected });
  };

  const comparison = compare.data;

  const radarData = React.useMemo(() => {
    if (!comparison) return [];
    const categories = new Set<string>();
    comparison.rows.forEach((row) => {
      Object.keys(row.benchmark_breakdown ?? {}).forEach((key) => categories.add(key));
    });
    return Array.from(categories).map((category) => {
      const point: Record<string, unknown> = { category };
      comparison.rows.forEach((row, index) => {
        point[`m${index}`] = row.benchmark_breakdown?.[category] ?? 0;
      });
      return point;
    });
  }, [comparison]);

  const suites = leaderboard?.suites ?? [];

  return (
    <Workspace
      inspectorTitle="Select models"
      inspector={
        <>
          <p className="text-xs text-ink-muted">
            Pick two or more models to compare. Metrics are reported individually — Foxtrot never
            collapses a model into a single quality number.
          </p>
          <div className="space-y-1">
            {(models?.items ?? []).map((model) => {
              const index = selected.indexOf(model.id);
              const isSelected = index >= 0;
              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => toggle(model.id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left text-xs transition-colors",
                    isSelected
                      ? "border-accent/60 bg-accent/10 text-ink"
                      : "border-border bg-panel text-ink-muted hover:border-border-strong",
                  )}
                >
                  {isSelected ? (
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                    />
                  ) : (
                    <span className="h-2 w-2 shrink-0 rounded-full border border-border" />
                  )}
                  <span className="truncate">{model.display_name ?? model.name}</span>
                </button>
              );
            })}
          </div>
          <Button
            size="sm"
            className="w-full"
            onClick={runComparison}
            disabled={selected.length < 2 || compare.isPending}
          >
            <Scale className="h-3.5 w-3.5" />
            {compare.isPending ? "Comparing…" : `Compare ${selected.length} models`}
          </Button>
        </>
      }
    >
      <PageHeader
        title="Evaluations"
        description="Leaderboard across every suite that has actually run here, plus a side-by-side model comparison."
      />

      {leaderboard?.note ? <WarningNote>{leaderboard.note}</WarningNote> : null}

      <Tabs defaultValue="leaderboard">
        <TabsList>
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          <TabsTrigger value="comparison">Model comparison</TabsTrigger>
        </TabsList>

        <TabsContent value="leaderboard">
          {error ? (
            <ErrorState error={error} onRetry={() => refetch()} />
          ) : isLoading ? (
            <LoadingState rows={4} />
          ) : !leaderboard?.rows.length ? (
            <EmptyState
              icon={<Gauge className="h-5 w-5" />}
              title="No evaluation results yet"
              description="Run a benchmark suite to populate the leaderboard. Nothing is pre-filled — every number here comes from a run on this machine."
            />
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead className="text-right">Params</TableHead>
                    {suites
                      .filter((suite) =>
                        leaderboard.rows.some((row) => row.scores[suite.key] !== undefined),
                      )
                      .map((suite) => (
                        <TableHead key={suite.key} className="text-right">
                          {suite.label}
                        </TableHead>
                      ))}
                    <TableHead className="text-right">Runs</TableHead>
                    <TableHead>Source</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {leaderboard.rows.map((row) => (
                    <TableRow key={row.model_id}>
                      <TableCell className="text-xs text-ink">{row.model_name}</TableCell>
                      <TableCell className="mono text-right text-xs">
                        {formatParameters(row.parameters)}
                      </TableCell>
                      {suites
                        .filter((suite) =>
                          leaderboard.rows.some((item) => item.scores[suite.key] !== undefined),
                        )
                        .map((suite) => {
                          const score = row.scores[suite.key];
                          return (
                            <TableCell key={suite.key} className="mono text-right text-xs">
                              {score ? (
                                <span className="inline-flex items-center gap-1">
                                  {score.score.toFixed(1)}%
                                  {!score.official_split ? (
                                    <span className="text-warning" title="bundled sample, not the official split">
                                      *
                                    </span>
                                  ) : null}
                                </span>
                              ) : (
                                <span className="text-ink-muted">—</span>
                              )}
                            </TableCell>
                          );
                        })}
                      <TableCell className="mono text-right text-xs">{row.runs}</TableCell>
                      <TableCell><ProvenanceBadge provenance={row.provenance} compact /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="border-t border-border px-4 py-2">
                <p className="text-2xs text-ink-muted">
                  <span className="text-warning">*</span> scored against a bundled sample, not the
                  official split.
                </p>
              </div>
            </Card>
          )}

          <SectionTitle className="mt-4">Available suites</SectionTitle>
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {suites.map((suite) => (
              <Card key={suite.key}>
                <CardContent className="space-y-2 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-ink">{suite.label}</span>
                    <Badge variant="muted">{suite.metric}</Badge>
                  </div>
                  <p className="line-clamp-2 text-2xs text-ink-muted">{suite.description}</p>
                  <DataSourceBadge official={suite.official} dataSource={suite.data_source} />
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-4">
          {compare.error ? <ErrorState error={compare.error} /> : null}
          {!comparison ? (
            <EmptyState
              icon={<Scale className="h-5 w-5" />}
              title="No comparison yet"
              description="Select models in the inspector and press Compare."
            />
          ) : (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Metric comparison</CardTitle>
                  <span className="text-2xs text-ink-muted">
                    each metric on its own — no composite score
                  </span>
                </CardHeader>
                <CardContent className="p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Metric</TableHead>
                        {comparison.rows.map((row, index) => (
                          <TableHead key={row.model_id} className="text-right">
                            <span className="inline-flex items-center gap-1.5">
                              <span
                                className="h-2 w-2 rounded-full"
                                style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                              />
                              {row.model_name}
                            </span>
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <MetricComparisonRow
                        label="Benchmark score (avg)"
                        rows={comparison.rows}
                        render={(row) =>
                          row.benchmark_score !== null && row.benchmark_score !== undefined
                            ? `${row.benchmark_score.toFixed(1)}%`
                            : "—"
                        }
                      />
                      <MetricComparisonRow
                        label="Validation loss"
                        rows={comparison.rows}
                        render={(row) => formatFloat(row.validation_loss, 4)}
                      />
                      <MetricComparisonRow
                        label="Parameters"
                        rows={comparison.rows}
                        render={(row) => formatParameters(row.parameters)}
                      />
                      <MetricComparisonRow
                        label="Model size"
                        rows={comparison.rows}
                        render={(row) => formatBytes(row.size_bytes)}
                      />
                      <MetricComparisonRow
                        label="Inference speed"
                        rows={comparison.rows}
                        render={(row) =>
                          row.inference_tokens_per_sec
                            ? `${formatFloat(row.inference_tokens_per_sec, 1)} tok/s`
                            : "—"
                        }
                      />
                      <MetricComparisonRow
                        label="VRAM usage"
                        rows={comparison.rows}
                        render={(row) =>
                          row.vram_usage_mb ? `${(row.vram_usage_mb / 1024).toFixed(1)} GB` : "—"
                        }
                      />
                      <MetricComparisonRow
                        label="Context length"
                        rows={comparison.rows}
                        render={(row) => formatCompact(row.context_length ?? 0)}
                      />
                      <MetricComparisonRow
                        label="Precision"
                        rows={comparison.rows}
                        render={(row) => (row.precision ?? "—").toUpperCase()}
                      />
                      <MetricComparisonRow
                        label="Provenance"
                        rows={comparison.rows}
                        render={(row) => row.provenance}
                      />
                      <MetricComparisonRow
                        label="Measured metrics"
                        rows={comparison.rows}
                        render={(row) =>
                          row.measured_metrics.length ? row.measured_metrics.join(", ") : "none measured"
                        }
                      />
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <div className="grid gap-3 lg:grid-cols-2">
                {radarData.length >= 3 ? (
                  <ChartFrame title="Benchmark profile" height={280} provenance={comparison.rows[0]?.provenance}>
                    <RadarProfileChart
                      data={radarData}
                      angleKey="category"
                      series={comparison.rows.map((row, index) => ({
                        key: `m${index}`,
                        label: row.model_name,
                        color: CHART_COLORS[index % CHART_COLORS.length],
                        unit: "%",
                      }))}
                    />
                  </ChartFrame>
                ) : null}

                <ChartFrame
                  title="Benchmark score"
                  subtitle="average across completed runs"
                  height={240}
                  provenance={comparison.rows[0]?.provenance}
                >
                  <BarSeriesChart
                    data={comparison.rows.map((row) => ({
                      model: row.model_name,
                      score: row.benchmark_score ?? 0,
                    }))}
                    xKey="model"
                    layout="vertical"
                    series={[{ key: "score", label: "score", unit: "%" }]}
                    colorByIndex
                    valueDomain={[0, 100]}
                    yTickFormatter={(value) => `${value}%`}
                  />
                </ChartFrame>

                <ChartFrame title="Inference throughput" height={240} provenance={comparison.rows[0]?.provenance}>
                  <BarSeriesChart
                    data={comparison.rows.map((row) => ({
                      model: row.model_name,
                      tps: row.inference_tokens_per_sec ?? 0,
                    }))}
                    xKey="model"
                    layout="vertical"
                    series={[{ key: "tps", label: "tokens/s", color: CHART_COLORS[2] }]}
                    colorByIndex
                  />
                </ChartFrame>

                <ChartFrame title="VRAM usage" height={240} provenance={comparison.rows[0]?.provenance}>
                  <BarSeriesChart
                    data={comparison.rows.map((row) => ({
                      model: row.model_name,
                      vram: row.vram_usage_mb ? Number((row.vram_usage_mb / 1024).toFixed(2)) : 0,
                    }))}
                    xKey="model"
                    layout="vertical"
                    series={[{ key: "vram", label: "VRAM", color: CHART_COLORS[4], unit: " GB" }]}
                    colorByIndex
                  />
                </ChartFrame>
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>
    </Workspace>
  );
}

function MetricComparisonRow({
  label,
  rows,
  render,
}: {
  label: string;
  rows: ComparisonRow[];
  render: (row: ComparisonRow) => React.ReactNode;
}) {
  return (
    <TableRow>
      <TableCell className="text-xs text-ink-muted">{label}</TableCell>
      {rows.map((row) => (
        <TableCell key={row.model_id} className="mono text-right text-xs text-ink-secondary">
          {render(row)}
        </TableCell>
      ))}
    </TableRow>
  );
}
