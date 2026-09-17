"use client";

import { Play, Target } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { DataSourceBadge, JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { EmptyState, ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useBenchmarkMutations,
  useBenchmarkRuns,
  useBenchmarkSuites,
  useDatasets,
  useModels,
} from "@/lib/hooks/queries";
import { useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { useWorkspace } from "@/lib/store";
import { BENCHMARK_CATEGORY_LABELS } from "@/lib/constants";
import { formatDuration, formatFloat, formatNumber, formatRelative } from "@/lib/format";
import type { BenchmarkProgressEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

function BenchmarksPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const presetModel = searchParams.get("model");

  const activeModelId = useWorkspace((state) => state.activeModelId);
  const { data: models } = useModels({ limit: 200 });
  const { data: datasets } = useDatasets({ limit: 200 });
  const { data: suites, isLoading: suitesLoading } = useBenchmarkSuites();
  const { data: runs, isLoading, error, refetch } = useBenchmarkRuns({ limit: 50 }, 6000);
  const { run, cancel } = useBenchmarkMutations();

  const [modelId, setModelId] = React.useState(presetModel ?? activeModelId ?? "");
  const [suiteKey, setSuiteKey] = React.useState("mmlu");
  const [datasetId, setDatasetId] = React.useState("");
  const [numExamples, setNumExamples] = React.useState(20);
  const [fewShot, setFewShot] = React.useState(0);
  const [temperature, setTemperature] = React.useState(0);
  const [maxTokens, setMaxTokens] = React.useState(256);
  const [seed, setSeed] = React.useState(42);
  const [progress, setProgress] = React.useState<Record<string, BenchmarkProgressEvent>>({});

  useRealtimeTopic<BenchmarkProgressEvent>("benchmark.progress", (event) => {
    setProgress((current) => ({ ...current, [event.data.run_id]: event.data }));
  });

  React.useEffect(() => {
    if (!modelId && models?.items.length) setModelId(activeModelId ?? models.items[0].id);
  }, [activeModelId, models, modelId]);

  const suite = suites?.find((item) => item.key === suiteKey);

  React.useEffect(() => {
    if (suite) {
      setFewShot(suite.default_shots);
      setNumExamples((current) =>
        suite.available_items > 0 ? Math.min(current, Math.max(1, suite.available_items)) : current,
      );
    }
  }, [suite]);

  const launch = () => {
    run.mutate(
      {
        model_id: modelId,
        suite: suiteKey,
        dataset_id: suiteKey === "custom" ? datasetId : null,
        config: {
          num_examples: numExamples,
          few_shot: fewShot,
          temperature,
          max_tokens: maxTokens,
          seed,
        },
      },
      { onSuccess: (created) => router.push(`/benchmarks/${created.id}`) },
    );
  };

  const grouped = React.useMemo(() => {
    const map = new Map<string, typeof suites>();
    (suites ?? []).forEach((item) => {
      const list = map.get(item.category) ?? [];
      map.set(item.category, [...(list ?? []), item] as typeof suites);
    });
    return Array.from(map.entries());
  }, [suites]);

  return (
    <Workspace
      inspectorTitle="Run configuration"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Configure</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 py-3">
              <div className="space-y-1.5">
                <Label>Model</Label>
                <Select value={modelId} onValueChange={setModelId}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Select model" /></SelectTrigger>
                  <SelectContent>
                    {(models?.items ?? []).map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.display_name ?? model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label>Suite</Label>
                <Select value={suiteKey} onValueChange={setSuiteKey}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {(suites ?? []).map((item) => (
                      <SelectItem key={item.key} value={item.key}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {suiteKey === "custom" ? (
                <div className="space-y-1.5">
                  <Label>Dataset</Label>
                  <Select value={datasetId} onValueChange={setDatasetId}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Pick a dataset" /></SelectTrigger>
                    <SelectContent>
                      {(datasets?.items ?? []).map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : null}

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label>Examples</Label>
                  <Input
                    type="number"
                    value={numExamples}
                    min={1}
                    onChange={(event) => setNumExamples(Number(event.target.value))}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Few-shot</Label>
                  <Input
                    type="number"
                    value={fewShot}
                    min={0}
                    onChange={(event) => setFewShot(Number(event.target.value))}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Temperature</Label>
                  <Input
                    type="number"
                    value={temperature}
                    step={0.05}
                    min={0}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Max tokens</Label>
                  <Input
                    type="number"
                    value={maxTokens}
                    min={1}
                    onChange={(event) => setMaxTokens(Number(event.target.value))}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Seed</Label>
                  <Input
                    type="number"
                    value={seed}
                    min={0}
                    onChange={(event) => setSeed(Number(event.target.value))}
                    className="h-8 text-xs"
                  />
                </div>
              </div>

              {suite ? (
                <div className="space-y-2 rounded-md border border-border bg-elevated/40 p-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-2xs text-ink-muted">available items</span>
                    <span className="mono text-2xs text-ink-secondary">{suite.available_items}</span>
                  </div>
                  <DataSourceBadge official={suite.official} dataSource={suite.data_source} />
                  {suite.notes ? <p className="text-2xs text-ink-muted">{suite.notes}</p> : null}
                </div>
              ) : null}

              <Button
                size="sm"
                className="w-full"
                onClick={launch}
                disabled={run.isPending || !modelId || (suiteKey === "custom" && !datasetId)}
              >
                <Play className="h-3.5 w-3.5" />
                {run.isPending ? "Starting…" : "Run benchmark"}
              </Button>
              {run.error ? <ErrorState error={run.error} /> : null}
            </CardContent>
          </Card>
        </>
      }
    >
      <PageHeader
        title="Benchmarks"
        description="Run evaluation suites against a model and inspect every graded item."
      />

      <WarningNote>
        Suites marked <strong>sample · not official</strong> use a small bundled item pool so the
        runner works offline. They are <em>not</em> the published splits and their scores are not
        comparable to public leaderboards. Drop a real split at{" "}
        <code className="mono">data/benchmarks/&lt;suite&gt;.jsonl</code> to run the official version.
      </WarningNote>

      <section>
        <SectionTitle>Suites</SectionTitle>
        {suitesLoading ? (
          <LoadingState rows={3} />
        ) : (
          <div className="space-y-4">
            {grouped.map(([category, list]) => (
              <div key={category}>
                <h3 className="mb-1.5 text-2xs uppercase tracking-[0.14em] text-ink-muted">
                  {BENCHMARK_CATEGORY_LABELS[category] ?? category}
                </h3>
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                  {(list ?? []).map((item) => (
                    <button
                      key={item.key}
                      type="button"
                      onClick={() => setSuiteKey(item.key)}
                      className={cn(
                        "rounded-md border px-3 py-2.5 text-left transition-colors",
                        suiteKey === item.key
                          ? "border-accent/60 bg-accent/10"
                          : "border-border bg-panel hover:border-border-strong",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-medium text-ink">{item.label}</span>
                        <Badge variant="muted">{item.metric}</Badge>
                      </div>
                      <p className="mt-1 line-clamp-2 text-2xs text-ink-muted">{item.description}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-1.5">
                        <DataSourceBadge official={item.official} dataSource={item.data_source} />
                        <span className="mono text-2xs text-ink-muted">{item.available_items} items</span>
                        {item.requires_execution ? <Badge variant="warning">no code execution</Badge> : null}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <SectionTitle>Runs</SectionTitle>
        {error ? (
          <ErrorState error={error} onRetry={() => refetch()} />
        ) : isLoading ? (
          <LoadingState rows={3} />
        ) : !runs?.items.length ? (
          <EmptyState
            icon={<Target className="h-5 w-5" />}
            title="No benchmark runs yet"
            description="Results appear here only after a suite has actually been executed against a model."
          />
        ) : (
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-[150px]">Progress</TableHead>
                  <TableHead className="text-right">Score</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Latency</TableHead>
                  <TableHead className="text-right">Runtime</TableHead>
                  <TableHead>Data</TableHead>
                  <TableHead className="text-right">When</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.items.map((item) => {
                  const live = progress[item.id];
                  const completed = live?.completed ?? item.completed_items;
                  const total = live?.total ?? item.total_items;
                  const pct = total ? (completed / total) * 100 : 0;
                  const isRunning = item.status === "running" || item.status === "queued";
                  return (
                    <TableRow
                      key={item.id}
                      className="cursor-pointer"
                      onClick={() => router.push(`/benchmarks/${item.id}`)}
                    >
                      <TableCell>
                        <Link
                          href={`/benchmarks/${item.id}`}
                          className="text-xs text-ink hover:text-accent"
                          onClick={(event) => event.stopPropagation()}
                        >
                          {item.suite_label}
                        </Link>
                        <div className="mono text-2xs text-ink-muted">{item.name}</div>
                      </TableCell>
                      <TableCell className="text-xs text-ink-secondary">
                        {models?.items.find((model) => model.id === item.model_id)?.display_name ?? item.model_id}
                      </TableCell>
                      <TableCell><JobStatusBadge status={item.status} /></TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <Progress value={pct} size="sm" tone={item.status === "failed" ? "critical" : "accent"} />
                          <span className="mono text-2xs text-ink-muted">
                            {completed}/{total}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="mono text-right text-xs">
                        {item.overall_score !== null && item.overall_score !== undefined
                          ? `${(item.overall_score * 100).toFixed(1)}%`
                          : "—"}
                      </TableCell>
                      <TableCell className="mono text-right text-xs">
                        {item.accuracy !== null && item.accuracy !== undefined
                          ? `${(item.accuracy * 100).toFixed(1)}%`
                          : "—"}
                      </TableCell>
                      <TableCell className="mono text-right text-xs">
                        {formatFloat(item.avg_latency_ms, 0)} ms
                      </TableCell>
                      <TableCell className="mono text-right text-xs">
                        {formatDuration(item.runtime_seconds)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <DataSourceBadge
                            official={item.config?.official_split}
                            dataSource={item.config?.data_source}
                          />
                          <ProvenanceBadge provenance={item.provenance} compact />
                        </div>
                      </TableCell>
                      <TableCell className="text-right text-2xs text-ink-muted">
                        {isRunning ? (
                          <Button
                            size="xs"
                            variant="ghost"
                            onClick={(event) => {
                              event.stopPropagation();
                              cancel.mutate(item.id);
                            }}
                          >
                            cancel
                          </Button>
                        ) : (
                          formatRelative(item.created_at)
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Card>
        )}
        {runs?.items.length ? (
          <p className="mt-2 text-2xs text-ink-muted">
            {formatNumber(runs.total)} run{runs.total === 1 ? "" : "s"} recorded.
          </p>
        ) : null}
      </section>
    </Workspace>
  );
}

/**
 * `useSearchParams` opts this route into client-side rendering, so the page body
 * lives behind a Suspense boundary (required by the Next.js app router).
 */
export default function BenchmarksPage() {
  return (
    <React.Suspense
      fallback={
        <div className="p-6 text-xs text-ink-muted">Loading benchmarks…</div>
      }
    >
      <BenchmarksPageContent />
    </React.Suspense>
  );
}
