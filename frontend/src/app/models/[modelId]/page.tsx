"use client";

import { ArrowLeft, GitBranch, Play, Square, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import * as React from "react";

import { ModelStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { CopyButton } from "@/components/common/copy-button";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/states";
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
import { useBenchmarkRuns, useModel, useModelMutations } from "@/lib/hooks/queries";
import {
  formatBytes,
  formatDateTime,
  formatFloat,
  formatNumber,
  formatParameters,
} from "@/lib/format";

export default function ModelDetailPage() {
  const params = useParams<{ modelId: string }>();
  const router = useRouter();
  const modelId = params?.modelId ?? null;

  const { data: model, isLoading, error, refetch } = useModel(modelId);
  const { data: runs } = useBenchmarkRuns(modelId ? { model_id: modelId, limit: 20 } : undefined);
  const { load, unload, remove } = useModelMutations();

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !model) return <div className="p-4"><LoadingState rows={4} /></div>;

  const configEntries = Object.entries(model.config ?? {});

  return (
    <Workspace
      inspectorTitle="Actions"
      inspector={
        <>
          <div className="space-y-2">
            <Button
              size="sm"
              className="w-full"
              onClick={() => load.mutate({ id: model.id })}
              disabled={load.isPending}
            >
              <Play className="h-3.5 w-3.5" />
              Load into engine
            </Button>
            <Button
              size="sm"
              variant="secondary"
              className="w-full"
              onClick={() => unload.mutate(model.id)}
            >
              <Square className="h-3.5 w-3.5" />
              Unload
            </Button>
            <Button size="sm" variant="secondary" className="w-full" asChild>
              <Link href={`/chat?model=${model.id}`}>Chat with this model</Link>
            </Button>
            <Button size="sm" variant="secondary" className="w-full" asChild>
              <Link href={`/benchmarks?model=${model.id}`}>Run a benchmark</Link>
            </Button>
            <Button size="sm" variant="secondary" className="w-full" asChild>
              <Link href={`/training?model=${model.id}`}>Fine-tune</Link>
            </Button>
            <Button
              size="sm"
              variant="danger"
              className="w-full"
              onClick={() => {
                if (confirm(`Delete ${model.name}?`)) {
                  remove.mutate(model.id, { onSuccess: () => router.push("/models") });
                }
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete
            </Button>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Identity</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="ID" value={<span className="flex items-center gap-1">{model.id}<CopyButton value={model.id} size="xs" /></span>} />
                <MetricRow label="Created" value={formatDateTime(model.created_at)} />
                <MetricRow label="Updated" value={formatDateTime(model.updated_at)} />
                <MetricRow label="Loaded at" value={formatDateTime(model.loaded_at)} />
                <MetricRow label="Parent" value={model.parent_model_id ?? "—"} />
              </div>
            </CardContent>
          </Card>
        </>
      }
    >
      <div>
        <Link href="/models" className="mb-2 inline-flex items-center gap-1 text-2xs text-ink-muted hover:text-ink">
          <ArrowLeft className="h-3 w-3" />
          Models
        </Link>
        <PageHeader
          title={model.display_name ?? model.name}
          description={model.description ?? undefined}
          actions={
            <div className="flex items-center gap-2">
              <ModelStatusBadge status={model.status} />
              <ProvenanceBadge provenance={model.is_demo ? "simulated" : "measured"} />
            </div>
          }
        />
      </div>

      {model.error ? (
        <div className="rounded-md border border-critical/40 bg-critical/5 px-3 py-2">
          <p className="mono text-xs text-critical">{model.error}</p>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <StatTile label="Parameters" value={formatParameters(model.parameters)} />
        <StatTile label="Context" value={formatNumber(model.context_length)} unit="tokens" />
        <StatTile label="Precision" value={model.precision.toUpperCase()} />
        <StatTile label="Size" value={formatBytes(model.size_bytes)} />
        <StatTile label="VRAM est." value={formatNumber(model.vram_estimate_mb)} unit="MB" />
        <StatTile
          label="Throughput"
          value={formatFloat(model.metrics?.inference_tokens_per_sec, 1)}
          unit="tok/s"
          hint={model.is_demo ? "simulated" : "last measured"}
        />
      </div>

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="checkpoints">Checkpoints ({model.checkpoints.length})</TabsTrigger>
          <TabsTrigger value="benchmarks">Benchmarks ({runs?.items.length ?? 0})</TabsTrigger>
          <TabsTrigger value="config">Raw config</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="grid gap-3 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Metadata</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Name" value={model.name} />
                <MetricRow label="Source" value={model.source} />
                <MetricRow label="Format" value={model.format} />
                <MetricRow label="Repository" value={model.repo_id ?? "—"} />
                <MetricRow label="Revision" value={model.revision ?? "—"} />
                <MetricRow label="Local path" value={model.local_path ?? "—"} />
                <MetricRow label="Architecture" value={model.architecture ?? "—"} />
                <MetricRow label="Quantization" value={model.quantization ?? "—"} />
                <MetricRow label="License" value={model.license ?? "—"} />
                <MetricRow
                  label="Tags"
                  value={
                    model.tags?.length ? (
                      <span className="flex flex-wrap justify-end gap-1">
                        {model.tags.map((tag) => (
                          <Badge key={String(tag)} variant="muted">
                            {String(tag)}
                          </Badge>
                        ))}
                      </span>
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
              <CardTitle>Tokenizer</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Type" value={String(model.tokenizer?.type ?? "—")} />
                <MetricRow label="Vocab size" value={formatNumber(Number(model.tokenizer?.vocab_size ?? 0))} />
                <MetricRow label="BOS token" value={String(model.tokenizer?.bos_token ?? "—")} />
                <MetricRow label="EOS token" value={String(model.tokenizer?.eos_token ?? "—")} />
                <MetricRow label="Pad token" value={String(model.tokenizer?.pad_token ?? "—")} />
                <MetricRow label="Unk token" value={String(model.tokenizer?.unk_token ?? "—")} />
                <MetricRow label="Chat template" value={model.tokenizer?.chat_template ? "present" : "none"} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="checkpoints">
          {model.checkpoints.length === 0 ? (
            <EmptyState
              icon={<GitBranch className="h-5 w-5" />}
              title="No checkpoints"
              description="Checkpoints appear here once a training job writes them."
            />
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-right">Step</TableHead>
                    <TableHead className="text-right">Epoch</TableHead>
                    <TableHead className="text-right">Train loss</TableHead>
                    <TableHead className="text-right">Val loss</TableHead>
                    <TableHead className="text-right">Size</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Flags</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {model.checkpoints.map((checkpoint) => (
                    <TableRow key={checkpoint.id}>
                      <TableCell className="mono text-right text-xs">{formatNumber(checkpoint.step)}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.epoch, 2)}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.train_loss, 4)}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatFloat(checkpoint.val_loss, 4)}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatBytes(checkpoint.size_bytes)}</TableCell>
                      <TableCell className="text-2xs text-ink-muted">{formatDateTime(checkpoint.created_at)}</TableCell>
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
          )}
        </TabsContent>

        <TabsContent value="benchmarks">
          {!runs?.items.length ? (
            <EmptyState
              title="No benchmark runs for this model"
              description="Run a suite from the Benchmarks page — results only appear after an actual run."
              action={
                <Button asChild size="sm">
                  <Link href={`/benchmarks?model=${model.id}`}>Run a benchmark</Link>
                </Button>
              }
            />
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Suite</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Score</TableHead>
                    <TableHead className="text-right">Items</TableHead>
                    <TableHead>Data source</TableHead>
                    <TableHead>Run</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs.items.map((run) => (
                    <TableRow key={run.id}>
                      <TableCell className="text-xs text-ink">{run.suite_label}</TableCell>
                      <TableCell className="text-xs">{run.status}</TableCell>
                      <TableCell className="mono text-right text-xs">
                        {run.overall_score !== null && run.overall_score !== undefined
                          ? `${(run.overall_score * 100).toFixed(1)}%`
                          : "—"}
                      </TableCell>
                      <TableCell className="mono text-right text-xs">{run.total_items}</TableCell>
                      <TableCell>
                        <Badge variant={run.config?.official_split ? "good" : "warning"}>
                          {run.config?.official_split ? "official split" : "sample"}
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
            </Card>
          )}
        </TabsContent>

        <TabsContent value="config">
          <Card>
            <CardHeader>
              <CardTitle>Model configuration</CardTitle>
              <CopyButton value={JSON.stringify(model.config, null, 2)} label="Copy JSON" size="xs" />
            </CardHeader>
            <CardContent className="p-0">
              {configEntries.length === 0 ? (
                <p className="p-4 text-xs text-ink-muted">No configuration recorded for this model.</p>
              ) : (
                <pre className="mono scrollbar-thin max-h-[420px] overflow-auto p-4 text-2xs text-ink-secondary">
                  {JSON.stringify(model.config, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </Workspace>
  );
}
