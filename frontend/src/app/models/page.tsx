"use client";

import { Boxes, Copy, Play, Search, Square, Trash2 } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { ModelStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { ImportModelDialog } from "@/components/models/import-model-dialog";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MetricRow } from "@/components/common/stat-tile";
import { useModelMutations, useModels } from "@/lib/hooks/queries";
import { useWorkspace } from "@/lib/store";
import {
  formatBytes,
  formatCompact,
  formatNumber,
  formatParameters,
  formatRelative,
} from "@/lib/format";
import type { Model, ModelStatus } from "@/lib/types";

const STATUS_FILTERS: (ModelStatus | "all")[] = [
  "all",
  "ready",
  "loaded",
  "training",
  "stopped",
  "error",
];

export default function ModelsPage() {
  const [search, setSearch] = React.useState("");
  const [status, setStatus] = React.useState<ModelStatus | "all">("all");
  const [selected, setSelected] = React.useState<Model | null>(null);

  const { data, isLoading, error, refetch } = useModels({
    limit: 200,
    search: search || undefined,
    status: status === "all" ? undefined : status,
  });
  const { load, unload, remove, clone, exportModel } = useModelMutations();
  const setActiveModel = useWorkspace((state) => state.setActiveModel);

  const models = data?.items ?? [];
  const active = selected ?? models[0] ?? null;

  return (
    <Workspace
      inspectorTitle="Model details"
      inspector={
        active ? (
          <>
            <Card>
              <CardHeader>
                <div className="min-w-0">
                  <CardTitle>{active.display_name ?? active.name}</CardTitle>
                  <p className="mono mt-1 truncate text-2xs text-ink-muted">{active.id}</p>
                </div>
                <ProvenanceBadge provenance={active.is_demo ? "simulated" : "measured"} compact />
              </CardHeader>
              <CardContent className="py-3">
                <div className="divide-y divide-border">
                  <MetricRow label="Status" value={active.status} />
                  <MetricRow label="Source" value={active.source} />
                  <MetricRow label="Format" value={active.format} />
                  <MetricRow label="Architecture" value={active.architecture ?? "—"} />
                  <MetricRow label="Parameters" value={formatParameters(active.parameters)} />
                  <MetricRow label="Context length" value={formatNumber(active.context_length)} />
                  <MetricRow label="Precision" value={active.precision.toUpperCase()} />
                  <MetricRow label="Quantization" value={active.quantization ?? "—"} />
                  <MetricRow label="Size on disk" value={formatBytes(active.size_bytes)} />
                  <MetricRow label="VRAM estimate" value={`${formatNumber(active.vram_estimate_mb)} MB`} />
                  <MetricRow label="License" value={active.license ?? "—"} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Tokenizer</CardTitle>
              </CardHeader>
              <CardContent className="py-3">
                <div className="divide-y divide-border">
                  <MetricRow label="Type" value={String(active.tokenizer?.type ?? "—")} />
                  <MetricRow label="Vocab size" value={formatNumber(Number(active.tokenizer?.vocab_size ?? 0))} />
                  <MetricRow label="BOS" value={String(active.tokenizer?.bos_token ?? "—")} />
                  <MetricRow label="EOS" value={String(active.tokenizer?.eos_token ?? "—")} />
                  <MetricRow label="Pad" value={String(active.tokenizer?.pad_token ?? "—")} />
                  <MetricRow
                    label="Chat template"
                    value={active.tokenizer?.chat_template ? "yes" : "no"}
                  />
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => load.mutate({ id: active.id })}
                disabled={load.isPending}
              >
                <Play className="h-3.5 w-3.5" />
                Load
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => unload.mutate(active.id)}
                disabled={unload.isPending}
              >
                <Square className="h-3.5 w-3.5" />
                Unload
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => clone.mutate({ id: active.id, name: `${active.name}-clone` })}
                disabled={clone.isPending}
              >
                <Copy className="h-3.5 w-3.5" />
                Clone config
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => exportModel.mutate({ id: active.id, format: "safetensors" })}
                disabled={exportModel.isPending}
              >
                Export
              </Button>
              <Button size="sm" variant="secondary" asChild className="col-span-2">
                <Link href={`/models/${active.id}`}>Open full details</Link>
              </Button>
              <Button
                size="sm"
                variant="danger"
                className="col-span-2"
                onClick={() => {
                  if (confirm(`Delete ${active.name}? This removes the record, not the weights.`)) {
                    remove.mutate(active.id);
                    setSelected(null);
                  }
                }}
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete model
              </Button>
            </div>

            {exportModel.data ? (
              <p className="mono text-2xs text-ink-muted">
                {String((exportModel.data as { message?: string }).message ?? "")}
              </p>
            ) : null}
          </>
        ) : (
          <p className="text-xs text-ink-muted">Select a model to inspect it.</p>
        )
      }
    >
      <PageHeader
        title="Models"
        description="Every base model, fine-tune and checkpoint-derived model registered in this workspace."
        actions={<ImportModelDialog />}
      />

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-muted" />
          <Input
            placeholder="Search models…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="h-8 w-[240px] pl-8 text-xs"
          />
        </div>
        <Select value={status} onValueChange={(value) => setStatus(value as ModelStatus | "all")}>
          <SelectTrigger className="h-8 w-[140px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_FILTERS.map((item) => (
              <SelectItem key={item} value={item}>
                {item === "all" ? "All statuses" : item}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="mono ml-auto text-2xs text-ink-muted">
          {models.length} model{models.length === 1 ? "" : "s"}
        </span>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <LoadingState rows={5} />
      ) : models.length === 0 ? (
        <EmptyState
          icon={<Boxes className="h-5 w-5" />}
          title="No models registered"
          description="Import a Hugging Face repository or point Foxtrot at a local directory to get started."
          action={<ImportModelDialog />}
        />
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Format</TableHead>
                <TableHead className="text-right">Params</TableHead>
                <TableHead className="text-right">Context</TableHead>
                <TableHead>Precision</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead>Origin</TableHead>
                <TableHead className="text-right">Added</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((model) => (
                <TableRow
                  key={model.id}
                  onClick={() => {
                    setSelected(model);
                    setActiveModel(model.id);
                  }}
                  data-state={active?.id === model.id ? "selected" : undefined}
                  className="cursor-pointer"
                >
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/models/${model.id}`}
                        className="truncate text-xs text-ink hover:text-accent"
                        onClick={(event) => event.stopPropagation()}
                      >
                        {model.display_name ?? model.name}
                      </Link>
                      {model.is_demo ? <ProvenanceBadge provenance="simulated" compact /> : null}
                    </div>
                    <div className="mono text-2xs text-ink-muted">{model.name}</div>
                  </TableCell>
                  <TableCell>
                    <ModelStatusBadge status={model.status} />
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{model.format}</Badge>
                  </TableCell>
                  <TableCell className="mono text-right text-xs">
                    {formatParameters(model.parameters)}
                  </TableCell>
                  <TableCell className="mono text-right text-xs">
                    {formatCompact(model.context_length ?? 0)}
                  </TableCell>
                  <TableCell className="mono text-xs">{model.precision.toUpperCase()}</TableCell>
                  <TableCell className="mono text-right text-xs">{formatBytes(model.size_bytes)}</TableCell>
                  <TableCell className="mono truncate text-2xs text-ink-muted">
                    {model.repo_id ?? model.local_path ?? model.source}
                  </TableCell>
                  <TableCell className="text-right text-2xs text-ink-muted">
                    {formatRelative(model.created_at)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}
    </Workspace>
  );
}
