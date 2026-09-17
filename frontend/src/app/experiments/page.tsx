"use client";

import { Copy, FlaskConical, GitCompare, Search, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";

import { JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useExperimentMutations, useExperiments } from "@/lib/hooks/queries";
import { formatDuration, formatFloat, formatRelative } from "@/lib/format";
import { cn } from "@/lib/utils";

export default function ExperimentsPage() {
  const router = useRouter();
  const [search, setSearch] = React.useState("");
  const [selected, setSelected] = React.useState<string[]>([]);

  const { data, isLoading, error, refetch } = useExperiments({ limit: 200, search: search || undefined });
  const { duplicate, remove, compare } = useExperimentMutations();

  const experiments = data?.items ?? [];
  const comparison = compare.data;

  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(0, 8),
    );

  return (
    <Workspace
      inspectorTitle="Compare"
      inspector={
        <>
          <p className="text-xs text-ink-muted">
            Select two or more experiments to line up their hyperparameters and outcomes. Differing
            values are highlighted.
          </p>
          <div className="mono text-2xs text-ink-muted">{selected.length} selected</div>
          <Button
            size="sm"
            className="w-full"
            disabled={selected.length < 2 || compare.isPending}
            onClick={() => compare.mutate(selected)}
          >
            <GitCompare className="h-3.5 w-3.5" />
            Compare selected
          </Button>

          {comparison ? (
            <Card>
              <CardHeader>
                <CardTitle>Differing hyperparameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 py-3">
                {comparison.differing_hyperparameters.length === 0 ? (
                  <p className="text-2xs text-ink-muted">Configurations are identical.</p>
                ) : (
                  comparison.differing_hyperparameters.map((key) => (
                    <div key={key} className="mono text-2xs text-warning">
                      {key}
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          ) : null}
        </>
      }
    >
      <PageHeader
        title="Experiments"
        description="Every training run records an experiment: configuration, outcome, best checkpoint and notes."
      />

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-muted" />
          <Input
            placeholder="Search experiments…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="h-8 w-[240px] pl-8 text-xs"
          />
        </div>
        <span className="mono ml-auto text-2xs text-ink-muted">{experiments.length} experiments</span>
      </div>

      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <LoadingState rows={4} />
      ) : experiments.length === 0 ? (
        <EmptyState
          icon={<FlaskConical className="h-5 w-5" />}
          title="No experiments recorded"
          description="Start a training run — an experiment is created automatically and tracks the whole configuration."
          action={<Button size="sm" asChild><Link href="/training">Configure a run</Link></Button>}
        />
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10" />
                <TableHead>Experiment</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Method</TableHead>
                <TableHead className="text-right">Train loss</TableHead>
                <TableHead className="text-right">Val loss</TableHead>
                <TableHead className="text-right">Duration</TableHead>
                <TableHead className="text-right">Started</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {experiments.map((experiment) => (
                <TableRow
                  key={experiment.id}
                  className={cn("cursor-pointer", selected.includes(experiment.id) && "bg-accent/5")}
                  onClick={() => router.push(`/experiments/${experiment.id}`)}
                >
                  <TableCell onClick={(event) => event.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected.includes(experiment.id)}
                      onChange={() => toggle(experiment.id)}
                      className="h-3 w-3 accent-[#3987e5]"
                      aria-label={`Select ${experiment.name}`}
                    />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/experiments/${experiment.id}`}
                        className="text-xs text-ink hover:text-accent"
                        onClick={(event) => event.stopPropagation()}
                      >
                        {experiment.name}
                      </Link>
                      <ProvenanceBadge provenance={experiment.provenance} compact />
                    </div>
                    {experiment.notes ? (
                      <div className="truncate text-2xs text-ink-muted">{experiment.notes}</div>
                    ) : null}
                  </TableCell>
                  <TableCell><JobStatusBadge status={experiment.status} /></TableCell>
                  <TableCell><Badge variant="outline">{experiment.method}</Badge></TableCell>
                  <TableCell className="mono text-right text-xs">{formatFloat(experiment.final_train_loss, 4)}</TableCell>
                  <TableCell className="mono text-right text-xs">{formatFloat(experiment.final_val_loss, 4)}</TableCell>
                  <TableCell className="mono text-right text-xs">{formatDuration(experiment.duration_seconds)}</TableCell>
                  <TableCell className="text-right text-2xs text-ink-muted">
                    {formatRelative(experiment.started_at ?? experiment.created_at)}
                  </TableCell>
                  <TableCell className="text-right" onClick={(event) => event.stopPropagation()}>
                    <div className="flex justify-end gap-1">
                      <Button
                        size="xs"
                        variant="ghost"
                        title="Duplicate configuration as a new run"
                        onClick={() => duplicate.mutate({ id: experiment.id })}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                      <Button
                        size="xs"
                        variant="ghost"
                        title="Delete"
                        onClick={() => {
                          if (confirm(`Delete experiment ${experiment.name}?`)) remove.mutate(experiment.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      {comparison ? (
        <Card>
          <CardHeader>
            <CardTitle>Comparison</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Field</TableHead>
                  {comparison.rows.map((row) => (
                    <TableHead key={row.experiment_id} className="text-right">{row.name}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  { label: "Model", get: (row: (typeof comparison.rows)[number]) => row.model_name ?? "—" },
                  { label: "Dataset", get: (row: (typeof comparison.rows)[number]) => row.dataset_name ?? "—" },
                  { label: "Method", get: (row: (typeof comparison.rows)[number]) => row.method },
                  {
                    label: "Final train loss",
                    get: (row: (typeof comparison.rows)[number]) => formatFloat(row.final_train_loss, 4),
                  },
                  {
                    label: "Final val loss",
                    get: (row: (typeof comparison.rows)[number]) => formatFloat(row.final_val_loss, 4),
                  },
                  {
                    label: "Duration",
                    get: (row: (typeof comparison.rows)[number]) => formatDuration(row.duration_seconds),
                  },
                  { label: "Provenance", get: (row: (typeof comparison.rows)[number]) => row.provenance },
                ].map((field) => (
                  <TableRow key={field.label}>
                    <TableCell className="text-xs text-ink-muted">{field.label}</TableCell>
                    {comparison.rows.map((row) => (
                      <TableCell key={row.experiment_id} className="mono text-right text-xs">
                        {field.get(row)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
                {comparison.differing_hyperparameters.map((key) => (
                  <TableRow key={key}>
                    <TableCell className="mono text-xs text-warning">{key}</TableCell>
                    {comparison.rows.map((row) => (
                      <TableCell key={row.experiment_id} className="mono text-right text-xs">
                        {String(readPath(row.hyperparameters, key) ?? "—")}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </Workspace>
  );
}

function readPath(source: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((current, key) => {
    if (current && typeof current === "object") return (current as Record<string, unknown>)[key];
    return undefined;
  }, source);
}
