"use client";

import { Database, FileJson, Upload } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { MetricRow } from "@/components/common/stat-tile";
import { EmptyState, ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDatasetMutations, useDatasetTemplates, useDatasets } from "@/lib/hooks/queries";
import { formatBytes, formatCompact, formatFloat, formatNumber, formatRelative } from "@/lib/format";
import type { DatasetTemplate } from "@/lib/types";

export default function DatasetsPage() {
  const { data, isLoading, error, refetch } = useDatasets({ limit: 200 });
  const { data: templates } = useDatasetTemplates();
  const datasets = data?.items ?? [];

  return (
    <Workspace
      inspectorTitle="Formatting templates"
      inspector={
        <>
          <p className="text-xs text-ink-muted">
            Rows are validated against the selected template on import. Malformed rows are reported
            individually — the import is never silently dropped.
          </p>
          {(templates ?? []).map((template) => (
            <Card key={template.key}>
              <CardHeader>
                <div>
                  <CardTitle>{template.label}</CardTitle>
                  <p className="mt-1 text-2xs text-ink-muted">{template.description}</p>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <pre className="mono scrollbar-thin overflow-auto p-3 text-2xs text-ink-secondary">
                  {JSON.stringify(template.schema_example, null, 2)}
                </pre>
                {template.required_fields.length ? (
                  <div className="border-t border-border px-3 py-2">
                    <span className="text-2xs text-ink-muted">required: </span>
                    <span className="mono text-2xs text-ink-secondary">
                      {template.required_fields.join(", ")}
                    </span>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ))}
        </>
      }
    >
      <PageHeader
        title="Datasets"
        description="Import, validate and preview the corpora your training and evaluation runs consume."
        actions={<ImportDatasetDialog />}
      />

      {error ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <LoadingState rows={4} />
      ) : datasets.length === 0 ? (
        <EmptyState
          icon={<Database className="h-5 w-5" />}
          title="No datasets imported"
          description="Upload a JSON, JSONL, CSV, TXT or Parquet file, or point Foxtrot at a Hugging Face dataset."
          action={<ImportDatasetDialog />}
        />
      ) : (
        <>
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Template</TableHead>
                  <TableHead>Format</TableHead>
                  <TableHead className="text-right">Rows</TableHead>
                  <TableHead className="text-right">Tokens</TableHead>
                  <TableHead className="text-right">Avg len</TableHead>
                  <TableHead>Splits</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead>Validation</TableHead>
                  <TableHead className="text-right">Added</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.map((dataset) => {
                  const invalid = dataset.validation_report?.invalid_rows ?? 0;
                  return (
                    <TableRow key={dataset.id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Link
                            href={`/datasets/${dataset.id}`}
                            className="text-xs text-ink hover:text-accent"
                          >
                            {dataset.name}
                          </Link>
                          {dataset.is_demo ? <ProvenanceBadge provenance="simulated" compact /> : null}
                        </div>
                        {dataset.description ? (
                          <div className="truncate text-2xs text-ink-muted">{dataset.description}</div>
                        ) : null}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{dataset.template}</Badge>
                      </TableCell>
                      <TableCell className="mono text-xs">{dataset.format}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatNumber(dataset.rows)}</TableCell>
                      <TableCell className="mono text-right text-xs">{formatCompact(dataset.tokens)}</TableCell>
                      <TableCell className="mono text-right text-xs">
                        {formatFloat(dataset.avg_sequence_length, 1)}
                      </TableCell>
                      <TableCell className="mono text-2xs text-ink-muted">
                        {(dataset.train_split * 100).toFixed(0)}/{(dataset.validation_split * 100).toFixed(0)}/
                        {(dataset.test_split * 100).toFixed(0)}
                      </TableCell>
                      <TableCell className="mono text-right text-xs">{formatBytes(dataset.size_bytes)}</TableCell>
                      <TableCell>
                        {invalid > 0 ? (
                          <Badge variant="critical">{invalid} invalid</Badge>
                        ) : (
                          <Badge variant="good">clean</Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right text-2xs text-ink-muted">
                        {formatRelative(dataset.created_at)}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Card>

          <section>
            <SectionTitle>Token accounting</SectionTitle>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {datasets.slice(0, 4).map((dataset) => (
                <Card key={dataset.id}>
                  <CardHeader>
                    <CardTitle>{dataset.name}</CardTitle>
                  </CardHeader>
                  <CardContent className="py-3">
                    <div className="divide-y divide-border">
                      <MetricRow label="Rows" value={formatNumber(dataset.rows)} />
                      <MetricRow label="Tokens" value={formatCompact(dataset.tokens)} />
                      <MetricRow label="Avg seq len" value={formatFloat(dataset.avg_sequence_length, 1)} />
                      <MetricRow label="Max seq len" value={formatNumber(dataset.max_sequence_length)} />
                      <MetricRow label="Counting" value={dataset.token_count_method} />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        </>
      )}
    </Workspace>
  );
}

function ImportDatasetDialog() {
  const [open, setOpen] = React.useState(false);
  const [mode, setMode] = React.useState<"upload" | "path" | "huggingface">("upload");
  const [file, setFile] = React.useState<File | null>(null);
  const [name, setName] = React.useState("");
  const [path, setPath] = React.useState("");
  const [repoId, setRepoId] = React.useState("");
  const [subset, setSubset] = React.useState("");
  const [split, setSplit] = React.useState("train");
  const [template, setTemplate] = React.useState<DatasetTemplate>("instruction");

  const { upload, importDataset } = useDatasetMutations();
  const pending = upload.isPending || importDataset.isPending;
  const error = (upload.error ?? importDataset.error) as Error | null;

  const submit = () => {
    if (mode === "upload" && file) {
      upload.mutate(
        { file, name: name || file.name.replace(/\.[^.]+$/, ""), template },
        { onSuccess: () => setOpen(false) },
      );
    } else if (mode === "path") {
      importDataset.mutate(
        { source: "path", name, path, template },
        { onSuccess: () => setOpen(false) },
      );
    } else {
      importDataset.mutate(
        { source: "huggingface", name: name || repoId, repo_id: repoId, subset: subset || undefined, split, template },
        { onSuccess: () => setOpen(false) },
      );
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Upload className="h-3.5 w-3.5" />
          Import dataset
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Import a dataset</DialogTitle>
          <DialogDescription>
            Supported: JSON, JSONL, CSV, TXT, Parquet and Hugging Face datasets. Rows are validated
            against the chosen template on import.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 p-4">
          <Tabs value={mode} onValueChange={(value) => setMode(value as typeof mode)}>
            <TabsList>
              <TabsTrigger value="upload">Upload file</TabsTrigger>
              <TabsTrigger value="path">Server path</TabsTrigger>
              <TabsTrigger value="huggingface">Hugging Face</TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="file">File</Label>
                <Input
                  id="file"
                  type="file"
                  accept=".json,.jsonl,.csv,.tsv,.txt,.parquet"
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
              </div>
            </TabsContent>

            <TabsContent value="path" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="ds-path">Path inside the data root</Label>
                <Input
                  id="ds-path"
                  placeholder="datasets/sft-mixture.jsonl"
                  value={path}
                  onChange={(event) => setPath(event.target.value)}
                  className="mono"
                />
              </div>
              <WarningNote>
                Paths resolve inside <code className="mono">FOXTROT_DATA_DIR</code>; traversal outside it
                is rejected.
              </WarningNote>
            </TabsContent>

            <TabsContent value="huggingface" className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="hf-repo">Dataset repository</Label>
                <Input
                  id="hf-repo"
                  placeholder="tatsu-lab/alpaca"
                  value={repoId}
                  onChange={(event) => setRepoId(event.target.value)}
                  className="mono"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <Label htmlFor="hf-subset">Subset</Label>
                  <Input id="hf-subset" placeholder="default" value={subset} onChange={(e) => setSubset(e.target.value)} />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="hf-split">Split</Label>
                  <Input id="hf-split" value={split} onChange={(e) => setSplit(e.target.value)} />
                </div>
              </div>
              <WarningNote>
                Requires the optional <code className="mono">datasets</code> package on the backend.
              </WarningNote>
            </TabsContent>
          </Tabs>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="ds-name">Name</Label>
              <Input id="ds-name" placeholder="auto" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label>Template</Label>
              <Select value={template} onValueChange={(value) => setTemplate(value as DatasetTemplate)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="instruction">Instruction tuning</SelectItem>
                  <SelectItem value="chat">Chat</SelectItem>
                  <SelectItem value="plain_text">Plain text</SelectItem>
                  <SelectItem value="raw">Raw / custom</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {error ? <p className="mono text-xs text-critical">{error.message}</p> : null}
        </div>

        <DialogFooter>
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={submit}
            disabled={pending || (mode === "upload" ? !file : mode === "path" ? !path || !name : !repoId)}
          >
            <FileJson className="h-3.5 w-3.5" />
            {pending ? "Importing…" : "Import"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
