"use client";

import { AlertTriangle, ArrowLeft, CheckCircle2, ChevronLeft, ChevronRight, Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import * as React from "react";

import { ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { ErrorState, LoadingState } from "@/components/common/states";
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
import {
  useDataset,
  useDatasetMutations,
  useDatasetPreview,
} from "@/lib/hooks/queries";
import {
  formatBytes,
  formatCompact,
  formatFloat,
  formatNumber,
  truncate,
} from "@/lib/format";

const PAGE_SIZE = 20;

export default function DatasetDetailPage() {
  const params = useParams<{ datasetId: string }>();
  const router = useRouter();
  const datasetId = params?.datasetId ?? null;
  const [offset, setOffset] = React.useState(0);

  const { data: dataset, isLoading, error, refetch } = useDataset(datasetId);
  const { data: preview, isFetching } = useDatasetPreview(datasetId, offset, PAGE_SIZE);
  const { validate, remove } = useDatasetMutations();

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !dataset) return <div className="p-4"><LoadingState rows={4} /></div>;

  const report = validate.data ?? dataset.validation_report;
  const issues = (validate.data?.issues ?? []) as { row: number; field?: string | null; severity: string; message: string }[];
  const invalid = report?.invalid_rows ?? 0;

  return (
    <Workspace
      inspectorTitle="Dataset"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Splits</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Train" value={`${(dataset.train_split * 100).toFixed(1)}% · ${formatNumber(Math.round(dataset.rows * dataset.train_split))}`} />
                <MetricRow label="Validation" value={`${(dataset.validation_split * 100).toFixed(1)}% · ${formatNumber(Math.round(dataset.rows * dataset.validation_split))}`} />
                <MetricRow label="Test" value={`${(dataset.test_split * 100).toFixed(1)}% · ${formatNumber(Math.round(dataset.rows * dataset.test_split))}`} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Source</CardTitle>
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Origin" value={dataset.source} />
                <MetricRow label="Format" value={dataset.format} />
                <MetricRow label="Template" value={dataset.template} />
                <MetricRow label="Repository" value={dataset.repo_id ?? "—"} />
                <MetricRow label="Token count" value={dataset.token_count_method} />
              </div>
              {dataset.local_path ? (
                <p className="mono mt-2 break-all text-2xs text-ink-muted">{dataset.local_path}</p>
              ) : null}
            </CardContent>
          </Card>

          <div className="space-y-2">
            <Button
              size="sm"
              variant="secondary"
              className="w-full"
              onClick={() => validate.mutate({ id: dataset.id, template: dataset.template })}
              disabled={validate.isPending}
            >
              {validate.isPending ? "Validating…" : "Re-validate rows"}
            </Button>
            <Button size="sm" variant="secondary" className="w-full" asChild>
              <Link href={`/training?dataset=${dataset.id}`}>Use for training</Link>
            </Button>
            <Button
              size="sm"
              variant="danger"
              className="w-full"
              onClick={() => {
                if (confirm(`Delete dataset ${dataset.name}?`)) {
                  remove.mutate(dataset.id, { onSuccess: () => router.push("/datasets") });
                }
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete dataset
            </Button>
          </div>
        </>
      }
    >
      <div>
        <Link href="/datasets" className="mb-2 inline-flex items-center gap-1 text-2xs text-ink-muted hover:text-ink">
          <ArrowLeft className="h-3 w-3" />
          Datasets
        </Link>
        <PageHeader
          title={dataset.name}
          description={dataset.description ?? undefined}
          actions={
            <>
              <Badge variant="outline">{dataset.template}</Badge>
              {dataset.is_demo ? <ProvenanceBadge provenance="simulated" /> : null}
            </>
          }
        />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatTile label="Rows" value={formatNumber(dataset.rows)} />
        <StatTile label="Tokens" value={formatCompact(dataset.tokens)} hint={dataset.token_count_method} />
        <StatTile label="Avg seq length" value={formatFloat(dataset.avg_sequence_length, 1)} unit="tok" />
        <StatTile label="Max seq length" value={formatNumber(dataset.max_sequence_length)} unit="tok" />
        <StatTile label="Size" value={formatBytes(dataset.size_bytes)} />
        <StatTile
          label="Invalid rows"
          value={formatNumber(invalid)}
          tone={invalid > 0 ? "critical" : "good"}
          hint={`${formatNumber(report?.checked_rows ?? 0)} checked`}
        />
      </div>

      <Tabs defaultValue="preview">
        <TabsList>
          <TabsTrigger value="preview">Preview</TabsTrigger>
          <TabsTrigger value="validation">Validation {invalid > 0 ? `(${invalid})` : ""}</TabsTrigger>
          <TabsTrigger value="schema">Schema</TabsTrigger>
        </TabsList>

        <TabsContent value="preview">
          <Card>
            <CardHeader>
              <CardTitle>
                Rows {offset + 1}–{offset + (preview?.rows.length ?? 0)} of {formatNumber(dataset.rows)}
              </CardTitle>
              <div className="flex items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  disabled={offset === 0 || isFetching}
                  onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  disabled={isFetching || (preview?.rows.length ?? 0) < PAGE_SIZE}
                  onClick={() => setOffset(offset + PAGE_SIZE)}
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              {preview?.rows.length ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12 text-right">#</TableHead>
                      {preview.columns.map((column) => (
                        <TableHead key={column}>{column}</TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {preview.rows.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell className="mono text-right text-2xs text-ink-muted">
                          {offset + index + 1}
                        </TableCell>
                        {preview.columns.map((column) => (
                          <TableCell key={column} className="max-w-[320px] align-top">
                            <span className="mono text-2xs leading-relaxed text-ink-secondary">
                              {typeof row[column] === "object"
                                ? truncate(JSON.stringify(row[column]), 220)
                                : truncate(String(row[column] ?? ""), 220)}
                            </span>
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="p-4 text-xs text-ink-muted">No preview rows available.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validation">
          <Card>
            <CardHeader>
              <CardTitle>Validation report</CardTitle>
              {invalid > 0 ? (
                <Badge variant="critical">
                  <AlertTriangle className="h-3 w-3" />
                  {invalid} invalid
                </Badge>
              ) : (
                <Badge variant="good">
                  <CheckCircle2 className="h-3 w-3" />
                  all rows valid
                </Badge>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <StatTile label="Checked" value={formatNumber(report?.checked_rows ?? 0)} />
                <StatTile label="Valid" value={formatNumber(report?.valid_rows ?? 0)} tone="good" />
                <StatTile
                  label="Invalid"
                  value={formatNumber(invalid)}
                  tone={invalid > 0 ? "critical" : "default"}
                />
              </div>

              {issues.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-20 text-right">Row</TableHead>
                      <TableHead className="w-40">Field</TableHead>
                      <TableHead className="w-24">Severity</TableHead>
                      <TableHead>Message</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {issues.slice(0, 100).map((issue, index) => (
                      <TableRow key={index}>
                        <TableCell className="mono text-right text-xs">{issue.row}</TableCell>
                        <TableCell className="mono text-2xs text-ink-muted">{issue.field ?? "—"}</TableCell>
                        <TableCell>
                          <Badge variant={issue.severity === "error" ? "critical" : "warning"}>
                            {issue.severity}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs">{issue.message}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-xs text-ink-muted">
                  Run &ldquo;Re-validate rows&rdquo; to list individual malformed entries.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="schema">
          <Card>
            <CardHeader>
              <CardTitle>Detected columns</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                {(dataset.columns ?? []).map((column) => (
                  <Badge key={String(column)} variant="outline" className="mono">
                    {String(column)}
                  </Badge>
                ))}
              </div>
              {dataset.preview?.length ? (
                <pre className="mono scrollbar-thin max-h-[380px] overflow-auto rounded-md border border-border bg-base p-3 text-2xs text-ink-secondary">
                  {JSON.stringify(dataset.preview[0], null, 2)}
                </pre>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </Workspace>
  );
}
