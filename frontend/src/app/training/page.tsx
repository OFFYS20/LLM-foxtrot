"use client";

import { Activity, Plus } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { JobStatusBadge, ProvenanceBadge } from "@/components/common/badges";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState, ErrorState, LoadingState } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { TrainingConfigForm } from "@/components/training/config-form";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDatasets, useModels, useTrainingJobs } from "@/lib/hooks/queries";
import { formatCompact, formatEta, formatFloat, formatNumber, formatRelative } from "@/lib/format";

function TrainingPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const wantsNew = searchParams.get("new") === "1";
  const presetModel = searchParams.get("model");
  const presetDataset = searchParams.get("dataset");

  const [tab, setTab] = React.useState(wantsNew || presetModel || presetDataset ? "configure" : "runs");

  const { data: jobs, isLoading, error, refetch } = useTrainingJobs({ limit: 100 });
  const { data: models } = useModels({ limit: 200 });
  const { data: datasets } = useDatasets({ limit: 200 });

  const items = jobs?.items ?? [];

  return (
    <Workspace>
      <PageHeader
        title="Training"
        description="Configure fine-tuning runs and follow them step by step. Jobs run on the server, independent of this browser session."
        actions={
          <Button size="sm" onClick={() => setTab("configure")}>
            <Plus className="h-3.5 w-3.5" />
            New run
          </Button>
        }
      />

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="runs">Runs ({items.length})</TabsTrigger>
          <TabsTrigger value="configure">Configure</TabsTrigger>
        </TabsList>

        <TabsContent value="runs">
          {error ? (
            <ErrorState error={error} onRetry={() => refetch()} />
          ) : isLoading ? (
            <LoadingState rows={4} />
          ) : items.length === 0 ? (
            <EmptyState
              icon={<Activity className="h-5 w-5" />}
              title="No training runs yet"
              description="Pick a model, a dataset and a method, then start a run. Every run creates an experiment automatically."
              action={<Button size="sm" onClick={() => setTab("configure")}>Configure a run</Button>}
            />
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Run</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Method</TableHead>
                    <TableHead className="w-[180px]">Progress</TableHead>
                    <TableHead className="text-right">Loss</TableHead>
                    <TableHead className="text-right">Val loss</TableHead>
                    <TableHead className="text-right">Tokens/s</TableHead>
                    <TableHead className="text-right">ETA</TableHead>
                    <TableHead className="text-right">Started</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {items.map((job) => {
                    const progress = job.total_steps ? (job.current_step / job.total_steps) * 100 : 0;
                    return (
                      <TableRow
                        key={job.id}
                        className="cursor-pointer"
                        onClick={() => router.push(`/training/${job.id}`)}
                      >
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Link
                              href={`/training/${job.id}`}
                              className="truncate text-xs text-ink hover:text-accent"
                              onClick={(event) => event.stopPropagation()}
                            >
                              {job.name}
                            </Link>
                            <ProvenanceBadge provenance={job.provenance} compact />
                          </div>
                          <div className="mono text-2xs text-ink-muted">
                            {job.id} · backend {job.backend}
                          </div>
                        </TableCell>
                        <TableCell><JobStatusBadge status={job.status} /></TableCell>
                        <TableCell><Badge variant="outline">{job.method}</Badge></TableCell>
                        <TableCell>
                          <div className="space-y-1">
                            <Progress value={progress} size="sm" />
                            <div className="mono text-2xs text-ink-muted">
                              {formatNumber(job.current_step)}/{formatNumber(job.total_steps)} ·{" "}
                              {formatFloat(job.current_epoch, 2)}ep
                            </div>
                          </div>
                        </TableCell>
                        <TableCell className="mono text-right text-xs">{formatFloat(job.loss, 4)}</TableCell>
                        <TableCell className="mono text-right text-xs">{formatFloat(job.val_loss, 4)}</TableCell>
                        <TableCell className="mono text-right text-xs">{formatCompact(job.tokens_per_sec ?? 0)}</TableCell>
                        <TableCell className="mono text-right text-xs">{formatEta(job.eta_seconds)}</TableCell>
                        <TableCell className="text-right text-2xs text-ink-muted">
                          {formatRelative(job.started_at ?? job.created_at)}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="configure">
          {!models?.items.length || !datasets?.items.length ? (
            <EmptyState
              title="A model and a dataset are required"
              description="Import at least one model and one dataset before configuring a training run."
              action={
                <div className="flex gap-2">
                  <Button size="sm" asChild><Link href="/models">Models</Link></Button>
                  <Button size="sm" variant="secondary" asChild><Link href="/datasets">Datasets</Link></Button>
                </div>
              }
            />
          ) : (
            <TrainingConfigForm
              models={models.items}
              datasets={datasets.items}
              initialModelId={presetModel}
              initialDatasetId={presetDataset}
              onLaunched={(jobId) => router.push(`/training/${jobId}`)}
            />
          )}
        </TabsContent>
      </Tabs>
    </Workspace>
  );
}

/**
 * `useSearchParams` opts this route into client-side rendering, so the page body
 * lives behind a Suspense boundary (required by the Next.js app router).
 */
export default function TrainingPage() {
  return (
    <React.Suspense
      fallback={
        <div className="p-6 text-xs text-ink-muted">Loading training…</div>
      }
    >
      <TrainingPageContent />
    </React.Suspense>
  );
}
