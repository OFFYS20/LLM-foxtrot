"use client";

import { Database, FlaskConical, RefreshCw, Server, Trash2 } from "lucide-react";
import * as React from "react";

import { ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { MetricRow } from "@/components/common/stat-tile";
import { ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { ProjectsCard } from "@/components/settings/projects-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, API_BASE_URL, isMockProvider } from "@/lib/api";
import { useSettings, useSystemInfo } from "@/lib/hooks/queries";
import { useRealtimeState } from "@/lib/hooks/use-realtime";
import { useWorkspace } from "@/lib/store";

export default function SettingsPage() {
  const { data: system, isLoading, error, refetch } = useSystemInfo();
  const { data: settings } = useSettings();
  const connection = useRealtimeState();

  const [busy, setBusy] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState<string | null>(null);

  const runAction = async (label: string, action: () => Promise<{ message?: string | null }>) => {
    setBusy(label);
    setMessage(null);
    try {
      const result = await action();
      setMessage(result.message ?? "Done");
      await refetch();
    } catch (actionError) {
      setMessage(actionError instanceof Error ? actionError.message : String(actionError));
    } finally {
      setBusy(null);
    }
  };

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !system) return <div className="p-4"><LoadingState rows={4} /></div>;

  const engineStatus = (settings?.engine_status ?? []) as {
    engine: string;
    available: boolean;
    detail?: string | null;
  }[];

  return (
    <Workspace>
      <PageHeader
        title="Settings"
        description="How this deployment is wired up: data source, engines, telemetry and demo data."
      />

      {isMockProvider ? (
        <WarningNote>
          The frontend is running on the <strong>mock provider</strong> — all data comes from
          client-side fixtures. Set <code className="mono">NEXT_PUBLIC_DATA_PROVIDER=http</code> to talk
          to the Foxtrot backend.
        </WarningNote>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Deployment</CardTitle>
            <ProvenanceBadge provenance={system.demo_mode ? "simulated" : "measured"} compact />
          </CardHeader>
          <CardContent className="py-3">
            <div className="divide-y divide-border">
              <MetricRow label="Application" value={`${system.app} ${system.version}`} />
              <MetricRow label="Data provider" value={isMockProvider ? "mock (client-side)" : "http"} />
              <MetricRow label="API base URL" value={API_BASE_URL} />
              <MetricRow label="WebSocket" value={connection} />
              <MetricRow label="Demo mode" value={system.demo_mode ? "on" : "off"} />
              <MetricRow label="Database" value={String(settings?.database_url ?? "—")} />
              <MetricRow label="Data directory" value={String(settings?.data_dir ?? "—")} />
              <MetricRow label="Max upload" value={`${settings?.max_upload_mb ?? "—"} MB`} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Compute</CardTitle>
            <Badge variant={system.gpu_available ? "good" : "warning"}>
              {system.compute_mode.toUpperCase()}
            </Badge>
          </CardHeader>
          <CardContent className="py-3">
            <div className="divide-y divide-border">
              <MetricRow label="GPU available" value={system.gpu_available ? "yes" : "no — CPU mode"} />
              <MetricRow label="Inference engine" value={system.inference_engine} />
              <MetricRow label="Hardware provider" value={system.hardware_provider} />
              <MetricRow label="Sample interval" value={`${settings?.hardware_sample_interval_s ?? "—"} s`} />
            </div>
          </CardContent>
        </Card>
      </div>

      <SectionTitle>Inference engines</SectionTitle>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {engineStatus.map((engine) => (
          <Card key={engine.engine}>
            <CardContent className="space-y-1.5 p-3">
              <div className="flex items-center justify-between">
                <span className="mono text-xs text-ink">{engine.engine}</span>
                <Badge variant={engine.available ? "good" : "muted"}>
                  {engine.available ? "available" : "unavailable"}
                </Badge>
              </div>
              {engine.detail ? <p className="text-2xs text-ink-muted">{engine.detail}</p> : null}
            </CardContent>
          </Card>
        ))}
      </div>

      <SectionTitle>Capabilities</SectionTitle>
      <Card>
        <CardContent className="divide-y divide-border p-0">
          {system.capabilities.map((capability) => (
            <div key={capability.name} className="flex items-start justify-between gap-3 px-4 py-2.5">
              <div>
                <div className="mono text-xs text-ink-secondary">{capability.name}</div>
                {capability.detail ? (
                  <div className="text-2xs text-ink-muted">{capability.detail}</div>
                ) : null}
              </div>
              <Badge variant={capability.available ? "good" : "muted"}>
                {capability.available ? "available" : "off"}
              </Badge>
            </div>
          ))}
        </CardContent>
      </Card>

      <SectionTitle>Projects</SectionTitle>
      <ProjectsCard />

      <SectionTitle>Demo data</SectionTitle>
      <Card>
        <CardContent className="space-y-3 py-3">
          <p className="text-xs text-ink-muted">
            Demo records let every screen be explored before real models exist. They are flagged
            <ProvenanceBadge provenance="simulated" compact /> throughout the UI and can be removed at
            any time — removing them never touches records produced by real runs.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={busy !== null || isMockProvider}
              onClick={() => runAction("seed", () => api.system.seedDemoData(false))}
            >
              <FlaskConical className="h-3.5 w-3.5" />
              {busy === "seed" ? "Seeding…" : "Seed demo data"}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={busy !== null || isMockProvider}
              onClick={() => runAction("reseed", () => api.system.seedDemoData(true))}
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Re-seed (replace)
            </Button>
            <Button
              size="sm"
              variant="danger"
              disabled={busy !== null || isMockProvider}
              onClick={() => runAction("clear", () => api.system.clearDemoData())}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {busy === "clear" ? "Clearing…" : "Clear demo data"}
            </Button>
          </div>
          {message ? <p className="mono text-2xs text-ink-secondary">{message}</p> : null}
        </CardContent>
      </Card>

      <SectionTitle>Environment reference</SectionTitle>
      <Card>
        <CardContent className="space-y-2 py-3">
          <div className="flex items-start gap-2">
            <Server className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-muted" />
            <div className="mono text-2xs text-ink-muted">
              <div>FOXTROT_DATABASE_URL — sqlite:///./data/foxtrot.db (PostgreSQL: postgresql+psycopg://…)</div>
              <div>FOXTROT_DATA_DIR — root for models, datasets, checkpoints, exports</div>
              <div>FOXTROT_INFERENCE_ENGINE — demo | transformers | llamacpp | vllm | ollama</div>
              <div>FOXTROT_HARDWARE_PROVIDER — auto | nvml | simulated</div>
              <div>FOXTROT_DEMO_MODE — enables simulated backends and demo seeding</div>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <Database className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-muted" />
            <div className="mono text-2xs text-ink-muted">
              <div>NEXT_PUBLIC_API_BASE_URL — {API_BASE_URL}</div>
              <div>NEXT_PUBLIC_DATA_PROVIDER — http | mock</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </Workspace>
  );
}
