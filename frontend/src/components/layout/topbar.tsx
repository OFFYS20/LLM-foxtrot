"use client";

import { Cpu, HardDrive, MemoryStick, Wifi, WifiOff, Zap } from "lucide-react";
import * as React from "react";

import { LiveDot, ProvenanceBadge } from "@/components/common/badges";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Hint } from "@/components/ui/tooltip";
import { useHardware, useModels, useProject, useSystemInfo } from "@/lib/hooks/queries";
import { useRealtimeState, useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { useWorkspace } from "@/lib/store";
import { formatPercent } from "@/lib/format";
import type { HardwareSnapshot } from "@/lib/types";
import { cn } from "@/lib/utils";
import { isMockProvider } from "@/lib/api";

function Reading({
  icon,
  label,
  value,
  sub,
  tone = "default",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  tone?: "default" | "warning" | "critical";
}) {
  return (
    <Hint
      label={
        <span>
          {label}
          {sub ? ` · ${sub}` : ""}
        </span>
      }
    >
      <div className="flex items-center gap-1.5 rounded border border-border bg-elevated/60 px-2 py-1">
        <span className="text-ink-muted">{icon}</span>
        <span
          className={cn(
            "mono text-2xs",
            tone === "critical" ? "text-critical" : tone === "warning" ? "text-warning" : "text-ink-secondary",
          )}
        >
          {value}
        </span>
      </div>
    </Hint>
  );
}

export function Topbar() {
  const { data: system } = useSystemInfo();
  const { data: models } = useModels({ limit: 100 });
  const activeModelId = useWorkspace((state) => state.activeModelId);
  const setActiveModel = useWorkspace((state) => state.setActiveModel);
  const activeProjectId = useWorkspace((state) => state.activeProjectId);
  const cachedProjectName = useWorkspace((state) => state.activeProjectName);
  const { data: project } = useProject(activeProjectId);
  // The live name wins; the cached one covers the first paint.
  const projectName = activeProjectId ? (project?.name ?? cachedProjectName) : "No project";
  const connection = useRealtimeState();

  const { data: polledHardware } = useHardware(isMockProvider ? 4000 : 15_000);
  const [liveHardware, setLiveHardware] = React.useState<HardwareSnapshot | null>(null);

  useRealtimeTopic<HardwareSnapshot>("hardware.sample", (event) => setLiveHardware(event.data));
  const hardware = liveHardware ?? polledHardware;

  // Memoised so the effect below does not re-run on every render.
  const items = React.useMemo(() => models?.items ?? [], [models]);
  const activeModel = items.find((model) => model.id === activeModelId) ?? null;

  React.useEffect(() => {
    if (!activeModelId && items.length > 0) {
      const loaded = items.find((model) => model.status === "loaded") ?? items[0];
      setActiveModel(loaded.id);
    }
  }, [activeModelId, items, setActiveModel]);

  const gpu = hardware?.gpus?.[0];
  const vramPct = gpu ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0;
  const ramPct = hardware ? (hardware.ram_used_gb / Math.max(1, hardware.ram_total_gb)) * 100 : 0;

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-panel px-3">
      <div className="flex min-w-0 items-center gap-2">
        <span className="hidden text-2xs uppercase tracking-[0.14em] text-ink-muted md:inline">
          Project
        </span>
        <span className="hidden max-w-[160px] truncate text-xs text-ink-secondary md:inline">
          {projectName}
        </span>
        <span className="hidden h-4 w-px bg-border md:inline" />
        <Select value={activeModelId ?? ""} onValueChange={setActiveModel}>
          <SelectTrigger className="h-7 w-[220px] border-border-strong bg-elevated text-xs">
            <SelectValue placeholder="No model selected" />
          </SelectTrigger>
          <SelectContent>
            {items.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                <span className="flex items-center gap-2">
                  <span className="truncate">{model.display_name ?? model.name}</span>
                  {model.is_demo ? <span className="text-2xs text-warning">demo</span> : null}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {activeModel ? (
          <Badge variant={activeModel.status === "loaded" ? "good" : "muted"} className="hidden lg:inline-flex">
            {activeModel.status}
          </Badge>
        ) : null}
      </div>

      <div className="ml-auto flex items-center gap-2">
        {system?.demo_mode || isMockProvider ? (
          <ProvenanceBadge provenance="simulated" compact />
        ) : null}

        <Badge variant={system?.gpu_available ? "accent" : "warning"} className="hidden sm:inline-flex">
          {system?.compute_mode?.toUpperCase() ?? "…"}
          {system && !system.gpu_available ? " · CPU MODE" : ""}
        </Badge>

        {gpu ? (
          <>
            <Reading
              icon={<Zap className="h-3 w-3" />}
              label={`${gpu.name} utilisation`}
              value={formatPercent(gpu.utilization, 0)}
              tone={gpu.utilization > 95 ? "warning" : "default"}
            />
            <Reading
              icon={<HardDrive className="h-3 w-3" />}
              label="VRAM"
              sub={`${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`}
              value={`${(gpu.memory_used_mb / 1024).toFixed(1)}G`}
              tone={vramPct > 92 ? "critical" : vramPct > 80 ? "warning" : "default"}
            />
          </>
        ) : null}

        <Reading
          icon={<Cpu className="h-3 w-3" />}
          label="CPU utilisation"
          value={formatPercent(hardware?.cpu_percent ?? 0, 0)}
        />
        <Reading
          icon={<MemoryStick className="h-3 w-3" />}
          label="System RAM"
          sub={`${hardware?.ram_used_gb?.toFixed(1) ?? "—"} / ${hardware?.ram_total_gb?.toFixed(0) ?? "—"} GB`}
          value={formatPercent(ramPct, 0)}
          tone={ramPct > 90 ? "warning" : "default"}
        />

        <Hint
          label={
            isMockProvider
              ? "Mock provider — no backend connection"
              : connection === "open"
                ? "Live updates connected"
                : "Reconnecting to the event stream…"
          }
        >
          <span className="flex items-center gap-1 rounded border border-border bg-elevated/60 px-2 py-1">
            {connection === "open" && !isMockProvider ? (
              <Wifi className="h-3 w-3 text-good" />
            ) : (
              <WifiOff className="h-3 w-3 text-ink-muted" />
            )}
            <LiveDot active={connection === "open" && !isMockProvider} label={isMockProvider ? "mock" : undefined} />
          </span>
        </Hint>
      </div>
    </header>
  );
}
