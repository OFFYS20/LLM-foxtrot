"use client";

import { Cpu, HardDrive, MemoryStick, Thermometer, Zap } from "lucide-react";
import * as React from "react";

import { CHART_COLORS, ChartFrame, TimeSeriesChart, UsageMeter } from "@/components/charts/chart-kit";
import { ProvenanceBadge } from "@/components/common/badges";
import { PageHeader, SectionTitle } from "@/components/common/page-header";
import { MetricRow, StatTile } from "@/components/common/stat-tile";
import { ErrorState, LoadingState, WarningNote } from "@/components/common/states";
import { Workspace } from "@/components/layout/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { isMockProvider } from "@/lib/api";
import { useHardware, useHardwareHistory } from "@/lib/hooks/queries";
import { useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { formatFloat, formatNumber, formatPercent } from "@/lib/format";
import type { HardwareSnapshot } from "@/lib/types";

export default function HardwarePage() {
  const { data: polled, isLoading, error, refetch } = useHardware(isMockProvider ? 3000 : 15_000);
  const { data: history } = useHardwareHistory(240, isMockProvider ? 6000 : 15_000);
  const [live, setLive] = React.useState<HardwareSnapshot | null>(null);

  useRealtimeTopic<HardwareSnapshot>("hardware.sample", (event) => setLive(event.data));
  const hardware = live ?? polled;

  const series = React.useMemo(
    () =>
      (history?.points ?? []).map((point, index) => {
        const row: Record<string, unknown> = {
          index,
          cpu: point.cpu_percent,
          ram: point.ram_used_gb,
        };
        point.gpu_utilization.forEach((value, gpuIndex) => {
          row[`gpu${gpuIndex}`] = value;
        });
        point.gpu_memory_used_mb.forEach((value, gpuIndex) => {
          row[`vram${gpuIndex}`] = Number((value / 1024).toFixed(2));
        });
        point.gpu_temperature_c.forEach((value, gpuIndex) => {
          row[`temp${gpuIndex}`] = value;
        });
        point.gpu_power_w.forEach((value, gpuIndex) => {
          row[`power${gpuIndex}`] = value;
        });
        return row;
      }),
    [history],
  );

  if (error) return <div className="p-4"><ErrorState error={error} onRetry={() => refetch()} /></div>;
  if (isLoading || !hardware) return <div className="p-4"><LoadingState rows={4} /></div>;

  const gpuSeries = hardware.gpus.map((gpu, index) => ({
    key: `gpu${index}`,
    label: `GPU ${index}`,
    color: CHART_COLORS[index % CHART_COLORS.length],
    unit: "%",
  }));
  const vramSeries = hardware.gpus.map((gpu, index) => ({
    key: `vram${index}`,
    label: `GPU ${index}`,
    color: CHART_COLORS[index % CHART_COLORS.length],
    unit: " GB",
  }));
  const tempSeries = hardware.gpus.map((gpu, index) => ({
    key: `temp${index}`,
    label: `GPU ${index}`,
    color: CHART_COLORS[index % CHART_COLORS.length],
    unit: "°C",
  }));
  const powerSeries = hardware.gpus.map((gpu, index) => ({
    key: `power${index}`,
    label: `GPU ${index}`,
    color: CHART_COLORS[index % CHART_COLORS.length],
    unit: " W",
  }));

  return (
    <Workspace
      inspectorTitle="System"
      inspector={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Platform</CardTitle>
              <ProvenanceBadge provenance={hardware.provenance} compact />
            </CardHeader>
            <CardContent className="py-3">
              <div className="divide-y divide-border">
                <MetricRow label="Compute mode" value={hardware.compute_mode.toUpperCase()} />
                <MetricRow label="GPU available" value={hardware.gpu_available ? "yes" : "no"} />
                <MetricRow label="GPUs" value={formatNumber(hardware.gpus.length)} />
                <MetricRow label="Driver" value={hardware.driver_version ?? "—"} />
                <MetricRow label="CUDA" value={hardware.cuda_version ?? "—"} />
                <MetricRow label="Platform" value={hardware.platform} />
                <MetricRow label="CPU" value={hardware.cpu_model ?? "—"} />
                <MetricRow label="Cores" value={formatNumber(hardware.cpu_cores)} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Storage</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 py-3">
              {hardware.disks.map((disk) => (
                <UsageMeter
                  key={disk.mount}
                  label={disk.mount}
                  used={disk.used_gb}
                  total={disk.total_gb}
                />
              ))}
              <UsageMeter label="Swap" used={hardware.swap_used_gb} total={hardware.swap_total_gb} />
            </CardContent>
          </Card>
        </>
      }
    >
      <PageHeader
        title="Hardware"
        description="Live telemetry for the machine this workspace runs on."
        actions={<ProvenanceBadge provenance={hardware.provenance} />}
      />

      {hardware.note ? <WarningNote>{hardware.note}</WarningNote> : null}
      {!hardware.gpu_available ? (
        <WarningNote>
          <strong>CPU mode.</strong> No GPU is visible to the platform — training and inference will be
          far slower, and VRAM-dependent configurations are unavailable.
        </WarningNote>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="CPU utilisation"
          value={formatPercent(hardware.cpu_percent, 1)}
          hint={hardware.cpu_model ?? undefined}
          icon={<Cpu className="h-3.5 w-3.5" />}
        />
        <StatTile
          label="System RAM"
          value={formatFloat(hardware.ram_used_gb, 1)}
          unit={`/ ${hardware.ram_total_gb.toFixed(0)} GB`}
          icon={<MemoryStick className="h-3.5 w-3.5" />}
          sparkline={<UsageMeter used={hardware.ram_used_gb} total={hardware.ram_total_gb} />}
        />
        <StatTile
          label="Swap"
          value={formatFloat(hardware.swap_used_gb, 1)}
          unit={`/ ${hardware.swap_total_gb.toFixed(0)} GB`}
          icon={<HardDrive className="h-3.5 w-3.5" />}
        />
        <StatTile
          label="CPU temperature"
          value={hardware.cpu_temperature_c ? formatFloat(hardware.cpu_temperature_c, 1) : "—"}
          unit="°C"
          icon={<Thermometer className="h-3.5 w-3.5" />}
        />
      </div>

      <SectionTitle>GPUs</SectionTitle>
      {hardware.gpus.length === 0 ? (
        <Card>
          <CardContent className="py-6 text-center">
            <p className="text-xs text-ink-muted">
              No GPU devices detected. Install NVIDIA drivers and{" "}
              <code className="mono">nvidia-ml-py</code> for real telemetry.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {hardware.gpus.map((gpu, index) => (
            <Card key={gpu.index}>
              <CardHeader>
                <div>
                  <CardTitle>
                    <span className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                      />
                      GPU {gpu.index}
                    </span>
                  </CardTitle>
                  <p className="mono mt-1 text-2xs text-ink-muted">{gpu.name}</p>
                </div>
                <Badge variant={gpu.utilization > 90 ? "warning" : "muted"}>
                  <Zap className="h-3 w-3" />
                  {formatPercent(gpu.utilization, 0)}
                </Badge>
              </CardHeader>
              <CardContent className="space-y-3">
                <UsageMeter
                  label="VRAM"
                  used={gpu.memory_used_mb / 1024}
                  total={gpu.memory_total_mb / 1024}
                />
                <div className="grid grid-cols-2 gap-x-4 divide-y-0">
                  <MetricRow label="Temperature" value={`${formatFloat(gpu.temperature_c, 1)} °C`} />
                  <MetricRow label="Power" value={`${formatFloat(gpu.power_draw_w, 0)} / ${formatFloat(gpu.power_limit_w, 0)} W`} />
                  <MetricRow label="Fan" value={formatPercent(gpu.fan_speed_pct ?? 0, 0)} />
                  <MetricRow label="Clock" value={`${formatNumber(gpu.clock_mhz)} MHz`} />
                  <MetricRow label="Processes" value={formatNumber(gpu.processes)} />
                  <MetricRow label="Compute" value={gpu.compute_capability ?? "—"} />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <SectionTitle>History</SectionTitle>
      <div className="grid gap-3 lg:grid-cols-2">
        <ChartFrame
          title="GPU utilisation"
          provenance={hardware.provenance}
          series={gpuSeries}
          subtitle={`sampled every ${history?.interval_seconds ?? 2}s`}
        >
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={gpuSeries}
            yDomain={[0, 100]}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>

        <ChartFrame title="VRAM usage" provenance={hardware.provenance} series={vramSeries}>
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={vramSeries}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>

        <ChartFrame title="GPU temperature" provenance={hardware.provenance} series={tempSeries}>
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={tempSeries}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>

        <ChartFrame title="Power draw" provenance={hardware.provenance} series={powerSeries}>
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={powerSeries}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>

        <ChartFrame
          title="CPU utilisation"
          provenance={hardware.provenance}
          series={[{ key: "cpu", label: "CPU", color: CHART_COLORS[3], unit: "%" }]}
        >
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={[{ key: "cpu", label: "CPU", color: CHART_COLORS[3], area: true, unit: "%" }]}
            yDomain={[0, 100]}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>

        <ChartFrame
          title="System RAM"
          provenance={hardware.provenance}
          series={[{ key: "ram", label: "RAM", color: CHART_COLORS[4], unit: " GB" }]}
        >
          <TimeSeriesChart
            data={series}
            xKey="index"
            series={[{ key: "ram", label: "RAM", color: CHART_COLORS[4], area: true, unit: " GB" }]}
            xTickFormatter={() => ""}
            tooltipLabelFormatter={() => "sample"}
          />
        </ChartFrame>
      </div>
    </Workspace>
  );
}
