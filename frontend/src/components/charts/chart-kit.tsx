"use client";

/**
 * Chart kit.
 *
 * Palette: the validated dark categorical steps (contrast >= 3:1 on the panel
 * surface, adjacent-pair CVD separation >= 8 ΔE). Slots are assigned in fixed
 * order and never cycled. Status colors are reserved and never used as a series.
 *
 * Conventions enforced here so every chart in the app agrees:
 *   · one y-axis per chart — never a second scale
 *   · recessive grid + axes, 2px lines, >= 8px hover markers
 *   · crosshair + tooltip by default on time series
 *   · a legend whenever there are two or more series
 */

import * as React from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ProvenanceBadge } from "@/components/common/badges";
import { cn } from "@/lib/utils";
import type { Provenance } from "@/lib/types";

/** Fixed categorical order — index 0 is always the primary series. */
export const CHART_COLORS = [
  "#3987e5", // 1 blue
  "#d95926", // 2 orange
  "#199e70", // 3 aqua
  "#c98500", // 4 yellow
  "#d55181", // 5 magenta
  "#008300", // 6 green
  "#9085e9", // 7 violet
  "#e66767", // 8 red
] as const;

export const STATUS_COLORS = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
} as const;

const GRID = "#2c2c2a";
const AXIS = "#383835";
const MUTED_INK = "#898781";

const axisProps = {
  stroke: AXIS,
  tick: { fill: MUTED_INK, fontSize: 10, fontFamily: "var(--font-mono)" },
  tickLine: false,
  axisLine: { stroke: AXIS },
} as const;

export interface SeriesSpec {
  key: string;
  label: string;
  color?: string;
  /** Rendered as an area with a faint fill instead of a bare line. */
  area?: boolean;
  strokeDasharray?: string;
  unit?: string;
  format?: (value: number) => string;
}

function defaultFormat(value: number): string {
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 1) return value.toFixed(2);
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001) return value.toExponential(1);
  return value.toFixed(4);
}

function ChartTooltip({
  active,
  payload,
  label,
  series,
  labelFormatter,
}: {
  active?: boolean;
  payload?: { dataKey?: string | number; value?: number; color?: string }[];
  label?: string | number;
  series: SeriesSpec[];
  labelFormatter?: (value: string | number) => string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-md border border-border bg-overlay px-2.5 py-2 shadow-xl">
      <div className="mono mb-1 text-2xs uppercase tracking-wider text-ink-muted">
        {labelFormatter ? labelFormatter(label ?? "") : label}
      </div>
      <div className="space-y-0.5">
        {payload.map((entry) => {
          const spec = series.find((s) => s.key === entry.dataKey);
          if (!spec || entry.value === undefined || entry.value === null) return null;
          return (
            <div key={String(entry.dataKey)} className="flex items-center gap-2 text-xs">
              <span
                className="h-2 w-2 shrink-0 rounded-[2px]"
                style={{ backgroundColor: entry.color }}
                aria-hidden
              />
              <span className="text-ink-muted">{spec.label}</span>
              <span className="mono ml-auto text-ink">
                {(spec.format ?? defaultFormat)(entry.value)}
                {spec.unit ? <span className="ml-0.5 text-ink-muted">{spec.unit}</span> : null}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChartLegend({ series }: { series: SeriesSpec[] }) {
  if (series.length < 2) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {series.map((spec, index) => (
        <span key={spec.key} className="flex items-center gap-1.5 text-2xs text-ink-muted">
          <span
            className="h-0.5 w-3 rounded-full"
            style={{ backgroundColor: spec.color ?? CHART_COLORS[index % CHART_COLORS.length] }}
            aria-hidden
          />
          {spec.label}
        </span>
      ))}
    </div>
  );
}

export function ChartFrame({
  title,
  subtitle,
  series,
  provenance,
  right,
  children,
  className,
  height = 200,
}: {
  title: string;
  subtitle?: React.ReactNode;
  series?: SeriesSpec[];
  provenance?: Provenance | string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  height?: number;
}) {
  return (
    <div className={cn("rounded-lg border border-border bg-panel", className)}>
      <div className="flex items-start justify-between gap-3 px-3.5 pb-1 pt-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-2xs font-semibold uppercase tracking-[0.14em] text-ink-secondary">
              {title}
            </h3>
            {provenance ? <ProvenanceBadge provenance={provenance} compact /> : null}
          </div>
          {subtitle ? <p className="mt-0.5 text-2xs text-ink-muted">{subtitle}</p> : null}
        </div>
        <div className="flex items-center gap-2">{right}</div>
      </div>
      {series && series.length > 1 ? (
        <div className="px-3.5 pb-1">
          <ChartLegend series={series} />
        </div>
      ) : null}
      <div style={{ height }} className="px-1 pb-2">
        {children}
      </div>
    </div>
  );
}

export interface TimeSeriesChartProps {
  data: Record<string, unknown>[];
  series: SeriesSpec[];
  xKey: string;
  xLabel?: string;
  yLabel?: string;
  yDomain?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];
  xTickFormatter?: (value: string | number) => string;
  yTickFormatter?: (value: number) => string;
  tooltipLabelFormatter?: (value: string | number) => string;
  referenceValue?: { value: number; label: string };
  height?: number;
}

/** Line/area time series with crosshair + tooltip. One y-axis, always. */
export function TimeSeriesChart({
  data,
  series,
  xKey,
  yDomain,
  xTickFormatter,
  yTickFormatter,
  tooltipLabelFormatter,
  referenceValue,
}: TimeSeriesChartProps) {
  const hasArea = series.some((s) => s.area);
  const Chart = hasArea ? AreaChart : LineChart;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <Chart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <defs>
          {series.map((spec, index) => {
            const color = spec.color ?? CHART_COLORS[index % CHART_COLORS.length];
            return (
              <linearGradient key={spec.key} id={`fill-${spec.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.22} />
                <stop offset="100%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            );
          })}
        </defs>
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey={xKey} {...axisProps} tickFormatter={xTickFormatter} minTickGap={28} />
        <YAxis
          {...axisProps}
          width={52}
          domain={yDomain ?? ["auto", "auto"]}
          tickFormatter={yTickFormatter}
        />
        <Tooltip
          cursor={{ stroke: AXIS, strokeWidth: 1, strokeDasharray: "3 3" }}
          content={<ChartTooltip series={series} labelFormatter={tooltipLabelFormatter} />}
        />
        {referenceValue ? (
          <ReferenceLine
            y={referenceValue.value}
            stroke={MUTED_INK}
            strokeDasharray="4 4"
            label={{ value: referenceValue.label, fill: MUTED_INK, fontSize: 10, position: "right" }}
          />
        ) : null}
        {series.map((spec, index) => {
          const color = spec.color ?? CHART_COLORS[index % CHART_COLORS.length];
          return hasArea ? (
            <Area
              key={spec.key}
              type="monotone"
              dataKey={spec.key}
              name={spec.label}
              stroke={color}
              strokeWidth={2}
              strokeDasharray={spec.strokeDasharray}
              fill={spec.area ? `url(#fill-${spec.key})` : "transparent"}
              fillOpacity={spec.area ? 1 : 0}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "#111113" }}
              isAnimationActive={false}
              connectNulls
            />
          ) : (
            <Line
              key={spec.key}
              type="monotone"
              dataKey={spec.key}
              name={spec.label}
              stroke={color}
              strokeWidth={2}
              strokeDasharray={spec.strokeDasharray}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "#111113" }}
              isAnimationActive={false}
              connectNulls
            />
          );
        })}
      </Chart>
    </ResponsiveContainer>
  );
}

export function BarSeriesChart({
  data,
  series,
  xKey,
  layout = "horizontal",
  yTickFormatter,
  colorByIndex = false,
  xTickFormatter,
  valueDomain,
}: {
  data: Record<string, unknown>[];
  series: SeriesSpec[];
  xKey: string;
  layout?: "horizontal" | "vertical";
  yTickFormatter?: (value: number) => string;
  xTickFormatter?: (value: string | number) => string;
  colorByIndex?: boolean;
  /** Fix the value axis (e.g. [0, 100] for percentages) so an all-zero
   *  series still reads against a meaningful scale instead of autoscaling. */
  valueDomain?: [number, number];
}) {
  const vertical = layout === "vertical";
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout={layout}
        margin={{ top: 8, right: 16, bottom: 4, left: vertical ? 8 : 4 }}
        barCategoryGap={vertical ? "22%" : "28%"}
      >
        <CartesianGrid stroke={GRID} strokeDasharray="2 4" vertical={vertical} horizontal={!vertical} />
        {vertical ? (
          <>
            <XAxis
              type="number"
              {...axisProps}
              domain={valueDomain ?? ["auto", "auto"]}
              tickFormatter={yTickFormatter}
            />
            <YAxis type="category" dataKey={xKey} {...axisProps} width={128} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} {...axisProps} tickFormatter={xTickFormatter} interval={0} />
            <YAxis
              {...axisProps}
              width={48}
              domain={valueDomain ?? ["auto", "auto"]}
              tickFormatter={yTickFormatter}
            />
          </>
        )}
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          content={<ChartTooltip series={series} />}
        />
        {series.map((spec, index) => (
          <Bar
            key={spec.key}
            dataKey={spec.key}
            name={spec.label}
            radius={vertical ? [0, 4, 4, 0] : [4, 4, 0, 0]}
            fill={spec.color ?? CHART_COLORS[index % CHART_COLORS.length]}
            isAnimationActive={false}
          >
            {colorByIndex
              ? data.map((_, cellIndex) => (
                  <Cell key={cellIndex} fill={CHART_COLORS[cellIndex % CHART_COLORS.length]} />
                ))
              : null}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RadarProfileChart({
  data,
  series,
  angleKey,
  max = 100,
}: {
  data: Record<string, unknown>[];
  series: SeriesSpec[];
  angleKey: string;
  max?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke={GRID} />
        <PolarAngleAxis dataKey={angleKey} tick={{ fill: MUTED_INK, fontSize: 10 }} />
        <PolarRadiusAxis
          domain={[0, max]}
          tick={{ fill: MUTED_INK, fontSize: 9 }}
          axisLine={false}
          tickCount={5}
        />
        <Tooltip content={<ChartTooltip series={series} />} />
        <Legend
          wrapperStyle={{ fontSize: 10, color: MUTED_INK }}
          formatter={(value) => <span style={{ color: MUTED_INK }}>{value}</span>}
        />
        {series.map((spec, index) => {
          const color = spec.color ?? CHART_COLORS[index % CHART_COLORS.length];
          return (
            <Radar
              key={spec.key}
              name={spec.label}
              dataKey={spec.key}
              stroke={color}
              strokeWidth={2}
              fill={color}
              fillOpacity={0.12}
              isAnimationActive={false}
            />
          );
        })}
      </RadarChart>
    </ResponsiveContainer>
  );
}

/** Tiny inline trend — no axes, no tooltip; it supports a number, never replaces it. */
export function Sparkline({
  data,
  dataKey,
  color = CHART_COLORS[0],
  fill = true,
}: {
  data: Record<string, unknown>[];
  dataKey: string;
  color?: string;
  fill?: boolean;
}) {
  if (!data.length) return null;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`spark-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={1.5}
          fill={fill ? `url(#spark-${dataKey})` : "transparent"}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Horizontal usage meter (VRAM, disk, RAM) — a bar, not a gauge. */
export function UsageMeter({
  used,
  total,
  label,
  unit = "GB",
  warnAt = 0.85,
}: {
  used: number;
  total: number;
  label?: string;
  unit?: string;
  warnAt?: number;
}) {
  const ratio = total > 0 ? Math.min(1, used / total) : 0;
  const tone =
    ratio >= warnAt ? STATUS_COLORS.critical : ratio >= 0.7 ? STATUS_COLORS.warning : CHART_COLORS[0];

  return (
    <div className="space-y-1">
      {label ? (
        <div className="flex items-baseline justify-between text-2xs">
          <span className="text-ink-muted">{label}</span>
          <span className="mono text-ink-secondary">
            {used.toFixed(1)} / {total.toFixed(1)} {unit}
          </span>
        </div>
      ) : null}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${ratio * 100}%`, backgroundColor: tone }}
        />
      </div>
    </div>
  );
}
