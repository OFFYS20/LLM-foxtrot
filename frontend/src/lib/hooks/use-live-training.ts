"use client";

import * as React from "react";

import { useTrainingJob, useTrainingMetrics } from "@/lib/hooks/queries";
import { useRealtimeTopic } from "@/lib/hooks/use-realtime";
import { isMockProvider } from "@/lib/api";
import type {
  TrainingJobDetail,
  TrainingMetricEvent,
  TrainingMetricPoint,
  TrainingStatusEvent,
} from "@/lib/types";

const MAX_POINTS = 1500;

/**
 * A training job plus its live metric stream.
 *
 * The initial series comes from the API; every subsequent point arrives over
 * the WebSocket and is appended locally, so an active run redraws at step
 * cadence without refetching. When the socket is unavailable (mock provider)
 * the query polls instead.
 */
export function useLiveTrainingJob(jobId: string | null) {
  const isLiveCandidate = Boolean(jobId);
  const jobQuery = useTrainingJob(jobId, {
    refetchInterval: isMockProvider ? false : 10_000,
  });
  const metricsQuery = useTrainingMetrics(jobId);

  const [streamed, setStreamed] = React.useState<TrainingMetricPoint[]>([]);
  const [liveStatus, setLiveStatus] = React.useState<TrainingStatusEvent | null>(null);

  React.useEffect(() => {
    setStreamed([]);
    setLiveStatus(null);
  }, [jobId]);

  useRealtimeTopic<TrainingMetricEvent>(
    jobId ? `training.metric.${jobId}` : null,
    (event) => {
      const point = event.data;
      setStreamed((current) => {
        if (current.some((existing) => existing.step === point.step)) return current;
        const next = [...current, point];
        return next.length > MAX_POINTS ? next.slice(next.length - MAX_POINTS) : next;
      });
    },
    isLiveCandidate,
  );

  useRealtimeTopic<TrainingStatusEvent>(
    jobId ? `training.status.${jobId}` : null,
    (event) => setLiveStatus(event.data),
    isLiveCandidate,
  );

  const metrics = React.useMemo(() => {
    const base = metricsQuery.data ?? [];
    if (streamed.length === 0) return base;
    const seen = new Set(base.map((point) => point.step));
    const merged = [...base, ...streamed.filter((point) => !seen.has(point.step))];
    return merged.sort((a, b) => a.step - b.step);
  }, [metricsQuery.data, streamed]);

  const latest = metrics.at(-1) ?? null;

  const job: TrainingJobDetail | undefined = React.useMemo(() => {
    if (!jobQuery.data) return undefined;
    const merged = { ...jobQuery.data } as TrainingJobDetail;
    if (liveStatus?.status) merged.status = liveStatus.status;
    if (liveStatus?.error) merged.error = liveStatus.error;
    if (latest) {
      merged.current_step = Math.max(merged.current_step, latest.step);
      merged.current_epoch = latest.epoch ?? merged.current_epoch;
      merged.loss = latest.loss ?? merged.loss;
      merged.val_loss = latest.val_loss ?? merged.val_loss;
      merged.learning_rate = latest.learning_rate ?? merged.learning_rate;
      merged.grad_norm = latest.grad_norm ?? merged.grad_norm;
      merged.tokens_per_sec = latest.tokens_per_sec ?? merged.tokens_per_sec;
      merged.samples_per_sec = latest.samples_per_sec ?? merged.samples_per_sec;
      merged.gpu_utilization = latest.gpu_utilization ?? merged.gpu_utilization;
      merged.vram_used_mb = latest.vram_used_mb ?? merged.vram_used_mb;
      const streamedPoint = latest as TrainingMetricEvent;
      if (typeof streamedPoint.tokens_processed === "number") {
        merged.tokens_processed = streamedPoint.tokens_processed;
      }
      if (typeof streamedPoint.eta_seconds === "number") {
        merged.eta_seconds = streamedPoint.eta_seconds;
      }
    }
    merged.metrics = metrics;
    return merged;
  }, [jobQuery.data, latest, liveStatus, metrics]);

  return {
    job,
    metrics,
    isLoading: jobQuery.isLoading,
    error: jobQuery.error,
    refetch: () => {
      void jobQuery.refetch();
      void metricsQuery.refetch();
    },
    streamedCount: streamed.length,
  };
}

/** Live console lines for a job (WebSocket, with the API tail as the seed). */
export function useTrainingConsole(jobId: string | null, seed: { message: string; level: string; ts: string }[]) {
  const [lines, setLines] = React.useState<{ message: string; level: string; ts: string }[]>([]);

  React.useEffect(() => {
    setLines(seed.map((entry) => ({ message: entry.message, level: entry.level, ts: entry.ts })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, seed.length]);

  useRealtimeTopic<{ message: string; level: string; ts: string }>(
    jobId ? `training.log.${jobId}` : null,
    (event) => {
      setLines((current) => {
        const next = [...current, event.data];
        return next.length > 600 ? next.slice(next.length - 600) : next;
      });
    },
  );

  return lines;
}
