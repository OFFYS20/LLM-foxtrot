"use client";

/**
 * WebSocket client for live training metrics, hardware samples, benchmark
 * progress and log lines.
 *
 * One socket per browser tab, shared by every subscriber, with exponential
 * reconnect. When the mock provider is active (or the socket cannot connect)
 * the hooks fall back to polling, so the UI degrades instead of freezing.
 */

import type { RealtimeEvent } from "@/lib/types";

type Handler = (event: RealtimeEvent) => void;

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ??
  (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000")
    .replace(/^http/, "ws")
    .replace(/\/$/, "") + "/ws";

export type ConnectionState = "idle" | "connecting" | "open" | "closed";

class RealtimeClient {
  private socket: WebSocket | null = null;
  private handlers = new Map<string, Set<Handler>>();
  private stateListeners = new Set<(state: ConnectionState) => void>();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private manuallyClosed = false;

  state: ConnectionState = "idle";

  private setState(state: ConnectionState) {
    this.state = state;
    this.stateListeners.forEach((listener) => listener(state));
  }

  connect() {
    if (typeof window === "undefined") return;
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.manuallyClosed = false;
    this.setState("connecting");

    try {
      this.socket = new WebSocket(WS_URL);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.setState("open");
    };

    this.socket.onmessage = (raw) => {
      try {
        const event = JSON.parse(raw.data as string) as RealtimeEvent;
        this.dispatch(event);
      } catch {
        /* ignore malformed frame */
      }
    };

    this.socket.onclose = () => {
      this.setState("closed");
      if (!this.manuallyClosed) this.scheduleReconnect();
    };

    this.socket.onerror = () => {
      this.socket?.close();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    const delay = Math.min(15_000, 800 * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private dispatch(event: RealtimeEvent) {
    this.handlers.forEach((handlers, pattern) => {
      const matches =
        pattern === "*" ||
        event.topic === pattern ||
        (pattern.endsWith("*") && event.topic.startsWith(pattern.slice(0, -1))) ||
        event.topic.startsWith(`${pattern}.`);
      if (matches) handlers.forEach((handler) => handler(event));
    });
  }

  subscribe(topic: string, handler: Handler): () => void {
    this.connect();
    const handlers = this.handlers.get(topic) ?? new Set<Handler>();
    handlers.add(handler);
    this.handlers.set(topic, handlers);

    return () => {
      const current = this.handlers.get(topic);
      if (!current) return;
      current.delete(handler);
      if (current.size === 0) this.handlers.delete(topic);
    };
  }

  onStateChange(listener: (state: ConnectionState) => void): () => void {
    this.stateListeners.add(listener);
    listener(this.state);
    return () => this.stateListeners.delete(listener);
  }

  close() {
    this.manuallyClosed = true;
    this.socket?.close();
    this.socket = null;
  }
}

export const realtime = new RealtimeClient();

export const TOPICS = {
  trainingMetric: "training.metric",
  trainingStatus: "training.status",
  trainingLog: "training.log",
  trainingCheckpoint: "training.checkpoint",
  hardware: "hardware.sample",
  benchmarkProgress: "benchmark.progress",
  log: "log.entry",
  modelStatus: "model.status",
} as const;
