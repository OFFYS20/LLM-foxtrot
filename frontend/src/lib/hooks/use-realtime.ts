"use client";

import { useEffect, useRef, useState } from "react";

import { isMockProvider } from "@/lib/api";
import { realtime, type ConnectionState } from "@/lib/realtime";
import type { RealtimeEvent } from "@/lib/types";

/** Subscribe to a topic (or topic prefix) for the lifetime of a component. */
export function useRealtimeTopic<T = Record<string, unknown>>(
  topic: string | null,
  handler: (event: RealtimeEvent<T>) => void,
  enabled = true,
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (!topic || !enabled || isMockProvider) return;
    return realtime.subscribe(topic, (event) => handlerRef.current(event as RealtimeEvent<T>));
  }, [topic, enabled]);
}

export function useRealtimeState(): ConnectionState {
  const [state, setState] = useState<ConnectionState>(isMockProvider ? "idle" : "connecting");

  useEffect(() => {
    if (isMockProvider) return;
    realtime.connect();
    return realtime.onStateChange(setState);
  }, []);

  return state;
}
