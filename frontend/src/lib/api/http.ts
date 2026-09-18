/** HTTP implementation of the DataProvider contract (FastAPI backend). */

import type { DataProvider, ListParams, StreamCallbacks } from "@/lib/api/provider";
import type { GenerationUsage } from "@/lib/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string = "error",
    readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function buildQuery(params?: ListParams): string {
  if (!params) return "";
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    search.set(key, String(value));
  });
  const query = search.toString();
  return query ? `?${query}` : "";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
      cache: "no-store",
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the Foxtrot API at ${API_BASE_URL}. Is the backend running?`,
      0,
      "network_error",
      cause,
    );
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let code = "error";
    let details: unknown;
    try {
      const body = await response.json();
      if (body?.error) {
        message = body.error.message ?? message;
        code = body.error.code ?? code;
        details = body.error.details;
      } else if (body?.detail) {
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(message, response.status, code, details);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "PATCH", body: JSON.stringify(body) });
const del = <T,>(path: string) => request<T>(path, { method: "DELETE" });

export const httpProvider: DataProvider = {
  kind: "http",

  system: {
    info: () => request("/system"),
    health: async () => {
      const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
      return response.json();
    },
    settings: () => request("/settings"),
    seedDemoData: (force = false) => post(`/settings/demo-data/seed?force=${force}`),
    clearDemoData: () => post("/settings/demo-data/clear"),
  },

  models: {
    list: (params) => request(`/models${buildQuery(params)}`),
    get: (id) => request(`/models/${id}`),
    import: (payload) => post("/models/import", payload),
    update: (id, payload) => patch(`/models/${id}`, payload),
    remove: (id) => del(`/models/${id}`),
    clone: (id, name, copyCheckpoints = false) =>
      post(`/models/${id}/clone`, { name, copy_checkpoints: copyCheckpoints }),
    load: (id, engine = "auto") => post(`/models/${id}/load`, { engine }),
    unload: (id) => post(`/models/${id}/unload`),
    export: (id, format, destination) =>
      post(`/models/${id}/export`, { format, destination, include_tokenizer: true }),
    checkpoints: (id) => request(`/models/${id}/checkpoints`),
  },

  projects: {
    list: (params) => request(`/projects${buildQuery(params)}`),
    get: (id) => request(`/projects/${id}`),
    create: (payload) => post("/projects", payload),
    update: (id, payload) => patch(`/projects/${id}`, payload),
    remove: (id) => del(`/projects/${id}`),
  },

  datasets: {
    list: (params) => request(`/datasets${buildQuery(params)}`),
    get: (id) => request(`/datasets/${id}`),
    templates: () => request("/datasets/templates"),
    import: (payload) => post("/datasets/import", payload),
    upload: (file, name, template) => {
      const form = new FormData();
      form.append("file", file);
      form.append("name", name);
      form.append("template", template);
      return request("/datasets/upload", { method: "POST", body: form });
    },
    preview: (id, offset = 0, limit = 25) =>
      request(`/datasets/${id}/preview?offset=${offset}&limit=${limit}`),
    validate: (id, template) => post(`/datasets/${id}/validate`, { template, max_rows: 5000 }),
    remove: (id) => del(`/datasets/${id}`),
  },

  training: {
    listJobs: (params) => request(`/training/jobs${buildQuery(params)}`),
    activeJobs: () => request("/training/active"),
    getJob: (id) => request(`/training/jobs/${id}`),
    createJob: (payload) => post("/training/jobs", payload),
    metrics: (id) => request(`/training/jobs/${id}/metrics`),
    logs: (id) => request(`/training/jobs/${id}/logs`),
    validateConfig: (config, datasetId) =>
      post(`/training/validate${datasetId ? `?dataset_id=${datasetId}` : ""}`, config),
    parseRawConfig: (content, format, datasetId) =>
      post("/training/config/parse", { content, format, dataset_id: datasetId }),
    start: (id) => post(`/training/jobs/${id}/start`),
    pause: (id) => post(`/training/jobs/${id}/pause`),
    resume: (id) => post(`/training/jobs/${id}/resume`),
    stop: (id) => post(`/training/jobs/${id}/stop`),
    checkpoint: (id) => post(`/training/jobs/${id}/checkpoint`, {}),
    remove: (id) => del(`/training/jobs/${id}`),
  },

  chat: {
    conversations: (params) => request(`/conversations${buildQuery(params)}`),
    conversation: (id) => request(`/conversations/${id}`),
    createConversation: (payload) => post("/conversations", payload),
    updateConversation: (id, payload) => patch(`/conversations/${id}`, payload),
    deleteConversation: (id) => del(`/conversations/${id}`),
    editMessage: (id, content) => patch(`/messages/${id}`, { content }),
    deleteMessage: (id, andAfter = false) => del(`/messages/${id}?and_after=${andAfter}`),
    tokenize: (text, modelId) => post("/chat/tokenize", { text, model_id: modelId }),

    async stream(payload, callbacks: StreamCallbacks) {
      const response = await fetch(`${API_BASE_URL}/api/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, stream: true }),
        signal: callbacks.signal,
      });

      if (!response.ok || !response.body) {
        const detail = await response.text().catch(() => "");
        callbacks.onError?.(`Generation failed (${response.status}). ${detail.slice(0, 200)}`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          try {
            const parsed = JSON.parse(data) as {
              type: string;
              text?: string;
              usage?: GenerationUsage;
              message_id?: string | null;
              message?: string;
            };
            if (parsed.type === "token" && parsed.text) callbacks.onToken(parsed.text);
            else if (parsed.type === "usage" && parsed.usage)
              callbacks.onUsage?.(parsed.usage, parsed.message_id);
            else if (parsed.type === "error") callbacks.onError?.(parsed.message ?? "Unknown error");
          } catch {
            /* ignore malformed frame */
          }
        }
      }
    },
  },

  playground: {
    run: (payload) => post("/playground/run", { ...payload, persist: true }),
    history: (params) => request(`/playground/comparisons${buildQuery(params)}`),
    vote: (comparisonId, winnerModelId) =>
      post(`/playground/comparisons/${comparisonId}/vote`, { winner_model_id: winnerModelId }),
  },

  benchmarks: {
    suites: () => request("/benchmarks/suites"),
    run: (payload) => post("/benchmarks/run", payload),
    results: (params) => request(`/benchmarks/results${buildQuery(params)}`),
    result: (id, onlyIncorrect = false) =>
      request(`/benchmarks/results/${id}?only_incorrect=${onlyIncorrect}`),
    cancel: (id) => post(`/benchmarks/results/${id}/cancel`),
    remove: (id) => del(`/benchmarks/results/${id}`),
  },

  evaluations: {
    leaderboard: (suite) => request(`/evaluations/leaderboard${suite ? `?suite=${suite}` : ""}`),
    compare: (modelIds, name) => post("/evaluations/compare", { model_ids: modelIds, name }),
  },

  experiments: {
    list: (params) => request(`/experiments${buildQuery(params)}`),
    get: (id) => request(`/experiments/${id}`),
    update: (id, payload) => patch(`/experiments/${id}`, payload),
    duplicate: (id, name, start = false) =>
      post(`/experiments/${id}/duplicate`, { name, start_immediately: start }),
    compare: (ids) => post("/experiments/compare", { experiment_ids: ids }),
    remove: (id) => del(`/experiments/${id}`),
  },

  checkpoints: {
    list: (params) => request(`/checkpoints${buildQuery(params)}`),
    load: (id) => post(`/checkpoints/${id}/load`),
    evaluate: (id, suite = "mmlu", numExamples = 20) =>
      post(`/checkpoints/${id}/evaluate?suite=${suite}&num_examples=${numExamples}`),
    export: (id) => post(`/checkpoints/${id}/export`),
    remove: (id) => del(`/checkpoints/${id}`),
  },

  hardware: {
    snapshot: () => request("/hardware"),
    history: (limit = 240) => request(`/hardware/history?limit=${limit}`),
    devices: () => request("/hardware/devices"),
  },

  logs: {
    list: (params) => request(`/logs${buildQuery(params)}`),
    sources: () => request("/logs/sources"),
    clear: () => del("/logs"),
  },
};
