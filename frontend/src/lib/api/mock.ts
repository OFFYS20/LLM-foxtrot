/**
 * Mock implementation of the DataProvider contract.
 *
 * Runs the entire UI with zero backend — useful for design work, offline demos
 * and tests. Everything it returns is flagged `provenance: "simulated"` /
 * `is_demo: true`, exactly like the backend's demo mode, so no screen can
 * present mock output as a measured result.
 */

import type { DataProvider, ListParams, StreamCallbacks } from "@/lib/api/provider";
import type {
  BenchmarkRun,
  BenchmarkRunDetail,
  BenchmarkSuite,
  Checkpoint,
  Conversation,
  ConversationDetail,
  Dataset,
  DatasetDetail,
  Experiment,
  HardwareSnapshot,
  LogEntry,
  Message,
  Model,
  Page,
  PlaygroundComparison,
  Project,
  ProjectDetail,
  TrainingConfig,
  TrainingJob,
  TrainingJobDetail,
  TrainingMetricPoint,
} from "@/lib/types";
import { DEFAULT_TRAINING_CONFIG } from "@/lib/constants";

const now = () => new Date().toISOString();
const id = (prefix: string) => `${prefix}_${Math.random().toString(16).slice(2, 12)}`;

function page<T>(items: T[], params?: ListParams): Page<T> {
  const limit = params?.limit ?? 50;
  const offset = params?.offset ?? 0;
  return { items: items.slice(offset, offset + limit), total: items.length, limit, offset };
}

function seededRandom(seed: number) {
  let value = seed;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
}

const MODELS: Model[] = [
  {
    id: "mdl_demo_base",
    name: "foxtrot-7b-base",
    display_name: "Foxtrot 7B Base",
    description: "Mock base model (client-side fixtures).",
    source: "demo",
    format: "safetensors",
    status: "ready",
    repo_id: "foxtrot-ai/foxtrot-7b-base",
    architecture: "LlamaForCausalLM",
    parameters: 7240,
    context_length: 8192,
    precision: "bf16",
    size_bytes: 15_246_000_000,
    vram_estimate_mb: 15_400,
    tokenizer: { type: "SentencePieceBPE", vocab_size: 32000, chat_template: true },
    config: { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32 },
    metrics: { inference_tokens_per_sec: 148.2 },
    tags: ["base", "demo"],
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
  {
    id: "mdl_demo_sft3",
    name: "foxtrot-7b-sft-e3",
    display_name: "Foxtrot 7B SFT · epoch 3",
    description: "Mock fine-tuned model (client-side fixtures).",
    source: "checkpoint",
    format: "safetensors",
    status: "loaded",
    architecture: "LlamaForCausalLM",
    parameters: 7240,
    context_length: 8192,
    precision: "bf16",
    size_bytes: 15_246_000_000,
    vram_estimate_mb: 15_700,
    tokenizer: { type: "SentencePieceBPE", vocab_size: 32000, chat_template: true },
    config: { hidden_size: 4096, num_hidden_layers: 32 },
    metrics: { inference_tokens_per_sec: 139.4 },
    tags: ["fine-tuned", "demo"],
    parent_model_id: "mdl_demo_base",
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
  {
    id: "mdl_demo_small",
    name: "foxtrot-1.5b-instruct",
    display_name: "Foxtrot 1.5B Instruct",
    description: "Small mock model for latency comparisons.",
    source: "demo",
    format: "gguf",
    status: "ready",
    architecture: "Qwen2ForCausalLM",
    parameters: 1540,
    context_length: 32768,
    precision: "int4",
    quantization: "Q4_K_M",
    size_bytes: 1_052_000_000,
    vram_estimate_mb: 2100,
    tokenizer: { type: "BPE", vocab_size: 151646, chat_template: true },
    config: { hidden_size: 1536, num_hidden_layers: 28 },
    metrics: { inference_tokens_per_sec: 312.5 },
    tags: ["small", "demo"],
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
];

const PROJECTS: Project[] = [
  {
    id: "prj_demo_workspace",
    name: "Foxtrot workspace",
    description: "Mock project (client-side fixtures).",
    base_model_id: "mdl_demo_base",
    default_dataset_id: "ds_demo_sft",
    tags: ["demo"],
    settings: {},
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
];

function projectDetail(project: Project): ProjectDetail {
  return {
    ...project,
    base_model_name: "Foxtrot 7B Base",
    default_dataset_name: "foxtrot-instruct-50k",
    experiment_count: 0,
    training_job_count: 0,
    running_job_count: 0,
    checkpoint_count: 0,
    last_activity_at: null,
  };
}

const DATASETS: Dataset[] = [
  {
    id: "ds_demo_sft",
    name: "foxtrot-instruct-50k",
    description: "Mock instruction dataset.",
    format: "jsonl",
    template: "instruction",
    status: "ready",
    source: "demo",
    rows: 50_000,
    tokens: 18_420_000,
    avg_sequence_length: 368.4,
    max_sequence_length: 4096,
    size_bytes: 1_503_238_553,
    train_split: 0.9,
    validation_split: 0.05,
    test_split: 0.05,
    columns: ["instruction", "input", "output"],
    validation_report: { checked_rows: 50_000, valid_rows: 49_987, invalid_rows: 13 },
    tags: ["sft", "demo"],
    token_count_method: "estimated",
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
  {
    id: "ds_demo_chat",
    name: "foxtrot-chat-12k",
    description: "Mock multi-turn chat dataset.",
    format: "jsonl",
    template: "chat",
    status: "ready",
    source: "demo",
    rows: 12_400,
    tokens: 9_130_000,
    avg_sequence_length: 736.2,
    max_sequence_length: 8192,
    size_bytes: 754_974_720,
    train_split: 0.9,
    validation_split: 0.05,
    test_split: 0.05,
    columns: ["messages"],
    validation_report: { checked_rows: 12_400, valid_rows: 12_400, invalid_rows: 0 },
    tags: ["chat", "demo"],
    token_count_method: "estimated",
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
];

function buildMetrics(steps: number, seed = 11): TrainingMetricPoint[] {
  const rand = seededRandom(seed);
  const points: TrainingMetricPoint[] = [];
  let val: number | null = null;
  for (let step = 10; step <= steps; step += 10) {
    const progress = step / steps;
    const loss = 1.05 + 1.6 * Math.exp(-3.2 * progress) + (rand() - 0.5) * 0.08;
    if (step % 100 === 0) val = loss + 0.06 + Math.max(0, progress - 0.7) * 0.8;
    points.push({
      step,
      epoch: Number((progress * 3).toFixed(3)),
      ts: new Date(Date.now() - (steps - step) * 1000).toISOString(),
      loss: Number(loss.toFixed(4)),
      val_loss: val ? Number(val.toFixed(4)) : null,
      learning_rate: 2e-5 * 0.5 * (1 + Math.cos(Math.PI * progress)),
      grad_norm: Number((0.6 + rand() * 0.5).toFixed(3)),
      tokens_per_sec: Math.round(17_000 + rand() * 2500),
      samples_per_sec: Number((8 + rand() * 2).toFixed(2)),
      gpu_utilization: Number((85 + rand() * 12).toFixed(1)),
      vram_used_mb: Math.round(18_400 + rand() * 1600),
    });
  }
  return points;
}

const TRAINING_JOB: TrainingJobDetail = {
  id: "job_demo_live",
  name: "LoRA r16 · chat mixture (mock)",
  model_id: "mdl_demo_base",
  dataset_id: "ds_demo_sft",
  experiment_id: "exp_demo_1",
  method: "lora",
  status: "running",
  provenance: "simulated",
  backend: "simulated",
  config: DEFAULT_TRAINING_CONFIG as TrainingConfig,
  total_steps: 1400,
  current_step: 612,
  total_epochs: 3,
  current_epoch: 1.31,
  loss: 1.362,
  val_loss: 1.441,
  best_val_loss: 1.428,
  learning_rate: 1.42e-5,
  grad_norm: 0.78,
  tokens_processed: 40_108_032,
  tokens_per_sec: 17_940,
  samples_per_sec: 8.9,
  gpu_utilization: 88.4,
  vram_used_mb: 18_950,
  eta_seconds: 1180,
  started_at: new Date(Date.now() - 48 * 60 * 1000).toISOString(),
  created_at: now(),
  updated_at: now(),
  metrics: buildMetrics(612),
  model_name: "Foxtrot 7B Base",
  dataset_name: "foxtrot-instruct-50k",
  warnings: ["Mock provider active — these metrics are client-side fixtures."],
};

const EXPERIMENTS: Experiment[] = [
  {
    id: "exp_demo_1",
    name: "LoRA r16 · 3 epochs (mock)",
    model_id: "mdl_demo_base",
    dataset_id: "ds_demo_sft",
    job_id: "job_demo_live",
    status: "running",
    provenance: "simulated",
    method: "lora",
    hyperparameters: DEFAULT_TRAINING_CONFIG,
    started_at: new Date(Date.now() - 2880_000).toISOString(),
    final_train_loss: 1.362,
    final_val_loss: 1.441,
    benchmark_summary: {},
    tags: ["demo"],
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
  {
    id: "exp_demo_2",
    name: "QLoRA r64 · 3 epochs (mock)",
    model_id: "mdl_demo_sft3",
    dataset_id: "ds_demo_sft",
    status: "completed",
    provenance: "simulated",
    method: "qlora",
    hyperparameters: { ...DEFAULT_TRAINING_CONFIG, method: "qlora", precision: "int4" },
    started_at: new Date(Date.now() - 86_400_000).toISOString(),
    ended_at: new Date(Date.now() - 79_200_000).toISOString(),
    duration_seconds: 7200,
    final_train_loss: 1.02,
    final_val_loss: 1.31,
    benchmark_summary: { mmlu: 0.583 },
    tags: ["demo"],
    is_demo: true,
    created_at: now(),
    updated_at: now(),
  },
];

const SUITES: BenchmarkSuite[] = [
  {
    key: "mmlu",
    label: "MMLU",
    category: "knowledge",
    description: "Multitask knowledge across academic subjects.",
    metric: "accuracy",
    default_shots: 5,
    available_items: 10,
    data_source: "bundled_sample",
    official: false,
    requires_execution: false,
    notes: "Mock provider — bundled sample, NOT the official MMLU split.",
  },
  {
    key: "gsm8k",
    label: "GSM8K",
    category: "math",
    description: "Grade-school multi-step arithmetic.",
    metric: "exact_match",
    default_shots: 8,
    available_items: 10,
    data_source: "bundled_sample",
    official: false,
    requires_execution: false,
    notes: "Mock provider — bundled sample, NOT the official GSM8K split.",
  },
];

function buildBenchmarkRun(modelId: string, suite: string, score: number): BenchmarkRun {
  return {
    id: id("bm"),
    name: `${suite.toUpperCase()} · mock`,
    suite,
    suite_label: suite.toUpperCase(),
    category: suite === "gsm8k" ? "math" : "knowledge",
    model_id: modelId,
    status: "completed",
    provenance: "simulated",
    config: {
      num_examples: 10,
      few_shot: 5,
      temperature: 0,
      max_tokens: 256,
      seed: 42,
      data_source: "bundled_sample",
      official_split: false,
      demo: true,
    },
    total_items: 10,
    completed_items: 10,
    overall_score: score,
    accuracy: score,
    pass_at_1: null,
    avg_latency_ms: 780,
    avg_tokens_per_sec: 132,
    total_tokens: 1800,
    runtime_seconds: 42,
    category_scores: { knowledge: score, reasoning: score - 0.05 },
    is_demo: true,
    created_at: now(),
  };
}

const BENCHMARK_RUNS: BenchmarkRun[] = [
  buildBenchmarkRun("mdl_demo_base", "mmlu", 0.548),
  buildBenchmarkRun("mdl_demo_sft3", "mmlu", 0.583),
  buildBenchmarkRun("mdl_demo_sft3", "gsm8k", 0.492),
  buildBenchmarkRun("mdl_demo_small", "mmlu", 0.402),
];

const CHECKPOINTS: Checkpoint[] = [250, 500, 750, 1000].map((step, index) => ({
  id: `ckpt_demo_${step}`,
  model_id: "mdl_demo_base",
  job_id: "job_demo_live",
  experiment_id: "exp_demo_1",
  step,
  epoch: Number(((step / 1400) * 3).toFixed(2)),
  path: `/data/checkpoints/job_demo_live/step-${step}`,
  size_bytes: 70_464_307,
  train_loss: Number((1.9 - index * 0.18).toFixed(3)),
  val_loss: Number((1.98 - index * 0.14).toFixed(3)),
  benchmark_score: 52 + index * 2,
  is_best: index === 3,
  provenance: "simulated",
  is_demo: true,
  created_at: now(),
}));

const CONVERSATIONS: ConversationDetail[] = [
  {
    id: "conv_demo",
    title: "LoRA rank sizing (mock)",
    model_id: "mdl_demo_sft3",
    system_prompt: "You are Foxtrot, a concise ML engineering assistant.",
    params: { temperature: 0.7, top_p: 0.9, max_tokens: 512 },
    pinned: false,
    is_demo: true,
    created_at: now(),
    updated_at: now(),
    messages: [
      {
        id: "msg_1",
        conversation_id: "conv_demo",
        role: "user",
        content: "How should I pick the LoRA rank?",
        provenance: "simulated",
        prompt_tokens: 12,
        completion_tokens: 0,
        total_tokens: 12,
        created_at: now(),
      },
      {
        id: "msg_2",
        conversation_id: "conv_demo",
        role: "assistant",
        content:
          "[mock output — simulated, no weights loaded]\n\nStart at rank 16 with alpha 32 and scale with dataset size.",
        model_id: "mdl_demo_sft3",
        provenance: "simulated",
        prompt_tokens: 12,
        completion_tokens: 34,
        total_tokens: 46,
        tokens_per_sec: 139.4,
        latency_ms: 412,
        time_to_first_token_ms: 131,
        finish_reason: "stop",
        created_at: now(),
      },
    ],
  },
];

function hardwareSnapshot(): HardwareSnapshot {
  const rand = seededRandom(Date.now() % 100000);
  const util = 60 + rand() * 30;
  return {
    ts: now(),
    provenance: "simulated",
    compute_mode: "cuda",
    gpu_available: true,
    gpus: [
      {
        index: 0,
        name: "NVIDIA GeForce RTX 4090 (mock)",
        utilization: Number(util.toFixed(1)),
        memory_used_mb: Math.round(13_000 + rand() * 6000),
        memory_total_mb: 24_564,
        temperature_c: Number((52 + util * 0.3).toFixed(1)),
        power_draw_w: Number((180 + util * 2).toFixed(1)),
        power_limit_w: 450,
        fan_speed_pct: Number((40 + util * 0.4).toFixed(1)),
        clock_mhz: 2400,
        driver_version: "mock",
        processes: 1,
      },
    ],
    cpu_percent: Number((20 + rand() * 30).toFixed(1)),
    cpu_cores: 32,
    cpu_model: "AMD Ryzen 9 7950X (mock)",
    ram_used_gb: Number((18 + rand() * 8).toFixed(1)),
    ram_total_gb: 64,
    swap_used_gb: 1.2,
    swap_total_gb: 16,
    disks: [{ mount: "/", used_gb: 412, total_gb: 1024, percent: 40.2 }],
    platform: "Mock provider",
    note: "Mock hardware telemetry — no physical device is being read.",
  };
}

const LOGS: LogEntry[] = [
  { id: 1, ts: now(), level: "info", source: "system", message: "Mock provider active — no backend connected", context: {} },
  { id: 2, ts: now(), level: "warn", source: "hardware", message: "Telemetry is simulated", context: {} },
  { id: 3, ts: now(), level: "info", source: "training", message: "[TRAIN] Step 612/1400 loss: 1.362 lr: 1.42e-5", context: {}, job_id: "job_demo_live" },
];

const notSupported = (feature: string) => async () => {
  throw new Error(`${feature} needs the backend — switch NEXT_PUBLIC_DATA_PROVIDER to "http".`);
};

export const mockProvider: DataProvider = {
  kind: "mock",

  system: {
    info: async () => ({
      app: "Foxtrot",
      version: "0.1.0",
      demo_mode: true,
      gpu_available: true,
      compute_mode: "cuda",
      inference_engine: "mock",
      hardware_provider: "mock",
      capabilities: [
        { name: "gpu", available: true, detail: "Mock GPU" },
        { name: "training.torch", available: false, detail: "Mock provider — no training stack" },
        { name: "inference.demo", available: true, detail: "Client-side fixtures" },
      ],
      server_time: now(),
    }),
    health: async () => ({ status: "ok", demo_mode: true, mock: true }),
    settings: async () => ({ demo_mode: true, provider: "mock" }),
    seedDemoData: async () => ({ ok: true, message: "Mock data is always present" }),
    clearDemoData: async () => ({ ok: true, message: "Mock data cannot be cleared" }),
  },

  models: {
    list: async (params) => page(MODELS, params),
    get: async (modelId) => {
      const model = MODELS.find((m) => m.id === modelId) ?? MODELS[0];
      return { ...model, checkpoints: CHECKPOINTS.filter((c) => c.model_id === model.id) };
    },
    import: notSupported("Importing models"),
    update: async (modelId) => ({
      ...(MODELS.find((m) => m.id === modelId) ?? MODELS[0]),
      checkpoints: [],
    }),
    remove: async () => ({ ok: true, message: "Mock models cannot be deleted" }),
    clone: notSupported("Cloning models"),
    load: async (modelId) => ({
      ...(MODELS.find((m) => m.id === modelId) ?? MODELS[0]),
      status: "loaded",
      checkpoints: [],
    }),
    unload: async (modelId) => ({
      ...(MODELS.find((m) => m.id === modelId) ?? MODELS[0]),
      status: "stopped",
      checkpoints: [],
    }),
    export: notSupported("Exporting models"),
    checkpoints: async (modelId) => CHECKPOINTS.filter((c) => c.model_id === modelId),
  },

  projects: {
    list: async (params) =>
      page(params?.include_demo === false ? PROJECTS.filter((p) => !p.is_demo) : PROJECTS, params),
    get: async (projectId) =>
      projectDetail(PROJECTS.find((p) => p.id === projectId) ?? PROJECTS[0]),
    create: async (payload) => {
      const project: Project = {
        id: id("prj"),
        name: payload.name,
        description: payload.description ?? null,
        base_model_id: payload.base_model_id ?? null,
        default_dataset_id: payload.default_dataset_id ?? null,
        tags: payload.tags ?? [],
        settings: payload.settings ?? {},
        is_demo: false,
        created_at: now(),
        updated_at: now(),
      };
      PROJECTS.unshift(project);
      return projectDetail(project);
    },
    update: async (projectId, payload) => {
      const project = PROJECTS.find((p) => p.id === projectId) ?? PROJECTS[0];
      Object.assign(project, payload, { updated_at: now() });
      return projectDetail(project);
    },
    remove: async (projectId) => {
      const index = PROJECTS.findIndex((p) => p.id === projectId);
      if (index >= 0) PROJECTS.splice(index, 1);
      return { ok: true, message: "Deleted", id: projectId };
    },
  },

  datasets: {
    list: async (params) => page(DATASETS, params),
    get: async (datasetId): Promise<DatasetDetail> => ({
      ...(DATASETS.find((d) => d.id === datasetId) ?? DATASETS[0]),
      preview: [
        { instruction: "Explain LoRA", input: "", output: "Low-rank adapters freeze base weights." },
        { instruction: "Summarise QLoRA", input: "", output: "4-bit base weights plus LoRA adapters." },
      ],
    }),
    templates: async () => [
      {
        key: "instruction",
        label: "Instruction tuning",
        description: "Alpaca-style supervised pairs.",
        schema_example: { instruction: "", input: "", output: "" },
        required_fields: ["instruction", "output"],
      },
      {
        key: "chat",
        label: "Chat",
        description: "Multi-turn conversations.",
        schema_example: {
          messages: [
            { role: "system", content: "" },
            { role: "user", content: "" },
            { role: "assistant", content: "" },
          ],
        },
        required_fields: ["messages"],
      },
      {
        key: "plain_text",
        label: "Plain text",
        description: "Raw documents for continued pretraining.",
        schema_example: { text: "" },
        required_fields: ["text"],
      },
    ],
    import: notSupported("Importing datasets"),
    upload: notSupported("Uploading datasets"),
    preview: async (datasetId, offset = 0, limit = 25) => ({
      dataset_id: datasetId,
      columns: ["instruction", "input", "output"],
      rows: Array.from({ length: Math.min(limit, 10) }, (_, index) => ({
        instruction: `Mock instruction ${offset + index + 1}`,
        input: "",
        output: `Mock response ${offset + index + 1}`,
      })),
      total_rows: 50_000,
      offset,
      limit,
    }),
    validate: async () => ({
      template: "instruction",
      checked_rows: 5000,
      valid_rows: 4998,
      invalid_rows: 2,
      issues: [{ row: 214, field: "output", severity: "error", message: "`output` is empty" }],
      truncated: true,
    }),
    remove: async () => ({ ok: true }),
  },

  training: {
    listJobs: async (params) => page([TRAINING_JOB as TrainingJob], params),
    activeJobs: async () => [TRAINING_JOB as TrainingJob],
    getJob: async () => TRAINING_JOB,
    createJob: notSupported("Creating training jobs"),
    metrics: async () => TRAINING_JOB.metrics,
    logs: async () => ({ entries: LOGS.filter((l) => l.job_id) }),
    validateConfig: async (config) => ({
      ok: true,
      config,
      errors: [],
      warnings: ["Mock provider — configuration is not executed."],
      estimated_steps: 1400,
      estimated_tokens: 55_260_000,
      effective_batch_size: config.batch_size * config.gradient_accumulation_steps,
    }),
    parseRawConfig: async () => ({
      ok: true,
      config: DEFAULT_TRAINING_CONFIG as TrainingConfig,
      errors: [],
      warnings: ["Mock provider — configuration is not executed."],
      effective_batch_size: 32,
    }),
    start: async () => TRAINING_JOB as TrainingJob,
    pause: async () => ({ ...TRAINING_JOB, status: "paused" }) as TrainingJob,
    resume: async () => ({ ...TRAINING_JOB, status: "running" }) as TrainingJob,
    stop: async () => ({ ...TRAINING_JOB, status: "stopped" }) as TrainingJob,
    checkpoint: async () => ({ ok: true, message: "Mock checkpoint requested" }),
    remove: async () => ({ ok: true }),
  },

  chat: {
    conversations: async (params) => page(CONVERSATIONS as Conversation[], params),
    conversation: async (conversationId) =>
      CONVERSATIONS.find((c) => c.id === conversationId) ?? CONVERSATIONS[0],
    createConversation: async (payload) => {
      const conversation: ConversationDetail = {
        id: id("conv"),
        title: payload.title ?? "New conversation",
        model_id: payload.model_id ?? MODELS[0].id,
        system_prompt: payload.system_prompt ?? "",
        params: payload.params ?? {},
        pinned: false,
        is_demo: true,
        created_at: now(),
        updated_at: now(),
        messages: [],
      };
      CONVERSATIONS.unshift(conversation);
      return conversation;
    },
    updateConversation: async (conversationId, payload) => {
      const conversation = CONVERSATIONS.find((c) => c.id === conversationId) ?? CONVERSATIONS[0];
      Object.assign(conversation, payload);
      return conversation;
    },
    deleteConversation: async (conversationId) => {
      const index = CONVERSATIONS.findIndex((c) => c.id === conversationId);
      if (index >= 0) CONVERSATIONS.splice(index, 1);
      return { ok: true };
    },
    editMessage: async (messageId, content) => {
      for (const conversation of CONVERSATIONS) {
        const message = conversation.messages.find((m) => m.id === messageId);
        if (message) {
          message.content = content;
          return message;
        }
      }
      return CONVERSATIONS[0].messages[0] as Message;
    },
    deleteMessage: async (messageId) => {
      for (const conversation of CONVERSATIONS) {
        const index = conversation.messages.findIndex((m) => m.id === messageId);
        if (index >= 0) conversation.messages.splice(index, 1);
      }
      return { ok: true };
    },
    tokenize: async (text) => ({ tokens: Math.max(1, Math.round(text.length / 4)), method: "estimated" }),

    async stream(payload, callbacks: StreamCallbacks) {
      const prompt = payload.messages.at(-1)?.content ?? "";
      const words = [
        "[mock",
        "output",
        "—",
        "simulated,",
        "no",
        "weights",
        "loaded]\n\n",
        `You`,
        `asked:`,
        `“${prompt.slice(0, 60)}”.`,
        "\n\nSwitch",
        "NEXT_PUBLIC_DATA_PROVIDER",
        "to",
        "http",
        "to",
        "generate",
        "through",
        "the",
        "Foxtrot",
        "backend.",
      ];
      const started = performance.now();
      let first = 0;
      for (const [index, word] of words.entries()) {
        if (callbacks.signal?.aborted) break;
        await new Promise((resolve) => setTimeout(resolve, 45));
        if (index === 0) first = performance.now() - started;
        callbacks.onToken(`${word} `);
      }
      const elapsed = performance.now() - started;
      callbacks.onUsage?.(
        {
          prompt_tokens: Math.round(prompt.length / 4),
          completion_tokens: words.length,
          total_tokens: Math.round(prompt.length / 4) + words.length,
          tokens_per_sec: Number(((words.length / elapsed) * 1000).toFixed(2)),
          latency_ms: Number(elapsed.toFixed(1)),
          time_to_first_token_ms: Number(first.toFixed(1)),
          finish_reason: "stop",
          provenance: "simulated",
          engine: "mock",
        },
        null,
      );
    },
  },

  playground: {
    run: async (payload) => ({
      id: id("pg"),
      prompt: payload.prompt,
      created_at: now(),
      entries: payload.model_ids.map((modelId, index) => {
        const model = MODELS.find((m) => m.id === modelId);
        return {
          model_id: modelId,
          model_name: model?.display_name ?? modelId,
          content: `[mock output — simulated] Response ${index + 1} to “${payload.prompt.slice(0, 48)}”.`,
          latency_ms: 320 + index * 140,
          time_to_first_token_ms: 48 + index * 20,
          tokens_per_sec: 310 - index * 60,
          completion_tokens: 64,
          prompt_tokens: Math.round(payload.prompt.length / 4),
          total_tokens: 64 + Math.round(payload.prompt.length / 4),
          memory_mb: model?.vram_estimate_mb ?? 2048,
          finish_reason: "stop",
          provenance: "simulated" as const,
          engine: "mock",
        };
      }),
    }),
    history: async (params) => page([] as PlaygroundComparison[], params),
    vote: notSupported("Voting"),
  },

  benchmarks: {
    suites: async () => SUITES,
    run: notSupported("Running benchmarks"),
    results: async (params) => page(BENCHMARK_RUNS, params),
    result: async (runId): Promise<BenchmarkRunDetail> => {
      const run = BENCHMARK_RUNS.find((r) => r.id === runId) ?? BENCHMARK_RUNS[0];
      return {
        ...run,
        model_name: MODELS.find((m) => m.id === run.model_id)?.display_name ?? null,
        items: Array.from({ length: 10 }, (_, index) => ({
          index,
          category: run.category,
          question: `Mock question ${index + 1}`,
          prompt: `Mock prompt ${index + 1}`,
          expected: "42",
          response: index % 3 === 0 ? "41" : "42",
          raw_output: index % 3 === 0 ? "41" : "42",
          correct: index % 3 !== 0,
          score: index % 3 !== 0 ? 1 : 0,
          latency_ms: 620 + index * 30,
          tokens_used: 140 + index * 4,
        })),
      };
    },
    cancel: async () => ({ ok: true }),
    remove: async () => ({ ok: true }),
  },

  evaluations: {
    leaderboard: async () => ({
      suites: SUITES,
      rows: MODELS.map((model) => ({
        model_id: model.id,
        model_name: model.display_name ?? model.name,
        parameters: model.parameters,
        provenance: "simulated" as const,
        runs: 1,
        scores: {
          mmlu: {
            score: model.id === "mdl_demo_sft3" ? 58.3 : 54.8,
            accuracy: model.id === "mdl_demo_sft3" ? 58.3 : 54.8,
            run_id: BENCHMARK_RUNS[0].id,
            suite_label: "MMLU",
            official_split: false,
            data_source: "bundled_sample",
            provenance: "simulated" as const,
          },
        },
      })),
      note: "Mock provider — all scores are client-side fixtures, not measured runs.",
    }),
    compare: async (modelIds, name) => ({
      name: name ?? "Model comparison",
      categories: ["MMLU", "GSM8K"],
      created_at: now(),
      rows: modelIds.map((modelId) => {
        const model = MODELS.find((m) => m.id === modelId) ?? MODELS[0];
        return {
          model_id: model.id,
          model_name: model.display_name ?? model.name,
          parameters: model.parameters,
          size_bytes: model.size_bytes,
          context_length: model.context_length,
          precision: model.precision,
          benchmark_score: model.id === "mdl_demo_sft3" ? 58.3 : 54.8,
          validation_loss: model.id === "mdl_demo_sft3" ? 1.31 : 1.48,
          inference_tokens_per_sec: model.metrics.inference_tokens_per_sec ?? 120,
          vram_usage_mb: model.vram_estimate_mb,
          benchmark_breakdown: { MMLU: model.id === "mdl_demo_sft3" ? 58.3 : 54.8, GSM8K: 49.2 },
          measured_metrics: [],
          provenance: "simulated" as const,
        };
      }),
    }),
  },

  experiments: {
    list: async (params) => page(EXPERIMENTS, params),
    get: async (experimentId) => {
      const experiment = EXPERIMENTS.find((e) => e.id === experimentId) ?? EXPERIMENTS[0];
      return {
        ...experiment,
        model_name: "Foxtrot 7B Base",
        dataset_name: "foxtrot-instruct-50k",
        checkpoint_count: CHECKPOINTS.length,
        benchmark_runs: [],
      };
    },
    update: async (experimentId) => EXPERIMENTS.find((e) => e.id === experimentId) ?? EXPERIMENTS[0],
    duplicate: notSupported("Duplicating experiments"),
    compare: async (ids) => ({
      rows: ids.map((experimentId) => {
        const experiment = EXPERIMENTS.find((e) => e.id === experimentId) ?? EXPERIMENTS[0];
        return {
          experiment_id: experiment.id,
          name: experiment.name,
          model_name: "Foxtrot 7B Base",
          dataset_name: "foxtrot-instruct-50k",
          method: experiment.method,
          hyperparameters: experiment.hyperparameters as Record<string, unknown>,
          final_train_loss: experiment.final_train_loss,
          final_val_loss: experiment.final_val_loss,
          duration_seconds: experiment.duration_seconds,
          benchmark_summary: experiment.benchmark_summary,
          provenance: "simulated" as const,
        };
      }),
      differing_hyperparameters: ["method", "precision", "lora.rank"],
    }),
    remove: async () => ({ ok: true }),
  },

  checkpoints: {
    list: async (params) => page(CHECKPOINTS, params),
    load: notSupported("Loading checkpoints"),
    evaluate: notSupported("Evaluating checkpoints"),
    export: async () => ({ ok: true, message: "Mock export" }),
    remove: async () => ({ ok: true }),
  },

  hardware: {
    snapshot: async () => hardwareSnapshot(),
    history: async (limit = 120) => ({
      provenance: "simulated",
      interval_seconds: 2,
      points: Array.from({ length: limit }, (_, index) => {
        const rand = seededRandom(index + 1);
        return {
          ts: new Date(Date.now() - (limit - index) * 2000).toISOString(),
          gpu_utilization: [Number((60 + rand() * 30).toFixed(1))],
          gpu_memory_used_mb: [Math.round(13_000 + rand() * 6000)],
          gpu_temperature_c: [Number((58 + rand() * 12).toFixed(1))],
          gpu_power_w: [Number((220 + rand() * 120).toFixed(1))],
          cpu_percent: Number((20 + rand() * 30).toFixed(1)),
          ram_used_gb: Number((18 + rand() * 6).toFixed(1)),
        };
      }),
    }),
    devices: async () => ({ provider: "mock", provenance: "simulated" }),
  },

  logs: {
    list: async (params) => page(LOGS, params),
    sources: async () => ["system", "training", "hardware", "benchmark"],
    clear: async () => ({ ok: true }),
  },
};
