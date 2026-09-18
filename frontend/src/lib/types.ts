/**
 * Shared API types — mirrors the Pydantic schemas in `backend/app/schemas`.
 * Every provider (HTTP or mock) is typed against these, so swapping the data
 * source never touches a component.
 */

export type Provenance = "measured" | "simulated" | "imported";

export type ModelStatus =
  | "ready"
  | "loaded"
  | "training"
  | "stopped"
  | "importing"
  | "error";

export type ModelFormat = "pytorch" | "safetensors" | "gguf" | "hf_repo";
export type ModelSource =
  | "huggingface"
  | "local"
  | "custom_path"
  | "checkpoint"
  | "demo";
export type Precision = "fp32" | "fp16" | "bf16" | "int8" | "int4";

export type DatasetFormat = "json" | "jsonl" | "csv" | "txt" | "parquet" | "hf_dataset";
export type DatasetTemplate = "instruction" | "chat" | "plain_text" | "raw";
export type DatasetStatus = "ready" | "importing" | "invalid" | "error";

export type TrainingMethod =
  | "full_finetune"
  | "lora"
  | "qlora"
  | "continued_pretraining"
  | "instruction_tuning";

export type OptimizerName = "adamw" | "adamw_8bit" | "adafactor" | "sgd" | "lion";
export type SchedulerName =
  | "linear"
  | "cosine"
  | "cosine_with_restarts"
  | "constant"
  | "constant_with_warmup"
  | "polynomial";

export type JobStatus =
  | "queued"
  | "running"
  | "paused"
  | "stopping"
  | "completed"
  | "stopped"
  | "failed";

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Ack {
  ok: boolean;
  message?: string | null;
  id?: string | null;
}

export interface Capability {
  name: string;
  available: boolean;
  detail?: string | null;
}

export interface SystemInfo {
  app: string;
  version: string;
  demo_mode: boolean;
  gpu_available: boolean;
  compute_mode: "cuda" | "rocm" | "mps" | "cpu";
  inference_engine: string;
  hardware_provider: string;
  capabilities: Capability[];
  server_time: string;
}

/* ------------------------------------------------------------------ models */

export interface TokenizerInfo {
  type?: string | null;
  vocab_size?: number | null;
  bos_token?: string | null;
  eos_token?: string | null;
  pad_token?: string | null;
  unk_token?: string | null;
  chat_template?: boolean;
  special_tokens?: string[];
}

export interface Checkpoint {
  id: string;
  model_id: string;
  job_id?: string | null;
  experiment_id?: string | null;
  step: number;
  epoch: number;
  path?: string | null;
  size_bytes: number;
  train_loss?: number | null;
  val_loss?: number | null;
  benchmark_score?: number | null;
  is_best: boolean;
  provenance: Provenance;
  notes?: string | null;
  is_demo: boolean;
  created_at: string;
}

export interface Model {
  id: string;
  name: string;
  display_name?: string | null;
  description?: string | null;
  source: ModelSource;
  format: ModelFormat;
  status: ModelStatus;
  repo_id?: string | null;
  revision?: string | null;
  local_path?: string | null;
  architecture?: string | null;
  /** Parameter count in millions. */
  parameters?: number | null;
  context_length?: number | null;
  precision: Precision;
  quantization?: string | null;
  size_bytes?: number | null;
  vram_estimate_mb?: number | null;
  tokenizer: TokenizerInfo & Record<string, unknown>;
  config: Record<string, unknown>;
  metrics: Record<string, number | undefined>;
  tags: string[];
  parent_model_id?: string | null;
  license?: string | null;
  is_demo: boolean;
  error?: string | null;
  loaded_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ModelDetail extends Model {
  checkpoints: Checkpoint[];
}

export interface ModelImportRequest {
  source: "huggingface" | "local" | "custom_path";
  name?: string;
  repo_id?: string;
  revision?: string;
  path?: string;
  format?: ModelFormat;
  precision?: Precision;
  context_length?: number | null;
  description?: string;
  tags?: string[];
}

/* ---------------------------------------------------------------- datasets */

export interface DatasetIssue {
  row: number;
  field?: string | null;
  severity: "error" | "warning";
  message: string;
}

export interface DatasetValidationReport {
  template: DatasetTemplate;
  checked_rows: number;
  valid_rows: number;
  invalid_rows: number;
  issues: DatasetIssue[];
  truncated: boolean;
}

export interface Project {
  id: string;
  name: string;
  description?: string | null;
  base_model_id?: string | null;
  default_dataset_id?: string | null;
  tags: string[];
  settings: Record<string, unknown>;
  is_demo: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProjectDetail extends Project {
  base_model_name?: string | null;
  default_dataset_name?: string | null;
  experiment_count: number;
  training_job_count: number;
  running_job_count: number;
  checkpoint_count: number;
  last_activity_at?: string | null;
}

export interface ProjectCreateRequest {
  name: string;
  description?: string | null;
  base_model_id?: string | null;
  default_dataset_id?: string | null;
  tags?: string[];
  settings?: Record<string, unknown>;
}

export interface ProjectUpdateRequest {
  name?: string;
  description?: string | null;
  base_model_id?: string | null;
  default_dataset_id?: string | null;
  tags?: string[];
  settings?: Record<string, unknown>;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string | null;
  format: DatasetFormat;
  template: DatasetTemplate;
  status: DatasetStatus;
  source: string;
  repo_id?: string | null;
  local_path?: string | null;
  rows: number;
  tokens: number;
  avg_sequence_length: number;
  max_sequence_length: number;
  size_bytes: number;
  train_split: number;
  validation_split: number;
  test_split: number;
  columns: string[];
  validation_report: Partial<DatasetValidationReport>;
  tags: string[];
  token_count_method: string;
  is_demo: boolean;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetDetail extends Dataset {
  preview: Record<string, unknown>[];
}

export interface DatasetPreview {
  dataset_id: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  offset: number;
  limit: number;
}

export interface DatasetTemplateInfo {
  key: DatasetTemplate;
  label: string;
  description: string;
  schema_example: Record<string, unknown>;
  required_fields: string[];
}

/* ---------------------------------------------------------------- training */

export interface LoRAConfig {
  rank: number;
  alpha: number;
  dropout: number;
  target_modules: string[];
  bias: "none" | "all" | "lora_only";
}

export interface CheckpointConfig {
  save_every_steps: number;
  keep_last: number;
  save_best: boolean;
  save_optimizer_state: boolean;
}

export interface TrainingConfig {
  method: TrainingMethod;
  epochs: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  warmup_steps: number;
  weight_decay: number;
  max_sequence_length: number;
  gradient_clipping: number;
  optimizer: OptimizerName;
  lr_scheduler: SchedulerName;
  seed: number;
  precision: Precision;
  gradient_checkpointing: boolean;
  flash_attention: boolean;
  lora: LoRAConfig;
  checkpointing: CheckpointConfig;
  eval_every_steps: number;
  log_every_steps: number;
  packing: boolean;
  shuffle: boolean;
  num_workers: number;
}

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  ts: string;
  loss?: number | null;
  val_loss?: number | null;
  learning_rate?: number | null;
  grad_norm?: number | null;
  tokens_per_sec?: number | null;
  samples_per_sec?: number | null;
  gpu_utilization?: number | null;
  vram_used_mb?: number | null;
}

export interface TrainingJob {
  id: string;
  name: string;
  model_id: string;
  dataset_id: string;
  experiment_id?: string | null;
  project_id?: string | null;
  method: TrainingMethod;
  status: JobStatus;
  provenance: Provenance;
  backend: string;
  config: TrainingConfig;
  total_steps: number;
  current_step: number;
  total_epochs: number;
  current_epoch: number;
  loss?: number | null;
  val_loss?: number | null;
  best_val_loss?: number | null;
  learning_rate?: number | null;
  grad_norm?: number | null;
  tokens_processed: number;
  tokens_per_sec?: number | null;
  samples_per_sec?: number | null;
  gpu_utilization?: number | null;
  vram_used_mb?: number | null;
  eta_seconds?: number | null;
  started_at?: string | null;
  ended_at?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TrainingJobDetail extends TrainingJob {
  metrics: TrainingMetricPoint[];
  model_name?: string | null;
  dataset_name?: string | null;
  warnings: string[];
}

export interface TrainingConfigValidation {
  ok: boolean;
  config?: TrainingConfig | null;
  errors: string[];
  warnings: string[];
  estimated_steps?: number | null;
  estimated_tokens?: number | null;
  effective_batch_size?: number | null;
}

export interface TrainingJobCreate {
  name?: string;
  model_id: string;
  dataset_id: string;
  config: TrainingConfig;
  notes?: string;
  start_immediately?: boolean;
}

/* -------------------------------------------------------------------- chat */

export interface SamplingParams {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  repetition_penalty: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  seed?: number | null;
  stop?: string[];
}

export interface GenerationUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tokens_per_sec: number;
  latency_ms: number;
  time_to_first_token_ms: number;
  finish_reason: string;
  provenance: Provenance;
  engine: string;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Message extends ChatMessage {
  id: string;
  conversation_id: string;
  model_id?: string | null;
  provenance: Provenance;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tokens_per_sec?: number | null;
  latency_ms?: number | null;
  time_to_first_token_ms?: number | null;
  finish_reason?: string | null;
  error?: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  model_id?: string | null;
  system_prompt: string;
  params: Partial<SamplingParams>;
  pinned: boolean;
  is_demo: boolean;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

/* -------------------------------------------------------------- playground */

export interface PlaygroundEntry {
  model_id: string;
  model_name: string;
  content: string;
  latency_ms: number;
  time_to_first_token_ms: number;
  tokens_per_sec: number;
  completion_tokens: number;
  prompt_tokens: number;
  total_tokens: number;
  memory_mb?: number | null;
  finish_reason: string;
  provenance: Provenance;
  engine: string;
  error?: string | null;
}

export interface PlaygroundRun {
  id?: string | null;
  prompt: string;
  entries: PlaygroundEntry[];
  created_at: string;
}

export interface PlaygroundComparison {
  id: string;
  prompt: string;
  system_prompt: string;
  params: Record<string, unknown>;
  entries: PlaygroundEntry[];
  winner_model_id?: string | null;
  votes: Record<string, number>;
  provenance: Provenance;
  is_demo: boolean;
  created_at: string;
}

/* -------------------------------------------------------------- benchmarks */

export interface BenchmarkSuite {
  key: string;
  label: string;
  category: string;
  description: string;
  metric: string;
  default_shots: number;
  available_items: number;
  data_source: string;
  official: boolean;
  requires_execution: boolean;
  notes?: string | null;
}

export interface BenchmarkRunConfig {
  num_examples: number;
  few_shot: number;
  temperature: number;
  max_tokens: number;
  seed: number;
  batch_size?: number;
  stop?: string[];
}

export interface BenchmarkItem {
  index: number;
  category: string;
  question: string;
  prompt: string;
  expected: string;
  response: string;
  raw_output: string;
  correct: boolean;
  score: number;
  latency_ms: number;
  tokens_used: number;
}

export interface BenchmarkRun {
  id: string;
  name: string;
  suite: string;
  suite_label: string;
  category: string;
  model_id: string;
  checkpoint_id?: string | null;
  dataset_id?: string | null;
  status: JobStatus;
  provenance: Provenance;
  config: BenchmarkRunConfig & {
    data_source?: string;
    official_split?: boolean;
    suite_notes?: string | null;
    demo?: boolean;
  };
  total_items: number;
  completed_items: number;
  overall_score?: number | null;
  accuracy?: number | null;
  pass_at_1?: number | null;
  avg_latency_ms?: number | null;
  avg_tokens_per_sec?: number | null;
  total_tokens: number;
  runtime_seconds?: number | null;
  category_scores: Record<string, number>;
  started_at?: string | null;
  ended_at?: string | null;
  error?: string | null;
  is_demo: boolean;
  created_at: string;
}

export interface BenchmarkRunDetail extends BenchmarkRun {
  items: BenchmarkItem[];
  model_name?: string | null;
}

export interface ComparisonRow {
  model_id: string;
  model_name: string;
  label?: string | null;
  parameters?: number | null;
  size_bytes?: number | null;
  context_length?: number | null;
  precision?: string | null;
  benchmark_score?: number | null;
  validation_loss?: number | null;
  inference_tokens_per_sec?: number | null;
  vram_usage_mb?: number | null;
  benchmark_breakdown: Record<string, number>;
  measured_metrics: string[];
  provenance: Provenance;
}

export interface ModelComparison {
  id?: string | null;
  name: string;
  rows: ComparisonRow[];
  categories: string[];
  created_at: string;
}

export interface LeaderboardEntry {
  model_id: string;
  model_name: string;
  parameters?: number | null;
  provenance: Provenance;
  runs: number;
  scores: Record<
    string,
    {
      score: number;
      accuracy?: number | null;
      run_id: string;
      suite_label: string;
      official_split: boolean;
      data_source: string;
      provenance: Provenance;
    }
  >;
}

export interface Leaderboard {
  suites: BenchmarkSuite[];
  rows: LeaderboardEntry[];
  note: string;
}

/* ------------------------------------------------------------- experiments */

export interface Experiment {
  id: string;
  name: string;
  model_id?: string | null;
  dataset_id?: string | null;
  job_id?: string | null;
  project_id?: string | null;
  status: JobStatus;
  provenance: Provenance;
  method: TrainingMethod;
  hyperparameters: Partial<TrainingConfig>;
  started_at?: string | null;
  ended_at?: string | null;
  duration_seconds?: number | null;
  final_train_loss?: number | null;
  final_val_loss?: number | null;
  best_checkpoint_id?: string | null;
  benchmark_summary: Record<string, unknown>;
  notes?: string | null;
  tags: string[];
  is_demo: boolean;
  created_at: string;
  updated_at: string;
}

export interface ExperimentDetail extends Experiment {
  model_name?: string | null;
  dataset_name?: string | null;
  checkpoint_count: number;
  benchmark_runs: {
    id: string;
    suite: string;
    suite_label: string;
    overall_score?: number | null;
    accuracy?: number | null;
    provenance: Provenance;
    official_split: boolean;
  }[];
}

export interface ExperimentCompareRow {
  experiment_id: string;
  name: string;
  model_name?: string | null;
  dataset_name?: string | null;
  method: string;
  hyperparameters: Record<string, unknown>;
  final_train_loss?: number | null;
  final_val_loss?: number | null;
  duration_seconds?: number | null;
  benchmark_summary: Record<string, unknown>;
  provenance: Provenance;
}

export interface ExperimentComparison {
  rows: ExperimentCompareRow[];
  differing_hyperparameters: string[];
}

/* ---------------------------------------------------------------- hardware */

export interface GPUInfo {
  index: number;
  name: string;
  utilization: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature_c?: number | null;
  power_draw_w?: number | null;
  power_limit_w?: number | null;
  fan_speed_pct?: number | null;
  clock_mhz?: number | null;
  compute_capability?: string | null;
  driver_version?: string | null;
  processes: number;
}

export interface DiskInfo {
  mount: string;
  used_gb: number;
  total_gb: number;
  percent: number;
}

export interface HardwareSnapshot {
  ts: string;
  provenance: "measured" | "simulated";
  compute_mode: "cuda" | "rocm" | "mps" | "cpu";
  gpu_available: boolean;
  gpus: GPUInfo[];
  cpu_percent: number;
  cpu_cores: number;
  cpu_model?: string | null;
  cpu_temperature_c?: number | null;
  ram_used_gb: number;
  ram_total_gb: number;
  swap_used_gb: number;
  swap_total_gb: number;
  disks: DiskInfo[];
  platform: string;
  driver_version?: string | null;
  cuda_version?: string | null;
  note?: string | null;
}

export interface HardwareHistoryPoint {
  ts: string;
  gpu_utilization: number[];
  gpu_memory_used_mb: number[];
  gpu_temperature_c: number[];
  gpu_power_w: number[];
  cpu_percent: number;
  ram_used_gb: number;
}

export interface HardwareHistory {
  points: HardwareHistoryPoint[];
  provenance: "measured" | "simulated";
  interval_seconds: number;
}

/* -------------------------------------------------------------------- logs */

export interface LogEntry {
  id: number;
  ts: string;
  level: LogLevel;
  source: string;
  message: string;
  job_id?: string | null;
  context: Record<string, unknown>;
}

/* ------------------------------------------------------------------ events */

export interface RealtimeEvent<T = Record<string, unknown>> {
  topic: string;
  ts?: number;
  seq?: number;
  data: T;
}

export interface TrainingMetricEvent extends TrainingMetricPoint {
  job_id: string;
  tokens_processed: number;
  eta_seconds?: number | null;
}

export interface TrainingStatusEvent {
  job_id: string;
  status: JobStatus;
  error?: string | null;
}

export interface BenchmarkProgressEvent {
  run_id: string;
  phase: string;
  completed?: number;
  total?: number;
  correct?: number;
  last_score?: number;
  latency_ms?: number;
  error?: string;
}
