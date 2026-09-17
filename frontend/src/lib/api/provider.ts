/**
 * The data-provider contract.
 *
 * Every screen talks to this interface, never to `fetch` directly. The HTTP
 * provider (FastAPI backend) and the mock provider (client-side demo fixtures)
 * both implement it, so swapping data sources is a one-line change in
 * `src/lib/api/index.ts` and no component is rewritten.
 */

import type {
  Ack,
  BenchmarkRun,
  BenchmarkRunConfig,
  BenchmarkRunDetail,
  BenchmarkSuite,
  Conversation,
  ConversationDetail,
  Dataset,
  DatasetDetail,
  DatasetPreview,
  DatasetTemplate,
  DatasetTemplateInfo,
  DatasetValidationReport,
  Experiment,
  ExperimentComparison,
  ExperimentDetail,
  HardwareHistory,
  HardwareSnapshot,
  Leaderboard,
  LogEntry,
  Message,
  Model,
  ModelComparison,
  ModelDetail,
  ModelImportRequest,
  Page,
  PlaygroundRun,
  PlaygroundComparison,
  SamplingParams,
  SystemInfo,
  TrainingConfig,
  TrainingConfigValidation,
  TrainingJob,
  TrainingJobCreate,
  TrainingJobDetail,
  TrainingMetricPoint,
  Checkpoint,
  ChatMessage,
} from "@/lib/types";

export interface ListParams {
  limit?: number;
  offset?: number;
  search?: string;
  [key: string]: string | number | boolean | undefined;
}

export interface StreamCallbacks {
  onToken: (text: string) => void;
  onUsage?: (usage: import("@/lib/types").GenerationUsage, messageId?: string | null) => void;
  onError?: (message: string) => void;
  signal?: AbortSignal;
}

export interface DataProvider {
  readonly kind: "http" | "mock";

  system: {
    info(): Promise<SystemInfo>;
    health(): Promise<Record<string, unknown>>;
    settings(): Promise<Record<string, unknown>>;
    seedDemoData(force?: boolean): Promise<Ack>;
    clearDemoData(): Promise<Ack>;
  };

  models: {
    list(params?: ListParams): Promise<Page<Model>>;
    get(id: string): Promise<ModelDetail>;
    import(payload: ModelImportRequest): Promise<ModelDetail>;
    update(id: string, payload: Partial<Model> & { notes?: string }): Promise<ModelDetail>;
    remove(id: string): Promise<Ack>;
    clone(id: string, name: string, copyCheckpoints?: boolean): Promise<ModelDetail>;
    load(id: string, engine?: string): Promise<ModelDetail>;
    unload(id: string): Promise<ModelDetail>;
    export(id: string, format: string, destination?: string): Promise<Record<string, unknown>>;
    checkpoints(id: string): Promise<Checkpoint[]>;
  };

  datasets: {
    list(params?: ListParams): Promise<Page<Dataset>>;
    get(id: string): Promise<DatasetDetail>;
    templates(): Promise<DatasetTemplateInfo[]>;
    import(payload: {
      source: "path" | "huggingface";
      name: string;
      path?: string;
      repo_id?: string;
      subset?: string;
      split?: string;
      template?: DatasetTemplate;
      description?: string;
    }): Promise<DatasetDetail>;
    upload(file: File, name: string, template: DatasetTemplate): Promise<DatasetDetail>;
    preview(id: string, offset?: number, limit?: number): Promise<DatasetPreview>;
    validate(id: string, template?: DatasetTemplate): Promise<DatasetValidationReport>;
    remove(id: string): Promise<Ack>;
  };

  training: {
    listJobs(params?: ListParams): Promise<Page<TrainingJob>>;
    activeJobs(): Promise<TrainingJob[]>;
    getJob(id: string): Promise<TrainingJobDetail>;
    createJob(payload: TrainingJobCreate): Promise<TrainingJobDetail>;
    metrics(id: string): Promise<TrainingMetricPoint[]>;
    logs(id: string): Promise<{ entries: LogEntry[] }>;
    validateConfig(
      config: TrainingConfig,
      datasetId?: string,
    ): Promise<TrainingConfigValidation>;
    parseRawConfig(
      content: string,
      format: "json" | "yaml",
      datasetId?: string,
    ): Promise<TrainingConfigValidation>;
    start(id: string): Promise<TrainingJob>;
    pause(id: string): Promise<TrainingJob>;
    resume(id: string): Promise<TrainingJob>;
    stop(id: string): Promise<TrainingJob>;
    checkpoint(id: string): Promise<Ack>;
    remove(id: string): Promise<Ack>;
  };

  chat: {
    conversations(params?: ListParams): Promise<Page<Conversation>>;
    conversation(id: string): Promise<ConversationDetail>;
    createConversation(payload: {
      title?: string;
      model_id?: string;
      system_prompt?: string;
      params?: Partial<SamplingParams>;
    }): Promise<ConversationDetail>;
    updateConversation(
      id: string,
      payload: Partial<{
        title: string;
        model_id: string;
        system_prompt: string;
        params: SamplingParams;
        pinned: boolean;
      }>,
    ): Promise<ConversationDetail>;
    deleteConversation(id: string): Promise<Ack>;
    editMessage(id: string, content: string): Promise<Message>;
    deleteMessage(id: string, andAfter?: boolean): Promise<Ack>;
    stream(
      payload: {
        model_id: string;
        messages: ChatMessage[];
        system_prompt?: string;
        params: SamplingParams;
        conversation_id?: string | null;
        persist?: boolean;
      },
      callbacks: StreamCallbacks,
    ): Promise<void>;
    tokenize(text: string, modelId?: string): Promise<{ tokens: number; method: string }>;
  };

  playground: {
    run(payload: {
      model_ids: string[];
      prompt: string;
      system_prompt?: string;
      params: SamplingParams;
    }): Promise<PlaygroundRun>;
    history(params?: ListParams): Promise<Page<PlaygroundComparison>>;
    vote(comparisonId: string, winnerModelId: string): Promise<PlaygroundComparison>;
  };

  benchmarks: {
    suites(): Promise<BenchmarkSuite[]>;
    run(payload: {
      model_id: string;
      suite: string;
      dataset_id?: string | null;
      name?: string;
      config: BenchmarkRunConfig;
    }): Promise<BenchmarkRun>;
    results(params?: ListParams): Promise<Page<BenchmarkRun>>;
    result(id: string, onlyIncorrect?: boolean): Promise<BenchmarkRunDetail>;
    cancel(id: string): Promise<Ack>;
    remove(id: string): Promise<Ack>;
  };

  evaluations: {
    leaderboard(suite?: string): Promise<Leaderboard>;
    compare(modelIds: string[], name?: string): Promise<ModelComparison>;
  };

  experiments: {
    list(params?: ListParams): Promise<Page<Experiment>>;
    get(id: string): Promise<ExperimentDetail>;
    update(id: string, payload: { name?: string; notes?: string; tags?: string[] }): Promise<Experiment>;
    duplicate(id: string, name?: string, start?: boolean): Promise<Experiment>;
    compare(ids: string[]): Promise<ExperimentComparison>;
    remove(id: string): Promise<Ack>;
  };

  checkpoints: {
    list(params?: ListParams): Promise<Page<Checkpoint>>;
    load(id: string): Promise<ModelDetail>;
    evaluate(id: string, suite?: string, numExamples?: number): Promise<BenchmarkRun>;
    export(id: string): Promise<Ack>;
    remove(id: string): Promise<Ack>;
  };

  hardware: {
    snapshot(): Promise<HardwareSnapshot>;
    history(limit?: number): Promise<HardwareHistory>;
    devices(): Promise<Record<string, unknown>>;
  };

  logs: {
    list(params?: ListParams): Promise<Page<LogEntry>>;
    sources(): Promise<string[]>;
    clear(): Promise<Ack>;
  };
}
