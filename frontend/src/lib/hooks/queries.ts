"use client";

/**
 * React Query hooks — the only place components touch the data provider.
 * Live screens combine a query with a WebSocket subscription that patches the
 * cache, so charts stream without refetching.
 */

import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ListParams } from "@/lib/api/provider";
import type {
  BenchmarkRunConfig,
  DatasetTemplate,
  ModelImportRequest,
  ProjectCreateRequest,
  ProjectUpdateRequest,
  SamplingParams,
  TrainingConfig,
  TrainingJobCreate,
} from "@/lib/types";

export const queryKeys = {
  system: ["system"] as const,
  settings: ["settings"] as const,
  models: (params?: ListParams) => ["models", params ?? {}] as const,
  model: (id: string) => ["model", id] as const,
  projects: (params?: ListParams) => ["projects", params ?? {}] as const,
  project: (id: string) => ["project", id] as const,
  datasets: (params?: ListParams) => ["datasets", params ?? {}] as const,
  dataset: (id: string) => ["dataset", id] as const,
  datasetPreview: (id: string, offset: number) => ["dataset-preview", id, offset] as const,
  datasetTemplates: ["dataset-templates"] as const,
  jobs: (params?: ListParams) => ["training-jobs", params ?? {}] as const,
  job: (id: string) => ["training-job", id] as const,
  jobMetrics: (id: string) => ["training-metrics", id] as const,
  jobLogs: (id: string) => ["training-logs", id] as const,
  activeJobs: ["training-active"] as const,
  conversations: ["conversations"] as const,
  conversation: (id: string) => ["conversation", id] as const,
  suites: ["benchmark-suites"] as const,
  benchmarkRuns: (params?: ListParams) => ["benchmark-runs", params ?? {}] as const,
  benchmarkRun: (id: string, onlyIncorrect: boolean) =>
    ["benchmark-run", id, onlyIncorrect] as const,
  leaderboard: (suite?: string) => ["leaderboard", suite ?? "all"] as const,
  experiments: (params?: ListParams) => ["experiments", params ?? {}] as const,
  experiment: (id: string) => ["experiment", id] as const,
  checkpoints: (params?: ListParams) => ["checkpoints", params ?? {}] as const,
  hardware: ["hardware"] as const,
  hardwareHistory: (limit: number) => ["hardware-history", limit] as const,
  logs: (params?: ListParams) => ["logs", params ?? {}] as const,
  playgroundHistory: ["playground-history"] as const,
};

/* ------------------------------------------------------------------ system */

export function useSystemInfo() {
  return useQuery({
    queryKey: queryKeys.system,
    queryFn: () => api.system.info(),
    refetchInterval: 30_000,
    retry: 1,
  });
}

export function useSettings() {
  return useQuery({ queryKey: queryKeys.settings, queryFn: () => api.system.settings(), retry: 1 });
}

/* ------------------------------------------------------------------ models */

export function useModels(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.models(params),
    queryFn: () => api.models.list(params),
    placeholderData: keepPreviousData,
  });
}

export function useModel(id: string | null) {
  return useQuery({
    queryKey: queryKeys.model(id ?? "none"),
    queryFn: () => api.models.get(id as string),
    enabled: Boolean(id),
  });
}

export function useImportModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: ModelImportRequest) => api.models.import(payload),
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });
}

export function useModelMutations() {
  const client = useQueryClient();
  const invalidate = () => {
    client.invalidateQueries({ queryKey: ["models"] });
    client.invalidateQueries({ queryKey: ["model"] });
  };

  return {
    load: useMutation({
      mutationFn: ({ id, engine }: { id: string; engine?: string }) => api.models.load(id, engine),
      onSuccess: invalidate,
    }),
    unload: useMutation({
      mutationFn: (id: string) => api.models.unload(id),
      onSuccess: invalidate,
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.models.remove(id),
      onSuccess: invalidate,
    }),
    clone: useMutation({
      mutationFn: ({ id, name, copyCheckpoints }: { id: string; name: string; copyCheckpoints?: boolean }) =>
        api.models.clone(id, name, copyCheckpoints),
      onSuccess: invalidate,
    }),
    exportModel: useMutation({
      mutationFn: ({ id, format }: { id: string; format: string }) => api.models.export(id, format),
    }),
  };
}

/* ---------------------------------------------------------------- datasets */

export function useProjects(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.projects(params),
    queryFn: () => api.projects.list(params),
    placeholderData: keepPreviousData,
  });
}

export function useProject(id: string | null) {
  return useQuery({
    queryKey: queryKeys.project(id ?? "none"),
    queryFn: () => api.projects.get(id as string),
    enabled: Boolean(id),
  });
}

export function useProjectMutations() {
  const client = useQueryClient();
  const invalidate = () => {
    client.invalidateQueries({ queryKey: ["projects"] });
    client.invalidateQueries({ queryKey: ["project"] });
  };

  return {
    create: useMutation({
      mutationFn: (payload: ProjectCreateRequest) => api.projects.create(payload),
      onSuccess: invalidate,
    }),
    update: useMutation({
      mutationFn: ({ id, payload }: { id: string; payload: ProjectUpdateRequest }) =>
        api.projects.update(id, payload),
      onSuccess: invalidate,
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.projects.remove(id),
      // Drop the deleted project's query instead of invalidating it — an
      // invalidation would refetch an id that no longer exists and 404.
      onSuccess: (_result, id) => {
        client.removeQueries({ queryKey: queryKeys.project(id) });
        client.invalidateQueries({ queryKey: ["projects"] });
      },
    }),
  };
}

export function useDatasets(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.datasets(params),
    queryFn: () => api.datasets.list(params),
    placeholderData: keepPreviousData,
  });
}

export function useDataset(id: string | null) {
  return useQuery({
    queryKey: queryKeys.dataset(id ?? "none"),
    queryFn: () => api.datasets.get(id as string),
    enabled: Boolean(id),
  });
}

export function useDatasetPreview(id: string | null, offset = 0, limit = 25) {
  return useQuery({
    queryKey: queryKeys.datasetPreview(id ?? "none", offset),
    queryFn: () => api.datasets.preview(id as string, offset, limit),
    enabled: Boolean(id),
    placeholderData: keepPreviousData,
  });
}

export function useDatasetTemplates() {
  return useQuery({
    queryKey: queryKeys.datasetTemplates,
    queryFn: () => api.datasets.templates(),
    staleTime: 10 * 60_000,
  });
}

export function useDatasetMutations() {
  const client = useQueryClient();
  const invalidate = () => client.invalidateQueries({ queryKey: ["datasets"] });

  return {
    upload: useMutation({
      mutationFn: ({ file, name, template }: { file: File; name: string; template: DatasetTemplate }) =>
        api.datasets.upload(file, name, template),
      onSuccess: invalidate,
    }),
    importDataset: useMutation({
      mutationFn: (payload: Parameters<typeof api.datasets.import>[0]) => api.datasets.import(payload),
      onSuccess: invalidate,
    }),
    validate: useMutation({
      mutationFn: ({ id, template }: { id: string; template?: DatasetTemplate }) =>
        api.datasets.validate(id, template),
      onSuccess: invalidate,
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.datasets.remove(id),
      onSuccess: invalidate,
    }),
  };
}

/* ---------------------------------------------------------------- training */

export function useTrainingJobs(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.jobs(params),
    queryFn: () => api.training.listJobs(params),
    placeholderData: keepPreviousData,
  });
}

export function useActiveJobs(refetchInterval = 10_000) {
  return useQuery({
    queryKey: queryKeys.activeJobs,
    queryFn: () => api.training.activeJobs(),
    refetchInterval,
  });
}

export function useTrainingJob(id: string | null, options?: Partial<UseQueryOptions<any>>) {
  return useQuery({
    queryKey: queryKeys.job(id ?? "none"),
    queryFn: () => api.training.getJob(id as string),
    enabled: Boolean(id),
    ...options,
  });
}

export function useTrainingMetrics(id: string | null) {
  return useQuery({
    queryKey: queryKeys.jobMetrics(id ?? "none"),
    queryFn: () => api.training.metrics(id as string),
    enabled: Boolean(id),
  });
}

export function useTrainingLogs(id: string | null) {
  return useQuery({
    queryKey: queryKeys.jobLogs(id ?? "none"),
    queryFn: () => api.training.logs(id as string),
    enabled: Boolean(id),
  });
}

export function useTrainingMutations() {
  const client = useQueryClient();
  const invalidate = () => {
    client.invalidateQueries({ queryKey: ["training-jobs"] });
    client.invalidateQueries({ queryKey: ["training-job"] });
    client.invalidateQueries({ queryKey: queryKeys.activeJobs });
    client.invalidateQueries({ queryKey: ["experiments"] });
  };

  return {
    create: useMutation({
      mutationFn: (payload: TrainingJobCreate) => api.training.createJob(payload),
      onSuccess: invalidate,
    }),
    start: useMutation({ mutationFn: (id: string) => api.training.start(id), onSuccess: invalidate }),
    pause: useMutation({ mutationFn: (id: string) => api.training.pause(id), onSuccess: invalidate }),
    resume: useMutation({ mutationFn: (id: string) => api.training.resume(id), onSuccess: invalidate }),
    stop: useMutation({ mutationFn: (id: string) => api.training.stop(id), onSuccess: invalidate }),
    checkpoint: useMutation({ mutationFn: (id: string) => api.training.checkpoint(id) }),
    remove: useMutation({ mutationFn: (id: string) => api.training.remove(id), onSuccess: invalidate }),
    validateConfig: useMutation({
      mutationFn: ({ config, datasetId }: { config: TrainingConfig; datasetId?: string }) =>
        api.training.validateConfig(config, datasetId),
    }),
    parseRaw: useMutation({
      mutationFn: ({ content, format, datasetId }: { content: string; format: "json" | "yaml"; datasetId?: string }) =>
        api.training.parseRawConfig(content, format, datasetId),
    }),
  };
}

/* -------------------------------------------------------------------- chat */

export function useConversations() {
  return useQuery({ queryKey: queryKeys.conversations, queryFn: () => api.chat.conversations() });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: queryKeys.conversation(id ?? "none"),
    queryFn: () => api.chat.conversation(id as string),
    enabled: Boolean(id),
  });
}

export function useChatMutations() {
  const client = useQueryClient();
  const invalidate = () => {
    client.invalidateQueries({ queryKey: queryKeys.conversations });
    client.invalidateQueries({ queryKey: ["conversation"] });
  };

  return {
    create: useMutation({
      mutationFn: (payload: Parameters<typeof api.chat.createConversation>[0]) =>
        api.chat.createConversation(payload),
      onSuccess: invalidate,
    }),
    update: useMutation({
      mutationFn: ({ id, payload }: { id: string; payload: Parameters<typeof api.chat.updateConversation>[1] }) =>
        api.chat.updateConversation(id, payload),
      onSuccess: invalidate,
    }),
    remove: useMutation({
      mutationFn: (id: string) => api.chat.deleteConversation(id),
      onSuccess: invalidate,
    }),
    editMessage: useMutation({
      mutationFn: ({ id, content }: { id: string; content: string }) => api.chat.editMessage(id, content),
      onSuccess: invalidate,
    }),
    deleteMessage: useMutation({
      mutationFn: ({ id, andAfter }: { id: string; andAfter?: boolean }) =>
        api.chat.deleteMessage(id, andAfter),
      onSuccess: invalidate,
    }),
  };
}

/* -------------------------------------------------------------- playground */

export function usePlaygroundHistory() {
  return useQuery({
    queryKey: queryKeys.playgroundHistory,
    queryFn: () => api.playground.history({ limit: 20 }),
  });
}

export function usePlaygroundMutations() {
  const client = useQueryClient();
  return {
    run: useMutation({
      mutationFn: (payload: {
        model_ids: string[];
        prompt: string;
        system_prompt?: string;
        params: SamplingParams;
      }) => api.playground.run(payload),
      onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.playgroundHistory }),
    }),
    vote: useMutation({
      mutationFn: ({ comparisonId, winner }: { comparisonId: string; winner: string }) =>
        api.playground.vote(comparisonId, winner),
      onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.playgroundHistory }),
    }),
  };
}

/* -------------------------------------------------------------- benchmarks */

export function useBenchmarkSuites() {
  return useQuery({
    queryKey: queryKeys.suites,
    queryFn: () => api.benchmarks.suites(),
    staleTime: 5 * 60_000,
  });
}

export function useBenchmarkRuns(params?: ListParams, refetchInterval?: number) {
  return useQuery({
    queryKey: queryKeys.benchmarkRuns(params),
    queryFn: () => api.benchmarks.results(params),
    refetchInterval,
    placeholderData: keepPreviousData,
  });
}

export function useBenchmarkRun(id: string | null, onlyIncorrect = false, refetchInterval?: number) {
  return useQuery({
    queryKey: queryKeys.benchmarkRun(id ?? "none", onlyIncorrect),
    queryFn: () => api.benchmarks.result(id as string, onlyIncorrect),
    enabled: Boolean(id),
    refetchInterval,
  });
}

export function useBenchmarkMutations() {
  const client = useQueryClient();
  const invalidate = () => client.invalidateQueries({ queryKey: ["benchmark-runs"] });
  return {
    run: useMutation({
      mutationFn: (payload: {
        model_id: string;
        suite: string;
        dataset_id?: string | null;
        name?: string;
        config: BenchmarkRunConfig;
      }) => api.benchmarks.run(payload),
      onSuccess: invalidate,
    }),
    cancel: useMutation({ mutationFn: (id: string) => api.benchmarks.cancel(id), onSuccess: invalidate }),
    remove: useMutation({ mutationFn: (id: string) => api.benchmarks.remove(id), onSuccess: invalidate }),
  };
}

/* ------------------------------------------------------------- evaluations */

export function useLeaderboard(suite?: string) {
  return useQuery({
    queryKey: queryKeys.leaderboard(suite),
    queryFn: () => api.evaluations.leaderboard(suite),
  });
}

export function useCompareModels() {
  return useMutation({
    mutationFn: ({ modelIds, name }: { modelIds: string[]; name?: string }) =>
      api.evaluations.compare(modelIds, name),
  });
}

/* ------------------------------------------------------------- experiments */

export function useExperiments(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.experiments(params),
    queryFn: () => api.experiments.list(params),
    placeholderData: keepPreviousData,
  });
}

export function useExperiment(id: string | null) {
  return useQuery({
    queryKey: queryKeys.experiment(id ?? "none"),
    queryFn: () => api.experiments.get(id as string),
    enabled: Boolean(id),
  });
}

export function useExperimentMutations() {
  const client = useQueryClient();
  const invalidate = () => client.invalidateQueries({ queryKey: ["experiments"] });
  return {
    update: useMutation({
      mutationFn: ({ id, payload }: { id: string; payload: { name?: string; notes?: string; tags?: string[] } }) =>
        api.experiments.update(id, payload),
      onSuccess: invalidate,
    }),
    duplicate: useMutation({
      mutationFn: ({ id, name, start }: { id: string; name?: string; start?: boolean }) =>
        api.experiments.duplicate(id, name, start),
      onSuccess: invalidate,
    }),
    compare: useMutation({ mutationFn: (ids: string[]) => api.experiments.compare(ids) }),
    remove: useMutation({ mutationFn: (id: string) => api.experiments.remove(id), onSuccess: invalidate }),
  };
}

/* ------------------------------------------------------------- checkpoints */

export function useCheckpoints(params?: ListParams) {
  return useQuery({
    queryKey: queryKeys.checkpoints(params),
    queryFn: () => api.checkpoints.list(params),
    placeholderData: keepPreviousData,
  });
}

export function useCheckpointMutations() {
  const client = useQueryClient();
  const invalidate = () => client.invalidateQueries({ queryKey: ["checkpoints"] });
  return {
    load: useMutation({ mutationFn: (id: string) => api.checkpoints.load(id), onSuccess: invalidate }),
    evaluate: useMutation({
      mutationFn: ({ id, suite }: { id: string; suite?: string }) => api.checkpoints.evaluate(id, suite),
    }),
    exportCheckpoint: useMutation({ mutationFn: (id: string) => api.checkpoints.export(id) }),
    remove: useMutation({ mutationFn: (id: string) => api.checkpoints.remove(id), onSuccess: invalidate }),
  };
}

/* ---------------------------------------------------------------- hardware */

export function useHardware(refetchInterval = 4000) {
  return useQuery({
    queryKey: queryKeys.hardware,
    queryFn: () => api.hardware.snapshot(),
    refetchInterval,
  });
}

export function useHardwareHistory(limit = 180, refetchInterval = 10_000) {
  return useQuery({
    queryKey: queryKeys.hardwareHistory(limit),
    queryFn: () => api.hardware.history(limit),
    refetchInterval,
  });
}

/* -------------------------------------------------------------------- logs */

export function useLogs(params?: ListParams, refetchInterval = 8000) {
  return useQuery({
    queryKey: queryKeys.logs(params),
    queryFn: () => api.logs.list(params),
    refetchInterval,
    placeholderData: keepPreviousData,
  });
}

export function useLogSources() {
  return useQuery({ queryKey: ["log-sources"], queryFn: () => api.logs.sources(), staleTime: 60_000 });
}
