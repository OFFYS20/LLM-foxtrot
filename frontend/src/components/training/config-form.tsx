"use client";

import { Play, RefreshCw, Save } from "lucide-react";
import * as React from "react";

import { WarningNote } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LORA_TARGET_MODULES,
  OPTIMIZERS,
  PRECISIONS,
  SCHEDULERS,
  TRAINING_METHODS,
  DEFAULT_TRAINING_CONFIG,
} from "@/lib/constants";
import { useTrainingMutations } from "@/lib/hooks/queries";
import { formatCompact, formatNumber } from "@/lib/format";
import type {
  Dataset,
  Model,
  OptimizerName,
  Precision,
  SchedulerName,
  TrainingConfig,
  TrainingConfigValidation,
  TrainingMethod,
} from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  models: Model[];
  datasets: Dataset[];
  initialModelId?: string | null;
  initialDatasetId?: string | null;
  onLaunched?: (jobId: string) => void;
}

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      <Input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-8 text-xs"
      />
      {hint ? <p className="text-2xs text-ink-muted">{hint}</p> : null}
    </div>
  );
}

export function TrainingConfigForm({
  models,
  datasets,
  initialModelId,
  initialDatasetId,
  onLaunched,
}: Props) {
  const [modelId, setModelId] = React.useState(initialModelId ?? models[0]?.id ?? "");
  const [datasetId, setDatasetId] = React.useState(initialDatasetId ?? datasets[0]?.id ?? "");
  const [name, setName] = React.useState("");
  const [notes, setNotes] = React.useState("");
  const [config, setConfig] = React.useState<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG } as TrainingConfig);
  const [raw, setRaw] = React.useState(() => JSON.stringify(DEFAULT_TRAINING_CONFIG, null, 2));
  const [rawFormat, setRawFormat] = React.useState<"json" | "yaml">("json");
  const [validation, setValidation] = React.useState<TrainingConfigValidation | null>(null);

  const { create, validateConfig, parseRaw } = useTrainingMutations();

  React.useEffect(() => {
    if (!modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);
  React.useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id);
  }, [datasets, datasetId]);

  const dataset = datasets.find((item) => item.id === datasetId);
  const model = models.find((item) => item.id === modelId);

  const patch = React.useCallback((changes: Partial<TrainingConfig>) => {
    setConfig((current) => {
      const next = { ...current, ...changes };
      setRaw(JSON.stringify(next, null, 2));
      return next;
    });
  }, []);

  const effectiveBatch = config.batch_size * config.gradient_accumulation_steps;
  const estimatedSteps = dataset
    ? Math.max(1, Math.ceil((dataset.rows * config.epochs) / Math.max(1, effectiveBatch)))
    : null;
  const estimatedTokens = dataset ? Math.round(dataset.tokens * config.epochs) : null;

  const runValidation = () => {
    validateConfig.mutate(
      { config, datasetId: datasetId || undefined },
      { onSuccess: (result) => setValidation(result) },
    );
  };

  const applyRaw = () => {
    parseRaw.mutate(
      { content: raw, format: rawFormat, datasetId: datasetId || undefined },
      {
        onSuccess: (result) => {
          setValidation(result);
          if (result.ok && result.config) setConfig(result.config);
        },
      },
    );
  };

  const launch = (startImmediately: boolean) => {
    create.mutate(
      {
        model_id: modelId,
        dataset_id: datasetId,
        name: name || undefined,
        notes: notes || undefined,
        config,
        start_immediately: startImmediately,
      },
      { onSuccess: (job) => onLaunched?.(job.id) },
    );
  };

  const isLora = config.method === "lora" || config.method === "qlora";

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Run</CardTitle>
          <div className="mono text-2xs text-ink-muted">
            effective batch {effectiveBatch}
            {estimatedSteps ? ` · ~${formatNumber(estimatedSteps)} steps` : ""}
            {estimatedTokens ? ` · ~${formatCompact(estimatedTokens)} tokens` : ""}
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          <div className="space-y-1.5">
            <Label>Base model</Label>
            <Select value={modelId} onValueChange={setModelId}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.display_name ?? item.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {model ? (
              <p className="mono text-2xs text-ink-muted">
                {model.architecture ?? model.format} · {model.precision.toUpperCase()}
              </p>
            ) : null}
          </div>

          <div className="space-y-1.5">
            <Label>Dataset</Label>
            <Select value={datasetId} onValueChange={setDatasetId}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Select a dataset" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {dataset ? (
              <p className="mono text-2xs text-ink-muted">
                {formatNumber(dataset.rows)} rows · {dataset.template}
              </p>
            ) : null}
          </div>

          <div className="space-y-1.5">
            <Label>Run name</Label>
            <Input
              placeholder="auto: model · method · dataset"
              value={name}
              onChange={(event) => setName(event.target.value)}
              className="h-8 text-xs"
            />
          </div>

          <div className="space-y-1.5">
            <Label>Notes</Label>
            <Input
              placeholder="What are you testing?"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              className="h-8 text-xs"
            />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="form">
        <TabsList>
          <TabsTrigger value="form">Visual configuration</TabsTrigger>
          <TabsTrigger value="raw">Advanced (JSON / YAML)</TabsTrigger>
        </TabsList>

        <TabsContent value="form" className="space-y-3">
          <Card>
            <CardHeader>
              <CardTitle>Method</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {TRAINING_METHODS.map((method) => (
                <button
                  key={method.value}
                  type="button"
                  onClick={() => patch({ method: method.value as TrainingMethod })}
                  className={cn(
                    "rounded-md border px-3 py-2 text-left transition-colors",
                    config.method === method.value
                      ? "border-accent/60 bg-accent/10"
                      : "border-border bg-elevated/40 hover:border-border-strong",
                  )}
                >
                  <div className="text-xs font-medium text-ink">{method.label}</div>
                  <div className="mt-0.5 text-2xs text-ink-muted">{method.hint}</div>
                </button>
              ))}
            </CardContent>
          </Card>

          <div className="grid gap-3 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Hyperparameters</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <NumberField label="Epochs" value={config.epochs} step={0.5} min={0.1} onChange={(v) => patch({ epochs: v })} />
                <NumberField label="Batch size" value={config.batch_size} min={1} onChange={(v) => patch({ batch_size: v })} />
                <NumberField
                  label="Gradient accumulation"
                  value={config.gradient_accumulation_steps}
                  min={1}
                  onChange={(v) => patch({ gradient_accumulation_steps: v })}
                  hint={`effective batch ${effectiveBatch}`}
                />
                <NumberField
                  label="Learning rate"
                  value={config.learning_rate}
                  step={0.000005}
                  onChange={(v) => patch({ learning_rate: v })}
                  hint={config.learning_rate.toExponential(2)}
                />
                <NumberField label="Warmup steps" value={config.warmup_steps} min={0} onChange={(v) => patch({ warmup_steps: v })} />
                <NumberField label="Weight decay" value={config.weight_decay} step={0.001} min={0} onChange={(v) => patch({ weight_decay: v })} />
                <NumberField
                  label="Max sequence length"
                  value={config.max_sequence_length}
                  min={16}
                  step={128}
                  onChange={(v) => patch({ max_sequence_length: v })}
                />
                <NumberField label="Gradient clipping" value={config.gradient_clipping} step={0.1} min={0} onChange={(v) => patch({ gradient_clipping: v })} />
                <NumberField label="Seed" value={config.seed} min={0} onChange={(v) => patch({ seed: v })} />

                <div className="space-y-1.5">
                  <Label>Optimizer</Label>
                  <Select value={config.optimizer} onValueChange={(value) => patch({ optimizer: value as OptimizerName })}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {OPTIMIZERS.map((item) => (
                        <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <Label>LR scheduler</Label>
                  <Select value={config.lr_scheduler} onValueChange={(value) => patch({ lr_scheduler: value as SchedulerName })}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {SCHEDULERS.map((item) => (
                        <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            <div className="space-y-3">
              <Card>
                <CardHeader>
                  <CardTitle>Precision & memory</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {PRECISIONS.map((item) => (
                      <button
                        key={item.value}
                        type="button"
                        onClick={() => patch({ precision: item.value as Precision })}
                        className={cn(
                          "rounded border px-2.5 py-1 text-2xs transition-colors",
                          config.precision === item.value
                            ? "border-accent/60 bg-accent/10 text-ink"
                            : "border-border bg-elevated/40 text-ink-muted hover:text-ink-secondary",
                        )}
                        title={item.hint}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="normal-case tracking-normal">Gradient checkpointing</Label>
                    <Switch
                      checked={config.gradient_checkpointing}
                      onCheckedChange={(checked) => patch({ gradient_checkpointing: checked })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="normal-case tracking-normal">Flash attention</Label>
                    <Switch
                      checked={config.flash_attention}
                      onCheckedChange={(checked) => patch({ flash_attention: checked })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="normal-case tracking-normal">Sequence packing</Label>
                    <Switch checked={config.packing} onCheckedChange={(checked) => patch({ packing: checked })} />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="normal-case tracking-normal">Shuffle</Label>
                    <Switch checked={config.shuffle} onCheckedChange={(checked) => patch({ shuffle: checked })} />
                  </div>
                </CardContent>
              </Card>

              <Card className={cn(!isLora && "opacity-50")}>
                <CardHeader>
                  <CardTitle>LoRA configuration</CardTitle>
                  {!isLora ? <Badge variant="muted">method is not LoRA</Badge> : null}
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-3 gap-3">
                    <NumberField
                      label="Rank"
                      value={config.lora.rank}
                      min={1}
                      onChange={(v) => patch({ lora: { ...config.lora, rank: v } })}
                    />
                    <NumberField
                      label="Alpha"
                      value={config.lora.alpha}
                      min={1}
                      onChange={(v) => patch({ lora: { ...config.lora, alpha: v } })}
                    />
                    <NumberField
                      label="Dropout"
                      value={config.lora.dropout}
                      step={0.01}
                      min={0}
                      max={0.9}
                      onChange={(v) => patch({ lora: { ...config.lora, dropout: v } })}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label>Target modules</Label>
                    <div className="flex flex-wrap gap-1.5">
                      {LORA_TARGET_MODULES.map((module) => {
                        const selected = config.lora.target_modules.includes(module);
                        return (
                          <button
                            key={module}
                            type="button"
                            onClick={() =>
                              patch({
                                lora: {
                                  ...config.lora,
                                  target_modules: selected
                                    ? config.lora.target_modules.filter((item) => item !== module)
                                    : [...config.lora.target_modules, module],
                                },
                              })
                            }
                            className={cn(
                              "mono rounded border px-2 py-0.5 text-2xs transition-colors",
                              selected
                                ? "border-accent/60 bg-accent/10 text-ink"
                                : "border-border text-ink-muted hover:text-ink-secondary",
                            )}
                          >
                            {module}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Checkpointing</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <NumberField
                  label="Save every N steps"
                  value={config.checkpointing.save_every_steps}
                  min={1}
                  onChange={(v) => patch({ checkpointing: { ...config.checkpointing, save_every_steps: v } })}
                />
                <NumberField
                  label="Keep last N"
                  value={config.checkpointing.keep_last}
                  min={1}
                  onChange={(v) => patch({ checkpointing: { ...config.checkpointing, keep_last: v } })}
                />
                <div className="col-span-2 flex items-center justify-between">
                  <Label className="normal-case tracking-normal">Always keep the best checkpoint</Label>
                  <Switch
                    checked={config.checkpointing.save_best}
                    onCheckedChange={(checked) =>
                      patch({ checkpointing: { ...config.checkpointing, save_best: checked } })
                    }
                  />
                </div>
                <div className="col-span-2 flex items-center justify-between">
                  <Label className="normal-case tracking-normal">Save optimizer state</Label>
                  <Switch
                    checked={config.checkpointing.save_optimizer_state}
                    onCheckedChange={(checked) =>
                      patch({ checkpointing: { ...config.checkpointing, save_optimizer_state: checked } })
                    }
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cadence</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3">
                <NumberField
                  label="Evaluate every N steps"
                  value={config.eval_every_steps}
                  min={1}
                  onChange={(v) => patch({ eval_every_steps: v })}
                />
                <NumberField
                  label="Log every N steps"
                  value={config.log_every_steps}
                  min={1}
                  onChange={(v) => patch({ log_every_steps: v })}
                />
                <NumberField
                  label="Dataloader workers"
                  value={config.num_workers}
                  min={0}
                  onChange={(v) => patch({ num_workers: v })}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="raw" className="space-y-3">
          <Card>
            <CardHeader>
              <CardTitle>Raw configuration</CardTitle>
              <div className="flex items-center gap-2">
                <Select value={rawFormat} onValueChange={(value) => setRawFormat(value as "json" | "yaml")}>
                  <SelectTrigger className="h-7 w-[92px] text-2xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="json">JSON</SelectItem>
                    <SelectItem value="yaml">YAML</SelectItem>
                  </SelectContent>
                </Select>
                <Button size="xs" variant="secondary" onClick={applyRaw} disabled={parseRaw.isPending}>
                  <RefreshCw className="h-3 w-3" />
                  Parse & apply
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <Textarea
                value={raw}
                onChange={(event) => setRaw(event.target.value)}
                spellCheck={false}
                className="mono h-[420px] resize-none text-2xs leading-relaxed"
              />
              <p className="mt-2 text-2xs text-ink-muted">
                Parsed and validated on the backend against the same schema as the form — invalid
                configurations are rejected before a job is created.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {validation ? (
        <div className="space-y-2">
          {validation.errors.map((message) => (
            <p key={message} className="mono rounded border border-critical/40 bg-critical/5 px-3 py-2 text-xs text-critical">
              {message}
            </p>
          ))}
          {validation.warnings.map((message) => (
            <WarningNote key={message}>{message}</WarningNote>
          ))}
          {validation.ok && validation.errors.length === 0 && validation.warnings.length === 0 ? (
            <p className="rounded border border-good/40 bg-good/5 px-3 py-2 text-xs text-good">
              Configuration validated — no warnings.
            </p>
          ) : null}
        </div>
      ) : null}

      {create.error ? (
        <p className="mono rounded border border-critical/40 bg-critical/5 px-3 py-2 text-xs text-critical">
          {(create.error as Error).message}
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="secondary" size="sm" onClick={runValidation} disabled={validateConfig.isPending}>
          <RefreshCw className={cn("h-3.5 w-3.5", validateConfig.isPending && "animate-spin")} />
          Validate
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => launch(false)}
          disabled={create.isPending || !modelId || !datasetId}
        >
          <Save className="h-3.5 w-3.5" />
          Queue without starting
        </Button>
        <Button size="sm" onClick={() => launch(true)} disabled={create.isPending || !modelId || !datasetId}>
          <Play className="h-3.5 w-3.5" />
          {create.isPending ? "Starting…" : "Start training"}
        </Button>
      </div>
    </div>
  );
}
