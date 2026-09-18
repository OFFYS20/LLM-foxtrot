import type {
  DatasetTemplate,
  OptimizerName,
  Precision,
  SchedulerName,
  TrainingMethod,
} from "@/lib/types";

export const TRAINING_METHODS: { value: TrainingMethod; label: string; hint: string }[] = [
  { value: "full_finetune", label: "Full fine-tuning", hint: "All weights updated — highest VRAM" },
  { value: "lora", label: "LoRA", hint: "Low-rank adapters on frozen weights" },
  { value: "qlora", label: "QLoRA", hint: "4-bit base + LoRA adapters" },
  { value: "continued_pretraining", label: "Continued pretraining", hint: "Raw corpus, no labels" },
  { value: "instruction_tuning", label: "Instruction tuning", hint: "Supervised prompt/response" },
];

export const PRECISIONS: { value: Precision; label: string; hint: string }[] = [
  { value: "fp32", label: "FP32", hint: "Full precision — safest on CPU" },
  { value: "fp16", label: "FP16", hint: "Half precision, needs loss scaling" },
  { value: "bf16", label: "BF16", hint: "Half precision, wider exponent" },
  { value: "int8", label: "INT8", hint: "8-bit quantized base weights" },
  { value: "int4", label: "INT4", hint: "4-bit quantized base (QLoRA)" },
];

export const OPTIMIZERS: { value: OptimizerName; label: string }[] = [
  { value: "adamw", label: "AdamW" },
  { value: "adamw_8bit", label: "AdamW 8-bit" },
  { value: "adafactor", label: "Adafactor" },
  { value: "sgd", label: "SGD" },
  { value: "lion", label: "Lion" },
];

export const SCHEDULERS: { value: SchedulerName; label: string }[] = [
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
  { value: "cosine_with_restarts", label: "Cosine w/ restarts" },
  { value: "constant", label: "Constant" },
  { value: "constant_with_warmup", label: "Constant w/ warmup" },
  { value: "polynomial", label: "Polynomial" },
];

export const LORA_TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
  "lm_head",
];

export const DATASET_TEMPLATE_LABELS: Record<DatasetTemplate, string> = {
  instruction: "Instruction tuning",
  chat: "Chat",
  plain_text: "Plain text",
  raw: "Raw / custom",
};

export const DEFAULT_TRAINING_CONFIG = {
  method: "lora" as TrainingMethod,
  epochs: 3,
  batch_size: 4,
  gradient_accumulation_steps: 8,
  learning_rate: 2e-5,
  warmup_steps: 100,
  weight_decay: 0.01,
  max_sequence_length: 2048,
  gradient_clipping: 1.0,
  optimizer: "adamw" as OptimizerName,
  lr_scheduler: "cosine" as SchedulerName,
  seed: 42,
  precision: "bf16" as Precision,
  gradient_checkpointing: true,
  flash_attention: false,
  lora: {
    rank: 16,
    alpha: 32,
    dropout: 0.05,
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
    bias: "none" as const,
  },
  checkpointing: {
    save_every_steps: 250,
    keep_last: 3,
    save_best: true,
    save_optimizer_state: false,
  },
  eval_every_steps: 100,
  log_every_steps: 10,
  packing: false,
  shuffle: true,
  num_workers: 2,
};

export const DEFAULT_SAMPLING = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  max_tokens: 512,
  repetition_penalty: 1.1,
  seed: null as number | null,
  stop: [] as string[],
};

export const BENCHMARK_CATEGORY_LABELS: Record<string, string> = {
  reasoning: "Reasoning",
  math: "Math",
  coding: "Coding",
  knowledge: "Knowledge",
  instruction_following: "Instruction following",
  long_context: "Long-context",
  language_understanding: "Language understanding",
  hallucination_resistance: "Hallucination resistance",
  safety: "Safety",
  custom: "Custom",
};
