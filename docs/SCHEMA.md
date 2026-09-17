# Database schema

Generated from the SQLAlchemy metadata in `backend/app/db/models/`. SQLite is the default;
no SQLite-specific types are used, so `FOXTROT_DATABASE_URL=postgresql+psycopg://…` is the
only change needed for PostgreSQL.

## `datasets`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  | indexed |
| `description` | TEXT | yes |  |  |
| `format` | VARCHAR(32) | no | `jsonl` |  |
| `template` | VARCHAR(32) | no | `raw` |  |
| `status` | VARCHAR(32) | no | `ready` |  |
| `source` | VARCHAR(40) | no | `upload` |  |
| `repo_id` | VARCHAR(200) | yes |  |  |
| `local_path` | VARCHAR(600) | yes |  |  |
| `rows` | INTEGER | no | `0` |  |
| `tokens` | INTEGER | no | `0` |  |
| `avg_sequence_length` | FLOAT | no | `0.0` |  |
| `max_sequence_length` | INTEGER | no | `0` |  |
| `size_bytes` | INTEGER | no | `0` |  |
| `train_split` | FLOAT | no | `0.9` |  |
| `validation_split` | FLOAT | no | `0.05` |  |
| `test_split` | FLOAT | no | `0.05` |  |
| `columns` | JSON | no | runtime |  |
| `preview` | JSON | no | runtime |  |
| `validation_report` | JSON | no | runtime |  |
| `tags` | JSON | no | runtime |  |
| `token_count_method` | VARCHAR(32) | no | `estimated` |  |
| `is_demo` | BOOLEAN | no | `False` | indexed |
| `error` | TEXT | yes |  |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `hardware_samples`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no |  | primary key |
| `ts` | DATETIME | no | runtime | indexed |
| `payload` | JSON | no | runtime |  |

## `log_entries`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no |  | primary key |
| `ts` | DATETIME | no | runtime | indexed |
| `level` | VARCHAR(12) | no | `info` | indexed |
| `source` | VARCHAR(48) | no | `system` | indexed |
| `message` | TEXT | no | `` |  |
| `job_id` | VARCHAR(40) | yes |  | indexed |
| `context` | JSON | no | runtime |  |

## `model_comparisons`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  |  |
| `model_ids` | JSON | no | runtime |  |
| `metrics` | JSON | no | runtime |  |
| `notes` | TEXT | yes |  |  |
| `is_demo` | BOOLEAN | no | `False` |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `models`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  | indexed |
| `display_name` | VARCHAR(200) | yes |  |  |
| `description` | TEXT | yes |  |  |
| `source` | VARCHAR(32) | no | `local` |  |
| `format` | VARCHAR(32) | no | `safetensors` |  |
| `status` | VARCHAR(32) | no | `ready` | indexed |
| `repo_id` | VARCHAR(200) | yes |  |  |
| `revision` | VARCHAR(80) | yes |  |  |
| `local_path` | VARCHAR(600) | yes |  |  |
| `architecture` | VARCHAR(120) | yes |  |  |
| `parameters` | INTEGER | yes |  |  |
| `context_length` | INTEGER | yes |  |  |
| `precision` | VARCHAR(16) | no | `bf16` |  |
| `quantization` | VARCHAR(40) | yes |  |  |
| `size_bytes` | INTEGER | yes |  |  |
| `vram_estimate_mb` | INTEGER | yes |  |  |
| `tokenizer` | JSON | no | runtime |  |
| `config` | JSON | no | runtime |  |
| `metrics` | JSON | no | runtime |  |
| `tags` | JSON | no | runtime |  |
| `parent_model_id` | VARCHAR(40) | yes |  | → `models.id` |
| `source_checkpoint_id` | VARCHAR(40) | yes |  |  |
| `license` | VARCHAR(120) | yes |  |  |
| `is_demo` | BOOLEAN | no | `False` | indexed |
| `loaded_at` | DATETIME | yes |  |  |
| `error` | TEXT | yes |  |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `playground_comparisons`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `prompt` | TEXT | no |  |  |
| `system_prompt` | TEXT | no | `` |  |
| `params` | JSON | no | runtime |  |
| `entries` | JSON | no | runtime |  |
| `winner_model_id` | VARCHAR(40) | yes |  |  |
| `votes` | JSON | no | runtime |  |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `is_demo` | BOOLEAN | no | `False` |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `settings`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `key` | VARCHAR(120) | no |  | unique, indexed |
| `value` | JSON | no | runtime |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `benchmark_runs`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  |  |
| `suite` | VARCHAR(60) | no |  | indexed |
| `suite_label` | VARCHAR(120) | no | `` |  |
| `category` | VARCHAR(48) | no | `knowledge` |  |
| `model_id` | VARCHAR(40) | no |  | → `models.id`, indexed |
| `checkpoint_id` | VARCHAR(40) | yes |  |  |
| `dataset_id` | VARCHAR(40) | yes |  | → `datasets.id` |
| `status` | VARCHAR(24) | no | `queued` | indexed |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `config` | JSON | no | runtime |  |
| `total_items` | INTEGER | no | `0` |  |
| `completed_items` | INTEGER | no | `0` |  |
| `overall_score` | FLOAT | yes |  |  |
| `accuracy` | FLOAT | yes |  |  |
| `pass_at_1` | FLOAT | yes |  |  |
| `avg_latency_ms` | FLOAT | yes |  |  |
| `avg_tokens_per_sec` | FLOAT | yes |  |  |
| `total_tokens` | INTEGER | no | `0` |  |
| `runtime_seconds` | FLOAT | yes |  |  |
| `category_scores` | JSON | no | runtime |  |
| `started_at` | DATETIME | yes |  |  |
| `ended_at` | DATETIME | yes |  |  |
| `error` | TEXT | yes |  |  |
| `is_demo` | BOOLEAN | no | `False` | indexed |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `checkpoints`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `model_id` | VARCHAR(40) | no |  | → `models.id`, indexed |
| `job_id` | VARCHAR(40) | yes |  | indexed |
| `experiment_id` | VARCHAR(40) | yes |  |  |
| `step` | INTEGER | no | `0` |  |
| `epoch` | FLOAT | no | `0.0` |  |
| `path` | VARCHAR(600) | yes |  |  |
| `size_bytes` | INTEGER | no | `0` |  |
| `train_loss` | FLOAT | yes |  |  |
| `val_loss` | FLOAT | yes |  |  |
| `benchmark_score` | FLOAT | yes |  |  |
| `is_best` | BOOLEAN | no | `False` |  |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `notes` | TEXT | yes |  |  |
| `is_demo` | BOOLEAN | no | `False` |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `conversations`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `title` | VARCHAR(200) | no | `New conversation` |  |
| `model_id` | VARCHAR(40) | yes |  | → `models.id` |
| `system_prompt` | TEXT | no | `` |  |
| `params` | JSON | no | runtime |  |
| `pinned` | BOOLEAN | no | `False` |  |
| `is_demo` | BOOLEAN | no | `False` |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `projects`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(160) | no |  | unique |
| `description` | TEXT | yes |  |  |
| `base_model_id` | VARCHAR(40) | yes |  | → `models.id` |
| `default_dataset_id` | VARCHAR(40) | yes |  | → `datasets.id` |
| `tags` | JSON | no | runtime |  |
| `settings` | JSON | no | runtime |  |
| `is_demo` | BOOLEAN | no | `False` |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `benchmark_items`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no |  | primary key |
| `run_id` | VARCHAR(40) | no |  | → `benchmark_runs.id` |
| `index` | INTEGER | no |  |  |
| `category` | VARCHAR(48) | no | `` |  |
| `question` | TEXT | no | `` |  |
| `prompt` | TEXT | no | `` |  |
| `expected` | TEXT | no | `` |  |
| `response` | TEXT | no | `` |  |
| `raw_output` | TEXT | no | `` |  |
| `correct` | BOOLEAN | no | `False` |  |
| `score` | FLOAT | no | `0.0` |  |
| `latency_ms` | FLOAT | no | `0.0` |  |
| `tokens_used` | INTEGER | no | `0` |  |

## `experiments`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  | indexed |
| `model_id` | VARCHAR(40) | yes |  | → `models.id` |
| `dataset_id` | VARCHAR(40) | yes |  | → `datasets.id` |
| `job_id` | VARCHAR(40) | yes |  |  |
| `project_id` | VARCHAR(40) | yes |  | → `projects.id` |
| `status` | VARCHAR(24) | no | `queued` |  |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `method` | VARCHAR(40) | no | `lora` |  |
| `hyperparameters` | JSON | no | runtime |  |
| `started_at` | DATETIME | yes |  |  |
| `ended_at` | DATETIME | yes |  |  |
| `duration_seconds` | FLOAT | yes |  |  |
| `final_train_loss` | FLOAT | yes |  |  |
| `final_val_loss` | FLOAT | yes |  |  |
| `best_checkpoint_id` | VARCHAR(40) | yes |  |  |
| `benchmark_summary` | JSON | no | runtime |  |
| `notes` | TEXT | yes |  |  |
| `tags` | JSON | no | runtime |  |
| `is_demo` | BOOLEAN | no | `False` | indexed |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `messages`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `conversation_id` | VARCHAR(40) | no |  | → `conversations.id`, indexed |
| `role` | VARCHAR(16) | no |  |  |
| `content` | TEXT | no | `` |  |
| `model_id` | VARCHAR(40) | yes |  |  |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `prompt_tokens` | INTEGER | no | `0` |  |
| `completion_tokens` | INTEGER | no | `0` |  |
| `total_tokens` | INTEGER | no | `0` |  |
| `tokens_per_sec` | FLOAT | yes |  |  |
| `latency_ms` | FLOAT | yes |  |  |
| `time_to_first_token_ms` | FLOAT | yes |  |  |
| `finish_reason` | VARCHAR(32) | yes |  |  |
| `error` | TEXT | yes |  |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `training_jobs`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | VARCHAR(40) | no | runtime | primary key, indexed |
| `name` | VARCHAR(200) | no |  |  |
| `model_id` | VARCHAR(40) | no |  | → `models.id`, indexed |
| `dataset_id` | VARCHAR(40) | no |  | → `datasets.id`, indexed |
| `experiment_id` | VARCHAR(40) | yes |  | → `experiments.id` |
| `project_id` | VARCHAR(40) | yes |  | → `projects.id` |
| `method` | VARCHAR(40) | no | `lora` |  |
| `status` | VARCHAR(24) | no | `queued` | indexed |
| `provenance` | VARCHAR(24) | no | `simulated` |  |
| `backend` | VARCHAR(40) | no | `simulated` |  |
| `config` | JSON | no | runtime |  |
| `total_steps` | INTEGER | no | `0` |  |
| `current_step` | INTEGER | no | `0` |  |
| `total_epochs` | FLOAT | no | `0.0` |  |
| `current_epoch` | FLOAT | no | `0.0` |  |
| `loss` | FLOAT | yes |  |  |
| `val_loss` | FLOAT | yes |  |  |
| `best_val_loss` | FLOAT | yes |  |  |
| `learning_rate` | FLOAT | yes |  |  |
| `grad_norm` | FLOAT | yes |  |  |
| `tokens_processed` | INTEGER | no | `0` |  |
| `tokens_per_sec` | FLOAT | yes |  |  |
| `samples_per_sec` | FLOAT | yes |  |  |
| `gpu_utilization` | FLOAT | yes |  |  |
| `vram_used_mb` | FLOAT | yes |  |  |
| `eta_seconds` | FLOAT | yes |  |  |
| `started_at` | DATETIME | yes |  |  |
| `ended_at` | DATETIME | yes |  |  |
| `error` | TEXT | yes |  |  |
| `created_at` | DATETIME | no | runtime |  |
| `updated_at` | DATETIME | no | runtime |  |

## `training_metrics`

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no |  | primary key |
| `job_id` | VARCHAR(40) | no |  | → `training_jobs.id` |
| `step` | INTEGER | no |  |  |
| `epoch` | FLOAT | no | `0.0` |  |
| `ts` | DATETIME | no |  |  |
| `loss` | FLOAT | yes |  |  |
| `val_loss` | FLOAT | yes |  |  |
| `learning_rate` | FLOAT | yes |  |  |
| `grad_norm` | FLOAT | yes |  |  |
| `tokens_per_sec` | FLOAT | yes |  |  |
| `samples_per_sec` | FLOAT | yes |  |  |
| `gpu_utilization` | FLOAT | yes |  |  |
| `vram_used_mb` | FLOAT | yes |  |  |

