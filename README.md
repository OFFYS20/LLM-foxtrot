# Foxtrot

A local platform for experimenting with, training, evaluating and chatting with language models —
a model catalog, dataset manager, training runner, inference playground, benchmark suite and
experiment tracker behind one API.

Foxtrot runs on a laptop with no GPU and no weights on disk (demo mode), and on a real workstation
with PyTorch and NVIDIA telemetry. **Nothing produced by a simulated backend is ever reported as a
measured result**: every run carries a `provenance` field (`measured` | `simulated` | `imported`)
that the UI badges on every chart, table and score.

```
┌──────────────┐    HTTP + WebSocket/SSE    ┌───────────────────────────────────┐
│  Next.js UI  │ ─────────────────────────► │  FastAPI                          │
│  (frontend)  │ ◄───────────────────────── │   ├── training engine (jobs)      │
└──────────────┘      live metrics          │   ├── inference adapters          │
                                            │   ├── evaluation adapters         │
                                            │   ├── hardware monitor (NVML)     │
                                            │   └── SQLAlchemy  ── SQLite/PG    │
                                            └───────────────────────────────────┘
```

---

## Quick start

Two terminals, about a minute.

### 1 · Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional — defaults work
python run.py                                          # http://localhost:8000
```

On first start Foxtrot creates `data/foxtrot.db`, detects your hardware, and (in demo mode) seeds a
set of clearly-labelled demo records so every screen is explorable.

* API docs: <http://localhost:8000/docs>
* Health: <http://localhost:8000/health>

### 2 · Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local                       # optional — defaults work
npm run dev                                            # http://localhost:3000
```

### Optional: real training and inference

The ML stack is optional and imported lazily — install only what your machine supports:

```bash
cd backend
pip install -r requirements-optional.txt               # torch, transformers, peft, datasets, …
# CUDA users also get GPU telemetry from nvidia-ml-py; QLoRA additionally needs bitsandbytes
```

Then point the backend at a real engine:

```bash
FOXTROT_INFERENCE_ENGINE=transformers FOXTROT_DEMO_MODE=false python run.py
```

Without a GPU the UI says so explicitly (**CPU MODE** in the top bar) rather than pretending.

### No backend at all

The frontend can run entirely on client-side fixtures:

```bash
NEXT_PUBLIC_DATA_PROVIDER=mock npm run dev
```

Same components, same types — only the provider changes.

---

## What each screen does

| Screen | What it is for |
|---|---|
| **Dashboard** | Active model, running job, loss/LR/throughput/GPU charts, recent experiments, latest scores |
| **Models** | Import from Hugging Face / local path, inspect metadata, tokenizer, checkpoints; load, clone, export, delete |
| **Training** | Configure a run (visual form *and* raw JSON/YAML), then watch it step: metrics, six live charts, terminal console, pause/resume/stop/checkpoint |
| **Datasets** | Import JSON/JSONL/CSV/TXT/Parquet/HF datasets, preview rows, validate against a template, see token accounting |
| **Chat** | Streaming conversation UI with full sampling controls, saved conversations, message edit/delete/regenerate and a token/latency debug panel |
| **Playground** | One prompt against 2–4 models side by side, with latency, throughput, token counts, memory and voting |
| **Benchmarks** | Suite catalog, run configuration, live progress, per-item results with prompt/response inspection |
| **Evaluations** | Leaderboard across everything actually run here, plus multi-model comparison (table, bars, radar) |
| **Experiments** | Every run's configuration and outcome, duplication, and side-by-side comparison with differing hyperparameters highlighted |
| **Checkpoints** | Browse every checkpoint; load as a model, evaluate, chat, export, delete |
| **Hardware** | GPU/CPU/RAM/disk telemetry with history, multi-GPU aware |
| **Logs** | Structured, filterable log history plus a live stream |
| **Settings** | Deployment wiring, engine availability, capabilities, demo-data controls |

---

## Architecture

```
backend/
  app/
    main.py                 FastAPI app, lifespan, error handlers
    config.py               settings (env-driven)
    core/                   errors, event bus, logging, path safety
    db/                     SQLAlchemy base, session, models/
    schemas/                Pydantic request/response contracts
    api/routes/             models, datasets, training, chat, playground,
                            benchmarks, evaluations, experiments, checkpoints,
                            hardware, logs, system, realtime
    services/
      training/             backend contract, simulated trainer, torch trainer, job manager
      inference/            adapter contract, demo, transformers, ollama, llama.cpp, vLLM
      evaluation/           benchmark adapter contract, suites, custom datasets, runner
      datasets/             format readers, templates, validation, analysis
      hardware/             NVML provider, simulated provider, sampler
      demo/                 demo-data seeding
  tests/                    API smoke, datasets, training lifecycle, chat/benchmarks

frontend/
  src/
    app/                    one route per sidebar section
    components/ui/          primitives (button, card, table, dialog, select, …)
    components/charts/      chart kit — validated palette, shared conventions
    components/common/      provenance badges, stat tiles, states
    components/layout/      sidebar, top bar, workspace + inspector shell
    lib/api/                DataProvider contract + http and mock implementations
    lib/hooks/              React Query hooks, live-training + realtime hooks
```

### The three engine abstractions

Each one is a small interface with interchangeable implementations, so adding an engine never
touches the API layer or the UI.

**Inference** — `app/services/inference/base.py`

```python
class InferenceAdapter(ABC):
    @classmethod
    def is_available(cls) -> bool: ...
    async def load(self) -> None: ...
    def stream(self, messages, params) -> AsyncIterator[GenerationChunk]: ...
    async def complete(self, messages, params) -> GenerationResult: ...
    def count_tokens(self, text: str) -> int: ...
```

Shipped: `demo` (synthetic, always available), `transformers`, `ollama`, `llamacpp`, `vllm`.

**Training** — `app/services/training/base.py`

```python
class TrainingBackend(ABC):
    async def run(self, ctx: TrainingContext) -> dict: ...
```

`TrainingContext` carries pause/stop/checkpoint events and the reporting callbacks; the backend
never touches the database or the event bus. Shipped: `simulated`, `torch` (Transformers + PEFT).

**Evaluation** — `app/services/evaluation/base.py`

```python
class BenchmarkAdapter(ABC):
    def items(self, *, limit: int, seed: int) -> list[BenchmarkItemSpec]: ...
    def build_prompt(self, item, shots) -> str: ...
    def score(self, item, response) -> ScoreResult: ...
```

Shipped: MMLU, MMLU-Pro, GSM8K, ARC, HellaSwag, TruthfulQA, Winogrande, BBH, HumanEval, MBPP,
long-context retrieval, instruction following, safety judgement, and any imported dataset
(`custom`). Register your own with `registry.register(MyAdapter)`.

### Realtime

An in-process event bus fans out to WebSocket (`/ws`) and SSE (`/api/events/stream`). Topics:
`training.metric.<job_id>`, `training.status.<job_id>`, `training.log.<job_id>`,
`training.checkpoint`, `hardware.sample`, `benchmark.progress.<run_id>`, `log.entry`,
`model.status`. The UI merges streamed points into the React Query cache, so a running job redraws
at step cadence without refetching.

### Database

SQLAlchemy 2.0, SQLite by default. No SQLite-specific types are used, so PostgreSQL is a URL change:

```bash
FOXTROT_DATABASE_URL=postgresql+psycopg://foxtrot:foxtrot@localhost:5432/foxtrot
```

Tables: `projects`, `models`, `datasets`, `training_jobs`, `training_metrics`, `experiments`,
`checkpoints`, `benchmark_runs`, `benchmark_items`, `model_comparisons`, `conversations`,
`messages`, `playground_comparisons`, `log_entries`, `hardware_samples`, `settings`.
Full schema: [`docs/SCHEMA.md`](docs/SCHEMA.md).

---

## API

Every route is available under `/api/...` and, for convenience, at the root path too.
Full interactive reference at `/docs`.

```
GET    /system                      deployment info + capability flags
GET    /models                      POST /models/import          GET /models/{id}
POST   /models/{id}/load            POST /models/{id}/clone      POST /models/{id}/export
GET    /datasets                    POST /datasets/import        POST /datasets/upload
GET    /datasets/{id}/preview       POST /datasets/{id}/validate GET  /datasets/templates
POST   /training/jobs               GET  /training/jobs          GET  /training/jobs/{id}
POST   /training/jobs/{id}/pause    POST /training/jobs/{id}/resume
POST   /training/jobs/{id}/stop     POST /training/jobs/{id}/checkpoint
POST   /training/validate           POST /training/config/parse  (JSON or YAML)
POST   /chat/completions            (SSE streaming or buffered)
GET    /conversations               PATCH /messages/{id}         DELETE /messages/{id}
POST   /playground/run              POST /playground/comparisons/{id}/vote
GET    /benchmarks/suites           POST /benchmarks/run
GET    /benchmarks/results          GET  /benchmarks/results/{id}?only_incorrect=true
GET    /evaluations/leaderboard     POST /evaluations/compare
GET    /experiments                 POST /experiments/{id}/duplicate  POST /experiments/compare
GET    /checkpoints                 POST /checkpoints/{id}/load  POST /checkpoints/{id}/evaluate
GET    /hardware                    GET  /hardware/history
GET    /logs
WS     /ws                          GET  /api/events/stream
```

---

## Safety and reliability

* **No arbitrary code execution.** Model-generated code is never run. The HumanEval/MBPP adapters
  grade structurally and report `pass@1: n/a`, because execution-based pass@1 needs a sandbox this
  platform does not provide.
* **Path traversal is blocked.** Every user-supplied path is resolved and checked against the data
  root (`app/core/paths.py`); uploads are name-sanitised, suffix-checked and size-capped.
* **Training parameters are validated before execution.** `TrainingConfig` rejects incoherent
  combinations (e.g. full fine-tuning on INT4 weights) and returns non-fatal warnings (CPU-only,
  VRAM headroom, oversized effective batch) before a job starts.
* **CUDA OOM is handled cleanly.** Translated into a typed 507 response, the cache is released and
  the job is marked failed with an actionable message — the worker does not die.
* **Jobs outlive requests.** Training and benchmark runs execute on background tasks; the HTTP call
  that starts them returns immediately, and a server restart marks interrupted runs as stopped
  rather than silently resurrecting them.
* **Results are never invented.** Simulated runs are flagged end to end; bundled benchmark samples
  are labelled "sample · not official" everywhere they appear.

## Demo mode

`FOXTROT_DEMO_MODE=true` (default) enables the simulated training and inference backends and, with
`FOXTROT_SEED_DEMO_DATA=true`, seeds demo models, datasets, experiments, checkpoints, benchmark
runs and conversations. All of it is flagged `is_demo` / `provenance="simulated"`, badged in the UI,
and removable from **Settings → Demo data** without touching anything produced by a real run.

## Development

```bash
# backend
cd backend
pip install -r requirements-dev.txt
pytest -q            # 26 tests
ruff check app tests
ruff format app tests

# frontend
cd frontend
npm run typecheck
npm run lint
npm run build
```

## Configuration

See [`.env.example`](.env.example) for the full list. The essentials:

| Variable | Default | Meaning |
|---|---|---|
| `FOXTROT_DATABASE_URL` | `sqlite:///./data/foxtrot.db` | SQLite or PostgreSQL |
| `FOXTROT_DATA_DIR` | `./data` | Root for models, datasets, checkpoints, uploads, exports |
| `FOXTROT_DEMO_MODE` | `true` | Simulated backends + demo seeding |
| `FOXTROT_INFERENCE_ENGINE` | `demo` | `demo` / `transformers` / `llamacpp` / `vllm` / `ollama` |
| `FOXTROT_HARDWARE_PROVIDER` | `auto` | `auto` / `nvml` / `simulated` |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Backend the UI talks to |
| `NEXT_PUBLIC_DATA_PROVIDER` | `http` | `http` or `mock` |

## License

Apache 2.0 — see [LICENSE](LICENSE).
