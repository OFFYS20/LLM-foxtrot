# Extending Foxtrot

Three extension points cover most of what people want to add: a new inference engine, a new
benchmark, and a new training backend. Each is a single class plus one registration line — no API
route or UI component changes.

## Add an inference engine

```python
# backend/app/services/inference/my_engine.py
from collections.abc import AsyncIterator

from app.schemas.chat import ChatMessage, SamplingParams
from app.services.inference.base import EngineInfo, GenerationChunk, InferenceAdapter


class MyEngineAdapter(InferenceAdapter):
    engine = "my_engine"
    provenance = "measured"        # "simulated" only for synthetic output

    @classmethod
    def is_available(cls) -> bool:
        # Return True only when the engine can really serve a request.
        return _my_runtime_is_installed()

    async def load(self) -> None:
        self._handle = my_runtime.load(self.model_record.local_path)
        self._loaded = True

    async def stream(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> AsyncIterator[GenerationChunk]:
        async for piece in self._handle.generate(self.render_prompt(messages), params.max_tokens):
            yield GenerationChunk(text=piece)
        yield GenerationChunk(text="", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        return len(self._handle.tokenize(text))

    def token_count_method(self) -> str:
        return "tokenizer"

    def info(self) -> EngineInfo:
        return EngineInfo(engine=self.engine, available=self.is_available(), provenance="measured")
```

Register it:

```python
# backend/app/services/inference/registry.py
ENGINES["my_engine"] = MyEngineAdapter
```

`complete()`, usage accounting and the streaming SSE endpoint come from the base class. Select it
with `FOXTROT_INFERENCE_ENGINE=my_engine` or per request via `POST /models/{id}/load`.

**Availability must be honest.** `is_available()` gates whether the UI shows the engine as usable;
probing a port is not enough if something else could be listening on it.

## Add a benchmark

```python
# backend/app/services/evaluation/my_suite.py
from app.services.evaluation.base import BenchmarkAdapter, BenchmarkItemSpec, ScoreResult


class MySuiteAdapter(BenchmarkAdapter):
    key = "my_suite"
    label = "My Suite"
    category = "reasoning"
    metric = "accuracy"
    description = "What this suite measures."
    data_source = "local_split"     # bundled_sample | local_split | user_dataset
    official = True                 # True only for the real, complete split

    def items(self, *, limit: int, seed: int = 42) -> list[BenchmarkItemSpec]:
        rows = load_my_split()
        return self.sample(
            [
                BenchmarkItemSpec(index=i, question=r["q"], expected=r["a"], choices=r.get("choices", []))
                for i, r in enumerate(rows)
            ],
            limit,
            seed,
        )

    def score(self, item: BenchmarkItemSpec, response: str) -> ScoreResult:
        correct = self.normalize(response) == self.normalize(item.expected)
        return ScoreResult(correct=correct, score=1.0 if correct else 0.0, method="exact_match")
```

Register it:

```python
# anywhere during startup
from app.services.evaluation.registry import registry
registry.register(MySuiteAdapter)
```

The suite then appears in `GET /benchmarks/suites`, in the Benchmarks picker, and in the
leaderboard.

**Two rules the runner enforces for you.** Set `official=False` unless the adapter really reads the
complete published split — the UI badges everything else as *sample · not official*. And if grading
would require executing model output, return `ScoreResult(..., graded=False)` or a clearly named
proxy method instead: Foxtrot never runs model-generated code.

### Evaluating an arbitrary dataset

No code needed. Import a dataset with question/answer-ish columns (`question`/`answer`,
`instruction`/`output`, or chat `messages`), then run the `custom` suite against it.

### Using a real benchmark split

Drop the split at `data/benchmarks/<suite_key>.jsonl` with one object per line:

```json
{"question": "…", "answer": "…", "choices": ["A", "B", "C", "D"], "category": "biology"}
```

The bundled adapters pick it up automatically and switch to `data_source="local_split"`,
`official=true`.

## Add a training backend

```python
# backend/app/services/training/my_backend.py
from app.services.training.base import StepMetrics, TrainingBackend, TrainingContext


class MyTrainingBackend(TrainingBackend):
    name = "my_backend"
    provenance = "measured"

    @classmethod
    def is_available(cls) -> bool:
        return _my_trainer_installed()

    async def run(self, ctx: TrainingContext) -> dict:
        for step in range(ctx.start_step + 1, ctx.total_steps + 1):
            if ctx.should_stop:
                return {"stopped_at": step}
            await ctx.wait_if_paused()

            loss = my_trainer.step()
            await ctx.report(StepMetrics(step=step, epoch=..., loss=loss, learning_rate=...))

            if step % ctx.config.log_every_steps == 0:
                await ctx.log(f"[TRAIN] Step {step}/{ctx.total_steps}  loss: {loss:.3f}")
            if step % ctx.config.checkpointing.save_every_steps == 0:
                await ctx.checkpoint(step, loss, val_loss=None, is_best=False)

        return {"final_loss": loss}
```

Wire it into selection:

```python
# backend/app/services/training/manager.py → TrainingManager._select_backend / _build_backend
```

The manager owns persistence, event publishing, checkpoint retention, experiment bookkeeping and
failure handling — the backend only reports through `TrainingContext`.

## Frontend: swap the data source

Components never call `fetch`. They use `api`, which implements `DataProvider`
(`frontend/src/lib/api/provider.ts`). Two implementations ship:

* `http` — the FastAPI backend (`lib/api/http.ts`)
* `mock` — client-side fixtures (`lib/api/mock.ts`)

Select with `NEXT_PUBLIC_DATA_PROVIDER`. To target a different backend entirely, write a third
implementation of the same interface and export it from `lib/api/index.ts`; no page changes.

## Frontend: add a chart

Use the chart kit (`frontend/src/components/charts/chart-kit.tsx`) rather than Recharts directly, so
every chart keeps the same conventions: the validated categorical palette in fixed order, one
y-axis per chart, recessive grid, crosshair tooltips, a legend whenever there are two or more
series, and a provenance badge in the frame header.

```tsx
<ChartFrame title="My metric" provenance={job.provenance} series={series}>
  <TimeSeriesChart data={data} xKey="step" series={series} />
</ChartFrame>
```
