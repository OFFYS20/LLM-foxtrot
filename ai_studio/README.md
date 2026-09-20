# AI Studio

A local laboratory for **creating, training, fine-tuning, evaluating, saving and
chatting with language models**. It is not a wrapper around a hosted API: models
are defined, trained and run on your own machine with PyTorch.

You can take raw material — books, PDFs, articles, notes, spreadsheets — turn it
into a training corpus, train a tokenizer and a transformer from scratch, or
fine-tune an existing model with LoRA/QLoRA, then chat with the result, ground it
in your documents with retrieval, and benchmark it.

The whole application is Python. The interface is [Gradio](https://gradio.app);
there is no Node.js, no build step and no JavaScript toolchain.

---

## Contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [GPU setup](#gpu-setup)
5. [Running it](#running-it)
6. [A first end-to-end run](#a-first-end-to-end-run)
7. [The screens](#the-screens)
8. [Importing documents](#importing-documents)
9. [Building datasets](#building-datasets)
10. [Training a tokenizer](#training-a-tokenizer)
11. [Training from scratch](#training-from-scratch)
12. [Fine-tuning, LoRA and QLoRA](#fine-tuning-lora-and-qlora)
13. [Resuming and checkpoints](#resuming-and-checkpoints)
14. [Chat and generation](#chat-and-generation)
15. [Knowledge / RAG](#knowledge--rag)
16. [Benchmarks](#benchmarks)
17. [Saving and exporting models](#saving-and-exporting-models)
18. [Configuration](#configuration)
19. [Storage layout](#storage-layout)
20. [Project structure](#project-structure)
21. [Tests](#tests)
22. [Honesty rules](#honesty-rules)
23. [Licensing](#licensing)
24. [Troubleshooting](#troubleshooting)

---

## What it does

| Area | What is real |
|---|---|
| **Data** | PDF, DOCX, EPUB, HTML, Markdown, TXT, CSV, JSON/JSONL and whole folders are parsed, cleaned and stored. Originals are never modified. |
| **Tokenizers** | BPE, byte-level BPE, WordPiece and Unigram trained with HuggingFace `tokenizers` on your corpus. |
| **Models** | A decoder-only transformer implemented here (RMSNorm/LayerNorm, SwiGLU/GELU, RoPE or learned positions, grouped-query attention, weight tying), plus any causal LM from the Hub. |
| **Training** | An explicit PyTorch loop: forward, backward, gradient clipping, accumulation, scheduling, AMP, evaluation. Pause, resume, stop and checkpoint-on-demand. |
| **Fine-tuning** | Full fine-tuning, instruction tuning, LoRA and QLoRA through PEFT, plus adapter merging. |
| **Inference** | Token-by-token streaming from a custom sampling loop (studio models) or `TextIteratorStreamer` (HF models). |
| **Retrieval** | sentence-transformers embeddings in a FAISS index, with an exact NumPy fallback. |
| **Evaluation** | Nine benchmark suites with per-question results and explicit provenance. |
| **Hardware** | Real NVML/psutil telemetry; anything unmeasurable is shown as *Unavailable*, never invented. |

---

## Requirements

* **Python 3.11 or newer**
* ~2 GB of disk for dependencies, plus whatever your models and datasets need
* A CUDA GPU is optional. Everything runs on CPU — small models (up to roughly
  50M parameters) train in minutes; larger ones are impractical and the app will
  warn you before starting.

---

## Installation

```bash
git clone https://github.com/offys20/llm-foxtrot.git
cd llm-foxtrot/ai_studio

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

To install it as a package instead (this also gives you an `ai-studio` command):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[quantization]"   # bitsandbytes — 8-bit/4-bit, CUDA only
pip install -e ".[dev]"            # pytest
```

Every heavy dependency is optional at runtime. If `pymupdf` is missing you simply
cannot import PDFs, and the app says so — it does not crash. **Settings → Dependencies**
lists what is installed and what each missing package would enable.

---

## GPU setup

The `torch` wheel on PyPI is CPU-only on Linux and Windows. For an NVIDIA GPU,
install the CUDA build **before** the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check what was detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

The **Hardware** screen shows per-GPU utilisation, memory, temperature and power
through NVML. On Apple Silicon, MPS is detected and used where PyTorch supports it.

4-bit/8-bit quantization needs `bitsandbytes` **and** a CUDA GPU. Without both,
QLoRA and quantized export are refused with an explanation — they never silently
fall back to something else.

---

## Running it

From the **repository root** (one level above `ai_studio/`):

```bash
python -m ai_studio.app
```

or, if you installed the package:

```bash
ai-studio
```

Then open <http://127.0.0.1:7860>.

```
--host 0.0.0.0     bind address
--port 7861        port
--share            public Gradio link
--no-browser       do not open a browser window
```

> Run it from the repository root, not from inside `ai_studio/` — `python -m`
> needs the package's parent directory on the path.

---

## A first end-to-end run

This takes a few minutes on a laptop CPU and exercises the whole pipeline.

1. **Data Library → Paste text** (or import a few `.txt` files). A few hundred KB
   is enough to see training work; more is better.
2. **Tokenizer → Train**: pick *byte-level BPE*, vocabulary 4,000, select your
   documents, train. It takes seconds.
3. **Dataset Builder**: mode *raw text*, select the same documents, chunk size
   256 tokens, build. Check the split summary it reports.
4. **Models → Create from scratch**: preset `nano-1m`, your new tokenizer.
   The parameter count shown is computed from the architecture.
5. **Training**: method *Train from scratch*, your model and dataset, batch size
   4, sequence length 128, learning rate 3e-4, 1 epoch. Press **Check
   configuration** (memory preflight), then **Start training**.
6. Watch the live loss chart and console. Pause, resume and save a checkpoint
   on demand.
7. **Checkpoints**: verify a checkpoint, then register it as a model.
8. **Chat**: select the model and talk to it. A model this small produces
   grammar, not knowledge — that is the expected result.
9. **Benchmarks**: run GSM8K on it. A tiny model scores ~0%, and the report says
   whether the official split, a local file or the bundled sample was used.

---

## The screens

| Screen | Purpose |
|---|---|
| **Dashboard** | Models, datasets, running experiments, recent activity, hardware at a glance |
| **Models** | Create from scratch, import from the Hub or a local path, inspect, export, quantize, delete |
| **Data Library** | Import documents, preview cleaning, see the corpus statistics |
| **Dataset Builder** | Turn documents or records into train/validation/test splits |
| **Tokenizer** | Train, inspect and test tokenizers |
| **Training** | Pretraining: from scratch or continued pretraining, with a live run panel |
| **Fine-Tuning** | Full fine-tuning, instruction tuning, LoRA, QLoRA, and adapter merging |
| **Chat** | Streaming conversation with parameters, system presets and a debug panel |
| **Knowledge / RAG** | Build vector indexes over your documents and query them |
| **Benchmarks** | Run suites, read per-question reports, compare models |
| **Experiments** | Every run with its configuration, metrics and history |
| **Checkpoints** | Browse, verify, load, resume from, register or delete |
| **Hardware** | Live CPU/RAM/GPU telemetry |
| **Logs** | Structured application log with level filtering |
| **Settings** | Paths, defaults, dependency status, licence information |

---

## Importing documents

**Data Library → Import.** Supported: `.pdf`, `.docx`, `.epub`, `.html`/`.htm`,
`.md`, `.txt`, `.csv`, `.json`, `.jsonl`, plus pasted text and recursive folder
import.

Two copies are kept, always:

```
storage/documents/original/   ← byte-for-byte as imported, never modified
storage/documents/cleaned/    ← the cleaned text used for training
```

Cleaning is configurable and previewable before it is applied: normalise
whitespace and Unicode, strip page numbers and repeated headers/footers, rejoin
words hyphenated across line breaks, drop boilerplate and empty sections,
de-duplicate repeated blocks, and optionally strip URLs. Code blocks keep their
significant whitespace. The report tells you exactly what each step removed.

Re-running cleaning re-derives the cleaned copy from the original, so no
information is ever lost by experimenting with the settings.

---

## Building datasets

**Dataset Builder.** Choose documents (or import records) and a mode:

| Mode | Shape | Use |
|---|---|---|
| `raw_lm` | `{"text": ...}` | Pretraining and continued pretraining |
| `instruction` | `{"instruction", "input", "output"}` | Instruction tuning |
| `qa` | `{"question", "answer"}` | Question answering |
| `chat` | `{"messages": [{"role", "content"}]}` | Multi-turn conversation |

Structured examples can be imported directly instead: **Import records** takes a
`.jsonl`, `.json` or `.csv` file and keeps each row's fields intact, which is
what instruction tuning needs — flattening instruction pairs into raw text loses
the structure. **Check file** reports how many rows are valid for the chosen
mode and names the problem with each row that is not, before anything is built.

Chunking uses the real tokenizer when one is selected, so "512 tokens" means 512
tokens. Splits are assigned **by document**, so overlapping chunks of the same
source cannot leak between train and test. With fewer than three documents that
is impossible, so it splits by row and says so in the report rather than
pretending the held-out set is clean.

Malformed records are counted, described and skipped — never silently dropped.
The builder shows the row counts, an estimated token count and a preview before
anything is written.

---

## Training a tokenizer

**Tokenizer → Train.** Algorithms: byte-level BPE (recommended — it round-trips
any input losslessly), BPE, WordPiece, Unigram. Set the vocabulary size, minimum
frequency and special tokens.

After training, **Inspect** shows the vocabulary, the special-token IDs and the
learned merges; **Test** tokenizes a string and reports the token count,
compression ratio and whether decoding round-trips exactly.

You can also register a tokenizer from any Hub model to reuse it.

---

## Training from scratch

**Models → Create from scratch** defines the architecture:

* `vocab_size` (from your tokenizer), `hidden_size`, `num_layers`, `num_heads`
* `num_kv_heads` for grouped-query attention
* activation: SwiGLU, GELU, ReLU or SiLU
* normalisation: RMSNorm or LayerNorm
* positions: RoPE (with theta and scaling) or learned
* weight tying, dropout, bias

Presets from `nano-1m` to `huge-1b` are a starting point. The parameter count
and the memory estimate update as you edit, and are computed analytically from
the architecture — they match the instantiated model exactly.

**Training** then runs the loop. Before starting, **Check configuration** runs a
preflight that estimates weights, gradients, optimizer state and activations
against the memory you actually have. If a run cannot fit it is *blocked*, with
concrete suggestions (smaller batch, more accumulation, gradient checkpointing,
shorter sequences, LoRA, a thriftier optimizer). The app will not start a
configuration that is likely to crash the machine.

During a run you get live loss and learning-rate charts, tokens/second, ETA, a
streaming console, and **Pause / Resume / Stop / Save checkpoint**. Stopping is
clean: the run finishes its current step, writes its state and is recorded as
`stopped` — existing checkpoints are never corrupted.

---

## Fine-tuning, LoRA and QLoRA

**Fine-Tuning** covers the methods that start from an existing model:

* **Full fine-tuning** — every weight updates. Needs the most memory.
* **Instruction tuning** — full fine-tuning on an instruction/chat dataset, with
  the prompt masked out of the loss so only the response is learned.
* **LoRA** — small adapters on the attention/MLP projections. *Suggest targets
  for this model* inspects the architecture and proposes the right module names.
* **QLoRA** — LoRA on a 4-bit quantized base. Requires `bitsandbytes` and CUDA;
  without them the run is refused rather than silently downgraded.

A LoRA run saves an **adapter**, not a full model. **Adapter tools** at the
bottom of the screen merges an adapter into a copy of its base weights to
produce a standalone model that can be chatted with, exported or quantized. The
adapter and the base model are left untouched, and the merged model inherits the
base model's licence.

---

## Resuming and checkpoints

A checkpoint is a directory containing model weights (`safetensors`), the
tokenizer, the architecture config, a readable `checkpoint.json` manifest, and
`training_state.pt` with the optimizer state, scheduler state, RNG state, step,
epoch, training config and dataset metadata.

Checkpoints are written **atomically**: everything goes to a staging directory
that is renamed into place only once complete, so an interrupted save can never
leave a half-written checkpoint that looks valid.

* **Checkpoints → Verify** reports what a checkpoint contains and whether it is
  resumable (weights + training state) or only loadable (weights alone).
* **Training → Resume from checkpoint** continues a run with the optimizer
  moments, schedule position and RNG restored.
* Pruning keeps the last *N* and always keeps the best one unless you say otherwise.
* **Register as model** promotes a checkpoint into the model registry.

---

## Chat and generation

**Chat** streams tokens as they are produced. You control temperature, top-p,
top-k, repetition penalty, max new tokens, seed and the system prompt (several
presets, or your own).

The context strategy decides what happens when a conversation outgrows the
model's window:

| Strategy | What it does |
|---|---|
| `trim_oldest` | Drops the oldest turns, always keeping the newest |
| `summarize` | Summarises earlier turns and keeps the summary in the prompt |
| `retrieve` | Keeps the earlier turns most relevant to your current question, in conversation order |
| `new_context` | Starts fresh, keeping only the latest message |

Whatever a strategy does is **reported in the transcript** — how many messages
were left out, summarised or retrieved. Nothing is deleted from the saved
conversation, and history is never discarded silently. Conversations can be
cleared (emptying the messages) or deleted outright from the same screen.

The debug panel shows the exact prompt sent to the model, token counts,
tokens/second, time to first token, the sampling parameters used and any
retrieved context.

### Tools

Tools are explicit, allow-listed Python callables. Each validates its own input,
and a tool that fails returns an error rather than taking the application down.

| Tool | What it does | Confirmation |
|---|---|---|
| `calculator` | Arithmetic, evaluated over an AST — never `eval` | — |
| `document_search` | Keyword search over your Data Library | — |
| `python_sandbox` | A restricted snippet, stdout only | required |
| `web_search` | Searches the web, returns titles, addresses and snippets | required |
| `read_web_page` | Fetches one page and returns its readable text | required |

The sandbox blocks imports, attribute access, dunder names and every call
outside a small allow-list, and stops a snippet that exceeds its time limit.
**Model-generated code never reaches the operating system.**

The two web tools are the only ones that leave this machine, which is why both
are marked as needing confirmation — `registry.call()` refuses them until the
caller passes `confirmed=True`, so the flag is a gate rather than a label. They
speak http and https only, refuse any address on this machine or its local
network (checked again after every redirect, so a public URL cannot bounce into
`localhost`), cap each page at 3 MB and 20 seconds, and follow only the address
they were given — never a link found inside a page.

Search goes through DuckDuckGo's HTML endpoint, with Wikipedia's own search as a
fallback when DuckDuckGo blocks or rate-limits the machine. Neither needs an API
key. If no engine answers, that is reported — no result is ever invented.

**Which models can use these.** Deciding to call a tool — noticing a question
needs a fact it does not have, choosing a query, reading the answer back — is an
ability that appears in large instruction-tuned models. A model trained here, at
1M to a few hundred million parameters, will not do it. These tools are useful
with a capable model you have imported, and to you. Teacher uses the same code
to gather *training material* from the web (`teacher web`), which is a different
job and works with any model.

Pages you fetch are someone else's writing under someone else's terms. A licence
on this software is not permission to train on or redistribute them.

---

## Knowledge / RAG

1. **Knowledge / RAG → Build index**: select documents, an embedding model
   (default `all-MiniLM-L6-v2`), chunk size and overlap.
2. Chunks are embedded with sentence-transformers and stored in a FAISS
   `IndexFlatIP` over normalised vectors (cosine similarity). Without FAISS, an
   exact NumPy search is used instead — same results, slower on large indexes.
3. **Query** returns the matching passages with their similarity scores and
   sources. Overlapping chunks that repeat the same passage are de-duplicated.
4. In **Chat**, attach an index to ground answers. Retrieved passages are shown
   with their sources and scores, and the prompt instructs the model to say when
   the excerpts do not contain the answer.

The first index build downloads the embedding model from the Hub; after that it
works offline.

---

## Benchmarks

Available suites: **MMLU, MMLU-Pro, GSM8K, ARC-Challenge, HellaSwag, TruthfulQA,
Winogrande, HumanEval, MBPP**.

Each run states where its questions came from:

| Provenance | Meaning |
|---|---|
| `official` | The real split, downloaded from the Hub |
| `local` | Your own `storage/benchmarks/<suite>.jsonl` |
| `sample` | A small bundled set in the same format — **for smoke-testing only** |

A sample run is labelled as such everywhere it appears: in the run panel, in the
report, in the comparison table and in the stored record. **A score is never
presented as an official benchmark result unless the official split was actually
used.**

Reports give the overall score, per-category breakdowns, and a per-question table
(prompt, expected answer, model output, pass/fail, latency) that can be filtered
to failures. Comparing models shows each metric separately — accuracy, latency,
tokens/second, per-category scores — rather than collapsing them into one number.

---

## Saving and exporting models

* **Export** copies the model into a Hugging Face-style folder —
  `model.safetensors`, `config.json` and the tokenizer files — plus an
  `ai_studio_export.json` manifest recording the architecture, parameter count,
  declared licence and every file written. The result loads with
  `AutoModelForCausalLM.from_pretrained(...)` for HF models, or
  `TransformerLM.from_pretrained(...)` for studio models.
* **Quantize** (8-bit/4-bit) needs `bitsandbytes` and CUDA. Without them it
  fails with an explanation. It never reports success it did not achieve.
* **GGUF** is *not* bundled: it needs llama.cpp's conversion script. The app
  tells you this and points at `convert_hf_to_gguf.py` instead of pretending.

---

## Configuration

`config.yaml` next to the package holds the defaults — storage paths, training
defaults, chunking, RAG, inference and UI. Any value can be overridden by
environment variable:

```
AISTUDIO_<SECTION>_<FIELD>

AISTUDIO_UI_PORT=7861
AISTUDIO_STORAGE_ROOT=/data/ai-studio
AISTUDIO_TRAINING_DEFAULT_PRECISION=bf16
```

Environment variables win over the file, and the file wins over the built-in
defaults. Relative storage paths resolve next to `config.yaml`, so moving the
project moves its storage with it.

---

## Storage layout

```
storage/
├── ai_studio.db              SQLite: models, datasets, experiments, metrics, …
├── models/                   model weights and configs
├── datasets/                 built splits as JSONL, plus metadata
├── checkpoints/<experiment>/ one directory per checkpoint
├── documents/
│   ├── original/             untouched source files
│   └── cleaned/              cleaned text
├── tokenizers/               trained tokenizers
├── indexes/                  FAISS/NumPy vector stores
├── exports/                  exported models
└── benchmarks/               your own benchmark splits (optional)
```

SQLite is used through a small repository layer with per-thread connections and
WAL enabled. It stores metadata and metrics; weights and datasets live on disk.

---

## Project structure

```
ai_studio/
├── app.py                 entry point and tab wiring only
├── config.yaml
├── ui/                    one module per screen
├── core/                  config, database, logging, errors, paths
├── models/                transformer, tokenizer manager, model manager
├── training/              config, data, trainer, worker, checkpoints, LoRA
├── inference/             model loading, generation, conversations
├── data/                  loaders, cleaning, chunking, dataset builder
├── rag/                   embeddings, vector store, retrieval
├── evaluation/            evaluators, suites, benchmark runner
├── hardware/              CPU/RAM/GPU telemetry
├── tools/                 calculator, document search, Python sandbox, web search
└── tests/
```

`app.py` wires screens together and does nothing else; every screen is its own
module, and no UI module contains training, data or evaluation logic.

---

## Tests

```bash
cd ai_studio
python -m pytest tests/ -q
```

Coverage includes document loading and cleaning, dataset splitting (including
that splits never share a row), tokenizer training and round-tripping, checkpoint
saving and loading (bit-identical weights, resumable optimizer state, atomicity),
configuration validation, RAG retrieval and persistence, training configuration
and preflight, the training loop itself (the loss must actually fall), and the
sandbox's refusal of every escape attempt.

The training tests run real PyTorch steps on a tiny model. If training were
simulated, they would fail.

---

## Honesty rules

These are enforced in the code, not just documented:

* If the UI says training is running, a PyTorch loop is running.
* If a checkpoint is listed, weights exist on disk at that path.
* If a benchmark shows a score, that benchmark was executed, and the provenance
  of its questions is stated.
* If a hardware metric cannot be read, it shows **Unavailable** — never a
  plausible-looking number.
* If an operation needs a package or a GPU you do not have, it is refused with a
  reason, not silently downgraded.
* Original documents are never modified, and cleaned copies are kept separately.
* Conversation history is never dropped without telling you.
* One failed operation reports an error; it does not take the application down.

---

## Licensing

AI Studio itself is under the Apache License 2.0. **That licence covers this application only.**

Model weights, datasets and benchmarks you download carry their own terms —
Llama's community licence, Gemma's terms, CC-BY-NC datasets, benchmark usage
restrictions and so on. The app records the licence a model declares and shows
it in **Models** and **Settings → Licensing**, and marks it *unknown* when the
source does not declare one. It never assumes a licence and never implies that
using this software grants permission to use or redistribute someone else's
weights or data. Check the terms of anything you import before you use or share
it.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ai_studio'`**
Run from the repository root (`python -m ai_studio.app`), not from inside
`ai_studio/`.

**Storage appears in an unexpected place**
Paths are relative to `config.yaml`. Set `AISTUDIO_STORAGE_ROOT` to an absolute
path to be certain.

**"No GPU detected"**
You have the CPU-only PyTorch wheel. See [GPU setup](#gpu-setup).

**Out of memory during training**
Run **Check configuration** first — it usually catches this before the run. The
suggestions it prints (smaller batch, more gradient accumulation, gradient
checkpointing, shorter sequences, LoRA, `adafactor`/`adamw_8bit`) are in order of
impact.

**QLoRA refuses to start**
It needs `bitsandbytes` *and* a CUDA GPU. On CPU, use LoRA in FP32.

**PDF/DOCX/EPUB import unavailable**
Install the matching package (`pymupdf`, `python-docx`, `EbookLib`). Check
**Settings → Dependencies**.

**Training loss is not falling**
On a tiny corpus, first confirm the loss starts near `ln(vocab_size)` — that is
the value of an untrained model, and it means the data pipeline is sound. Then
raise the learning rate or train longer.

**A benchmark says "NOT the official split"**
The official split could not be downloaded, so the bundled sample was used. That
result is a smoke test, not a benchmark score. Put your own split in
`storage/benchmarks/<suite>.jsonl`, or check your network access to the Hub.
