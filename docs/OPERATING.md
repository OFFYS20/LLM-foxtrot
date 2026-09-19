# Operating LLM Foxtrot

**Paste this whole file into ChatGPT, Claude, or any assistant that can run
commands on your machine, then tell it what you want.** It contains everything
needed to drive the software without you touching a terminal.

> Running `python -m teacher guide` prints a shorter version of this with your
> own paths already filled in. This file is the complete one.

---

## Part 0 — For the assistant reading this

You are operating a program on someone's own computer that builds, trains and
runs language models. Everything is local; nothing is sent anywhere except when
the web-search command is used, and that is explained below.

**Talk to them in plain language.** They do not want to read terminal output.
Tell them what happened and what it means.

**Add `--json` to every Teacher command.** You get back exactly one JSON object
and nothing else — no progress bars, no training logs, no prose — on every
path, including failures:

```json
{"ok": true, "command": "teach", "stage": "adapting well", "held_out_loss": 2.41,
 "rounds": 5, "reason": "it stopped improving"}
```

```json
{"ok": false, "command": "show", "error": "There is no model called 'bookbot'."}
```

Read that object. Never invent a result you did not run. Twenty tests hold that
contract in place, so if a command prints nothing, something is genuinely wrong.

**Rules that matter:**

- Ask where their text is before you start. Do not guess a path.
- Start from a pretrained model (`--base`) unless they say they want to watch
  one learn from nothing. The output is far better.
- Use `--until best` rather than picking an epoch count.
- Training takes minutes to hours. Say so before starting a long one.
- If a lesson raises the held-out loss, say so and offer to roll it back.
- Before trying something risky, `branch` the model and train the branch.
- Only go to the web if they asked for it or agreed to it, and say which pages
  you took.
- If a command returns `ok: false`, tell them the error and what to do about
  it. Do not retry the same thing.

---

## Part 1 — Where things are

| | |
|---|---|
| Run commands from | the repository folder (where this file's parent `docs/` lives) |
| Models are stored in | `~/teacher-models` on macOS/Linux, `%USERPROFILE%\teacher-models` on Windows |
| To move that | set the `TEACHER_HOME` environment variable |
| Python | the one in `ai_studio/.venv` if it exists, otherwise the system `python` |

A model is a **folder**. It holds `model.safetensors` (the weights),
`config.json` (the architecture), `tokenizer.json`, `history.json` (every
lesson it has had), and `checkpoints/` (the weights as they were before each
lesson). Copy the folder and you have copied the model.

**First-time setup**, if it has not been done:

```bash
python -m venv ai_studio/.venv
source ai_studio/.venv/bin/activate        # Windows: ai_studio\.venv\Scripts\activate
pip install -r ai_studio/requirements.txt
```

On Windows there is also `Start-Teacher.bat` to double-click, and on
macOS/Linux `./start-teacher.sh`, which do the install and open the window.

---

## Part 2 — The three programs

| Program | What it is | Start it with |
|---|---|---|
| **Teacher** | Make a model, teach it, talk to it. The main thing. | `python -m teacher ui` or the commands below |
| **Bench** | One HTML file that runs a model in a browser, no install | open `web_chat/index.html` |
| **AI Studio** | The full workbench: datasets, tokenizers, LoRA, RAG, benchmarks | `python -m ai_studio.app` |

---

## Part 3 — Teacher, command by command

Every command takes `--json`. Put it before the command name:
`python -m teacher --json teach bookbot ...`

### Making a model

```bash
python -m teacher new NAME --base small --from PATH --until best
```

`--base` starts from a model that already writes English. **This is almost
always what you want.** Shortcuts: `small` (SmolLM2-135M, 135M, fine-tunes on a
CPU), `gpt2` (124M), `medium` (SmolLM2-360M, wants a GPU), `tiny-test` (a stub
for checking plumbing only). Any Hugging Face name works, and so does a local
folder. `python -m teacher --json bases` lists them.

Leave `--base` out to build from scratch instead. Then `--size` applies:

| `--size` | Parameters | What it needs |
|---|---|---|
| `1m` | 1M | minutes on a laptop CPU |
| `10m` | 10M | a few MB of text, still fine on a CPU |
| `100m` | 100M | a GPU and a library's worth of text |
| `200m` | 200M | a GPU with 8GB or so, and a lot of text |
| `500m` | 500M | a GPU with room to spare |
| `1b` | 1B | a serious GPU (24GB+) and gigabytes of text |

The older names `tiny`, `small`, `medium`, `large`, `huge` still work and mean
1m, 10m, 100m, 500m, 1b.

Note: the vocabulary is sized to the corpus, so a small corpus produces a model
somewhat smaller than the tier name. That is deliberate — a vocabulary bigger
than the text can support wastes parameters on embeddings that never get
trained.

Other flags on `new`: `--context N` (context length), `--replace` (overwrite a
model of that name), `--no-teach` (build it but do not train yet),
`--trust-remote-code` (let a base model run its own code — only for repos they
trust).

### Teaching it

```bash
python -m teacher teach NAME --from PATH --until best
```

**Material** — `--from` takes a file *or a folder*, and folders are walked
recursively. It reads `.txt` `.md` `.pdf` `.docx` `.epub` `.html` `.csv`
`.json` `.jsonl` `.tsv` `.rst` `.log` and more. Repeat `--from` to combine
sources. `--text "..."` passes text directly. `--raw` skips the cleaning step.

Junk directories (`.git`, `node_modules`, `.venv`, `__pycache__`) are skipped.
Row files (CSV/JSON/JSONL) are read in full, one record per passage. A very
large single file should be split — the whole file is read into memory.

**How long** — `--until STAGE` teaches in rounds and stops when it gets there
or stops improving. Stages: `words`, `sentences`, `best`. `best` means "until
it stops improving", which is usually what you want. `--epochs N` sets the size
of one round (default 3), `--max-rounds N` caps them (default 20),
`--max-minutes N` puts a clock on the whole thing.

**Speed** — `--batch auto` picks the largest batch that fits in memory and
still leaves enough optimizer steps to learn from. Use it when a GPU is sitting
at half load. `--rate` sets the learning rate (default 3e-4).

**Memory** — `--lora` trains a small adapter instead of every weight, which
lets a bigger base fit on a smaller machine. Pretrained models only; it is
refused on a from-scratch model, with a reason. `--lora-rank N` sets how much
the adapter can change (8 thrifty, 16 default, 64 more capable and costlier).
The adapter is merged back into the weights when the lesson ends.

**Safety** — `--keep N` keeps N saved states behind the model instead of 2.
Each is a full copy of the weights, so raise it before experimenting, not
routinely.

### Talking to it

```bash
python -m teacher ask NAME "once upon a"
```

`--tokens N` (default 120), `--temperature`, `--top-p`, `--top-k`, `--seed N`
to repeat an exact answer. With no prompt it opens a back-and-forth, which is
not useful to an assistant — always pass a prompt.

### Comparing two models

```bash
python -m teacher compare A B --prompt "The keeper" --tokens 60
```

Runs the same prompt through each at the same seed, so a difference in the
output is a difference in the models and not in the dice. Returns each one's
reply, stage and held-out loss.

It also returns `losses_comparable`. **If that is false, do not say one model
scored better.** A loss is an average over a model's vocabulary, and two models
that carve text up differently are not on the same scale. Judge by reading.

### Saved states, going back, and branching

```bash
python -m teacher checkpoints NAME            # what is saved
python -m teacher rollback NAME               # undo the last lesson
python -m teacher rollback NAME --to STAMP    # go back to a particular one
python -m teacher branch NAME NEW --at STAMP  # carry on from one, safely
```

Teacher copies the weights aside before every lesson. `rollback` restores one
over the current weights. `branch` copies one into a **new model**, leaving the
original completely untouched — this is how to try something risky. Train the
branch, then `compare` the two and keep the better.

### Material from the web

```bash
python -m teacher web "victorian lighthouses" --list      # look, download nothing
python -m teacher web "victorian lighthouses" --results 8 # gather and keep
python -m teacher web --url https://... -o ./material     # a page you know
```

Searches DuckDuckGo, falling back to Wikipedia's own search when DuckDuckGo
rate-limits the machine. If neither answers it says so rather than returning
nothing quietly. Every page is saved as a text file headed with its address and
the date it was read.

`--web "something"` on `new` and `teach` does gather-and-train in one step.

**Tell them which pages you took.** Those pages are someone else's writing
under someone else's terms. Five to ten pages is a normal run; do not pull
hundreds. Pages run 10,000–34,000 characters each, so a five-page run is
roughly 50,000–150,000 characters — plenty to fine-tune a pretrained model, and
nowhere near enough to build one from scratch. **Always pair `--web` with
`--base`.**

### Looking at what exists

```bash
python -m teacher list            # every model, with size and lessons
python -m teacher show NAME       # architecture, lesson history, where it stands
python -m teacher bases           # pretrained models worth starting from
```

### Getting a model out

```bash
python -m teacher pack NAME -o DIR
```

Copies the three files the Bench web page needs. **Only works for models built
from scratch** — Bench implements that architecture in JavaScript. A pretrained
model is refused with an explanation; chat with it in Teacher instead.

### Deleting

```bash
python -m teacher forget NAME --yes
```

The `--yes` is required. This cannot be undone.

---

## Part 4 — Reading the results

Every lesson is measured on text the model never trained on — the last 5% of
the material, held back. That number is `held_out_loss`. Lower is better.

**Falling between lessons** means it is still learning. **Flat** means it has
stopped, and more epochs will not help — it needs more material. Teacher says
this explicitly when it detects it.

`stage` is where it stands. For a model built from scratch, the baseline is a
model still guessing uniformly:

> barely started → learning the alphabet → learning words → learning sentences
> → has the shape of your text

For one started from a pretrained model, the baseline is its own first lesson,
since it could already write before you began:

> barely moved → picking up your material → adapting well → closely fitted

"barely moved" after one pass over a couple of hundred thousand characters is
the *expected* place to be, not a failure. Keep teaching, or gather more.

**Do not compare held-out losses between models unless `compare` says they are
comparable.**

---

## Part 5 — What to expect, honestly

A small model learns the *shape* of text: sentence structure, punctuation, the
rhythm of the source. Starting from a pretrained model gets fluent sentences in
the register of their material.

**Neither gives something that knows facts or answers questions reliably.**
That needs orders of magnitude more parameters, text and compute than a home
machine has. Say this plainly if they expect a chatbot.

Measured, on the same six Wikipedia articles (206,000 characters) on a CPU:

| | Result |
|---|---|
| From scratch, 1m | 7.32 → 6.18 over 7 rounds, then stopped improving: *"barely started"* |
| `--base small` (135M) | held-out 2.60 after one pass, 4m 36s: *"barely moved"* |

From scratch has to learn the language *and* the material. Below a few hundred
KB it never gets there.

Nothing in this software pretends otherwise. If a benchmark shows a score, the
benchmark ran. If a checkpoint is listed, the weights are on disk. If a
hardware reading is unavailable, it says *Unavailable* rather than inventing a
number.

---

## Part 6 — The window, for when they want to do it themselves

```bash
python -m teacher ui
```

Opens at `http://127.0.0.1:7861`. Four tabs — **Make**, **Teach**, **Chat**,
**Keep** — and a shared **Text** panel underneath for bringing in material
(files, a folder, pasted text, or *…or fetch it from the web*). **Keep** holds
saved states, rollback, branching, copying files for Bench, and deleting.

Flags: `--port`, `--host`, `--share` (public link), `--no-browser`.

The window and the commands share one store: a model made in one is visible in
the other immediately.

---

## Part 7 — Bench, the browser page

`web_chat/index.html` is a single file with no install and no server. Open it,
drop in a model folder (the output of `teacher pack`), and talk to the model —
the weights are read, run and sampled inside the page.

It implements the from-scratch architecture in JavaScript: RMSNorm/LayerNorm,
SwiGLU/GELU/SiLU/ReLU, RoPE or learned positions, grouped-query attention, a KV
cache. Its output matches PyTorch to six decimal places on the same input. It
also shows the five tokens the model weighed for each one it wrote.

Only models built from scratch run here.

---

## Part 8 — AI Studio, the full workbench

```bash
python -m ai_studio.app
```

Fifteen screens: dashboard, data library, dataset builder, tokenizer trainer,
models, training, fine-tuning (LoRA/QLoRA), chat, knowledge/RAG, benchmarks,
experiments, checkpoints, hardware, logs, settings. Flags: `--host`, `--port`,
`--share`, `--no-browser`.

Use this when they want every dial. Teacher is the same machinery with the
dials preset.

**Its tools** (what a capable chat model can call): `calculator`,
`document_search`, `python_sandbox`, `web_search`, `read_web_page`. The last
three need confirmation before they run, and the registry enforces that rather
than just displaying it. The Python sandbox blocks imports, attribute access,
dunder names and everything outside a small allow-list — model-generated code
never reaches the operating system.

A model trained here will not use those tools. Deciding to call a tool is an
ability of far larger instruction-tuned models.

---

## Part 9 — When something goes wrong

| Symptom | What it means |
|---|---|
| `TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'` | An old copy on Gradio 6. `git pull`; it works on both 5 and 6 now. |
| `ModuleNotFoundError: No module named 'teacher'` | Not in the repository folder, or the virtual environment is not active. |
| "This lesson would not fit in memory" | The preflight check refused it *before* starting. Follow its suggestions: smaller batch, `--lora`, a smaller size. |
| "Only N characters of material — too little" | Under 2,000 characters. Add more. |
| "Nothing readable was found" | `--from` pointed somewhere with no supported files. |
| "The search found nothing" | Both search engines refused or rate-limited. Wait a minute, or name pages with `--url`. |
| Output is garbled, with `�` characters | Undertrained. It is picking half of a multi-byte letter. Teach it more. |
| GPU sitting at half load | Use `--batch auto`. |
| Killed with no message during training | The machine ran out of memory. Lower `--batch`, or use `--lora`. |

Training can be interrupted with Ctrl-C at any point. Every round is saved, so
the last completed round is on disk and nothing is corrupted.

---

## Part 10 — Licensing

The software is MIT. **That covers this application only.** Model weights,
datasets, benchmark splits and web pages carry their own terms — a licence here
is not permission to use or redistribute someone else's work. Every page
fetched from the web is saved with its address and the date so the provenance
can be checked later. That judgement is theirs to make, per source.
