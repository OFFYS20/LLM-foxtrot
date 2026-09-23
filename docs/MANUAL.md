# The Teacher manual

How every part of Teacher works, and what to do when one of them breaks.

The README says what Teacher can do. This says **how** it does it — the real
mechanism, the real numbers, where in the code it lives — so that when
something goes wrong you can see which part went wrong and why, and fix it
rather than guess.

**If something has just broken,** go to
[Part 26 — When something breaks](#part-26--when-something-breaks). Every
message Teacher can stop with is there, word for word, with what it means and
what to do. A test keeps that list complete: adding a message to the code
without explaining it here fails the build.

**If you want to understand a stage,** each part below covers one, in the order
a model goes through them: material in, a model made, lessons taught, the model
used.

## Contents

- [Part 1 — The shape of it](#part-1--the-shape-of-it)
- [Part 2 — Installing and starting](#part-2--installing-and-starting)
- [Part 3 — Where everything lives](#part-3--where-everything-lives)
- [Part 4 — Material: reading, cleaning, repeats](#part-4--material-reading-cleaning-repeats)
- [Part 5 — Material from the web](#part-5--material-from-the-web)
- [Part 6 — Making a model from scratch](#part-6--making-a-model-from-scratch)
- [Part 7 — Adopting a pretrained model](#part-7--adopting-a-pretrained-model)
- [Part 8 — A lesson, step by step](#part-8--a-lesson-step-by-step)
- [Part 9 — Memory: the check before every lesson](#part-9--memory-the-check-before-every-lesson)
- [Part 10 — Batch size](#part-10--batch-size)
- [Part 11 — Learning rates](#part-11--learning-rates)
- [Part 12 — Teaching until it is done](#part-12--teaching-until-it-is-done)
- [Part 13 — Teaching it to answer](#part-13--teaching-it-to-answer)
- [Part 14 — LoRA](#part-14--lora)
- [Part 15 — More than one GPU](#part-15--more-than-one-gpu)
- [Part 16 — Crashes and resuming](#part-16--crashes-and-resuming)
- [Part 17 — Saved states, rollback and branches](#part-17--saved-states-rollback-and-branches)
- [Part 18 — Reading the numbers](#part-18--reading-the-numbers)
- [Part 19 — Talking to it](#part-19--talking-to-it)
- [Part 20 — Benchmarks](#part-20--benchmarks)
- [Part 21 — Model cards](#part-21--model-cards)
- [Part 22 — GGUF export](#part-22--gguf-export)
- [Part 23 — Serving it to other programs](#part-23--serving-it-to-other-programs)
- [Part 24 — The window](#part-24--the-window)
- [Part 25 — Speed: what the training loop does to be fast](#part-25--speed-what-the-training-loop-does-to-be-fast)
- [Part 26 — When something breaks](#part-26--when-something-breaks)

---

## Part 1 — The shape of it

Teacher is a thin layer over AI Studio's machinery. The model, the training
loop, the data loaders, the benchmark suites and the memory estimate are AI
Studio's, tested there and reused; Teacher adds the short path from "a folder
of text" to "a model that has learned it", and a record of everything done to
each model.

```
material ──► model ──► lessons ──► using it
 Part 4-5    Part 6-7   Part 8-17   Part 19-23
```

| Module | What it does |
|---|---|
| `teacher/__main__.py` | The command line: parses arguments, prints, and turns every result into one JSON object with `--json` |
| `teacher/ui.py` | The window: the same functions, behind buttons |
| `teacher/material.py` | Reads files and folders into one text, cleaned, with repeats dropped |
| `teacher/websearch.py` | Searches the web and saves pages as material |
| `teacher/lessons.py` | Builds, adopts and teaches models; judges how far along they are |
| `teacher/workspace.py` | Where models live; their record; saved states, rollback, branches, lineage |
| `teacher/interrupted.py` | What a lesson writes down as it runs, so a crash can be resumed |
| `teacher/answers.py` | Question-and-answer pairs: reading them, templates, masking |
| `teacher/generation.py` | Loading a model once and generating from it |
| `teacher/exams.py` | Benchmarks, with the chance rate and the noise line |
| `teacher/card.py` | Model cards, from the record |
| `teacher/export.py` | GGUF, through llama.cpp's converter |
| `teacher/serve.py` | The OpenAI-style API |
| `ai_studio/models/transformer.py` | The architecture built from scratch, and the search that sizes it |
| `ai_studio/training/trainer.py` | The training loop |
| `ai_studio/training/config.py` | Training settings, the memory estimate, the batch finder |
| `ai_studio/training/distributed.py` | Several GPUs |

Every command is in `teacher/__main__.py` as a `cmd_*` function; each does
its work through the modules above, which is why the window and the command
line always produce the same model.

| Command | Part |
|---|---|
| `teacher new NAME` — make a model and give it its first lesson | 6, 7, 8 |
| `teacher teach NAME` — another lesson | 8–14 |
| `teacher resume NAME` — carry on an interrupted lesson | 16 |
| `teacher ask NAME "…"` — talk to it | 19 |
| `teacher compare A B` — one prompt through several models | 18 |
| `teacher test NAME SUITE` — a benchmark | 20 |
| `teacher card NAME` — its model card | 21 |
| `teacher export NAME` — GGUF | 22 |
| `teacher serve NAME` — the API | 23 |
| `teacher pack NAME` — the three files the Bench page needs | 3 |
| `teacher checkpoints NAME`, `teacher rollback NAME`, `teacher branch NAME NEW` — saved states | 17 |
| `teacher list`, `teacher show NAME` — what exists, and one model's record | 3 |
| `teacher web "…"` — material from the web | 5 |
| `teacher bases` — pretrained models worth starting from | 7 |
| `teacher forget NAME --yes` — delete a model | 3 |
| `teacher ui` — the window | 24 |
| `teacher guide` — instructions to paste into an assistant | — |

Each is `python -m teacher COMMAND`; `--json` before the command turns its
result into one JSON object, on success and on failure alike.

---

## Part 2 — Installing and starting

Teacher runs on Python 3.10 or newer with AI Studio's requirements:

```bash
pip install -r ai_studio/requirements.txt
python -m teacher ui          # the window, at http://127.0.0.1:7861
python -m teacher --help      # the commands
```

**PyTorch decides whether a GPU is used.** The default PyTorch wheel on PyPI is
CPU-only on Windows and Linux. For an NVIDIA GPU install the CUDA build *first*:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check it took: `python -c "import torch; print(torch.cuda.is_available())"`
must print `True`. If it prints `False`, every lesson runs on the CPU — slowly,
but correctly — and `--device cuda` stops with a message saying no GPU is
visible.

**Gradio 5 and 6 both work.** Gradio 6 removed or moved several things Gradio 5
required (`Chatbot(type=...)`, `show_copy_button`, `show_api`, styling on
`Blocks`), and refuses a slider whose two ends are the same number.
`ai_studio/core/gradio_compat.py` builds each component the way the installed
version wants, and the GPU control is a slider only on a machine with two GPUs
or more. An error from inside Gradio at startup means an older copy of Teacher;
update it.

**On Windows**, run commands from the repository folder in a Command Prompt or
PowerShell: `cd C:\Users\you\LLM-foxtrot` then `python -m teacher ui`.

---

## Part 3 — Where everything lives

Models live in `~/teacher-models`, or wherever the `TEACHER_HOME` environment
variable points. One folder per model, named after it (lower case; anything
that is not a letter, digit, `-` or `_` becomes `-`):

```
~/teacher-models/bookbot/
├── model.safetensors     the weights
├── config.json           the architecture
├── tokenizer.json        the tokenizer
├── history.json          the record: what it was made from, every lesson, every result
├── README.md             its model card, once `teacher card` has written one
├── checkpoints/
│   └── before-20260919-172927/   the weights as they were before a lesson
└── .interrupted/         only while a lesson is unfinished — what `resume` needs
```

A pretrained model's folder also holds the extra files its tokenizer came
with (`tokenizer_config.json`, `generation_config.json` and so on). A model is
complete when it has the first three files; one that is missing any of them
is reported as incomplete rather than loaded.

### The record: `history.json`

Everything Teacher knows about a model's past is in this one file, and nothing
already in it is rewritten — lessons are appended, never edited. The fields
that matter:

| Field | What it holds |
|---|---|
| `made` | `from scratch` or `adopted` |
| `base_repo`, `base_license` | For an adopted model: where it came from, and its licence as its authors declare it |
| `origin_weights` | The stamp of the weights as first made — where every lineage walk ends |
| `lessons` | One entry per lesson (below) |
| `base_loss` | For an adopted model: its held-out loss before its first lesson |
| `answer_style` | The template it was taught to answer in, if any |
| `rollbacks` | Every rollback: when, which saved state, the weights it restored |
| `exams` | Every benchmark result, and the weights it was taken on |
| `exports` | Every GGUF export: file name, precision, size, SHA-256 |
| `branched_from`, `branched_at`, `branched_from_history` | For a branch: its source, and the source's lessons |
| `replaced` | The record of a model that used to occupy this folder, kept apart |

Each lesson records its sources, how much it read, epochs, steps, learning rate
and where that rate came from, batch size, the held-out loss **before** and
**after** (and the lowest seen on the way), what the held-out loss was
measured on, the device, and the stamps of the weights it started from and
saved.

**A weights stamp** is the size and modification time of `model.safetensors`,
read with one `stat` call instead of hashing gigabytes. Every copy Teacher
makes — a saved state, a rollback, a branch — preserves the file's time, so the
same weights carry the same stamp wherever they are put back. That is what
lets the record answer "which lessons are in the weights on disk now?"
(Part 17). Copying a model folder by hand with a tool that resets file times
breaks the chain; nothing else does.

**What is safe to delete by hand:** `checkpoints/` (you lose the ability to roll
back), `.interrupted/` (you lose the ability to resume), `README.md` (it can be
written again). Not the first three files, and not `history.json` unless you
want Teacher to forget everything about the model's past.

---

## Part 4 — Material: reading, cleaning, repeats

`teacher/material.py`, `gather()`.

**What is read.** `--from` takes files and folders, repeatedly; folders are
walked recursively and in sorted order. These file types are read: `.txt`
`.text` `.md` `.markdown` `.mdx` `.rst` `.log` `.pdf` `.docx` `.epub` `.html`
`.htm` `.xhtml` `.csv` `.tsv` `.json` `.jsonl` `.ndjson`. Anything else is
passed over. These folders are skipped wherever they appear: `.git`,
`__pycache__`, `.venv`, `node_modules`, `.idea`, `storage`. `--text "..."`
adds text typed on the command line.

**Row files are read in full.** A CSV or JSON file shown in AI Studio's preview
stops at 2,000 rows; for training, every row is read. A file that still could
not be read in full is reported as read only in part, with how many rows were
left out — never silently cut short.

**Cleaning** (`ai_studio/data/preprocessing.py`) removes what is not writing:
page numbers, headers and footers repeated on every page, hyphens splitting a
word across a line break, runs of blank lines, empty sections, and repeated
paragraphs inside a document. What applies depends on the kind of file: a PDF
gets all of it; HTML and EPUB lose their tags but keep their numbers; Markdown
keeps its headings and anything that looks like a page number; code blocks are
kept as they are. `--raw` skips cleaning, for text that is exactly what you want
already.

**Repeats across sources are dropped.** The same book in two folders, or a
chapter that is also on a saved web page, would teach the model to recite it —
and the held-out loss would fall while the model got worse. Any passage of
**200 characters or more** that has already appeared in an earlier source is
dropped, and the count is reported. Shorter passages repeat for honest
reasons — a heading, a name, a refrain — and are kept. `--keep-duplicates`
turns this off.

**How much is needed.** A lesson needs at least **2,000** characters; a model
built from scratch needs far more to learn anything useful (Part 6). One
lesson's material is held in memory whole, so split a single file of many
gigabytes into several.

**What goes wrong here:**

- *Nothing readable was found* — the path is wrong, or holds only file types
  that are not read. `teacher` prints what it skipped and why.
- *Only N characters* — too little. Add more.
- A file shows as *empty after cleaning* — it was all page furniture, or it is
  a scanned PDF with no text layer. Run OCR on it first; Teacher does not.
- *Read only in part* — a row file too big to read whole. Split it.

---

## Part 5 — Material from the web

`teacher/websearch.py` over `ai_studio/data/web.py`.

`teacher web "QUERY"` searches, reads each result, and saves every page as its
own text file in a new folder under `TEACHER_HOME/web-material/`. `--web` on
`new` and `teach` does the same and teaches from the folder in one go.

**The search** posts the query to DuckDuckGo's HTML endpoint (a plain GET gets
its home page back), and falls back to Wikipedia's search API when that returns
nothing. Up to 25 results can be asked for; five is the default.

**Reading a page** fetches it with a 20-second timeout and at most 3 MB, strips
scripts, styles, navigation, headers, footers, asides and forms, and keeps the
text. A page with under 200 characters of text left is skipped as unreadable.
There is a one-second pause between pages.

**What it refuses.** Only `http` and `https`. No address on this machine or its
network — `localhost`, `*.local`, `*.internal`, and any private, loopback,
link-local, reserved or multicast IP — checked on the address as written, on
what the name resolves to, and again on every redirect (at most five). A page
that tries to redirect somewhere private is refused at that redirect.

**What is saved.** Each file starts with three lines — the page's title,
`Source: <address>`, `Retrieved: <date>` — then the text. Those lines are how a
model card later lists where web material came from, with its date (Part 21).
Web pages are someone else's writing under someone else's terms; the address is
kept for that reason.

**How much is enough.** Pages run about 10,000–34,000 characters each, so five
pages is roughly 50,000–150,000 characters. That is enough to fine-tune a
pretrained model and far too little to build one from scratch.

**What goes wrong here:**

- *Nothing readable came back* — every result failed; the reasons are listed.
  Search for something else, or give addresses directly with `--url`.
- A page is *refused* — its address, or a redirect, points inside your network.
  That is deliberate and cannot be turned off.
- Searches return nothing at all — you are offline, or behind a proxy that
  blocks DuckDuckGo and Wikipedia.

---

## Part 6 — Making a model from scratch

`teacher/lessons.py`, `create()`; `ai_studio/models/transformer.py`.

**The architecture** is a decoder-only transformer of the kind every current
small language model uses: RMSNorm, SwiGLU feed-forward layers, rotary position
embeddings, grouped-query attention, a key-value cache for generation, and
input and output embeddings tied together.

**The tokenizer is trained first**, on your material: byte-level BPE, which
round-trips any text exactly because every byte is in its alphabet. Its size is
capped twice. The model's budget caps it at one token per 256 parameters (a
vocabulary is embeddings, and a model that is all embedding table learns
nothing). The material caps it at one token per 40 characters (a vocabulary
larger than the text can fill is wasted). Never under 256, never over 32,000.
Four special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`.

**The size is the number you ask for.** `--size` takes any count — `70K`,
`4M`, `51M`, `1.5B`, `250000` — and `1m`, `10m`, `100m`, `200m`, `500m`, `1b`
are shortcuts for those numbers, nothing more (`tiny`, `small`, `medium`,
`large`, `huge` are older names for them). The number is met by a search, not a
table:

1. A depth is suggested from the size — about 8 layers at 1M parameters, six
   more for every factor of ten (`8 + 6·log10(size/1M)`), between 2 and 96.
2. A context length is suggested: 256 tokens under 5M parameters, 512 under
   100M, 1,024 under 1B, 2,048 above. `--context` overrides it.
3. Depths within six of the suggestion, and head widths of 64, 32, 16 and 8,
   are tried; for each, the width that lands closest to the target is found.
   The search stops early once a shape is within 1% of the target.
4. Of the shapes within 0.5% of the best, the one nearest the suggested depth
   wins.

Widths move in steps, so the model built lands *near* the number, and the
output says by how much: `asked for 51.0M, built 50.97M (-0.1%)`. The number
Teacher reports afterwards is always the count of the model it built.

**It checks the model can exist.** Before building, the weights alone — four
bytes a parameter — are compared with the memory this machine has free. Above
60% of it, the build is refused with the arithmetic, because training needs
roughly four times the weights again for gradients and the optimizer. `100T`
is a valid number and a refused model.

**What goes wrong here:**

- *…cannot be built here* — the arithmetic above. Choose a smaller size.
- *This architecture will not build* — a shape that fails validation, usually
  from `--context` set to something odd. Leave `--context` out.
- *Only N characters of material* — the tokenizer needs material to be trained
  on. Give it at least a few thousand characters.
- The size is not understood — write it as a number with an optional K, M, B,
  T, P or Q after it.

---

## Part 7 — Adopting a pretrained model

`teacher/lessons.py`, `adopt()`.

`--base small` (or `gpt2`, `medium`, or any name on the Hugging Face Hub)
downloads a model that already writes a language, and makes it the starting
point. Your lessons then teach it your material rather than English itself —
far better output from far less text.

| Shortcut | Model | |
|---|---|---|
| `small` | HuggingFaceTB/SmolLM2-135M | writes real English, fine-tunes on a CPU |
| `gpt2` | openai-community/gpt2 | 124M, older, still capable |
| `medium` | HuggingFaceTB/SmolLM2-360M | better prose, wants a GPU |
| `tiny-test` | sshleifer/tiny-gpt2 | a stub for checking the plumbing, not for use |

The download goes through Hugging Face's cache, so a second model from the
same base does not download again. The weights are saved as `safetensors` in
the model's folder with the base's own tokenizer. The base model's licence is
looked up on the Hub at the same time and kept in the record for its model card.

`--trust-remote-code` lets a base model run code from its repository. Only use
it for repositories you trust; nothing else needs it.

**What goes wrong here:**

- *Could not fetch …* — the name is wrong, the repository is private or gated,
  or you are offline. Check the name on huggingface.co.
- *…ships a tokenizer Teacher cannot save as tokenizer.json* — some
  repositories only ship a slow, SentencePiece-only tokenizer. Pick another base.
- The download is slow — it happens once; the cache keeps it.

---

## Part 8 — A lesson, step by step

`teacher/lessons.py`, `teach()`; the loop is `ai_studio/training/trainer.py`.

1. **The material is tokenized** with the model's own tokenizer.
2. **The block length is chosen**: the model's context, capped at 512 tokens —
   a pretrained model's 8,192-token context would demand more material per
   training example than most people have.
3. **Text is held out.** The last 5% of the tokens (at least one block) is
   never trained on; the held-out loss is measured on it. `--eval-from PATH`
   measures on separate text instead (Part 18).
4. **Examples are packed**: the training tokens are cut into consecutive
   blocks. Each block is one training example, and the model learns to predict
   every token from the ones before it.
5. **Memory is checked** before anything heavy happens (Part 9). A lesson that
   would not fit is refused, with suggestions — Teacher does not start a run
   that is likely to crash the machine.
6. **The batch size and learning rate are settled** (Parts 10 and 11).
7. **The held-out loss is measured before the first step.** This is where the
   lesson starts from.
8. **The loop runs.** Forward, loss, backward, gradients clipped to a norm of
   1.0, an AdamW step (weight decay 0.01), and the learning rate moved along a
   cosine schedule after a warm-up of 3% of the steps (at least five). The
   held-out loss is measured about four times an epoch — every 10 steps when
   an epoch is shorter than 40 — on at most 20 batches of the held-out text;
   progress is logged about ten times an epoch.
9. **It writes itself down as it goes** — about ten times over the lesson, and
   never more often than every 25 steps — so a crash can be resumed (Part 16).
10. **The held-out loss is measured again, on the weights about to be saved.**
    Before and after are the lesson's result: `on held-out: 2.7079 -> 2.5567`.
    If after is higher than before, the lesson says it made the model worse.
    The lowest value seen on the way is kept too (`best_held_out`); when it is
    clearly lower than the end, the lesson went on past its best.
11. **The previous weights are copied aside** into `checkpoints/before-<time>/`,
    and only then are the new weights written over them (Part 17).
12. **The lesson is appended to the record.**

A lesson reports the held-out loss of **the weights it saved**. Earlier
versions reported the lowest value seen part-way through while saving the
weights from the end — two different models whenever a lesson went on too
long, with the number describing the one thrown away. A measured example: a
pretrained model taught at too high a rate reported 2.6108, while the weights
it saved scored 2.7152 — worse than the 2.7079 it started from.

**Precision** is chosen per device: bfloat16 on a GPU that supports it, float16
with loss scaling on one that does not, float32 on a CPU (half precision on a
CPU is slower and less reliable). **Device**: `--device auto` takes the GPU if
PyTorch can see one.

**What goes wrong here:**

- *The material is only N tokens, and one training block is M* — there is not
  enough for even two blocks. Add material, or make a model with a shorter
  context (`--context 64`).
- *This lesson would not fit in memory* — Part 9.
- *This lesson made it worse on text it did not train on* — roll it back
  (`teacher rollback NAME`), then fewer `--epochs` or a lower `--rate`.
- *The lesson failed: …* — the loop raised an error. The message after the
  colon is the real cause; the weights on disk are untouched.
- Out of memory part-way through, despite the check — the estimate is an
  estimate. Halve `--batch`.

---

## Part 9 — Memory: the check before every lesson

`ai_studio/training/config.py`, `preflight()` and `activation_estimate()`.

Before a lesson starts, the memory it needs is estimated and compared with
what is available — the GPU's free memory when training on a GPU, free RAM on a
CPU. It adds up:

- **weights**: parameters × 2 bytes in half precision, × 4 in full;
- **gradients**: the same again, for every parameter being trained;
- **optimizer**: AdamW keeps two numbers per trained parameter, 8 bytes;
- **activations**, from the architecture: per layer, the batch × block length ×
  width × 8 values for the attention and norm paths, × the MLP width × 3 for
  the feed-forward, and the batch × heads × block length² × 2 for attention
  scores; plus the batch × block length × vocabulary × 3 for the output logits
  and their copies in the loss;
- **a safety margin** of 1.25 on the activations, for what the arithmetic
  leaves out (allocator fragmentation, library workspace, the data itself).

Over what is available, the lesson is **refused**; over 90% of it, it is
allowed with a warning. With LoRA only the adapter's parameters count for
gradients and the optimizer. Gradient checkpointing, where used, cuts
activations to a quarter.

**How close is it?** Measured on SmolLM2-135M at a 512-token block on a CPU:
batch 4 peaked at 7.1 GB and batch 8 at 11.9 GB, against an estimate of
12.7 GB for batch 8 before the margin and 15.4 GB after it. The arithmetic runs
a little high, and the margin is deliberately on top of that. So on a machine
with 16 GB of RAM and no GPU, batch 8 on the `small` base is refused although
it would just fit — batch 4 or `--batch auto` is the answer.

**What goes wrong here:** *This lesson would not fit in memory* lists what
was estimated and against what, and suggests in order: a smaller batch; more
gradient accumulation to keep the effective batch; gradient checkpointing;
LoRA for a pretrained model. `--batch auto` does the first for you.

---

## Part 10 — Batch size

`--batch N` is how many blocks go through the model per step; default 8.

`--batch auto` (`fit_batch_size()` in `ai_studio/training/config.py`) climbs
the ladder 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 and takes the largest
batch whose memory estimate stays under 80% of what is available — **and**
that still leaves at least 20 optimizer steps an epoch. The second condition
matters on small material: a batch so large that an epoch is two steps learns
almost nothing from it, however well it fills the GPU. The output says which
limit it hit.

A larger batch is not free accuracy: it takes fewer, larger steps per epoch.
If the GPU is at half load, `auto` is the right fix; if the model is learning
too slowly, it is not.

With several GPUs the batch is **per GPU** — `--batch 8 --gpus 4` steps over
32 blocks at once, and the output says so.

---

## Part 11 — Learning rates

The learning rate is how far each step moves the weights. Too low and a lesson
learns a fraction of what it could; too high and it overshoots — it memorises
the training text, gets *worse* on text it has not seen, and, for a pretrained
model, forgets what it knew.

**Leave `--rate` out.** The default depends on the kind of lesson, because one
rate for every kind was measured to be wrong in both directions. They are
`RATES` in `teacher/lessons.py`:

| Kind of lesson | Default | |
|---|---|---|
| A pretrained base, every weight trained | **5e-5** | best of 2e-5, 5e-5, 1e-4 and 3e-4 |
| LoRA on a pretrained base | **1e-3** | best of 1e-4, 3e-4, 1e-3 and 3e-3 |
| A model built from scratch, up to 20M parameters | **3e-3** | best of 3e-4 to 3e-2, at 1M and 10M |
| A model built from scratch, larger | 3e-3 × 20M ÷ its size, never under 3e-4 | a rule of thumb — not measured |

`--rate 1e-4` sets one by hand. Every lesson records the rate it used and
where it came from (`given`, or `the default for this kind of lesson`), and a
model card shows it for every lesson.

### How the defaults were measured

Every lesson below read the same material — five Wikipedia articles about
lighthouses, 171,391 characters — for three epochs, with the same seed and
data order; only the rate changed. Every number is measured on **the weights
the lesson saved**, on up to three texts: the held-out tail of the material,
a sixth article on the same subject that was never trained on, and three
unrelated articles (photosynthesis, Baroque music, volcanoes) for how much a
pretrained model forgot. Lower is better; the change from before the lesson
is in brackets.

**A pretrained base, every weight trained** — SmolLM2-135M (the `small` base),
batch 4:

| Rate | Held-out tail | Unseen article | Unrelated text |
|---|---|---|---|
| before the lesson | 2.7079 | 2.7181 | 2.1860 |
| 3e-4 (the old default) | 2.7152 **(+0.007)** | 2.7912 (+0.073) | 2.3443 **(+0.158)** |
| 1e-4 | 2.5657 (−0.142) | 2.6388 (−0.079) | 2.1834 (−0.003) |
| **5e-5** | **2.5567 (−0.151)** | **2.6205 (−0.098)** | 2.1519 (−0.034) |
| 2e-5 | 2.5844 (−0.124) | 2.6400 (−0.078) | 2.1487 (−0.037) |

At 3e-4 the model memorised its material — a training loss of 1.23 against a
held-out 2.72 — got worse on text from its own material than it had been before
the lesson, and forgot general English. At 5e-5 it improved on every text. 2e-5
forgot no less and learned less in the time. Run again with a different data
order, 5e-5 came out within 0.002 on all three, so the differences in this
table are not noise.

**LoRA**, rank 16 on the same base, batch 4 (changes from the same 2.7079 /
2.7181 / 2.1860):

| Rate | Held-out tail | Unseen article | Unrelated text |
|---|---|---|---|
| 1e-4 | −0.036 | −0.023 | −0.012 |
| 3e-4 (the old default) | −0.086 | −0.043 | −0.020 |
| **1e-3** | **−0.104** | **−0.055** | −0.009 |
| 3e-3 | −0.104 | −0.036 | **+0.031** |

An adapter starts at zero and is small, so it takes larger steps than the
whole model does. At 3e-3 it learned no more and began to forget.

**Built from scratch**, batch 8, 256-token context. The held-out tail here is
ten blocks and moves by a few hundredths from one fresh model to the next; the
unseen article, forty times larger, is the steadier number:

| Rate | 1M: held-out tail | 1M: unseen article | 10M: held-out tail | 10M: unseen article |
|---|---|---|---|---|
| 3e-4 (the old default) | 7.772 | 7.748 | 6.850 | 6.795 |
| 7.6e-4 | | | 6.779 | 6.654 |
| 1e-3 | 7.143 | 7.010 | 6.907 | 6.722 |
| **3e-3** | **6.846** | **6.739** | 6.824 | 6.662 |
| 6.3e-3 | 7.008 | 6.788 | | |
| 1e-2 | 7.005 | 6.798 | 6.828 | 6.676 |
| 3e-2 | 7.014 | 6.773 | | |

(before any lesson: 8.28 for the 1M model, 8.41–8.44 for the 10M.) A model
starting from noise has nothing to protect and everything to learn: at 3e-4 the
1M model barely left the starting line. Larger models take smaller steps as a
rule, and nothing above 10M was measured, so the rate is scaled down above 20M
parameters rather than carried up unchanged.

### Why there is no `--rate auto`

A learning-rate range test — a few dozen steps at a rate climbing from 1e-7
to 1e-1, reading off where the loss falls fastest and where it bottoms out,
and taking a tenth of the bottom — was built, and measured against the tables
above before being offered. It is not offered:

| Kind of lesson | The range test picked | The best rate | |
|---|---|---|---|
| Pretrained, every weight | 7.6e-4 | 5e-5 | 3e-4 already made the model worse and made it forget; this is 2.5 times that |
| LoRA | 6.5e-8 | 1e-3 | The loss moved less than it varies from batch to batch, so the curve was noise |
| From scratch, 1M | 6.3e-3 | 3e-3 | A little worse than the default |
| From scratch, 10M | 7.6e-4 | 7.6e-4 to 1e-2 | As good as the default, no better |

A range test rewards how fast the *training* loss falls over a few dozen
steps — which is exactly what a rate that is about to overshoot does best. It
was designed for training from scratch, and there it was no better than a
fixed default. So it would have picked a damaging rate for the most common
lesson, noise for LoRA, and nothing better than the default anywhere else.

**What goes wrong here:**

- *A learning rate of N makes no sense; they sit between about 1e-6 and 1e-2*
  — the number given is not a learning rate (zero, negative, or one or more).
- A lesson makes the model worse — the rate is too high for it, or the lesson
  too long. Leave `--rate` out; fewer `--epochs` do the same job.
- A lesson from scratch learns very little — check the rate it recorded (its
  model card lists every lesson's, and so does `history.json`); a hand-set 3e-4
  is a tenth of what a small model from scratch can use.

---

## Part 12 — Teaching until it is done

`teach_until()` in `teacher/lessons.py`; `--until STAGE` on `new` and `teach`.

Teaching in rounds removes the guess at `--epochs`: each round is a lesson of
`--epochs` epochs (default 3), and after each Teacher measures the held-out
loss and decides whether to go on. It stops when:

- **it reached the stage** asked for — `words` or `sentences` (Part 18);
- **it stopped improving** — a round bought less than 1% of the loss;
- **a round made it worse** — then that round's weights are rolled back, so
  what is left on disk is the best the run reached, not merely the last thing
  it did (`rounds_kept` says how many rounds are in the saved weights);
- it ran out of rounds (`--max-rounds`, default 20) or time (`--max-minutes`).

`best` means "until it stops improving, whatever stage that is". A model
already past the stage asked for is left alone.

Every round is saved as it finishes, so Ctrl+C leaves the last completed round
on disk. The whole run is one entry in the record, with each round's loss and
the weights each round saved kept inside it — so a later rollback into the
middle of the run is placed exactly, and a model card says how many of its
rounds are in the weights (Part 17).

---

## Part 13 — Teaching it to answer

`teacher/answers.py`; `--answers PATH` on `new` and `teach`.

Plain text teaches a model to *continue*. Pairs teach it to *reply*: each
example is a question and its answer, laid out in a template, and **the loss is
taken on the answer only** — the question's tokens are labelled -100 and
ignored. The model is never rewarded for reproducing the question, only for
what follows it.

**The file** is `.jsonl`, `.json` or `.csv` (folders are walked), in any of
three shapes:

```json
{"question": "...", "answer": "..."}
{"instruction": "...", "input": "...", "output": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`prompt`/`task` work for `instruction`, `response`/`completion` for `output`,
`q`/`a` for `question`/`answer`. Rows that fit none are skipped and counted.
Each shape has its template:

```
### Question:            ### Instruction:          <|user|>
{question}               {instruction}             {message}
                                                   <|assistant|>
### Answer:              ### Response:             {reply}
{answer}                 {output}
```

5% of the pairs are held out to measure on. At least 8 pairs are needed; a
few hundred is a sensible start, and a dozen epochs is normal for pairs.

**The template is part of the model now.** It is written into the record, and
`ask`, the window, benchmarks and `serve` all wrap a question in it — a model
taught with `### Question:` and asked something bare answers as if continuing
a document, which looks like the training failed. Rolling back past the answer
lesson, or branching from a state before it, drops the template again.

**How the answer is found in the tokens.** The template's prompt part and the
whole example are tokenized separately, and the answer starts where the two
stop agreeing token by token. (Taking the length of the prompt tokenized on its
own counted the end-of-text token a tokenizer appends to a lone piece of text —
so the first token of every answer was masked as if it were prompt, and a model
taught that way learned to reply with nothing. Comparing token by token also
survives a tokenizer that merges the prompt's last character with the
answer's first.)

**What goes wrong here:**

- *No question-and-answer pairs were found* — the column names are not ones it
  knows. The message lists the shapes that work.
- *Only N pair(s) — too few to learn a habit of answering* — at least 8.
- *Every pair was empty once tokenized* — the fields are there but empty.
- `--gpus` is ignored for an answer lesson, and it says so; answer lessons are
  not resumable, and `resume` says so.

---

## Part 14 — LoRA

`--lora` and `--lora-rank N` on `new` and `teach`.

Instead of training every weight, LoRA trains a small adapter beside the
attention layers: two thin matrices per layer whose product is added to the
layer's output. Rank 16 by default (8 thrifty, 64 more capable and costlier),
with a scale of twice the rank and 5% dropout. Only the adapter needs
gradients and optimizer state, so a lesson needs a fraction of the memory —
which is what lets a bigger base fit on a smaller machine.

**The adapter is merged into the weights when the lesson ends.** What LoRA buys
here is a cheaper lesson, not a different kind of model: an adapter left on
its own would leave the folder unable to load, and would break rollback,
branches, packing and export.

LoRA is for pretrained models. On a model built from scratch there is nothing
to adapt — its weights are noise — so it is refused with that reason.

**What goes wrong here:**

- *LoRA trains a small adapter on top of a model that already knows a
  language…* — drop `--lora` for a model built from scratch.
- *The lesson ran, but its adapter could not be merged back in* — the weights on
  disk are unchanged; the message names the error from PEFT.

---

## Part 15 — More than one GPU

`ai_studio/training/distributed.py`; `--gpus N` or `--gpus auto`.

One process per GPU, each with a full copy of the model and a different slice
of every epoch (`DistributedSampler`); after every backward pass the gradients
are averaged across them (`DistributedDataParallel`), so the copies stay
identical. The held-out loss is pooled across processes, so it does not depend
on how many cards there were. Process 0 saves.

The communication backend is NCCL on Linux with CUDA, and gloo everywhere else
(Windows has no NCCL). Workers are started with `spawn`, which re-imports the
program in each process: a script of your own that calls Teacher across GPUs
needs the usual `if __name__ == "__main__":` guard, and without it the error
says so.

**What it does not do:** use the CPU and the GPU together. Every step waits for
the slowest process, so adding a CPU to a GPU makes each step slower, not
faster. It is not offered. `--gpus 2` on a machine with no GPU trains in one
CPU process and says why.

Answer lessons run on one GPU for now, and say so.

---

## Part 16 — Crashes and resuming

`teacher/interrupted.py`; `teacher resume NAME`.

A lesson writes itself down about ten times as it runs, into `.interrupted/`
in the model's folder: the weights, the optimizer's state (AdamW's two moments
for every weight), the schedule's position, the step it reached, and the plan
— sources, epochs, batch, rate, and a fingerprint (SHA-1) of the material.

**The write cannot be half done.** It goes to `.interrupted.writing/` first
and is renamed into place only when complete, so a crash during the save
leaves the previous save intact. A save that fails leaves nothing rather than
something unusable. The final step is not saved — the finished lesson is about
to replace it.

**When the lesson finishes, the folder is deleted.** A `.interrupted/` folder
that exists means a lesson that did not finish.

`teacher resume NAME` rebuilds the material from the recorded sources, checks
its fingerprint, and puts everything back:

- **the weights** as they were at the last save;
- **the optimizer's state** — AdamW's running averages of every weight's
  gradient and of its square, and its step count. Without them the first steps
  after resuming are as large and as noisy as a brand-new run's;
- **the place in the schedule** — the learning rate at the saved step, not
  back at the warm-up;
- **the place in the data** — the shuffle draws from a generator of its own,
  seeded from the lesson's seed and used by nothing else, so the order of each
  epoch can be drawn again; the epochs already finished are drawn and set
  aside, and the batches of the current epoch already trained on are passed
  over.

For a model without dropout — every model built from scratch here — a lesson
interrupted and resumed ends on the same weights as one that ran straight
through, to within a millionth; a test holds it to that. The shared random
state that dropout draws on is not restored, so a model that uses dropout
(LoRA's adapters do, at 5%) ends close to, not exactly on, the same weights.
The window shows a *Resume* button whenever there is something to resume.

**A lesson across several GPUs does not write itself down yet**, so it cannot
be resumed; if one is interrupted, the weights from before it are still in the
model's folder, untouched. Answer lessons are not resumable either (Part 13).

**What goes wrong here:**

- *…has no interrupted lesson — nothing to resume.* — the good case: the last
  lesson finished.
- *The material has changed since that lesson started…* — the files are
  different from when the lesson began; resuming would train on something else.
  Start the lesson again.
- *The interrupted lesson did not record where its material came from* — it
  cannot be rebuilt. Teach it again.
- *Answer lessons are not resumable yet* — they are short; start again.
- *The saved optimizer does not fit this model* — the saved state belongs to
  different settings; start the lesson again.

---

## Part 17 — Saved states, rollback and branches

`teacher/workspace.py`.

**Before a lesson writes its weights**, the weights, `config.json` and
`tokenizer.json` are copied into `checkpoints/before-<date>-<time>/`. Two are
kept by default (`--keep N` keeps more; each is a full copy of the weights).
`teacher checkpoints NAME` lists them.

**`teacher rollback NAME`** copies the newest back; `--to STAMP` a named one.
The lessons stay in the record — they happened — and the rollback is recorded
beside them. If the rollback goes past the lesson that taught the model to
answer, its answer template is dropped with it.

**`teacher branch NAME NEW [--at STAMP]`** copies the model — or one of its
saved states — into a new model. The original is not touched. The branch starts
with no lessons of its own and carries the source's under
`branched_from_history`, plus its base model, licence and template.

### Which lessons are in these weights: the lineage

Every lesson records the stamp of the weights it started from and the stamp of
the weights it saved (Part 3). From the weights on disk, the record can be
walked backwards — which lesson saved these? what did that one start from?
— until it reaches the weights the model was made with. The lessons on that
path are the ones in the weights; any other is **undone**: rolled back, or
taught to a branch's source after the branch was taken. `workspace.lineage()`
does the walk; model cards and the answer template use it.

Lessons recorded before this was tracked have no stamps, so the walk cannot see
through them: they are counted as in effect, and the answer is marked as not
certain.

**What goes wrong here:**

- *…has no earlier state saved* — it has not been taught, or the states were
  deleted.
- *…has no saved state 'X'. It has: …* — use one of the stamps listed.
- *'NEW' already exists* — branch to another name, or `teacher forget` the old one.

---

## Part 18 — Reading the numbers

**Loss** is how surprised the model is by the next token, averaged: the
negative log of the probability it gave the right one. Lower is better. It is
only comparable between models with the same vocabulary, on the same text.

**Held-out loss** is the loss on text the lesson did not train on — the only
number that says whether the model learned rather than memorised. Two kinds:

- **the tail of the same text** (the default) — never trained on, but the same
  source, voice and subject. It flatters a model that memorised its material.
- **a separate file** (`--eval-from PATH`) — nothing there to have memorised.
  Each lesson records which kind it was (`measured_on`) and, for a separate
  file, which files.

**Perplexity** is `e` to the power of the loss: roughly how many tokens the
model is choosing between at each step.

**The stage** is the held-out loss as a share of where the model started. For a
model built from scratch the start is `ln(vocabulary)` — the loss of a model
that guesses uniformly:

| Share of the start | Stage |
|---|---|
| 70% or more | barely started |
| 55% | learning the alphabet |
| 28% | learning words |
| 15% | learning sentences |
| under 15% | has the shape of your text |

For an adopted model the start is its own held-out loss before its first
lesson (`base_loss`):

| Share of the start | Stage |
|---|---|
| 95% or more | barely moved |
| 75% | picking up your material |
| 55% | adapting well |
| under 55% | closely fitted |

**Comparing two models** (`teacher compare A B`): their losses are said to be
comparable only when their vocabularies are the same **and** their last lessons
were measured on the same text. Otherwise read the replies.

---

## Part 19 — Talking to it

`teacher/generation.py`; `teacher ask NAME "prompt"`, the window's *Chat* tab.

The model is **loaded once and kept** — one model at a time — and reused for as
long as its weights stamp is unchanged; a lesson that rewrites the weights
means the next request loads the new ones. A lesson lets go of a kept model
before it starts, rather than training beside it.

The prompt gets the start token if the tokenizer has one, and the answer
template if the model was taught one (Part 13). Then one token at a time:

- **repetition penalty** (1.1): every token already in the text is made less
  likely — its score divided by 1.1 if positive, multiplied if negative. (It
  used to be divided either way, which made a token with a negative score
  *more* likely; the browser page always did it right.)
- **temperature** (0.8) divides the scores: lower is safer and duller, 0 is
  always the most likely token (greedy).
- **top-k** (40) keeps only the 40 most likely tokens; **top-p** (0.95) keeps
  the smallest set whose probability adds up to 95%.
- It stops at the end-of-text token, at the length limit (`--tokens`, 120), or
  at a stop string (the API's `stop`).

`--seed N` repeats an exact answer. Text is decoded as it comes; half of a
multi-byte character is held back until the rest of it arrives.

**What goes wrong here:**

- It repeats itself — raise the temperature a little, or teach it more varied
  material. A small model reproduces the repetition in what it read.
- ◆ or � in the output — the tokenizer works on bytes, and a model still
  guessing picks half a character. Teach it more.
- *The prompt is N tokens and this model reads at most M* — shorten the prompt.

---

## Part 20 — Benchmarks

`teacher/exams.py` over `ai_studio/evaluation`; `teacher test NAME SUITE`.

Nine suites: `mmlu`, `mmlu_pro`, `gsm8k`, `arc`, `hellaswag`, `truthfulqa`,
`winogrande`, `humaneval`, `mbpp`. Each item is generated **greedily** (so a
run repeats exactly), with a length limit that suits its kind — 8 tokens for a
multiple-choice letter, up to 160 for an answer graded by content — and graded
by the suite's own method.

**Where the questions came from.** The official split is loaded through the
`datasets` package, or a local `.jsonl`; failing both, a handful of bundled
example items. Six questions is a check that the plumbing works, not a score,
and the result says so (`official: false`). `--offline` skips the download.

**What chance looks like.** For multiple choice, the score a model that knows
nothing would get is printed beside every result — 25% for four choices.

**Whether it means anything.** A score has to clear the chance rate by two
binomial standard deviations before it is read as evidence: with 20
four-choice items, expected 5 right, spread about 1.9, so 9 or more. Below
that the verdict is *indistinguishable from guessing*. Models of a few hundred
million parameters score at chance on knowledge benchmarks; that is their size,
not a failed lesson.

Each result is kept in the record — the score, not every answer — with the
stamp of the weights it was taken on, for the model card.

**What goes wrong here:**

- *Name a model to test, or pass --list to see the suites.*
- *After N worked examples there were no questions left* — raise `--items` or
  lower `--shots`.
- *The X suite has no items to run* — neither the official split nor the
  bundled items could be loaded.

---

## Part 21 — Model cards

`teacher/card.py`; `teacher card NAME`.

A card is written from the record and nothing else, as `README.md` in the model's
folder — the file the Hugging Face Hub shows as a model's page, with the Hub's
front matter (`base_model`, `library_name`, `pipeline_tag`, tags).

- **Parameters** are counted from the weights file's header, without loading it.
- **Lessons** are those in the weights (Part 17's lineage); undone ones are
  listed apart. Lessons recorded before weights were tracked are listed with a
  note that a rollback from then cannot be told apart.
- **Sources**: a file whose first three lines are a title, `Source:` and
  `Retrieved:` is a saved web page and is listed with its address and date;
  anything else by file name only — never the path, because a card is for
  sharing and your folder layout is not.
- **Scores**: only results whose weights stamp is the current one; older ones
  are counted and left out.
- **Licence**: the base model's licence as declared on the Hub; a statement
  that Teacher's own licence covers the program, not the weights or the
  material; and **no licence chosen for the model** — the `license:` field is
  left commented out for the person who knows what the material was.

`--show` prints it; `-o FILE` writes it elsewhere; `--offline` skips the licence
lookup. **A `README.md` Teacher did not write is never overwritten** — the
card carries a marker comment, and only a file with that marker is replaced.
`teacher export` writes the card beside the GGUF too (`name.gguf` → `name.md`).

**What goes wrong here:** *…already exists and was not written by Teacher, so
it was left alone* — write the card elsewhere with `-o`.

---

## Part 22 — GGUF export

`teacher/export.py`; `teacher export NAME`.

GGUF is what llama.cpp, Ollama and LM Studio read. The conversion is done by
**llama.cpp's own converter**, `convert_hf_to_gguf.py`, run as a separate
process — it is the thing that knows the format, and a second copy of it here
would rot. It is looked for at `--converter PATH`, then `LLAMA_CPP_CONVERT`,
then `convert_hf_to_gguf.py`, `llama.cpp/`, `../llama.cpp/`, `~/llama.cpp/`
and `~/src/llama.cpp/` — the relative ones from the folder Teacher is run in,
so a clone beside the repository is found when Teacher is run from the
repository, as the instructions have it.

`--precision`: `f16` (the usual choice), `bf16`, `f32` or `q8_0`. Smaller
quantisations are a second step through llama.cpp's `llama-quantize`.

**Nothing is written unless it worked.** The converter writes to
`NAME.gguf.partial` beside the destination, and the file is renamed into place
only when the converter has finished and succeeded. If it is missing, refuses
the model, dies half-way or runs past an hour, the partial file is deleted and
the reason is shown verbatim — and a good file from an earlier export at the
same path is left exactly as it was. A reported success with no file is
reported as a failure.

Only models started from a pretrained base convert: llama.cpp implements the
architectures it knows, and Teacher's own is not one of them. Beside the file
goes its model card, with its size and SHA-256, and the export is noted in
the record.

**What goes wrong here:**

- *llama.cpp's converter was not found* — `git clone
  https://github.com/ggerganov/llama.cpp` and `pip install -r
  llama.cpp/requirements.txt`, then point at it.
- *llama.cpp's converter refused this model:* — its last lines follow; usually
  a missing Python package from llama.cpp's requirements (`sentencepiece`,
  `gguf`), or an architecture that version of llama.cpp does not know.
- *…was built here, and llama.cpp does not know this architecture* — a model
  built from scratch. Use `teacher pack` and the Bench page.
- *Could not run the converter* / *The converter ran for an hour without
  finishing* / *The converter reported success but … is not there* — what
  they say; nothing was kept.

---

## Part 23 — Serving it to other programs

`teacher/serve.py`; `teacher serve NAME`.

A FastAPI application on uvicorn (both come with Gradio) that answers in
OpenAI's format at `http://127.0.0.1:8008/v1`:

| Route | |
|---|---|
| `GET /v1/models` | the models being served |
| `POST /v1/completions` | continue `prompt` |
| `POST /v1/chat/completions` | reply to `messages` |
| `GET /health` | `{"status": "ok"}` |

**A request's settings**: `max_tokens` (or `max_completion_tokens`; default
256, at most 2,048, and never more than the context has room for),
`temperature` (0–2), `top_p`, `stop` (up to four strings), `seed`,
`repetition_penalty` (1–2), `stream`, and `stream_options.include_usage`. One
choice per request (`n` must be 1). Prompts over 200,000 characters are refused
before tokenizing.

**Conversations** are laid out the way the model was taught (Part 13): each
user turn in its template, each earlier reply after it, and generation stops
where the template would begin the next turn. A model taught only text gets a
`User:` / `Assistant:` transcript and stops at the next `User:`.

**Streaming** is server-sent events, one per piece of text, then a final event
with the finish reason, then `data: [DONE]`. Generation runs in a worker
thread; if the client disconnects, it stops.

**Keeping up**: each request compares the weights stamp with what is loaded
and reloads if a lesson has changed them. **One reply at a time**: a lock
around generation, because two at once would each take half the speed in twice
the memory.

**Who can reach it**: `127.0.0.1` — this machine only — unless `--host` says
otherwise. `--api-key KEY` or `TEACHER_API_KEY` makes every `/v1/` request
carry `Authorization: Bearer KEY`, compared in constant time; `/health` then
stops naming the models. There are no routes that touch files or run anything.

**Errors come back in OpenAI's shape** (`{"error": {"message", "type",
"code"}}`): 404 `model_not_found` naming the models being served; 400 for a bad
request, with `context_length_exceeded` for a prompt the model cannot read;
401 `invalid_api_key`.

**What goes wrong here:**

- The client says *model not found* — the `model` it sends must be the name
  Teacher serves (`GET /v1/models` lists them).
- *Address already in use* — another program has port 8008; `--port 8010`.
- Replies from a text-only model wander — it was taught to continue text, not to
  reply. Teach it with `--answers`.
- *Serving needs FastAPI* / *Serving needs uvicorn* — `pip install gradio`
  brings both.
- *A conversation has to end with something from the user.* — the last message
  must have the role `user`.

---

## Part 24 — The window

`teacher/ui.py`; `teacher ui` (port 7861, `--port`, `--host`, `--share`,
`--no-browser`).

The window calls the same functions as the command line, so a model made in
one is the same model in the other.

| Tab | Does | The command |
|---|---|---|
| Make | build from scratch, or adopt a base | `new` |
| Teach | text or pairs; until done or one lesson; batch, rate, GPUs; LoRA, device, saved states, a separate file to measure on; resume | `teach`, `resume` |
| Chat | talk to it; compare models side by side | `ask`, `compare` |
| Test | a benchmark, with the chance rate | `test` |
| Keep | Bench files, model card, GGUF, serving, saved states, rollback, branch, delete | `pack`, `card`, `export`, `serve`, `rollback`, `branch`, `forget` |

The material box at the bottom (files, a folder, pasted text, or a web search)
feeds both *Make* and *Teach*; for an answer lesson, its files and folder are
read as pairs. One lesson runs at a time — a second click while one is running
is told so. Every action catches its own failure and shows it as a message: one
failed operation never closes the window.

The API started from *Keep* listens on this machine only; the command line has
`--host` for wider.

**What goes wrong here:** the window's messages are in Part 26; the rest is
Part 2 (Gradio versions).

---

## Part 25 — Speed: what the training loop does to be fast

`ai_studio/training/trainer.py` and `ai_studio/models/transformer.py`.

- **The GPU is not stopped to report numbers.** Reading a loss off the GPU
  makes the CPU wait for every queued operation to finish. The loss is summed
  on the GPU and read once per optimizer step, together with the gradient norm
  in a single transfer.
- **The loss shifts the labels, not the logits** — the same numbers without a
  vocabulary-sized copy per step.
- **Attention takes PyTorch's fused path** (`scaled_dot_product_attention` with
  `is_causal=True`); passing an explicit mask would force the slow path.
- **AdamW is fused** on CUDA — one kernel for every parameter.
- **TF32 matrix maths and cuDNN autotuning** are on for GPUs that have them.
- **Half precision** on GPUs (Part 8).
- **CPU threads**: on a CPU, PyTorch is told to use every core the operating
  system reports (`os.cpu_count()`).
- **No data-loader workers**: a batch is a slice of a tensor already in memory,
  which a worker process would only add copying to.

`torch.compile` is not used: it needs Triton, which is not available on
Windows. Measured on a CPU with the same data, seed and result, these changes
took one run from 22.75 s to 18.77 s.

**A GPU at half load** is usually a batch too small to fill it — `--batch auto`.

---

## Part 26 — When something breaks

### Symptoms without a message

| What you see | What it is | What to do |
|---|---|---|
| Training is very slow | Running on the CPU — check `torch.cuda.is_available()` (Part 2) | Install the CUDA build of PyTorch |
| The GPU sits at half load | The batch is too small to fill it | `--batch auto` (Part 10) |
| The window will not open, with an error inside Gradio | An older copy of Teacher on Gradio 6 | Update the repository (Part 2) |
| The model writes nonsense | Built from scratch and barely taught, or taught far too little | Check its stage (Part 18); teach more, or start from `--base small` |
| The held-out loss went up | The lesson overshot | `teacher rollback NAME`; lower `--rate` or fewer `--epochs` (Part 11) |
| It repeats itself | Small model, repetitive material | Part 19 |
| ◆ or � in replies | Half-learned byte tokens | Part 19 |
| An answer-taught model replies to nothing | Asked outside its template — or taught by a Teacher from before the masking fix, which hid the first word of every answer (Part 13) | Ask through Teacher, which applies the template; check `answer_style` in `history.json`; if it was taught before the fix, teach it again |
| A lesson on the `small` base is refused for memory on a 16 GB machine without a GPU | Batch 8 is estimated past the RAM there is (Part 9) | `--batch 4`, or `--batch auto` |
| `resume` says the material changed, though you did not touch it | The fingerprint is taken on the cleaned text; updating Teacher between the crash and the resume can change the cleaning | Start the lesson again |
| The window shows the GPU count as a number you cannot change | One GPU or none — there is nothing to choose | Nothing; it trains on what there is |
| A benchmark score looks good but says "indistinguishable" | Inside the noise | Part 20 — more `--items`, or accept it |
| A client of `serve` gets 404 | Wrong model name | Part 23 |
| Killed by the operating system mid-lesson | Out of memory beyond the estimate | Halve `--batch`; `teacher resume NAME` picks up from the last save |

### Every message, and what to do

Teacher stops with a sentence rather than a traceback when the problem is one
you can fix. Here is each, word for word (`NAME`, `N` and the like stand for
what is filled in).

#### Making a model

| Message | What to do |
|---|---|
| `'NAME' already exists. Teach it more, or pass --replace.` | `teacher teach NAME`, or `--replace` to build over it (the old record is kept under `replaced`) |
| `Only N characters of material — too little to learn anything. Give it at least 2,000; a few hundred KB is a sensible start.` | More material (Part 4) |
| `100T parameters cannot be built here. The weights alone would be … GB at four bytes each, and this machine has … GB free. Training needs roughly four times the weights again, for gradients and the optimizer. The largest that would fit here is around …` | A smaller `--size` (Part 6) |
| `This architecture will not build: …` | Leave `--context` out (Part 6) |
| `… There are shortcuts too: 1m, 10m, …` | The size was not a number Teacher reads — write it like `51M` (Part 6) |
| `transformers is not installed.` | `pip install -r ai_studio/requirements.txt` |
| `Could not fetch 'REPO': … Check the name on huggingface.co, and that you are online.` | Part 7 |
| `'REPO' ships a tokenizer Teacher cannot save as tokenizer.json. Pick another base model.` | Part 7 |
| `That name has no usable characters in it.` | Use letters or digits in the name |
| `There is no model called 'NAME'. Try: teacher list` | `teacher list` shows the names |
| `'NAME' is missing model.safetensors. It may not have finished its first lesson.` | The folder is incomplete — teach it, or delete the folder |

#### Material

| Message | What to do |
|---|---|
| `Nothing readable was found. Point --from at a file or folder of text.` | Check the path and the file types (Part 4) |
| `Nothing readable in what --eval-from pointed at, so there would be nothing to measure against.` | Point `--eval-from` at text |
| `Only N characters — too little for a lesson. Give it at least 2,000.` | More material |
| `The material is only N tokens, and one training block is M. Add more material, or make a model with a shorter context.` | Part 8 |
| `The text to measure against is only N tokens, and one block is M. Give it more, or drop --eval-from.` | More separate text, or no `--eval-from` |

#### The web

| Message | What to do |
|---|---|
| `Give something to search for, or --url ADDRESS.` | A query or an address |
| `Give something to search for, or at least one address.` | The same, from the window |
| `--list shows what a search found; give it something to search for.` | `--list` needs a query |
| `Nothing readable came back.` | Every page failed; the reasons follow (Part 5) |
| `The search found nothing. Tried web: …; wikipedia: …` | Both engines failed or returned nothing — a search engine may be rate-limiting you. Wait a minute, or give the pages' addresses with `--url` |

#### Teaching

| Message | What to do |
|---|---|
| `This lesson would not fit in memory: …` | Part 9 — follow the suggestions, or `--batch auto` |
| `A learning rate of N makes no sense; they sit between about 1e-6 and 1e-2.` | Part 11 |
| `Unknown device 'X'. Use auto, cpu or cuda.` | One of those three |
| `--device cuda was asked for, but no CUDA GPU is visible here. Check that a GPU driver and a CUDA build of PyTorch are installed, or drop the flag to use the CPU.` | Part 2 |
| `The lesson failed: …` | The cause follows the colon; the weights on disk are untouched |
| `LoRA trains a small adapter on top of a model that already knows a language. This one was built here, starting from noise — there is nothing to adapt yet, so train all of it instead (drop --lora).` | Part 14 |
| `The lesson ran, but its adapter could not be merged back in: … The weights on disk are unchanged.` | Part 14 |
| `Unknown target 'X'. Choose one of: words, sentences, best` | `--until` takes one of those |

#### Answer lessons

| Message | What to do |
|---|---|
| `No question-and-answer pairs were found. Each row needs a question and its answer. Any of these work: {"instruction": "...", "output": "..."} {"question": "...", "answer": "..."} {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} as JSONL, JSON or CSV with those column names.` | Part 13 — the column names |
| `Only N pair(s) — too few to learn a habit of answering. A few hundred is a sensible start.` | At least 8 |
| `Every pair was empty once tokenized.` | The fields are empty |

#### Resuming

| Message | What to do |
|---|---|
| `NAME has no interrupted lesson — nothing to resume.` | Nothing to do |
| `NAME has no interrupted lesson to resume.` | The same, from deeper in; nothing to resume |
| `The interrupted lesson did not record where its material came from, so it cannot be rebuilt. Teach it again from the start.` | Part 16 |
| `Answer lessons are not resumable yet — they are short enough that starting again costs little: teacher teach NAME --answers …` | Run the command it gives |
| `The material has changed since that lesson started, so resuming would train on a different text than the one it was part-way through. Start it again: teacher teach NAME --from …` | Run the command it gives |
| `The saved optimizer does not fit this model (…). The settings must match the ones the run started with.` | Start the lesson again |

#### Saved states, branches, deleting

| Message | What to do |
|---|---|
| `NAME has no earlier state saved — it has not been taught yet, or the saved states were removed.` | Nothing to roll back to |
| `NAME has no saved state 'X'. It has: …` | Use one of the stamps listed |
| `The saved state before-X is empty.` | That state's files were removed by hand; choose another |
| `'NEW' already exists. Choose another name, or delete it with: teacher forget NEW --yes` | Part 17 |
| `NAME is missing model.safetensors.` | The source of a branch is incomplete |
| `This would delete PATH and everything NAME has learned. Pass --yes if you mean it.` | Add `--yes` if you mean it |

#### Comparing and testing

| Message | What to do |
|---|---|
| `Name at least two models to compare.` | Two names or more |
| `Name a model to test, or pass --list to see the suites.` | Part 20 |
| `After N worked examples there were no questions left. Raise --items, or lower --shots.` | Part 20 |
| `The X suite has no items to run.` | Part 20 |

#### Talking, cards, export, serving, Bench

| Message | What to do |
|---|---|
| `The prompt is N tokens and this model reads at most M. Shorten it.` | Part 19 |
| `PATH already exists and was not written by Teacher, so it was left alone. Write the card somewhere else: teacher card NAME --out CARD.md` | Part 21 |
| `Unknown precision 'X'. Choose one of: f32, f16, bf16, q8_0.` | Part 22 |
| `llama.cpp's converter was not found, so nothing was written. GGUF is llama.cpp's format and its converter is the thing that knows how to write it. Writing a file here without it would produce something that only looks like a model. Get it with: git clone https://github.com/ggerganov/llama.cpp pip install -r llama.cpp/requirements.txt Then point at it with --converter /path/to/convert_hf_to_gguf.py, or set LLAMA_CPP_CONVERT. It is looked for beside this repository automatically.` | Part 22 — "beside this repository" holds when Teacher is run from the repository folder |
| `NAME was built here, and llama.cpp does not know this architecture. GGUF export works for models started from a pretrained base (teacher new NAME --base small ...), whose architecture llama.cpp already supports. A from-scratch model runs in Teacher, and in the Bench web page: teacher pack NAME -o ./for-bench` | Part 22 |
| `llama.cpp's converter refused this model: …` | Part 22 |
| `Could not run the converter: …` | Part 22 |
| `The converter ran for an hour without finishing.` | Part 22 |
| `The converter reported success but PATH is not there.` | Part 22 |
| `A conversation has to end with something from the user.` | Part 23 |
| `Serving needs FastAPI, which comes with Gradio: pip install gradio` | Part 23 |
| `Serving needs uvicorn, which comes with Gradio: pip install gradio` | Part 23 |
| `NAME is built on BASE, and the Bench web page only runs models built here. Chat with it in the Teacher UI or with: teacher ask NAME` | Bench runs from-scratch models only |
| `NAME has no model.safetensors — teach it something first.` | Nothing to pack yet |
| `The window needs gradio: pip install gradio (…)` | `pip install gradio` |

#### The window

| Message | What to do |
|---|---|
| `Give the model a name.` | Type one on *Make* |
| `'NAME' already exists. Teach it instead, or pick a new name.` | *Teach* it, or another name |
| `Choose a model first.` | Pick one in the list on the left |
| `Add some text first — the tokenizer is built from it.` | Files, a folder or pasted text, below |
| `Add some text to teach it from.` | The same, for *Teach* |
| `Add a file of question-and-answer pairs — .jsonl, .json or .csv — or a folder of them.` | Pairs come from files, not pasted text |
| `Nothing readable in PATH, so there would be nothing to measure against.` | The *Measure against* box points at nothing readable |
| `Batch size 'X' is not a number. Use a whole number, or auto.` | Part 10 |
| `Batch size is a whole number, 1 or more — or auto.` | Part 10 |
| `Learning rate 'X' is not a number. Leave it empty for the measured default, or type something like 5e-5.` | Part 11 |
| `N is not a learning rate — they sit between about 1e-6 and 1e-2.` | Part 11 |
| `Pick at least two models to compare.` | Two or more in *Compare models* |
| `Write a prompt for them all to continue.` | A prompt for *Compare models* |
| `Could not listen on port N — is something else using it? Try another port.` | Another port on *Keep* |
| `Tick the box to confirm — this cannot be undone.` | Tick it if you mean it |

If a message is not here, it came from a library underneath (PyTorch,
Transformers, llama.cpp) and is shown as it was given. The window shows such
errors as *Something went wrong* with the error's type, rather than closing.
