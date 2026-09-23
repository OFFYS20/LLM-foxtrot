# Teacher

A small command-line program for teaching a language model from your own text.

Where [AI Studio](../ai_studio/README.md) is a workbench with every dial exposed,
Teacher is the short path: point it at some writing, and it builds a tokenizer, a
model and a training run for you. Each model lives in its own folder — and that
folder is exactly what the [Bench](../web_chat/README.md) web page opens, with
nothing to export.

Training is real PyTorch, borrowed from AI Studio rather than reimplemented, so
the loss you see is measured, not simulated.

---

## Install

Teacher runs on AI Studio's dependencies:

```bash
cd LLM-foxtrot/ai_studio
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

### The window

```bash
python -m teacher ui
```

Or double-click `Start-Teacher.bat` (Windows) / run `./start-teacher.sh`, which
installs anything missing on the first run and opens it. Nothing has to be typed.

| Tab | What is there |
|---|---|
| **Make** | Name it, then either adopt a pretrained model or build one from scratch at a chosen size |
| **Teach** | Give it material and press Teach. *Keep going until it is done* is on by default, and each round's loss and stage appear as they happen |
| **Chat** | Talk to it, with temperature, top-p, top-k and length behind a *Sampling* panel |
| **Keep** | Copy the files for Bench, roll back the last lesson, or delete the model |

Below those sits **Text**, shared by Make and Teach: drop files, name a folder on
this machine, or paste. *Check what this adds up to* reports what it found and
what it skipped, before you commit to anything. Inside it, *…or fetch it from
the web* searches, reads the pages and points the folder box at the result, so
material you do not have can be gathered without leaving the window.

The left-hand card shows the selected model's size, what it was built on, how far
along it is, and the loss of each of its last eight lessons.

Everything there is also a command, and the two share one store — make a model in
the window, teach it from a script, chat with it back in the window.

### The commands

```bash
python -m teacher --help
```

Models are stored in `~/teacher-models` by default. Set `TEACHER_HOME` to put
them somewhere else.

---

## Teach something in four commands

```bash
# 1. Build a model and give it its first lesson
python -m teacher new bookbot --from ./my-books --size tiny

# 2. Teach it more, whenever you like
python -m teacher teach bookbot --from ./more-books --epochs 4

# 3. Talk to it
python -m teacher ask bookbot "once upon a"

# 4. Copy the three files the Bench web page needs
python -m teacher pack bookbot -o ./for-bench
```

`--from` takes a file or a folder, and reads `.txt`, `.md`, `.pdf`, `.docx`,
`.epub`, `.html`, `.csv`, `.json` and `.jsonl` — a folder is walked recursively.
Pass it more than once to combine sources, or use `--text "..."` for something
short.

**Row files.** A `.csv`, `.json` or `.jsonl` is turned into text one record at a
time, as `key: value` lines, and *every* record is used — the 2,000-row ceiling
that the Data Library previews with does not apply to training. The whole file
is read into memory first, though, so a very large one wants splitting: a few
hundred MB per file is comfortable, tens of GB in one file is not. If anything
ever is left out, the lesson says so rather than quietly training on a fraction
of your data.

---

## The commands

| Command | What it does |
|---|---|
| `new NAME` | Trains a tokenizer on your material, builds a model sized to it, and runs the first lesson |
| `teach NAME` | Another lesson on more material. Picks up from what it already knows |
| `ask NAME [prompt]` | Generates a continuation. With no prompt, opens a back-and-forth |
| `list` | Every model you have, with size and how much it has been taught |
| `show NAME` | One model's architecture and its full lesson history |
| `pack NAME -o DIR` | Copies `model.safetensors`, `config.json` and `tokenizer.json` |
| `compare A B ...` | The same prompt through two or more models, side by side |
| `test NAME SUITE` | Runs a real benchmark suite and says what the score means |
| `export NAME` | Converts to GGUF for llama.cpp, Ollama or LM Studio, with a model card beside it |
| `card NAME` | Writes a model card: what it is, what it was taught, from where, on whose terms |
| `serve NAME` | Answers over a local API in OpenAI's format, for other programs to use |
| `resume NAME` | Carries on a lesson that was interrupted part-way through |
| `checkpoints NAME` | The saved states this model can go back to or branch from |
| `branch NAME NEW` | Copies a model, or one of its saved states, into a new model |
| `rollback NAME` | Undoes the last lesson, restoring the weights saved before it |
| `forget NAME --yes` | Deletes a model and everything it learned |
| `ui` | Opens the window — everything above, without the terminal |
| `web [QUERY]` | Searches the web, reads what it finds, and saves it as material |
| `bases` | Pretrained models worth starting from |
| `guide` | Prints instructions to paste into ChatGPT or Claude so it drives Teacher |

Add `--json` to any command for one machine-readable object instead of prose.

### Every option

| On | Flag | What it does |
|---|---|---|
| any | `--json` | One JSON object instead of prose |
| any | `--version` | Print the version |
| `new` | `--size` | Any parameter count (`70K`, `51M`, `1.5B`) or a rung — from scratch only |
| `new` | `--base` | Start from a pretrained model instead |
| `new` | `--context` | Context length in tokens, from scratch |
| `new` | `--replace` | Overwrite a model of the same name |
| `new` | `--no-teach` | Build it but do not train yet |
| `new` | `--trust-remote-code` | Let a base model run its own code — only for repos you trust |
| `new`, `teach` | `--from`, `--text`, `--raw` | Where the material comes from; `--raw` skips cleaning |
| `new`, `teach` | `--answers PATH` | Question-and-answer pairs, to teach it to reply rather than continue |
| `new`, `teach` | `--eval-from PATH` | Measure the held-out loss on separate text, not the tail of your own |
| `new`, `teach` | `--keep-duplicates` | Keep passages that appear in more than one source |
| `new`, `teach` | `--web`, `--web-results` | Gather material from the web first, and how many pages to read |
| `new`, `teach` | `--epochs`, `--batch` | Passes, and sequences per step (`auto` to fill the hardware) |
| `new`, `teach` | `--rate` | Learning rate — leave it out for the measured default, or `auto` to measure one |
| `new`, `teach` | `--until`, `--max-rounds`, `--max-minutes` | Teach in rounds until done, and the limits on that |
| `new`, `teach` | `--gpus N` | Spread the lesson across N GPUs, or `auto` for all of them |
| `new`, `teach` | `--device` | `auto`, `cpu` or `cuda` |
| `new`, `teach` | `--keep N` | How many saved states to keep behind the model (default 2) |
| `new`, `teach` | `--lora`, `--lora-rank N` | Train a small adapter instead of every weight (pretrained only) |
| `compare` | `--prompt`, `--tokens`, `--seed`, sampling | What to say, how much, and with which dice |
| `test` | `--items`, `--shots`, `--offline`, `--list` | How many questions, worked examples, and where the items come from |
| `branch` | `--at STAMP` | Branch from a saved state instead of the current weights |
| `rollback` | `--to STAMP` | Restore a named saved state instead of the newest |
| `ask` | `--tokens`, `--temperature`, `--top-p`, `--top-k` | How much to generate and how adventurously |
| `ask` | `--seed` | Repeat an exact answer |
| `web` | `--results`, `--url`, `--list`, `-o` | How many pages, extra addresses, look without downloading, where to keep them |
| `export` | `-o`, `--precision`, `--converter` | Where to write it, how much of each weight to keep, and where llama.cpp is |
| `card` | `-o`, `--show`, `--offline` | Where to write it, print it instead, and whether to look the base model's licence up |
| `serve` | `--host`, `--port`, `--api-key` | Where it listens (this machine only, by default) and the key every request must carry |
| `pack` | `-o`, `--out` | Where to copy the files |
| `forget` | `--yes` | Confirm the deletion |
| `ui` | `--port`, `--host`, `--share`, `--no-browser` | Where the window listens and whether it opens itself |

### Starting from a pretrained model

Building from scratch means the model learns English *and* your material out of
your text alone, which takes a lot of both. Adopting one that can already write
skips the first half:

```bash
python -m teacher new bookbot --base small --from ./my-books
python -m teacher bases          # what else is on offer
```

`--base` takes a shortcut (`small`, `gpt2`, `medium`) or any Hugging Face name.
The download happens once.

Progress is judged differently for these: the baseline is the model's own first
lesson rather than a uniform guess, so the stages read *barely moved → picking up
your material → adapting well → closely fitted*.

One limit: the Bench web page only runs models built from scratch here, so
`pack` refuses an adopted one and says so. Chat with it in the window or with
`teacher ask`.

### How big to make it

**`--size` takes any number of parameters you care to name.**

```bash
python -m teacher new bookbot --size 4M --from ./my-books
```

```
  asked for:  4.00M parameters
  built:      4.01M  (+0.2%)
  widths move in steps, so a number lands near rather than on
  shape:      160 wide, 10 layers, 5 heads
```

`70K`, `4m`, `51M`, `1.5B`, `250000` — all of them work, and all of them go
through the same search.

The parameter count of a transformer is exact arithmetic, so Teacher *searches*
for the architecture nearest your number rather than estimating one: depths
around a sensible aspect ratio, widths in steps of the head size. Most numbers
land within half a per cent. **It reports what it built, not what you asked
for** — reporting the number you asked for would be a small lie that compounds
every time someone repeats it.

Heads are 64 wide, the size attention kernels are tuned for. Narrower heads are
tried only when a 64-wide one cannot get within one per cent — for a very small
target the narrowest ordinary model is already bigger than you asked for.

The search runs against the vocabulary *your corpus actually produced*, which
is why it hits the number where a fixed preset would not: a preset sized for a
32,000-token vocabulary is ~18% light once a small corpus caps the vocabulary
at five thousand.

These are shortcuts, nothing more — each is just a number, and each goes
through the same search as one you type:

| Shortcut | Parameters | What it needs |
|---|---|---|
| `1m` | 1M | learns grammar in minutes on a laptop CPU |
| `10m` | 10M | a few MB of text and some patience; still fine on a CPU |
| `100m` | 100M | a GPU and a library's worth of text |
| `200m` | 200M | a GPU with 8GB or so, and a lot of text |
| `500m` | 500M | a GPU with room to spare and a great deal of text |
| `1b` | 1B | a serious GPU (24GB+) and gigabytes of text |

The older names still work: `tiny`, `small`, `medium`, `large` and `huge` mean
1m, 10m, 100m, 500m and 1b.

A number this machine cannot hold is refused before anything is built, with the
arithmetic and with what would fit:

```
Stopped: 100T parameters cannot be built here.
  The weights alone would be 372,529.0 GB at four bytes each, and this machine
  has 15.0 GB free.
  Training needs roughly four times the weights again, for gradients and the
  optimizer.
  The largest that would fit here is around 2.41B.
```

Start around `1M`. It will learn the shape of your text — sentence structure,
punctuation, the rhythm of the source — within minutes on a CPU. Every tenfold
step up needs roughly ten times the text and far more compute to be worth the
wait; a model too large for your material just memorises it.

Teacher will not start a run that cannot fit. Asking for `1b` on a laptop is
refused before anything happens:

```
Stopped: This lesson would not fit in memory:
  1019M parameters on CPU is not practical (expect days per epoch).
  Estimated 17.1 GB exceeds the 15.0 GB available.

Try:
  Choose a smaller model, or use LoRA on a small base
  Reduce batch size from 8 to 4
  Enable gradient checkpointing (~4x less activation memory)
```

### Making the hardware work

If your GPU sits at half load, it is almost always waiting rather than
thinking. Three things account for most of it, and Teacher now handles all
three.

**Precision.** Training runs in `auto`, which means bf16 on a card that
supports it, fp16 on one that does not, and fp32 on CPU where half precision is
slower and less reliable. Holding a modern GPU at fp32 roughly halves its
throughput and doubles its activation memory for nothing.

**TF32 and the tensor cores.** Turned on for every CUDA run, along with
`cudnn.benchmark` — the block size never changes during a lesson, so letting
cuDNN pick its best kernel once pays off for the rest of the run. On CPU,
Teacher instead sets the thread count to the cores you actually have.

**Batch size.** This is usually the real culprit: a small batch leaves the card
queueing tiny kernels and idling between them.

```bash
python -m teacher teach bookbot --from ./text --batch auto
```

```
  batch 96 is the largest that fits — 8.9 GB of the 14.5 GB available
```

It walks a ladder of batch sizes, estimates each with the same memory model the
preflight check uses, and stops one rung below the cliff. It also stops before
the batch grows so large that an epoch has too few optimizer steps to learn
anything — a full GPU and a wasted afternoon is still a wasted afternoon:

```
  batch 8 is the largest that fits — 8 keeps 28 steps in an epoch; a bigger
  batch would leave too few to learn from
```

When it says that, more material is what you need, not a bigger batch.

### What the training loop no longer wastes

A profile of a training step turned up five things spending time on nothing.
Each is fixed, and each fix was checked to leave the result unchanged:

| What it was doing | What it cost | Now |
|---|---|---|
| **Copying the logits to compute the loss.** Slicing the last position off a batch × sequence × vocabulary tensor and making it contiguous copied the whole thing every step, and the backward pass allocated a zero-filled one to scatter into. | 10% of every step was spent filling vocabulary-sized tensors with zeros | The *labels* are shifted instead. Loss and gradient are bit-identical; the loss and its backward run **1.62× faster** |
| **An attention mask of all ones.** Packed training data has no padding, yet every block carried a mask saying so. The model built a full sequence × sequence mask from it each step and called attention with `is_causal=False`. | On a GPU, that rules out the FlashAttention kernel entirely | No mask, so attention takes its causal fast path |
| **Waiting on the GPU every micro-batch.** Reading the loss into Python forces the CPU to wait for the GPU to finish, so it cannot queue the next step's work in the meantime. It did this once per micro-batch, and again for the gradient norm. | Two stalls per step, more with gradient accumulation | The loss stays on the device; one transfer per optimizer step carries both numbers |
| **Updating parameters one at a time.** | A loop over every weight tensor per step | Fused AdamW on CUDA: one kernel for all of them |
| **Saving on the very last step.** The crash-recovery record was also written at the end of the lesson — weights and optimizer state — moments before the finished lesson deleted it. | At 500M parameters, about 6 GB written for nothing | Skipped |

**Measured, on a CPU**, a real lesson start to finish (a 10M model, one epoch,
three runs each): **22.75 s before, 18.77 s after — 1.21× faster**, with the
held-out loss identical at 7.7886 in both. The numbers came out the same
because the arithmetic is the same.

**On a GPU it should be more, and that is not measured.** Three of the five
fixes — FlashAttention, the synchronisation stalls and fused AdamW — only exist
on a GPU, and this machine has none. The CPU figure is the floor, not the
estimate.

Two things were looked at and deliberately *not* changed:

* **Reloading the model each round.** An `--until` run reloads the model and
  re-tokenises the material every round, which looks wasteful. Measured, it is
  **2% of a round**. Not worth the complexity of keeping state between rounds.
* **`torch.compile`.** It can be a large win, but it needs Triton on a GPU,
  and Triton does not support Windows. Turning it on by default would break
  the platform most people double-click this on.

### More than one GPU

```bash
python -m teacher teach bookbot --from ./text --gpus auto
```

```
  2 GPUs over nccl: NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090
  effective batch 16 (8 per GPU across 2)
```

One process per card, each holding a full copy of the model and training on a
different slice of every epoch; PyTorch averages the gradients between them at
each backward pass. The result is *one* model trained on all the data, not two
models that have to be reconciled. Speedup is close to linear until the
gradient exchange starts to dominate, which for models this size it does not.

`--gpus` takes a number or `auto`. The default is 1. `--device auto|cpu|cuda`
picks where to train; `--device cpu` forces the CPU even when a card is
present, which is occasionally useful for a small model where the transfer
costs more than the compute.

Two things worth knowing:

**The batch you type is per GPU.** `--batch 8 --gpus 4` takes a gradient step
over 32 sequences. That is usually what you want — it is why it is faster —
but it changes the learning dynamics, so the effective number is printed.
`--batch auto` accounts for it, and counts an epoch's steps per process rather
than over the whole dataset.

**Windows uses gloo, not NCCL.** NCCL is not built for Windows. Gloo is
slower, but a slower two-card run still beats using one card, and Teacher
picks the right one without being asked.

**What will not help.**

*Using the CPU and the GPU together.* Every step ends with a gradient
exchange, so a step takes as long as the slowest process, and a CPU is one to
two orders of magnitude behind a GPU at this arithmetic — adding one makes the
GPUs wait. Splitting a model's layers across both is worse still, since
activations would cross the bus twice per step. Mixing the two is a way to
*fit* a model that does not fit, never a way to go faster, which is why it is
not offered.

Teacher also does not spin up DataLoader worker processes,
because the dataset is one tensor in memory and a batch is a slice of it —
workers would add process spawn and IPC cost to a memcpy. Raise `num_workers`
only for a dataset that reads from disk.

### Training options

`--epochs` (default 3), `--batch` (default 8) and `--rate` are on both `new` and
`teach`. Before any lesson starts, Teacher estimates the memory it needs and
refuses the run if it will not fit, with suggestions — it will not start
something that is likely to crash the machine.

`--rate` is best left out: the default depends on the kind of lesson, and was
measured rather than guessed — see [Learning rates](#learning-rates) below.

### Learning rates

One rate for every kind of lesson was wrong in both directions. Teaching a
pretrained model at 3e-4 made it **worse** on text it had not seen and made it
forget what it knew; building a model from scratch at 3e-4 left it learning a
fraction of what it could. Each default is now the best of lessons run to the
end at a spread of rates — the same material, seed and data order every time,
each number measured on the weights the lesson saved:

| Kind of lesson | Default | What the old 3e-4 did, measured |
|---|---|---|
| A pretrained base (`--base`) | **5e-5** | held-out 2.708 → 2.715 (worse), unrelated text 2.186 → 2.344 (forgot); at 5e-5: 2.557 and 2.152 |
| `--lora` on a pretrained base | **1e-3** | learned about four-fifths as much as 1e-3; at 3e-3 it began to forget |
| From scratch, up to 20M parameters | **3e-3** | a 1M model reached 7.77 held-out; at 3e-3, 6.85 |
| From scratch, larger | falls with size to 3e-4 | not measured — a rule of thumb |

`--rate 1e-4` sets one by hand, and every lesson records the rate it used and
where it came from. There is no `--rate auto`: a learning-rate range test was
built and measured against these tables first, and it picked a rate 15 times
too high for a pretrained model, noise for LoRA, and nothing better than the
default from scratch. The full tables, and that test, are in the
[manual](../docs/MANUAL.md#part-11--learning-rates).

### Training until it is actually done

Guessing at `--epochs` is how you end up with word soup. `--until` removes the
guess: Teacher teaches in rounds, checks the held-out loss after each, and stops
when the model reaches the stage you asked for **or** stops improving.

```bash
python -m teacher teach bookbot --from ./my-books --until sentences
```

```
  teaching until 'sentences' — 3 epoch(s) per round, at most 20
  every round is saved, so Ctrl-C is safe

  round  1  held-out  3.4958  #########                learning the alphabet
  round  2  held-out  2.5015  #############            learning words
  round  3  held-out  1.8004  ################         learning words
  round  4  held-out  1.2909  ##################       learning sentences
  round  5  held-out  1.0020  ###################      learning sentences
  round  6  held-out  0.8487  ####################     has the shape of your text

  rounds:        6 (18 epochs total)
  held-out loss: 3.4958 -> 0.8487
  took:          19s on cpu
  stopped:       it reached 'has the shape of your text'
```

Targets are `words`, `sentences` and `best` — `best` runs until it stops
improving, whatever stage that turns out to be. `--epochs` sets the size of one
round, `--max-rounds` caps how many (default 20) and `--max-minutes` puts a clock
on the whole thing.

It stops for one of four reasons, and says which: it reached the target, it
stopped improving, it ran out of rounds, or it ran out of time. A model already
past the target is left alone rather than trained pointlessly.

Every round saves the model, so Ctrl-C at any point leaves the last completed
round on disk. The whole run is written to the history as one entry, keeping each
round's loss inside it.

**A round that makes it worse is undone.** If a round ends with a higher held-out
loss than the round before it — or, for the first round, than the model had
before it started — its weights are rolled back, and the run stops and says so.
`rounds kept` in the output is how many rounds are in the saved weights. What is
left on disk is the best the run reached, not merely the last thing it did.

---

## Material from the web

If you have nothing to hand, Teacher can go and find some.

```bash
python -m teacher web "grace darling rescue 1838" --results 3
```

It searches, reads each page, strips it down to the text a person would read,
and writes every page to its own file:

```
Gathering from the web
  searching the web for 'grace darling rescue 1838'
    1/3  https://en.wikipedia.org/wiki/Grace_Darling
    2/3  https://www.missedhistory.com/article/grace-darling-lighthouse-rescue-1838
    3/3  https://rnli.org/about-us/our-history/timeline/1838-grace-darling
  kept 2 page(s), 25,091 characters, 1 skipped in ~/teacher-models/web-material/grace-darling-rescue-1838-20260919-134056
    skipped https://rnli.org/about-us/our-history/timeline/1838-grace-darling — the site answered 403
  each file records the address it came from — web pages carry their own terms

  2 page(s), 25,091 characters

  Teach from it:  python -m teacher teach NAME --from "~/teacher-models/web-material/grace-darling-..."
```

A page that refuses is named and skipped; the rest of the run carries on.

Every file starts with its title, its address and the date it was read:

```
The Lighthouse Keeper's Daughter Who Rowed Into a Storm | Missed History
Source: https://www.missedhistory.com/article/grace-darling-lighthouse-rescue-1838
Retrieved: 2026-09-19

At 4:45 in the morning on September 7, 1838, Grace Darling looked out from the
upper window of Longstone Lighthouse and saw a paddle steamer broken in half on
the rocks...
```

That header is not decoration. A page you found is someone else's writing under
someone else's terms, and a corpus with no record of where it came from cannot
be checked later. **A licence on this software is not permission to train on, or
redistribute, what you download with it** — that is yours to judge, per source.

| Flag | What it does |
|---|---|
| `--results N` | How many pages to read (default 5) |
| `--url ADDRESS` | Read this page too, or instead of searching. Repeatable |
| `--list` | Show what the search found and download nothing |
| `-o DIR` | Where to keep the files. The default is a dated folder under your models directory |

### Training a model on it, start to finish

```bash
# 1. Gather. Look first if you like — --list downloads nothing.
python -m teacher web "history of lighthouses" --list

# 2. Read the pages and keep them. The folder it prints is yours to keep.
python -m teacher web "history of lighthouses" --results 8

# 3. Train on that folder. Start from a pretrained model; see below for why.
python -m teacher new seabot --base small \
    --from "~/teacher-models/web-material/history-of-lighthouses-20260919-141500" \
    --until best

# 4. Talk to it.
python -m teacher ask seabot "The keeper climbed"
```

Steps 1–3 collapse into one command when you already know what you want:

```bash
python -m teacher new seabot --base small --web "history of lighthouses" --until best
```

The pages are still saved to a folder and the path is still printed — nothing
is fetched into memory and thrown away.

In the window it is the same four steps: **Text → …or fetch it from the web**,
type what you want and press *Search and keep*; the folder box fills in by
itself, so **Make** and **Teach** are ready to press.

### How much is enough

Six Wikipedia articles came to 206,000 characters — about 34,000 per page.
Ordinary web pages are shorter, nearer 10,000. So a five-page run is somewhere
around 50,000–150,000 characters, and that number decides which route works:

| | What 200,000 characters of web pages gets you |
|---|---|
| **From a pretrained model** | Works. It already writes English; the pages move its subject matter and register |
| **From scratch** | Does not work. It has to learn the language itself, and this is nowhere near enough |

Both runs below are real, on the same six articles, on a CPU:

```
from scratch, tiny:   held-out 7.32 -> 6.18 over 7 rounds, then stopped improving
                      "barely started — more epochs will not help, it needs more material"

from a pretrained
model, --base small:  held-out 2.60 after one pass, 4m 36s
                      "barely moved — it still writes like its base model"
```

Neither is a failure of the tool; they are two honest readings. From scratch
genuinely needs a few hundred KB at minimum — twenty to forty pages — and even
then `tiny` is the only tier worth attempting on a laptop. From a pretrained
model, *barely moved* after one pass over 200,000 characters is the expected
place to be: keep teaching, or gather more, and watch the held-out number.

**So: use `--base small` with web material unless you have gathered a lot.**

**What it will and will not do.** It searches DuckDuckGo, and falls back to
Wikipedia's own search when DuckDuckGo blocks or rate-limits the machine; if
neither answers it says so rather than returning nothing quietly. It reads only
the addresses a search returned or you named — it never follows links found
inside a page, so it cannot wander. Addresses on this machine or its local
network are refused, before and after redirects, so a page cannot redirect it
into `localhost`. Pages are read one at a time with a second between them, a
20-second timeout and a 3 MB cap.

**It does not make your model able to browse.** Deciding to search, choosing a
query and reading the answer back is something large instruction-tuned models
do; a model of 1M to 360M parameters will not. This gathers *material for
training*. AI Studio separately registers `web_search` and `read_web_page` as
tools, which are useful to a capable model you have imported — see
[ai_studio/README.md](../ai_studio/README.md).

---

## What a lesson actually does

1. Reads and cleans your material (page numbers, repeated headers, hyphens split
   across line breaks). Pass `--raw` to skip cleaning.
2. Tokenizes it, holding back the last 5% as text the model never trains on, so
   the reported loss is measured on writing it has not seen.
3. Measures the held-out loss **before the first step** — where this lesson
   starts from.
4. Runs a real training loop — forward, backward, gradient clipping, a cosine
   schedule with warmup — writing itself down about ten times on the way, so a
   crash can be resumed.
5. Measures the held-out loss again, **on the weights it is about to save**.
   That pair — before and after — is what it reports:
   `on held-out: 2.7079 -> 2.5567`. If the second number is higher, it says the
   lesson made the model worse and how to undo it.
6. Copies the previous weights into `checkpoints/` **before** overwriting them,
   so a disappointing lesson can be undone with `teacher rollback` or the window's
   *Roll back* button.
7. Appends what happened to `history.json` — including which weights the lesson
   started from and which it saved, which is how a model card can later tell a
   lesson that stuck from one that was rolled back. Nothing already recorded is
   rewritten.

Earlier versions reported the *lowest* held-out loss seen part-way through the
lesson, while saving the weights from the end. When a lesson went on too long,
those were different models, and the number described the one that was thrown
away. It now reports the model on disk; the lowest point is kept alongside
(`best_held_out`) and mentioned when it was clearly lower.

```
~/teacher-models/bookbot/
├── model.safetensors     the weights
├── config.json           the architecture
├── tokenizer.json        the tokenizer
├── history.json          every lesson, with its sources and losses
├── README.md             its model card, once you have run `teacher card`
├── checkpoints/          the two previous states, kept in case you want them back
└── .interrupted/         only while a lesson is unfinished — what `resume` needs
```

The first three files are all Bench needs, which is why `pack` is a copy rather
than a conversion.

### Comparing two models

Branching gives you two models from one ancestor. `compare` is how you tell
which one turned out better:

```bash
python -m teacher compare bookbot bookbot-v2 --prompt "The keeper" --tokens 60
```

```
Prompt  'The keeper'   (seed 0, 60 tokens, temperature 0.8)

bookbot  —  learning words
  4 lesson(s) · held-out 6.4433 · vocabulary 4,096
  The keeper of the Lighthouse,, the, in the a the of...

bookbot-v2  —  learning sentences
  6 lesson(s) · held-out 5.9012 · vocabulary 4,096 · branched from bookbot
  The keeper walked the stair and the lamp turned through the night...

  Lowest held-out loss: bookbot-v2 (5.9012)
```

The seed is the same for every model, so a difference in what comes back is a
difference in the models and not in the dice.

**On comparing the numbers.** A held-out loss is an average over a model's
vocabulary. Two models that carve text up differently are not being scored on
the same scale, so `compare` says so and refuses to pick a winner:

```
  These use different vocabularies, so their losses are not comparable — a loss
  is an average over the tokens a model has, and these carve text up
  differently. Judge by reading.
```

Even with matching vocabularies, each model's loss was measured on *its own*
held-out text. The comparison only means something if you taught them the same
material — which is exactly the case after a branch.

### Training a big model on a small machine: LoRA

Fine-tuning normally updates every weight, which needs memory for the weights,
their gradients and the optimizer's two moments — roughly four copies. LoRA
freezes the model and trains a small adapter instead:

```bash
python -m teacher teach seabot --from ./material --lora --lora-rank 8
```

```
  adapter:       rank 8 on q_proj, k_proj, v_proj, o_proj
                 921.6K of 135.44M weights trained (0.68%), then merged in
```

That is a real run: 921,600 trainable parameters out of 135 million.

**The adapter is folded back in when the lesson ends.** Teacher keeps one
model per folder, and an adapter on its own would leave that folder unable to
load — quietly breaking `rollback`, `branch`, `pack` and Bench. What LoRA buys
here is a cheaper lesson, not a different kind of artefact.

`--lora-rank` sets how much the adapter can change: 8 is thrifty, 16 is the
default, 64 learns more and costs more.

**Only for pretrained models.** LoRA adapts something that already knows a
language. A model built here starts from noise, so there is nothing to adapt —
Teacher refuses and says so rather than wasting your afternoon:

```
Stopped: LoRA trains a small adapter on top of a model that already knows a
language. This one was built here, starting from noise — there is nothing to
adapt yet, so train all of it instead (drop --lora).
```

LoRA lowers the optimizer and gradient cost, not the activation cost. A long
sequence or a big batch still needs the memory it always did, which is what
the preflight check is for — and it now accounts for depth, vocabulary and MLP
width rather than guessing from a parameter count.

### Trusting the number

Three things decide whether the held-out loss means what it looks like it
means.

**What it is measured on.** By default it is the last 5% of your own text —
never trained on, but from the same source, in the same voice, often about the
same thing. A model that memorised your corpus still scores well on it.
`--eval-from` measures against writing from somewhere else entirely:

```bash
python -m teacher teach bookbot --from ./my-books --eval-from ./other-books
```

```
  measuring against 1 source(s), 35,354 characters of separate text
  on held-out:   6.6808  (perplexity 796.9)
```

There is nothing in that file to have memorised, so the number is about
learning rather than recall. The lesson records which was used, so you can tell
later which kind of number you are looking at.

**Whether it saw the same thing twice.** The cleaner already drops repeated
paragraphs inside a document. Across sources it could not — the same book in
two folders, a chapter that is also on a web page you gathered — and repetition
teaches a model to recite. Repeats are now dropped, and counted:

```
  dropped 340 repeated passage(s), 412,908 characters — the same text twice
  teaches memorising
```

Only passages of 200 characters or more, because a heading or a refrain repeats
for honest reasons and dropping it would quietly edit your text.
`--keep-duplicates` turns it off.

**Whether you get to see it at all.** A lesson now writes itself down about ten
times as it runs — the weights, the optimizer's momentum, the schedule and the
step it reached — so a crash costs minutes rather than the whole round:

```bash
python -m teacher resume bookbot
```

```
bookbot — interrupted at step 312/1040 on 2026-09-22 14:56, 40 epoch(s) over 207,173 characters
  restored the weights saved at step 312, with the optimizer's state and the place in the schedule and the data
```

Everything comes back: the weights, the optimizer's momentum, the place in the
learning-rate schedule and the place in the data. For a model without dropout
an interrupted-and-resumed lesson ends on the same weights as one that ran
straight through — a test checks that to a millionth. (An earlier version
restored only the weights and the step count: the momentum started from
nothing, the schedule went back to its warm-up, and the data order started
again from the first batch.)

The record is deleted when a lesson finishes, so a record that is there means a
lesson that is not. The material is fingerprinted when the run starts and
checked when it resumes — if it changed, resuming would continue onto a
different text than the run was part-way through, so it refuses and says to
start again. Answer lessons are not resumable yet, and say so rather than
appearing to work.

### Teaching it to answer, not just continue

A model trained on plain text learns to *continue* it. Give it half a sentence
and you get the other half. That is what a language model is, and it is not
what most people mean when they ask a question.

Instruction tuning is the difference. Give it pairs — a question and its
answer — and it learns the **answer only**:

```bash
python -m teacher teach bookbot --answers ./pairs.jsonl --epochs 12
```

```
320 pair(s), 30,240 characters, qa style
on held-out:   0.1623  (perplexity 1.2)
```

```
$ python -m teacher ask bookbot "What is the capital of France?"
The capital of France is Paris.
```

That is a real 2M model, built from scratch on this machine. Before the
pairs it could only continue text.

**The file.** Any of these shapes, as `.jsonl`, `.json` or `.csv`, and folders
are walked:

```json
{"question": "...", "answer": "..."}
{"instruction": "...", "input": "...", "output": "..."}
{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
```

`prompt`/`response` and `q`/`a` are understood too. A row with only half a
pair is not a pair, and is skipped.

**Why only the answer.** The loss is masked over the question, so the model is
never rewarded for reproducing what you asked — only for what comes after it.
That masking is the whole mechanism. Get it wrong by a single token and the
model never learns which word *starts* an answer, so it emits end-of-sequence
and returns nothing; it looks exactly like training that failed. (It was wrong
by a single token. There is a test now.)

**Ask it the way it was taught.** The template used in training is written into
the model, and `teacher ask` puts your question into that same template. A
model taught with `### Question:` and then asked something bare answers as if
continuing a document.

Instruction tuning runs on one GPU for now — `--gpus` is ignored for an answer
lesson, and says so rather than training the wrong thing quietly.

### Using it somewhere else: GGUF

```bash
python -m teacher export bookbot --precision f16
```

```
/home/you/bookbot-f16.gguf  258.3MB
/home/you/bookbot-f16.md  its model card

llama.cpp:  llama-cli -m /home/you/bookbot-f16.gguf -p "your prompt"
Ollama:     printf 'FROM /home/you/bookbot-f16.gguf\n' > Modelfile && ollama create bookbot -f Modelfile
LM Studio:  put it in your models folder and pick it from the list
```

GGUF is what llama.cpp, Ollama and LM Studio read. The conversion is done by
**llama.cpp's own converter**, because it is the thing that knows how to write
the format and a second, worse copy of it here would rot.

So you need it:

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
```

It is looked for beside this repository automatically; `--converter PATH` or
the `LLAMA_CPP_CONVERT` variable point at it anywhere else. **If it is not
there, nothing is written and the command says so** — it will not leave a file
named `.gguf` that llama.cpp cannot open. If the converter itself refuses a
model, its reason is printed verbatim and nothing is written either.

`--precision` is `f16` (the usual choice), `bf16`, `f32` or `q8_0`. Smaller
than that is a second step through llama.cpp's own `llama-quantize`, which is
not run from here.

**Only for models started from a pretrained base.** llama.cpp implements the
architectures it knows; a model built from scratch here is not one of them, and
`export` refuses it and points at `teacher pack` and the Bench page instead.

Every export writes the model's card beside the file (`bookbot-f16.md`), with
the file's size and SHA-256 added, and notes the export in the model's record.
A file of that name that Teacher did not write is left alone.

### What it is, and where it came from: model cards

```bash
python -m teacher card bookbot
```

```
Model card for bookbot
  /home/you/teacher-models/bookbot/README.md
  lessons in these weights: 2  (1 undone, listed apart)
  material: 7 source(s), 6 of them web pages
  benchmark results on these weights: 1
  base model licence: apache-2.0
  no licence is chosen for the model itself — that is yours to decide
```

A model card is what someone reads before trusting a model they did not make.
Teacher writes one from the model's own record — `README.md` in its folder,
which is the file the Hugging Face Hub shows as a model's page — and nothing
in it is estimated:

- **What it is**: parameters (counted from the weights file), architecture,
  vocabulary, context, and what it started from.
- **What it was taught**: every lesson still in these weights, with its date,
  kind, amount, epochs, steps, learning rate, held-out loss and what that was
  measured on. A lesson that was rolled back is still in the record — it
  happened — but it is not in the weights, so the card lists it apart, as
  undone. Every lesson notes the weights it started from and the weights it
  saved, and the card follows those links back from the weights on disk.
- **Where the material came from**: web pages by title, address and the date
  they were read; local files by name only. Paths stay on your machine — a
  card is for sharing, and your folder layout is not.
- **How it scores**: benchmark results taken on *these* weights, each beside
  the score guessing would get and whether the difference means anything.
  A result taken before the last lesson describes a model that no longer
  exists, and is left out.
- **Licence**: Teacher's licence covers Teacher, not the weights and not the
  material. The card reports the base model's licence as its authors declare
  it on the Hub (looked up once, `--offline` to skip), says web pages keep
  their authors' terms, and **chooses no licence for the model** — the
  `license:` field is left empty for the person who knows what the material
  was.

`--show` prints it instead of writing it; `-o FILE` writes it elsewhere. A
`README.md` that Teacher did not write is never overwritten.

### Letting other programs use it: `teacher serve`

```bash
python -m teacher serve bookbot
```

```
Serving bookbot
  http://127.0.0.1:8008/v1
  point any OpenAI client at that address, with the model's name as the model
  Ctrl+C stops it
```

A local API in OpenAI's format, so anything written for OpenAI's API — chat
front-ends, editor plugins, scripts using the `openai` package — can use your
model instead:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8008/v1", api_key="unused")
reply = client.chat.completions.create(
    model="bookbot", messages=[{"role": "user", "content": "Who kept the light?"}])
print(reply.choices[0].message.content)
```

| Route | What it does |
|---|---|
| `GET /v1/models` | The models being served |
| `POST /v1/completions` | Continues a prompt |
| `POST /v1/chat/completions` | Replies to a conversation |
| `GET /health` | Whether it is up |

Both generation routes stream with `"stream": true`, and take `max_tokens`
(capped at 2,048 and at what the context has room for), `temperature`,
`top_p`, `stop` (up to four), `seed` and `repetition_penalty`. Tested against
the official `openai` Python package, streaming included.

- **The template is applied for you.** A model taught with `--answers` is asked
  in the form it was taught, and each earlier turn of a conversation is laid
  out the same way. A model taught only text gets the conversation as a plain
  `User:` / `Assistant:` transcript to continue — it will continue it, but it
  was never taught to reply; `--answers` is what teaches that.
- **One reply at a time.** Two at once would each run at half the speed in
  twice the memory, so requests wait their turn.
- **It keeps up with lessons.** Teach the model in another terminal and the
  next request uses the new weights — no restart.
- **It listens on this machine only.** `--host 0.0.0.0` opens it to your
  network; do that only with `--api-key KEY` (or `TEACHER_API_KEY`, which
  keeps the key out of your shell history), and every request must then carry
  `Authorization: Bearer KEY`.
- **Errors come back in OpenAI's shape**: a model name that is not being
  served is a 404 naming the ones that are; a prompt longer than the model's
  context is a 400 with `context_length_exceeded`.

Several models at once: `teacher serve bookbot seabot` — the request's
`model` picks between them. The window's **Keep** tab has the same thing with
a start and a stop button.

### Is it any good? Benchmarks

The held-out loss says whether a model is still learning. It says nothing
about whether the model knows anything. For that there are nine real suites,
the same ones AI Studio runs:

```bash
python -m teacher test --list
python -m teacher test bookbot arc --items 20
```

```
ARC-Challenge  —  7/20 (35%)
guessing would score  25%
0 worked example(s) in the prompt, 76s

Indistinguishable from guessing. A model this size is expected to score at
chance on this; it is not a fault.
Benchmarks like this measure knowledge a model of this size was never going
to hold.
```

That is a real run of SmolLM2-135M against the official ARC-Challenge split,
and it is worth looking at closely. **35% against a 25% chance rate looks like
a result, and is not one.** Twenty questions at a quarter has an expected
score of five and a standard deviation of about two, so anything under nine
right is inside the noise. A tool that printed "35%" and stopped would have
told you something untrue. **Expect this.** A model of one to a few hundred million
parameters scores at chance on knowledge benchmarks; they were built to
separate models a thousand times larger. A score here is not a verdict on your
training — it is a verdict on the size of the thing you trained.

Three things this command insists on:

**Where the questions came from.** Each suite loads the official split through
`datasets`, or a local `.jsonl` you provide, and falls back to a handful of
bundled example items when neither is available. Six questions is a check that
the plumbing works, and it is labelled as such rather than reported as a score.
`test --list` shows which you would get. `pip install datasets` for the real
ones.

**What chance looks like.** A four-choice question is 25% for a model that has
learned nothing. That number is printed beside every score, because 25% on its
own reads like a result.

**Whether the number means anything.** With twenty items a score has to clear
about two standard deviations of the chance rate before it is evidence of
anything. Below that the command says so instead of letting you read a lead
that is noise — 8 out of 20 against a quarter is not a finding; 160 out of 400
is.

| Suite | Measures |
|---|---|
| `mmlu`, `mmlu_pro` | knowledge and reasoning across academic subjects |
| `arc` | grade-school science |
| `gsm8k` | multi-step arithmetic |
| `hellaswag`, `winogrande` | commonsense and pronoun resolution |
| `truthfulqa` | questions where the common answer is wrong |
| `humaneval`, `mbpp` | writing Python |

`--items N` sets how many questions, `--shots N` how many worked examples go in
the prompt (the default is whatever the suite specifies), and `--offline`
skips the download and uses whatever is already here.

Generation is greedy — temperature zero — so a run repeats exactly. The model
is loaded once for the whole run rather than once per question.

Each result is kept in the model's record — the score, not every answer — tied
to the weights it was taken on, so `teacher card` can show it for as long as
those weights are the ones on disk.

### Saved states: going back, and branching off

Teacher copies the weights aside before every lesson. Those copies are what
make experimenting safe — a lesson that ruins a model is one command away from
being undone, and a promising one is one command away from being explored two
ways at once.

```bash
python -m teacher checkpoints bookbot
```

```
bookbot — 4 saved state(s)
  written before each lesson; the newest is what rollback restores

  STAMP             TAKEN                    SIZE
  20260919-172847   2026-09-19 17:28        5.1MB
  20260919-172902   2026-09-19 17:29        5.1MB
  20260919-172915   2026-09-19 17:29        5.1MB
  20260919-172927   2026-09-19 17:29        5.1MB
```

**Going back** restores one of them over the model's current weights:

```bash
python -m teacher rollback bookbot                      # the newest
python -m teacher rollback bookbot --to 20260919-172902 # a particular one
```

The history is left alone either way: it records what happened, and the lesson
did happen.

**Branching** is the other direction — carry on *from* a saved state without
giving up where you are now:

```bash
python -m teacher branch bookbot bookbot-v2 --at 20260919-172902
python -m teacher teach bookbot-v2 --from ./different-material --until best
```

`bookbot` is not touched at all. You now have two models from the same
ancestor, trained differently, and you can compare them with `teacher ask` and
keep whichever is better. Drop `--at` to branch from the current weights.

A branch starts with an empty lesson list, because its weights have not had
those lessons *in the form the branch holds them*. The original's record is
kept inside it under `branched_from_history` rather than thrown away, and
`teacher show` names where it came from.

**How many are kept.** Two by default, because each one is a full copy of the
weights — four states of a 500M model is 8 GB. Raise it when you are about to
experiment:

```bash
python -m teacher teach bookbot --from ./text --epochs 3 --keep 10
```

In the window this is all under **Keep → Saved states**: pick a state, then
either *Roll this model back to it* or name a copy and press *Branch*.

---

## Reading the numbers

```
  learned from:  59.9K tokens over 4 epoch(s)
  loss:          0.9464
  on held-out:   0.9528  (perplexity 2.6)
  took:          11.4s on cpu
```

**Loss** is how surprised the model was by the next token, averaged. Lower is
better. An untrained model sits near `ln(vocabulary size)` — about 5.9 for a
369-token vocabulary — so if your first lesson starts near that number, the
pipeline is working.

**Held-out loss** is the same measure on text the model never trained on. If it
stops falling while the training loss keeps dropping, the model has started
memorising rather than learning; stop there, or give it more material.

**Perplexity** is just `e^loss` — roughly, how many tokens it was choosing
between. 2.6 means it was almost sure each time.

After every lesson Teacher reads those numbers against `ln(vocabulary)` — the
loss of a model that is still guessing — and says where the model actually
stands, with the next command to run:

| Share of the untrained baseline | What it writes |
|---|---|
| above 70% | noise, and broken characters where it picks half a letter |
| 55–70% | word fragments |
| 28–55% | whole words, in no particular order |
| 15–28% | sentences in the shape of your material |
| under 15% | about as far as a model this size can go on this material |

Those bands are calibrated against real runs, not guessed. If a lesson leaves
the model in the top band, it will say so rather than claiming it is ready.

---

## What to expect

A model this size learns the *shape* of your text, not its content. Teaching a
one-million-parameter model on a few hundred KB gives you grammatical sentences
in the style of the source, not a chatbot that knows things. That is the honest
result, and Teacher will not dress it up as more.

For something that answers questions, you need orders of magnitude more
parameters, text and compute — and at that point you want AI Studio's
fine-tuning screen and an existing base model rather than training from scratch.

---

## Troubleshooting

**`TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'`**
An older copy of Teacher on Gradio 6. Gradio 6 removed several things Gradio 5
required, so the window would not open at all. Update the repository — Teacher
now works on both. There is nothing to install or downgrade.

**`ValueError: Slider minimum must be less than maximum.`**
The same story, one step later: the GPU slider ran from 1 to the number of GPUs,
and on a machine with one GPU or none that is 1 to 1, which Gradio 6 refuses —
so the window would not open. Update the repository; with fewer than two GPUs
the window now shows the count instead of a slider.

**"This lesson made it worse on text it did not train on"**
The held-out loss after the lesson is higher than before it. Undo it with
`teacher rollback NAME`, then teach again with fewer `--epochs` or a lower
`--rate`. If you set `--rate` yourself, try leaving it out — see
[Learning rates](#learning-rates).


**"Only N characters of material — too little to learn anything"**
Give it at least a few thousand characters; a few hundred KB is a sensible start.

**"The material is only N tokens, and one training block is 128"**
Add more material, or build the model with a shorter context:
`teacher new small-ctx --context 64 --from ...`

**"This lesson would not fit in memory"**
The preflight check caught it before the crash. Follow the suggestions it prints
— a smaller batch is usually enough.

**The model repeats itself**
Raise `--temperature`, or teach it more varied material. A small model trained on
repetitive text will faithfully reproduce that repetition.

**The output has ◆ or � characters in it**
Not an encoding problem. The tokenizer works on bytes, so some tokens are half of
a multi-byte character; a model that is still guessing picks them at random and
half a character cannot be shown. They disappear as it learns — teach it more.

**I want the previous version back**
Copy the three files out of `checkpoints/before-<timestamp>/` over the ones in
the model folder.
