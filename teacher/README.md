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
`.epub`, `.html`, `.csv` and `.json` — a folder is walked recursively. Pass it
more than once to combine sources, or use `--text "..."` for something short.

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
| `new` | `--size` | `tiny`, `small`, `medium`, `large`, `huge` — from scratch only |
| `new` | `--base` | Start from a pretrained model instead |
| `new` | `--context` | Context length in tokens, from scratch |
| `new` | `--replace` | Overwrite a model of the same name |
| `new` | `--no-teach` | Build it but do not train yet |
| `new` | `--trust-remote-code` | Let a base model run its own code — only for repos you trust |
| `new`, `teach` | `--from`, `--text`, `--raw` | Where the material comes from; `--raw` skips cleaning |
| `new`, `teach` | `--web`, `--web-results` | Gather material from the web first, and how many pages to read |
| `new`, `teach` | `--epochs`, `--batch`, `--rate` | Passes, sequences per step, learning rate |
| `new`, `teach` | `--until`, `--max-rounds`, `--max-minutes` | Teach in rounds until done, and the limits on that |
| `ask` | `--tokens`, `--temperature`, `--top-p`, `--top-k` | How much to generate and how adventurously |
| `ask` | `--seed` | Repeat an exact answer |
| `web` | `--results`, `--url`, `--list`, `-o` | How many pages, extra addresses, look without downloading, where to keep them |
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

### Sizes (from scratch)

| Size | Parameters | What it needs |
|---|---|---|
| `tiny` | 1M | learns grammar in minutes on a laptop CPU |
| `small` | 10M | a few MB of text and some patience; still fine on a CPU |
| `medium` | 100M | a GPU and a library's worth of text |
| `large` | 500M | a GPU with room to spare and a lot of text |
| `huge` | 1B | a serious GPU (24GB+) and gigabytes of text |

Start with `tiny`. It will learn the shape of your text — sentence structure,
punctuation, the rhythm of the source — within minutes on a CPU. Each step up
needs roughly ten times the text and far more compute to be worth the wait; a
tier too large for your material just memorises it.

Teacher will not start a run that cannot fit. Asking for `huge` on a laptop is
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

### Training options

`--epochs` (default 3), `--batch` (default 8) and `--rate` (default 3e-4) are on
both `new` and `teach`. Before any lesson starts, Teacher estimates the memory it
needs and refuses the run if it will not fit, with suggestions — it will not
start something that is likely to crash the machine.

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

`new` and `teach` take `--web "SOMETHING"` to do both steps at once:

```bash
python -m teacher new seabot --base small --web "history of lighthouses" --until best
```

The pages are still saved to a folder and the path is printed — nothing is
fetched into memory and thrown away.

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
3. Runs a real training loop — forward, backward, gradient clipping, a cosine
   schedule with warmup.
4. Copies the previous weights into `checkpoints/` **before** overwriting them,
   so a disappointing lesson can be undone with `teacher rollback` or the window's
   *Roll back* button.
5. Appends what happened to `history.json`. Nothing already recorded is rewritten.

```
~/teacher-models/bookbot/
├── model.safetensors     the weights
├── config.json           the architecture
├── tokenizer.json        the tokenizer
├── history.json          every lesson, with its sources and losses
└── checkpoints/          the two previous states, kept in case you want them back
```

The first three files are all Bench needs, which is why `pack` is a copy rather
than a conversion.

### Undoing a lesson

```bash
python -m teacher rollback bookbot
```

Restores the weights saved before the most recent lesson. The history is left
alone: it records what happened, and the lesson did happen. Two earlier states
are kept, so this can be done twice before the older one is gone.

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
