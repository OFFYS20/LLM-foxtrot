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

Run it from the repository root:

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
| `forget NAME --yes` | Deletes a model and everything it learned |

### Sizes

Each tier is ten times the one below it:

| Size | Parameters | What it needs |
|---|---|---|
| `tiny` | 1M | learns grammar in minutes on a laptop CPU |
| `small` | 10M | a few MB of text and some patience; still fine on a CPU |
| `medium` | 100M | a GPU and a library's worth of text |
| `large` | 1B | a serious GPU (24GB+) and gigabytes of text |

Start with `tiny`. It will learn the shape of your text — sentence structure,
punctuation, the rhythm of the source — within minutes on a CPU. Each step up
needs roughly ten times the text and far more compute to be worth the wait; a
tier too large for your material just memorises it.

Teacher will not start a run that cannot fit. Asking for `large` on a laptop is
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

## What a lesson actually does

1. Reads and cleans your material (page numbers, repeated headers, hyphens split
   across line breaks). Pass `--raw` to skip cleaning.
2. Tokenizes it, holding back the last 5% as text the model never trains on, so
   the reported loss is measured on writing it has not seen.
3. Runs a real training loop — forward, backward, gradient clipping, a cosine
   schedule with warmup.
4. Copies the previous weights into `checkpoints/` **before** overwriting them,
   so a disappointing lesson can be rolled back by hand.
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
