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

```
  tiny       1.3M  about a million parameters — learns grammar in minutes on a laptop
  small     10.1M  about ten million — needs a few MB of text and some patience
  medium    48.7M  about fifty million — wants real hardware and a lot of text
  large    102.5M  about a hundred million — a GPU and a large library
```

Start with `tiny`. It will learn the shape of your text — sentence structure,
punctuation, the rhythm of the source — within minutes on a CPU. The larger
sizes need proportionally more text and a GPU to be worth the wait.

### Training options

`--epochs` (default 3), `--batch` (default 8) and `--rate` (default 3e-4) are on
both `new` and `teach`. Before any lesson starts, Teacher estimates the memory it
needs and refuses the run if it will not fit, with suggestions — it will not
start something that is likely to crash the machine.

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

**I want the previous version back**
Copy the three files out of `checkpoints/before-<timestamp>/` over the ones in
the model folder.
