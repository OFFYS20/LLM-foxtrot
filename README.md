# LLM Foxtrot

Four programs for making, training and using your own language models. Everything
runs on your machine; nothing is sent anywhere.

| | What it is | Who it is for |
|---|---|---|
| **[Teacher](teacher/README.md)** | Make a model, teach it your writing, talk to it — as a window or a command | Start here |
| **[Bench](web_chat/README.md)** | One HTML file that runs your model in a browser | Sharing a model, or chatting with no install |
| **[AI Studio](ai_studio/README.md)** | The full workbench: datasets, tokenizers, LoRA, RAG, benchmarks | When you want every dial |
| **Foxtrot** | A web platform for teams, with a FastAPI backend and a Next.js front end | Running this as a service |

---

## Start here

**Windows:** double-click **`Start-Teacher.bat`**.
**macOS / Linux:** run **`./start-teacher.sh`**.

The first run installs what it needs (a few minutes), then Teacher opens in your
browser. There is nothing else to set up.

If you would rather do it by hand:

```bash
git clone -b claude/foxtrot-llm-platform-edxmef https://github.com/OFFYS20/LLM-foxtrot.git
cd LLM-foxtrot
python3.11 -m venv ai_studio/.venv
source ai_studio/.venv/bin/activate      # Windows: ai_studio\.venv\Scripts\activate
pip install -r ai_studio/requirements.txt
python -m teacher ui
```

---

## Your first model, in the window

1. **Make** — give it a name, choose **Start from a pretrained model**, press *Make it*.
   A model that already writes English downloads once (about 270 MB).
2. **Text** — drop in files, point at a folder, or paste. Press *Check what this
   adds up to* to see what it found.
3. **Teach** — leave *Keep going until it is done* ticked and press *Teach*. It
   trains in rounds and stops on its own when it stops improving.
4. **Chat** — talk to it.

That is the whole loop. **Keep** holds the rest: copy the files for Bench, roll
back a lesson you regret, or delete a model.

Teacher saves the weights before every lesson, so one that made things worse is a
button — or `teacher rollback NAME` — away from being undone. You can also
**branch** a saved state into a new model and train that instead, which leaves
the one that already works exactly as it is.

### From scratch instead

Choose **Build from scratch** to watch a model learn a language from nothing.
It is slower to anything readable and the ceiling is much lower, but you see the
whole thing happen — and only these models run in Bench.

| | From scratch | From a pretrained model |
|---|---|---|
| First useful output | tens of minutes | minutes |
| Needs | a few hundred KB of text | a few KB is enough |
| Learns | the language *and* your material | your material |
| Runs in Bench | yes | no — chat in Teacher |
| Sizes | 1M / 10M / 100M / 200M / 500M / 1B | whatever you adopt |

---

## The same things from a terminal

Everything in the window is a command, and the two share one store — make a model
in the window, teach it from a script.

```bash
python -m teacher ui                                   # open the window
python -m teacher bases                                # pretrained models to start from
python -m teacher new bookbot --base small --from ./my-books
python -m teacher teach bookbot --from ./more --until best
python -m teacher ask bookbot "once upon a"
python -m teacher web "victorian lighthouses" --results 5   # material from the web
python -m teacher list
python -m teacher show bookbot
python -m teacher pack bookbot -o ./for-bench
python -m teacher compare bookbot bookbot-v2           # two models, same prompt, same seed
python -m teacher checkpoints bookbot                  # saved states, one per lesson
python -m teacher branch bookbot bookbot-v2 --at ...   # carry on from one, safely
python -m teacher rollback bookbot                     # undo the last lesson
```

`--from` takes a file or a folder and reads `.txt`, `.md`, `.pdf`, `.docx`,
`.epub`, `.html`, `.csv` and `.json`. Full reference: **[teacher/README.md](teacher/README.md)**.

### No text of your own?

`teacher web` searches, reads what it finds and saves each page as a text file —
or `--web "something"` on `new` and `teach` does it in one step. Every file keeps
the address it came from and the date, because a page you found is someone
else's writing under someone else's terms. It reads only what a search returned
or you named, never links inside a page, and refuses anything on this machine or
its local network. The same thing is in the window, under **Text → …or fetch it
from the web**.

```bash
python -m teacher web "history of lighthouses" --results 8       # gather
python -m teacher new seabot --base small --from "<the folder it printed>" --until best
python -m teacher ask seabot "The keeper climbed"
```

Use `--base` with web material. A handful of pages is plenty to fine-tune a
model that already writes English, and nowhere near enough to teach one the
language from scratch — the walkthrough in
[teacher/README.md](teacher/README.md#how-much-is-enough) shows both runs and
what each actually reaches.

This gathers *material*. It does not make a small model able to browse — that is
an ability of far larger instruction-tuned models. AI Studio registers
`web_search` and `read_web_page` as tools for a capable model you have imported.

---

## Letting an AI assistant do it for you

**[docs/OPERATING.md](docs/OPERATING.md)** is written to be pasted whole into
ChatGPT or Claude. It covers all three programs, every command and flag, how to
read the results, and what to tell you when something goes wrong. Paste it,
then say what you want.


If you would rather not run anything yourself, Teacher can write the instructions
for an assistant that can:

```bash
python -m teacher guide
```

Paste what it prints into ChatGPT, Claude or any assistant with access to your
machine, then say what you want — *"train a model on the book in my Documents
folder"*. The guide tells it where everything is, which commands exist, how to
read the results, and what not to do.

It works because every command takes `--json` and answers with one object:

```bash
python -m teacher --json teach bookbot --from ./my-books --until best
```

```json
{
  "ok": true,
  "command": "teach",
  "model": "bookbot",
  "stage": "adapting well",
  "held_out_loss": 2.41,
  "rounds": 5,
  "reason": "it stopped improving"
}
```

Failures come back the same shape with `"ok": false` and an `"error"`, so an
assistant can tell you what went wrong instead of guessing.

**Every command, on every path, prints exactly one object** — an empty workspace,
a model that was built but not taught, a failure. Nothing else reaches stdout,
not even during a live training run, so there is nothing for an assistant to
misread. Twenty-five tests hold that contract in place.

---

## Reading the numbers

Teacher measures every lesson on text the model never trained on, and says where
it stands rather than claiming it is finished.

**Built from scratch** — the baseline is a model still guessing:

> barely started → learning the alphabet → learning words → learning sentences → has the shape of your text

**Started from a pretrained model** — the baseline is its own first lesson, since
it could already write before you began:

> barely moved → picking up your material → adapting well → closely fitted

The number behind it is the **held-out loss**. Falling between lessons means it is
still learning. Flat means it has stopped, and more training will not help — it
needs more material.

---

## What to expect

A small model learns the *shape* of your text: sentence structure, punctuation,
the rhythm of the source. Starting from a pretrained model gets you fluent
sentences in your register. Neither gives you something that knows facts or
answers questions reliably — that needs orders of magnitude more parameters,
text and compute than a home machine has.

Nothing here will pretend otherwise. If a benchmark shows a score, the benchmark
ran. If a checkpoint is listed, the weights are on disk. If a hardware reading is
unavailable, it says *Unavailable* rather than inventing a number.

---

## The other two programs

### Bench — your model in a browser

`web_chat/index.html` is one file with no install and no server. Drop in a model
folder and talk to it; the weights are read, run and sampled in the page.

It implements the from-scratch architecture in JavaScript — RMSNorm/LayerNorm,
SwiGLU/GELU/SiLU/ReLU, RoPE or learned positions, grouped-query attention, a KV
cache — and its output matches PyTorch to six decimal places on the same input.
It also shows the five tokens the model weighed for each one it wrote.

→ **[web_chat/README.md](web_chat/README.md)**

### AI Studio — the full workbench

A fifteen-screen Gradio application for when you want every dial: document
cleaning, dataset building, tokenizer training, LoRA and QLoRA fine-tuning,
retrieval over your own documents, nine benchmark suites, experiment tracking,
checkpoint management and hardware monitoring.

```bash
python -m ai_studio.app
```

→ **[ai_studio/README.md](ai_studio/README.md)**

---

## Requirements

* **Python 3.11 or newer**
* ~2 GB of disk for dependencies, plus what your models and text need
* A CUDA GPU is optional. Everything runs on CPU — models up to roughly 50M
  parameters train in reasonable time; bigger ones are refused before they start,
  with an explanation.

For an NVIDIA GPU, install the CUDA build of PyTorch **first**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r ai_studio/requirements.txt
```

---

## Where things are kept

```
LLM-foxtrot/
├── Start-Teacher.bat / start-teacher.sh   one-press launchers
├── teacher/            the trainer: window, commands, tests
├── web_chat/           Bench — the single-file browser page
├── ai_studio/          the full workbench
├── backend/ frontend/  Foxtrot, the team platform
└── docs/
```

Your models live outside the repository, in `~/teacher-models` (set `TEACHER_HOME`
to move them). Each one is a folder holding its weights, architecture, tokenizer
and a record of every lesson it has had.

---

## Tests

```bash
python -m pytest teacher/tests ai_studio/tests -q
cd backend && .venv/bin/python -m pytest tests -q
```

The training tests run real PyTorch steps and assert the loss beats an untrained
model. If training were simulated, they would fail.

---

## Licence

The software is under the Apache License 2.0 — see [LICENSE](LICENSE). **That covers this application only.** Model weights,
datasets, benchmark splits and web pages you download carry their own terms — a
licence here is not permission to use or redistribute someone else's work.
Teacher and AI Studio record what each model declares and show it, and every
page fetched from the web is saved with its address and the date; neither
assumes anything beyond that.
