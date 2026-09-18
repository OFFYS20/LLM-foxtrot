# Bench

A single HTML file that loads a language model you trained yourself and lets you
talk to it. No install, no server, no Node.js — open `index.html` in a browser
and drop in your model folder.

The weights are read, run and sampled entirely in the page. Nothing is uploaded
anywhere; the page makes no network requests except for its two fonts.

---

## Use it

1. Open `index.html` (double-click it, or serve the folder — both work).
2. Drag in a model folder, or use **choose files** to pick three files:
   - `model.safetensors`
   - `config.json`
   - `tokenizer.json`
3. Type something and press **Send**.

Those are exactly the files that [Teacher](../teacher/README.md) keeps in a model
folder, and that AI Studio writes into a checkpoint — so there is nothing to
convert:

```bash
python -m teacher pack bookbot -o ./for-bench    # then drop ./for-bench in
```

---

## What it runs

Bench implements the `ai_studio_transformer` architecture in plain JavaScript:

- RMSNorm and LayerNorm
- SwiGLU, GELU, SiLU and ReLU feed-forwards
- RoPE (with theta and scaling) or learned position embeddings
- Grouped-query attention, with a KV cache so generation stays linear
- Tied or separate output heads
- fp32 weights

If a model uses something outside that set, the page says so and names the part
it cannot run, rather than producing plausible nonsense.

**Verified against PyTorch.** For the same input tokens, Bench's logits match
`TransformerLM`'s to six decimal places and pick the same top five candidates.
It is the same model, not an approximation of it.

---

## What it shows

Alongside the conversation, the page reports what is actually happening:

- **Model** — parameters counted from the weight file, tensor count, layers,
  heads (with KV heads when they differ), head dimension, context, vocabulary,
  activation, normalisation, position scheme, merge count, whether weights are tied
- **Next-token candidates** — the five tokens the model weighed for the token it
  just wrote, with their probabilities. Sampling stops being a black box
- **Meters** — prompt and generated token counts, tokens per second, how much of
  the context is used, and which condition ended the run (length, end token, a
  full context, or you pressing Stop)

Sampling is yours to set: temperature, top-p, top-k, repetition penalty and a
token budget. Temperature 0 is greedy.

---

## A note on prompting

Bench prepends the `<s>` token and stops at `</s>`. It deliberately does *not*
append `</s>` to your prompt, which is what the Python tokenizer's post-processor
does by default — appending it tells the model your text just ended, so it starts
a new passage instead of continuing yours. That is why a continuation here reads
on from what you wrote, and the same prompt in AI Studio's Chat tab may not.

---

## Speed and size

Everything runs on one CPU thread in JavaScript, so this suits the small models
you can train at home. A 0.8M-parameter model generates at roughly 200–500
tokens per second in a browser. Cost per token scales with parameters and with
how much context has accumulated, so a 10M model will be noticeably slower and a
100M one impractical.

The weight file is read into memory in full, so the model must fit in the tab's
memory budget — a few hundred MB is the practical ceiling.

---

## If something goes wrong

**"This does not look like a safetensors file"**
The file is not safetensors, or it is truncated. Re-copy it.

**"Tensor X is BF16; Bench reads F32 weights"**
Export or train in fp32. Teacher and AI Studio both save fp32 by default.

**"The tokenizer holds N tokens but the model expects M"**
The three files came from different runs. Take all three from the same folder.

**"The weight file is missing N tensor(s)"**
An incomplete checkpoint — most often a LoRA adapter rather than a full model.
Merge the adapter into its base first (AI Studio → Fine-Tuning → Adapter tools).

**The output is gibberish**
Check the model was actually trained: `python -m teacher show <name>` lists its
lessons. An untrained model produces noise, correctly.
