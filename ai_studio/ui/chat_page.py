"""Chat screen: streaming conversation with generation stats and optional RAG."""

from __future__ import annotations

from typing import Any

import gradio as gr

from ai_studio.core.errors import StudioError
from ai_studio.inference import conversation as convo
from ai_studio.inference.conversation import SYSTEM_PRESETS, Message
from ai_studio.inference.generator import GenerationSettings, StopSignal, stream_generate
from ai_studio.inference.model_loader import cache
from ai_studio.models.model_manager import list_models
from ai_studio.rag.retrieval import RetrievalSettings, list_indexes, retrieve_for_prompt
from ai_studio.ui import theme as t

_stop = StopSignal()


def model_choices() -> list[tuple[str, str]]:
    return [(f"{r['name']} · {t.fmt_params(r.get('parameters'))}", r["id"]) for r in list_models()]


def index_choices() -> list[tuple[str, str]]:
    return [("(no retrieval)", "")] + [
        (f"{r['name']} · {r['chunk_count']:,} chunks", r["id"]) for r in list_indexes()
    ]


def conversation_choices() -> list[tuple[str, str]]:
    return [(f"{r['title'][:44]}", r["id"]) for r in convo.list_conversations()]


def _history(conversation_id: str | None) -> list[dict[str, str]]:
    if not conversation_id:
        return []
    return [
        {"role": message.role, "content": message.content}
        for message in convo.get_messages(conversation_id)
        if message.role in {"user", "assistant"}
    ]


def render() -> None:
    gr.HTML('<div class="studio-title">Chat</div>'
            '<div class="studio-sub">Talk to any trained model, checkpoint or imported base model</div>')

    state_conversation = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=460, show_copy_button=True, label="Conversation")
            with gr.Row():
                message_box = gr.Textbox(placeholder="Ask something…", scale=6, show_label=False, lines=2)
                send_button = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                stop_button = gr.Button("Stop", size="sm")
                regenerate_button = gr.Button("Regenerate", size="sm")
                clear_button = gr.Button("Clear", size="sm")
                new_button = gr.Button("New conversation", size="sm")
            stats_html = gr.HTML()
            retrieved_html = gr.HTML()

        with gr.Column(scale=1):
            model = gr.Dropdown(choices=model_choices(), label="Model")
            load_button = gr.Button("Load model", size="sm")
            load_status = gr.HTML()
            conversation_picker = gr.Dropdown(choices=conversation_choices(), label="Conversation")
            preset = gr.Dropdown(list(SYSTEM_PRESETS), value="General Assistant", label="Assistant preset")
            system_prompt = gr.Textbox(
                value=SYSTEM_PRESETS["General Assistant"], label="System prompt", lines=4
            )
            gr.Markdown("**Generation**")
            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top P")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top K")
            max_new_tokens = gr.Slider(16, 4096, value=256, step=16, label="Max new tokens")
            repetition_penalty = gr.Slider(0.8, 2.0, value=1.1, step=0.01, label="Repetition penalty")
            seed = gr.Number(value=None, label="Seed (blank = random)", precision=0)
            gr.Markdown("**Knowledge (RAG)**")
            rag_index = gr.Dropdown(choices=index_choices(), value="", label="Index")
            rag_top_k = gr.Slider(1, 10, value=4, step=1, label="Chunks retrieved")
            gr.Markdown("**Context**")
            context_strategy = gr.Dropdown(
                ["trim_oldest", "summarize", "new_context"], value="trim_oldest",
                label="When context is full",
            )
            refresh_button = gr.Button("↻ Refresh lists", size="sm")

    preset.change(lambda name: SYSTEM_PRESETS.get(name, ""), preset, system_prompt)

    def do_load(model_id):
        if not model_id:
            return t.note("Select a model.", "warn")
        try:
            loaded = cache.load(model_id)
        except StudioError as exc:
            return t.error_message(exc)
        return t.note(
            f"Loaded <b>{loaded.name}</b> on {loaded.device} — context {loaded.context_length:,} tokens, "
            f"{t.fmt_params(loaded.parameters)} parameters.", "good",
        )

    load_button.click(do_load, model, load_status)

    def do_new():
        record = convo.create_conversation()
        return (record["id"], [], gr.update(choices=conversation_choices(), value=record["id"]),
                t.note("New conversation started."), "")

    new_button.click(do_new, outputs=[state_conversation, chatbot, conversation_picker, stats_html, retrieved_html])

    def do_pick(conversation_id):
        if not conversation_id:
            return None, [], ""
        return conversation_id, _history(conversation_id), ""

    conversation_picker.change(do_pick, conversation_picker, [state_conversation, chatbot, stats_html])

    def do_clear(conversation_id):
        if conversation_id:
            convo.clear_conversation(conversation_id)
        return [], t.note("Conversation cleared."), ""

    clear_button.click(do_clear, state_conversation, [chatbot, stats_html, retrieved_html])
    stop_button.click(lambda: (_stop.stop(), t.note("Stopping…", "warn"))[1], outputs=stats_html)

    def chat_stream(text, conversation_id, model_id, system, temperature_v, top_p_v, top_k_v,
                    max_tokens_v, penalty, seed_v, index_id, top_k_chunks, strategy):
        """Streamed turn: yields the chatbot history, stats and retrieval panel."""
        if not model_id:
            yield [], t.note("Select and load a model first.", "warn"), "", conversation_id, text
            return
        text = (text or "").strip()
        if not text:
            yield _history(conversation_id), "", "", conversation_id, ""
            return

        _stop.reset()
        if not conversation_id:
            conversation_id = convo.create_conversation(
                model_id=model_id, system_prompt=system
            )["id"]

        convo.add_message(conversation_id, "user", text)
        history = _history(conversation_id)
        yield history + [{"role": "assistant", "content": "…"}], t.note("Generating…"), "", conversation_id, ""

        try:
            loaded = cache.load(model_id)
        except StudioError as exc:
            yield history, t.error_message(exc), "", conversation_id, ""
            return

        retrieval_html = ""
        context_block = ""
        if index_id:
            settings = RetrievalSettings(index_id=index_id, top_k=int(top_k_chunks))
            context_block, hits = retrieve_for_prompt(settings, text)
            if hits:
                retrieval_html = t.table(
                    ["#", "Source", "Similarity", "Excerpt"],
                    [[hit.rank, hit.chunk.source[:28], f"{hit.score:.3f}",
                      hit.chunk.text[:160].replace("<", "&lt;") + "…"] for hit in hits],
                )
            else:
                retrieval_html = t.note("No chunks passed the similarity threshold.", "warn")

        prompt, report = convo.build_prompt(
            convo.get_messages(conversation_id),
            system_prompt=system,
            tokenizer=loaded.tokenizer,
            context_limit=loaded.context_length,
            reserve_for_response=int(max_tokens_v),
            strategy=strategy,
            retrieved_context=context_block,
        )

        settings = GenerationSettings(
            temperature=float(temperature_v), top_p=float(top_p_v), top_k=int(top_k_v),
            max_new_tokens=int(max_tokens_v), repetition_penalty=float(penalty),
            seed=int(seed_v) if seed_v not in (None, "") else None,
        )

        answer = ""
        stats = None
        try:
            for delta, final in stream_generate(prompt, loaded=loaded, settings=settings, stop_signal=_stop):
                if final is not None:
                    stats = final
                    break
                answer += delta
                yield (history + [{"role": "assistant", "content": answer}],
                       t.note("Generating…"), retrieval_html, conversation_id, "")
        except StudioError as exc:
            yield history, t.error_message(exc), retrieval_html, conversation_id, ""
            return

        convo.add_message(conversation_id, "assistant", answer, stats=stats.to_dict() if stats else {})

        cards = ""
        if stats:
            cards = t.stat_grid([
                t.stat("Prompt tokens", t.fmt_number(stats.prompt_tokens)),
                t.stat("Generated", t.fmt_number(stats.completion_tokens)),
                t.stat("Tokens/sec", t.fmt_number(stats.tokens_per_second, 1)),
                t.stat("Time to first token", f"{stats.time_to_first_token_ms:.0f} ms"),
                t.stat("Generation time", f"{stats.generation_seconds:.2f} s"),
                t.stat("Context", f"{stats.context_used:,} / {stats.context_limit:,}",
                       f"{stats.context_percent:.0f}% used"),
                t.stat("Finish reason", stats.finish_reason),
            ])
        if report.note:
            cards += t.note(report.note, "warn")

        yield (history + [{"role": "assistant", "content": answer}], cards, retrieval_html,
               conversation_id, "")

    chat_inputs = [message_box, state_conversation, model, system_prompt, temperature, top_p, top_k,
                   max_new_tokens, repetition_penalty, seed, rag_index, rag_top_k, context_strategy]
    chat_outputs = [chatbot, stats_html, retrieved_html, state_conversation, message_box]

    send_button.click(chat_stream, chat_inputs, chat_outputs)
    message_box.submit(chat_stream, chat_inputs, chat_outputs)

    def do_regenerate(conversation_id, *args):
        if not conversation_id:
            yield [], t.note("Nothing to regenerate.", "warn"), "", conversation_id, ""
            return
        messages = convo.get_messages(conversation_id)
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is None:
            yield _history(conversation_id), t.note("No user message found.", "warn"), "", conversation_id, ""
            return
        for message in reversed(messages):
            if message.role == "assistant" and message.created_at > last_user.created_at and message.id:
                convo.delete_message(message.id)
        convo.delete_message(last_user.id)
        yield from chat_stream(last_user.content, conversation_id, *args)

    regenerate_button.click(
        do_regenerate,
        [state_conversation, model, system_prompt, temperature, top_p, top_k, max_new_tokens,
         repetition_penalty, seed, rag_index, rag_top_k, context_strategy],
        chat_outputs,
    )

    refresh_button.click(
        lambda: (gr.update(choices=model_choices()), gr.update(choices=index_choices()),
                 gr.update(choices=conversation_choices())),
        outputs=[model, rag_index, conversation_picker],
    )
