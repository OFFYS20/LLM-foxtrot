"""Demo data seeding.

Everything created here is flagged ``is_demo=True`` / ``provenance="simulated"``
so the UI can badge it and a single call to :func:`clear_demo_data` removes it.
The point is to make every screen explorable before any real model exists —
never to make the platform look like it has results it does not have.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.core.logging import get_logger
from app.db.models.catalog import Dataset, Model, Project
from app.db.models.chat import Conversation, Message, PlaygroundComparison
from app.db.models.enums import (
    DatasetFormat,
    DatasetStatus,
    DatasetTemplate,
    JobStatus,
    LogLevel,
    ModelFormat,
    ModelSource,
    ModelStatus,
    Precision,
    RunProvenance,
    TrainingMethod,
)
from app.db.models.evaluation import BenchmarkItem, BenchmarkRun
from app.db.models.system import LogEntry
from app.db.models.training import Checkpoint, Experiment, TrainingJob, TrainingMetric
from app.db.session import session_scope
from app.schemas.training import TrainingConfig
from app.services.evaluation.registry import registry as suite_registry
from app.services.training.schedules import lr_at_step

logger = get_logger("foxtrot.demo")

DEMO_TAG = "demo"


def demo_data_present() -> bool:
    with session_scope() as session:
        return session.scalar(select(Model.id).where(Model.is_demo.is_(True)).limit(1)) is not None


def clear_demo_data() -> dict[str, int]:
    """Remove everything marked as demo data."""
    removed: dict[str, int] = {}
    with session_scope() as session:
        for model_cls, label in (
            (BenchmarkRun, "benchmark_runs"),
            (PlaygroundComparison, "playground_comparisons"),
            (Conversation, "conversations"),
            (Experiment, "experiments"),
            (Checkpoint, "checkpoints"),
            (Dataset, "datasets"),
            (Project, "projects"),
        ):
            rows = session.scalars(select(model_cls).where(model_cls.is_demo.is_(True))).all()
            removed[label] = len(rows)
            for row in rows:
                session.delete(row)

        jobs = session.scalars(
            select(TrainingJob).where(TrainingJob.provenance == RunProvenance.SIMULATED)
        ).all()
        removed["training_jobs"] = len(jobs)
        for job in jobs:
            session.delete(job)

        models = session.scalars(select(Model).where(Model.is_demo.is_(True))).all()
        removed["models"] = len(models)
        for model in models:
            session.delete(model)
    logger.info("cleared demo data: %s", removed)
    return removed


def seed_demo_data(force: bool = False) -> dict[str, Any]:
    if demo_data_present() and not force:
        return {"skipped": True, "reason": "demo data already present"}
    if force:
        clear_demo_data()

    rng = random.Random(1337)
    now = datetime.now(timezone.utc)
    created: dict[str, Any] = {}

    with session_scope() as session:
        # ---------------------------------------------------------- models
        base = Model(
            name="foxtrot-7b-base",
            display_name="Foxtrot 7B Base",
            description="Demo base checkpoint used as the starting point for fine-tuning runs.",
            source=ModelSource.DEMO,
            format=ModelFormat.SAFETENSORS,
            status=ModelStatus.READY,
            repo_id="foxtrot-ai/foxtrot-7b-base",
            architecture="LlamaForCausalLM",
            parameters=7_240,
            context_length=8192,
            precision=Precision.BF16,
            size_bytes=int(14.2 * 1024**3),
            vram_estimate_mb=15_400,
            tokenizer={
                "type": "SentencePieceBPE",
                "vocab_size": 32000,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "chat_template": True,
                "special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
            },
            config={
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rope_theta": 10000,
                "vocab_size": 32000,
            },
            metrics={"inference_tokens_per_sec": 148.2, "first_token_ms": 128},
            tags=["base", DEMO_TAG],
            license="apache-2.0",
            is_demo=True,
        )
        tuned_e1 = Model(
            name="foxtrot-7b-sft-e1",
            display_name="Foxtrot 7B SFT · epoch 1",
            description="LoRA instruction-tuned adapter merged after one epoch.",
            source=ModelSource.CHECKPOINT,
            format=ModelFormat.SAFETENSORS,
            status=ModelStatus.READY,
            architecture="LlamaForCausalLM",
            parameters=7_240,
            context_length=8192,
            precision=Precision.BF16,
            size_bytes=int(14.2 * 1024**3),
            vram_estimate_mb=15_600,
            tokenizer=dict(base.tokenizer),
            config=dict(base.config),
            metrics={"inference_tokens_per_sec": 141.7, "first_token_ms": 133},
            tags=["fine-tuned", "epoch-1", DEMO_TAG],
            is_demo=True,
        )
        tuned_e3 = Model(
            name="foxtrot-7b-sft-e3",
            display_name="Foxtrot 7B SFT · epoch 3",
            description="Same recipe, three epochs — better instruction following, slight overfit.",
            source=ModelSource.CHECKPOINT,
            format=ModelFormat.SAFETENSORS,
            status=ModelStatus.LOADED,
            architecture="LlamaForCausalLM",
            parameters=7_240,
            context_length=8192,
            precision=Precision.BF16,
            size_bytes=int(14.2 * 1024**3),
            vram_estimate_mb=15_700,
            tokenizer=dict(base.tokenizer),
            config=dict(base.config),
            metrics={"inference_tokens_per_sec": 139.4, "first_token_ms": 131},
            tags=["fine-tuned", "epoch-3", DEMO_TAG],
            loaded_at=now - timedelta(minutes=42),
            is_demo=True,
        )
        small = Model(
            name="foxtrot-1.5b-instruct",
            display_name="Foxtrot 1.5B Instruct",
            description="Small, fast demo model for latency comparisons.",
            source=ModelSource.DEMO,
            format=ModelFormat.GGUF,
            status=ModelStatus.READY,
            architecture="Qwen2ForCausalLM",
            parameters=1_540,
            context_length=32768,
            precision=Precision.INT4,
            quantization="Q4_K_M",
            size_bytes=int(0.98 * 1024**3),
            vram_estimate_mb=2_100,
            tokenizer={"type": "BPE", "vocab_size": 151_646, "chat_template": True},
            config={"hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12},
            metrics={"inference_tokens_per_sec": 312.5, "first_token_ms": 46},
            tags=["small", "gguf", DEMO_TAG],
            is_demo=True,
        )
        mixtral = Model(
            name="foxtrot-moe-12b",
            display_name="Foxtrot MoE 12B",
            description="Mixture-of-experts demo entry for architecture comparisons.",
            source=ModelSource.HUGGINGFACE,
            format=ModelFormat.HF_REPO,
            status=ModelStatus.STOPPED,
            repo_id="foxtrot-ai/foxtrot-moe-12b",
            architecture="MixtralForCausalLM",
            parameters=12_880,
            context_length=32768,
            precision=Precision.FP16,
            size_bytes=int(25.8 * 1024**3),
            vram_estimate_mb=27_200,
            tokenizer={"type": "SentencePieceBPE", "vocab_size": 32000, "chat_template": True},
            config={"num_local_experts": 8, "num_experts_per_tok": 2, "num_hidden_layers": 32},
            metrics={"inference_tokens_per_sec": 96.3, "first_token_ms": 210},
            tags=["moe", DEMO_TAG],
            is_demo=True,
        )
        session.add_all([base, tuned_e1, tuned_e3, small, mixtral])
        session.flush()
        tuned_e1.parent_model_id = base.id
        tuned_e3.parent_model_id = base.id

        # -------------------------------------------------------- datasets
        datasets = [
            Dataset(
                name="foxtrot-instruct-50k",
                description="Instruction/response pairs across reasoning, coding and writing.",
                format=DatasetFormat.JSONL,
                template=DatasetTemplate.INSTRUCTION,
                status=DatasetStatus.READY,
                source="demo",
                rows=50_000,
                tokens=18_420_000,
                avg_sequence_length=368.4,
                max_sequence_length=4096,
                size_bytes=int(1.4 * 1024**3),
                train_split=0.9,
                validation_split=0.05,
                test_split=0.05,
                columns=["instruction", "input", "output"],
                preview=[
                    {
                        "instruction": "Explain the difference between LoRA and full fine-tuning.",
                        "input": "",
                        "output": "LoRA freezes the base weights and learns low-rank update matrices…",
                    },
                    {
                        "instruction": "Rewrite the sentence in the passive voice.",
                        "input": "The trainer saved the checkpoint.",
                        "output": "The checkpoint was saved by the trainer.",
                    },
                ],
                validation_report={
                    "checked_rows": 50_000,
                    "valid_rows": 49_987,
                    "invalid_rows": 13,
                },
                tags=["sft", DEMO_TAG],
                is_demo=True,
            ),
            Dataset(
                name="foxtrot-chat-12k",
                description="Multi-turn conversations with system prompts.",
                format=DatasetFormat.JSONL,
                template=DatasetTemplate.CHAT,
                status=DatasetStatus.READY,
                source="demo",
                rows=12_400,
                tokens=9_130_000,
                avg_sequence_length=736.2,
                max_sequence_length=8192,
                size_bytes=int(720 * 1024**2),
                columns=["messages"],
                preview=[
                    {
                        "messages": [
                            {"role": "system", "content": "You are a concise assistant."},
                            {"role": "user", "content": "How do I pick a LoRA rank?"},
                            {
                                "role": "assistant",
                                "content": "Start at 8–16 and scale with dataset size…",
                            },
                        ]
                    }
                ],
                validation_report={"checked_rows": 12_400, "valid_rows": 12_400, "invalid_rows": 0},
                tags=["chat", DEMO_TAG],
                is_demo=True,
            ),
            Dataset(
                name="domain-corpus-raw",
                description="Raw domain documents for continued pretraining.",
                format=DatasetFormat.TXT,
                template=DatasetTemplate.PLAIN_TEXT,
                status=DatasetStatus.READY,
                source="demo",
                rows=214_000,
                tokens=64_800_000,
                avg_sequence_length=302.8,
                max_sequence_length=16_384,
                size_bytes=int(4.8 * 1024**3),
                train_split=0.98,
                validation_split=0.02,
                test_split=0.0,
                columns=["text"],
                preview=[
                    {
                        "text": "Chapter 4 — Optimizer states dominate memory during full fine-tuning…"
                    }
                ],
                validation_report={"checked_rows": 5000, "valid_rows": 4996, "invalid_rows": 4},
                tags=["pretraining", DEMO_TAG],
                is_demo=True,
            ),
            Dataset(
                name="eval-holdout-2k",
                description="Held-out evaluation questions with reference answers.",
                format=DatasetFormat.JSON,
                template=DatasetTemplate.INSTRUCTION,
                status=DatasetStatus.READY,
                source="demo",
                rows=2_000,
                tokens=612_000,
                avg_sequence_length=306.0,
                max_sequence_length=2048,
                size_bytes=int(62 * 1024**2),
                columns=["instruction", "output"],
                preview=[
                    {
                        "instruction": "What does gradient accumulation trade off?",
                        "output": "Memory for wall-clock time per optimizer step.",
                    }
                ],
                validation_report={"checked_rows": 2000, "valid_rows": 2000, "invalid_rows": 0},
                tags=["eval", DEMO_TAG],
                is_demo=True,
            ),
        ]
        session.add_all(datasets)
        session.flush()

        project = Project(
            name="Foxtrot instruction tuning",
            description="Demo project: instruction-tune the 7B base on the 50k SFT mixture.",
            base_model_id=base.id,
            default_dataset_id=datasets[0].id,
            tags=[DEMO_TAG],
            is_demo=True,
        )
        session.add(project)

        # ------------------------------------------------- training history
        configs = [
            ("LoRA r16 · 1 epoch", TrainingMethod.LORA, 1.0, 16, 2e-5, 1.42, 1.51),
            ("LoRA r32 · 3 epochs", TrainingMethod.LORA, 3.0, 32, 2e-5, 1.08, 1.24),
            ("QLoRA r64 · 3 epochs", TrainingMethod.QLORA, 3.0, 64, 1e-4, 1.02, 1.31),
        ]
        experiments: list[Experiment] = []
        for offset, (label, method, epochs, rank, lr, final_train, final_val) in enumerate(configs):
            config = TrainingConfig(
                method=method,
                epochs=epochs,
                batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=lr,
                warmup_steps=100,
                max_sequence_length=2048,
                precision=Precision.INT4 if method == TrainingMethod.QLORA else Precision.BF16,
            )
            config.lora.rank = rank
            config.lora.alpha = rank * 2

            started = now - timedelta(days=6 - offset * 2, hours=offset)
            duration = 3600 * (2.2 + offset * 0.6)
            total_steps = 1400 + offset * 350

            experiment = Experiment(
                name=f"{label}",
                model_id=base.id,
                dataset_id=datasets[0].id,
                project_id=project.id,
                status=JobStatus.COMPLETED,
                provenance=RunProvenance.SIMULATED,
                method=method,
                hyperparameters=config.model_dump(mode="json"),
                started_at=started,
                ended_at=started + timedelta(seconds=duration),
                duration_seconds=duration,
                final_train_loss=final_train,
                final_val_loss=final_val,
                notes="Simulated demo experiment — synthetic metrics, no weights were trained.",
                tags=[DEMO_TAG, str(method)],
                is_demo=True,
            )
            session.add(experiment)
            session.flush()

            job = TrainingJob(
                name=label,
                model_id=base.id,
                dataset_id=datasets[0].id,
                project_id=project.id,
                experiment_id=experiment.id,
                method=method,
                status=JobStatus.COMPLETED,
                provenance=RunProvenance.SIMULATED,
                backend="simulated",
                config=config.model_dump(mode="json"),
                total_steps=total_steps,
                current_step=total_steps,
                total_epochs=epochs,
                current_epoch=epochs,
                loss=final_train,
                val_loss=final_val,
                best_val_loss=final_val - 0.03,
                learning_rate=lr * 0.05,
                grad_norm=0.62,
                tokens_processed=total_steps * 32 * 2048,
                tokens_per_sec=18_420,
                samples_per_sec=9.4,
                gpu_utilization=91.2,
                vram_used_mb=19_400,
                started_at=started,
                ended_at=started + timedelta(seconds=duration),
            )
            session.add(job)
            session.flush()
            experiment.job_id = job.id

            metrics, checkpoints = _synth_curve(
                job_id=job.id,
                total_steps=total_steps,
                epochs=epochs,
                config=config,
                start_loss=2.65,
                final_loss=final_train,
                final_val=final_val,
                started=started,
                duration=duration,
                rng=rng,
            )
            session.add_all(metrics)
            for step, train_loss, val_loss, is_best in checkpoints:
                session.add(
                    Checkpoint(
                        model_id=base.id,
                        job_id=job.id,
                        experiment_id=experiment.id,
                        step=step,
                        epoch=round(step / total_steps * epochs, 2),
                        path=str(settings.checkpoints_dir / job.id / f"step-{step}"),
                        size_bytes=int(rank * 4.2 * 1024 * 1024),
                        train_loss=train_loss,
                        val_loss=val_loss,
                        benchmark_score=round(52 + offset * 4 + step / total_steps * 6, 2),
                        is_best=is_best,
                        provenance=RunProvenance.SIMULATED,
                        is_demo=True,
                    )
                )
            experiments.append(experiment)

        session.flush()
        for experiment in experiments:
            best = session.scalars(
                select(Checkpoint)
                .where(Checkpoint.experiment_id == experiment.id, Checkpoint.is_best.is_(True))
                .limit(1)
            ).first()
            if best:
                experiment.best_checkpoint_id = best.id

        # ---------------------------------------------- a running demo job
        live_config = TrainingConfig(
            method=TrainingMethod.LORA,
            epochs=2.0,
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            max_sequence_length=2048,
        )
        live_experiment = Experiment(
            name="LoRA r16 · chat mixture (live demo)",
            model_id=tuned_e1.id,
            dataset_id=datasets[1].id,
            project_id=project.id,
            status=JobStatus.PAUSED,
            provenance=RunProvenance.SIMULATED,
            method=TrainingMethod.LORA,
            hyperparameters=live_config.model_dump(mode="json"),
            started_at=now - timedelta(minutes=48),
            notes="Paused simulated run — press Resume on the Training page to watch it stream.",
            tags=[DEMO_TAG, "live"],
            is_demo=True,
        )
        session.add(live_experiment)
        session.flush()

        live_total = 900
        live_step = 348
        live_job = TrainingJob(
            name="LoRA r16 · chat mixture (live demo)",
            model_id=tuned_e1.id,
            dataset_id=datasets[1].id,
            project_id=project.id,
            experiment_id=live_experiment.id,
            method=TrainingMethod.LORA,
            status=JobStatus.PAUSED,
            provenance=RunProvenance.SIMULATED,
            backend="simulated",
            config=live_config.model_dump(mode="json"),
            total_steps=live_total,
            current_step=live_step,
            total_epochs=2.0,
            current_epoch=round(live_step / live_total * 2, 3),
            loss=1.431,
            val_loss=1.512,
            best_val_loss=1.498,
            learning_rate=1.42e-5,
            grad_norm=0.78,
            tokens_processed=live_step * 32 * 2048,
            tokens_per_sec=17_940,
            samples_per_sec=8.9,
            gpu_utilization=88.4,
            vram_used_mb=18_950,
            eta_seconds=1180,
            started_at=now - timedelta(minutes=48),
        )
        session.add(live_job)
        session.flush()
        live_experiment.job_id = live_job.id

        metrics, checkpoints = _synth_curve(
            job_id=live_job.id,
            total_steps=live_total,
            epochs=2.0,
            config=live_config,
            start_loss=2.58,
            final_loss=1.431,
            final_val=1.512,
            started=now - timedelta(minutes=48),
            duration=48 * 60,
            rng=rng,
            up_to_step=live_step,
        )
        session.add_all(metrics)
        for step, train_loss, val_loss, is_best in checkpoints:
            session.add(
                Checkpoint(
                    model_id=tuned_e1.id,
                    job_id=live_job.id,
                    experiment_id=live_experiment.id,
                    step=step,
                    epoch=round(step / live_total * 2, 2),
                    path=str(settings.checkpoints_dir / live_job.id / f"step-{step}"),
                    size_bytes=int(16 * 4.2 * 1024 * 1024),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    is_best=is_best,
                    provenance=RunProvenance.SIMULATED,
                    is_demo=True,
                )
            )

        # ------------------------------------------------------ benchmarks
        bench_plan = [
            (base.id, "mmlu", 0.548),
            (base.id, "gsm8k", 0.352),
            (tuned_e1.id, "mmlu", 0.571),
            (tuned_e1.id, "gsm8k", 0.446),
            (tuned_e3.id, "mmlu", 0.583),
            (tuned_e3.id, "gsm8k", 0.492),
            (tuned_e3.id, "humaneval", 0.318),
            (tuned_e3.id, "truthfulqa", 0.474),
            (small.id, "mmlu", 0.402),
            (small.id, "gsm8k", 0.238),
        ]
        for index, (model_id, suite_key, target) in enumerate(bench_plan):
            adapter = suite_registry.get(suite_key)
            specs = adapter.items(limit=min(8, adapter.available_items()), seed=42 + index)
            started = now - timedelta(hours=30 - index * 2)
            latencies = []
            run = BenchmarkRun(
                name=f"{adapter.label} · demo",
                suite=suite_key,
                suite_label=adapter.label,
                category=str(adapter.category),
                model_id=model_id,
                status=JobStatus.COMPLETED,
                provenance=RunProvenance.SIMULATED,
                config={
                    "num_examples": len(specs),
                    "few_shot": adapter.default_shots,
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "seed": 42,
                    "data_source": adapter.data_source,
                    "official_split": adapter.official,
                    "suite_notes": adapter.notes,
                    "demo": True,
                },
                total_items=len(specs),
                completed_items=len(specs),
                overall_score=round(target, 4),
                accuracy=round(target, 4),
                pass_at_1=None,
                total_tokens=len(specs) * 180,
                started_at=started,
                ended_at=started + timedelta(minutes=6),
                runtime_seconds=362.0,
                is_demo=True,
            )
            session.add(run)
            session.flush()

            correct_target = round(target * len(specs))
            for position, spec in enumerate(specs):
                correct = position < correct_target
                latency = round(rng.uniform(420, 1650), 1)
                latencies.append(latency)
                session.add(
                    BenchmarkItem(
                        run_id=run.id,
                        index=position,
                        category=spec.category or str(adapter.category),
                        question=spec.question
                        if not spec.context
                        else f"{spec.context}\n\n{spec.question}",
                        prompt=adapter.build_prompt(spec),
                        expected=spec.expected,
                        response=spec.expected
                        if correct
                        else "[demo] The model's answer did not match the reference.",
                        raw_output=spec.expected if correct else "…",
                        correct=correct,
                        score=1.0 if correct else 0.0,
                        latency_ms=latency,
                        tokens_used=rng.randint(90, 320),
                    )
                )
            run.avg_latency_ms = round(sum(latencies) / len(latencies), 1)
            run.avg_tokens_per_sec = round(rng.uniform(98, 168), 1)
            run.category_scores = {
                str(adapter.category): round(target, 4),
                **(
                    {
                        spec.category: round(min(1.0, target + rng.uniform(-0.1, 0.1)), 3)
                        for spec in specs[:4]
                        if spec.category
                    }
                ),
            }

        # ---------------------------------------------------- conversations
        conversation = Conversation(
            title="LoRA rank sizing",
            model_id=tuned_e3.id,
            system_prompt="You are Foxtrot, a concise ML engineering assistant.",
            params={"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 512},
            is_demo=True,
        )
        session.add(conversation)
        session.flush()
        session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content="How should I pick the LoRA rank for a 50k-example instruction set?",
                    prompt_tokens=18,
                    total_tokens=18,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    model_id=tuned_e3.id,
                    content=(
                        "[demo output — simulated, no weights loaded]\n\n"
                        "Start at rank 16 with alpha 32. With 50k examples you have enough signal "
                        "to justify rank 32 if validation loss is still improving at the end of "
                        "epoch 2 — but raise alpha with it, otherwise the effective update scale "
                        "changes and the run destabilises."
                    ),
                    prompt_tokens=18,
                    completion_tokens=62,
                    total_tokens=80,
                    tokens_per_sec=141.7,
                    latency_ms=612.0,
                    time_to_first_token_ms=131.0,
                    finish_reason="stop",
                    provenance=RunProvenance.SIMULATED,
                ),
            ]
        )

        session.add(
            PlaygroundComparison(
                prompt="Summarise the trade-off between gradient accumulation and batch size.",
                system_prompt="Answer in two sentences.",
                params={"temperature": 0.7, "max_tokens": 256},
                entries=[
                    {
                        "model_id": tuned_e3.id,
                        "model_name": "Foxtrot 7B SFT · epoch 3",
                        "content": "[demo] Gradient accumulation trades wall-clock time for memory…",
                        "latency_ms": 742.0,
                        "time_to_first_token_ms": 131.0,
                        "tokens_per_sec": 139.4,
                        "completion_tokens": 88,
                        "prompt_tokens": 22,
                        "total_tokens": 110,
                        "memory_mb": 15_700,
                        "provenance": "simulated",
                        "engine": "demo",
                    },
                    {
                        "model_id": small.id,
                        "model_name": "Foxtrot 1.5B Instruct",
                        "content": "[demo] Accumulation simulates a larger batch without more VRAM…",
                        "latency_ms": 318.0,
                        "time_to_first_token_ms": 46.0,
                        "tokens_per_sec": 312.5,
                        "completion_tokens": 74,
                        "prompt_tokens": 22,
                        "total_tokens": 96,
                        "memory_mb": 2_100,
                        "provenance": "simulated",
                        "engine": "demo",
                    },
                ],
                winner_model_id=tuned_e3.id,
                votes={tuned_e3.id: 1},
                provenance=RunProvenance.SIMULATED,
                is_demo=True,
            )
        )

        # ----------------------------------------------------------- logs
        for index, (level, source, message) in enumerate(
            [
                (
                    LogLevel.INFO,
                    "system",
                    "Foxtrot started in demo mode — simulated providers active",
                ),
                (LogLevel.INFO, "models", "Registered demo model foxtrot-7b-base (7.24B params)"),
                (
                    LogLevel.INFO,
                    "datasets",
                    "Imported foxtrot-instruct-50k — 50,000 rows / 18.4M tokens",
                ),
                (
                    LogLevel.WARN,
                    "hardware",
                    "No NVIDIA device detected — hardware telemetry is simulated",
                ),
                (
                    LogLevel.INFO,
                    "training",
                    "[TRAIN] Step 348/900 loss: 1.431 lr: 1.42e-5 tokens/sec: 17,940",
                ),
                (
                    LogLevel.INFO,
                    "benchmark",
                    "MMLU (bundled sample) completed on foxtrot-7b-sft-e3 — 0.583",
                ),
                (
                    LogLevel.WARN,
                    "training",
                    "Validation loss increased for 2 consecutive evals — possible overfit",
                ),
            ]
        ):
            session.add(
                LogEntry(
                    ts=now - timedelta(minutes=90 - index * 11),
                    level=level,
                    source=source,
                    message=message,
                    context={"demo": True},
                )
            )

        created = {
            "models": 5,
            "datasets": len(datasets),
            "experiments": len(configs) + 1,
            "training_jobs": len(configs) + 1,
            "benchmark_runs": len(bench_plan),
            "conversations": 1,
            "projects": 1,
        }

    logger.info("seeded demo data: %s", json.dumps(created))
    return {"skipped": False, "created": created}


def _synth_curve(
    *,
    job_id: str,
    total_steps: int,
    epochs: float,
    config: TrainingConfig,
    start_loss: float,
    final_loss: float,
    final_val: float,
    started: datetime,
    duration: float,
    rng: random.Random,
    up_to_step: int | None = None,
) -> tuple[list[TrainingMetric], list[tuple[int, float, float | None, bool]]]:
    """Build a plausible loss/LR/throughput trajectory for demo charts."""
    metrics: list[TrainingMetric] = []
    checkpoints: list[tuple[int, float, float | None, bool]] = []
    last_step = up_to_step or total_steps
    stride = max(1, last_step // 120)
    decay = math.log(max(1e-3, (start_loss - final_loss + 0.05) / 0.05)) / max(1, total_steps)
    best_val = math.inf
    val_loss: float | None = None

    for step in range(stride, last_step + 1, stride):
        progress = step / total_steps
        loss = (
            final_loss + (start_loss - final_loss) * math.exp(-decay * step) + rng.gauss(0, 0.035)
        )
        loss = max(0.05, loss)
        lr = lr_at_step(
            base_lr=config.learning_rate,
            step=step,
            total_steps=total_steps,
            warmup_steps=config.warmup_steps,
            scheduler=config.lr_scheduler,
        )
        if step % max(stride, config.eval_every_steps) < stride:
            val_loss = max(0.05, loss + 0.05 + max(0.0, progress - 0.7) * 0.9 + rng.gauss(0, 0.02))
            if val_loss < best_val:
                best_val = val_loss

        metrics.append(
            TrainingMetric(
                job_id=job_id,
                step=step,
                epoch=round(progress * epochs, 4),
                ts=started + timedelta(seconds=duration * progress),
                loss=round(loss, 5),
                val_loss=round(val_loss, 5) if val_loss is not None else None,
                learning_rate=lr,
                grad_norm=round(max(0.05, rng.gauss(0.8, 0.22)), 4),
                tokens_per_sec=round(rng.uniform(16_500, 19_800), 1),
                samples_per_sec=round(rng.uniform(8.2, 9.8), 2),
                gpu_utilization=round(min(99.0, rng.gauss(89, 4)), 1),
                vram_used_mb=round(rng.uniform(18_400, 20_100), 1),
            )
        )

        if step % config.checkpointing.save_every_steps < stride:
            checkpoints.append(
                (step, round(loss, 5), round(val_loss, 5) if val_loss else None, False)
            )

    if checkpoints:
        best_index = min(
            range(len(checkpoints)),
            key=lambda i: checkpoints[i][2] if checkpoints[i][2] is not None else math.inf,
        )
        step, train_loss, val, _ = checkpoints[best_index]
        checkpoints[best_index] = (step, train_loss, val, True)
        checkpoints = checkpoints[-max(3, config.checkpointing.keep_last) :]
    if not any(c[3] for c in checkpoints) and checkpoints:
        step, train_loss, val, _ = checkpoints[-1]
        checkpoints[-1] = (step, train_loss, val, True)

    return metrics, checkpoints
