"""Training configuration and pre-flight checks.

``TrainingConfig`` is validated before anything is allocated, and
``preflight`` estimates whether the run will fit in memory — the application
refuses to start a configuration that is very likely to crash the machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai_studio.core.errors import ValidationError

METHODS = (
    "scratch",
    "continued_pretraining",
    "finetune",
    "instruction_tuning",
    "lora",
    "qlora",
)

METHOD_LABELS = {
    "scratch": "Train from scratch",
    "continued_pretraining": "Continued pretraining",
    "finetune": "Full fine-tuning",
    "instruction_tuning": "Instruction tuning",
    "lora": "LoRA",
    "qlora": "QLoRA",
}

OPTIMIZERS = ("adamw", "adamw_8bit", "adafactor", "sgd", "adam")
SCHEDULERS = ("cosine", "linear", "constant", "constant_with_warmup", "polynomial")
PRECISIONS = ("fp32", "fp16", "bf16")

#: Sensible LoRA target modules per known architecture family.
LORA_TARGETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "phi3": ["qkv_proj", "o_proj"],
    "gpt2": ["c_attn", "c_proj"],
    "gptj": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "gpt_neox": ["query_key_value"],
    "falcon": ["query_key_value"],
    "mpt": ["Wqkv"],
    "ai_studio_transformer": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def suggest_lora_targets(model_type: str | None) -> list[str]:
    """Best-known target modules for an architecture, with a safe fallback."""
    if not model_type:
        return list(DEFAULT_LORA_TARGETS)
    key = str(model_type).lower()
    # Longest family name first so "phi3" is not captured by "phi".
    for family in sorted(LORA_TARGETS, key=len, reverse=True):
        if family in key:
            return list(LORA_TARGETS[family])
    return list(DEFAULT_LORA_TARGETS)


@dataclass
class LoRASettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    bias: str = "none"

    def validate(self) -> None:
        if not 1 <= self.rank <= 512:
            raise ValidationError("LoRA rank must be between 1 and 512")
        if not 1 <= self.alpha <= 2048:
            raise ValidationError("LoRA alpha must be between 1 and 2048")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValidationError("Select at least one LoRA target module")


@dataclass
class TrainingConfig:
    """Everything the Training screen can set."""

    method: str = "finetune"
    epochs: float = 1.0
    max_steps: int = 0                 # 0 → derive from epochs
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_sequence_length: int = 512
    gradient_clipping: float = 1.0
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    seed: int = 42
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    flash_attention: bool = False
    compile_model: bool = False
    quantization: str = "none"          # none | 8bit | 4bit
    packing: bool = True                 # concatenate short sequences (raw LM)
    eval_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 500
    keep_last_checkpoints: int = 3
    save_best: bool = True
    eval_max_batches: int = 20
    num_workers: int = 0
    lora: LoRASettings = field(default_factory=LoRASettings)

    # ----------------------------------------------------------- validation
    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValidationError(f"Unknown training method {self.method!r}")
        if self.optimizer not in OPTIMIZERS:
            raise ValidationError(f"Unknown optimizer {self.optimizer!r}")
        if self.lr_scheduler not in SCHEDULERS:
            raise ValidationError(f"Unknown scheduler {self.lr_scheduler!r}")
        if self.precision not in PRECISIONS:
            raise ValidationError(f"Unknown precision {self.precision!r}")
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ValidationError("Set either epochs > 0 or max_steps > 0")
        if self.batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValidationError("gradient_accumulation_steps must be at least 1")
        if not 0 < self.learning_rate <= 1.0:
            raise ValidationError("learning_rate must be in (0, 1]")
        if self.max_sequence_length < 8:
            raise ValidationError("max_sequence_length must be at least 8")
        if self.quantization not in {"none", "8bit", "4bit"}:
            raise ValidationError("quantization must be none, 8bit or 4bit")
        if self.method == "qlora" and self.quantization == "none":
            self.quantization = "4bit"
        if self.method in {"lora", "qlora"}:
            self.lora.validate()
        if self.method == "finetune" and self.quantization != "none":
            raise ValidationError(
                "Full fine-tuning cannot update quantized weights.",
                hint="Use LoRA/QLoRA for quantized bases, or set quantization to none.",
            )

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def uses_peft(self) -> bool:
        return self.method in {"lora", "qlora"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        data = dict(data or {})
        lora = data.pop("lora", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        config = cls(**{k: v for k, v in data.items() if k in known})
        if isinstance(lora, dict):
            config.lora = LoRASettings(
                **{k: v for k, v in lora.items() if k in LoRASettings.__dataclass_fields__}
            )
        return config


@dataclass
class PreflightResult:
    ok: bool
    device: str
    estimated_mb: float
    available_mb: float | None
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        head = "Ready to train" if self.ok else "Configuration is likely to fail"
        available = f"{self.available_mb / 1024:.1f} GB available" if self.available_mb else "available memory unknown"
        return f"{head} — needs ~{self.estimated_mb / 1024:.2f} GB, {available} ({self.device})"


#: How many tensors of each shape autograd keeps alive per transformer block,
#: and how many vocabulary-sized copies the loss makes. Calibrated against a
#: measured run: SmolLM2-135M, batch 8, sequence 512, fp32 on CPU reached
#: 13.9 GB before the kernel killed it, where the old estimate said 0.7 GB.
ACTIVATIONS_PER_HIDDEN = 8
ACTIVATIONS_PER_MLP = 3
ATTENTION_COPIES = 2
LOGIT_COPIES = 3

#: Estimates land under the truth — fused kernels, allocator caching and
#: fragmentation all cost memory this arithmetic cannot see. Erring high costs
#: a warning nobody had to act on; erring low costs the whole run.
SAFETY_MARGIN = 1.25


def activation_estimate(
    config: TrainingConfig,
    *,
    parameter_count: int,
    bytes_per_param: int,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    intermediate_size: int | None = None,
    vocab_size: int | None = None,
) -> float:
    """Bytes of activations one training step keeps alive.

    Given the architecture this is arithmetic. Without it the answer is a
    guess, and the guess is deliberately pessimistic: telling someone a run
    fits when it does not costs them the run.
    """
    batch, length = config.batch_size, config.max_sequence_length
    checkpointing = 0.25 if config.gradient_checkpointing else 1.0

    if not (hidden_size and num_layers):
        # No architecture to hand. sqrt(parameters) stands in for the hidden
        # size and a depth of 24 for the layer count, which is the shape of an
        # ordinary model of that size.
        hidden_size = hidden_size or max(1, int(parameter_count ** 0.5))
        num_layers = num_layers or 24

    per_layer = batch * length * hidden_size * ACTIVATIONS_PER_HIDDEN * bytes_per_param
    per_layer += batch * length * (intermediate_size or hidden_size * 4) \
        * ACTIVATIONS_PER_MLP * bytes_per_param
    if num_heads:
        # Eager attention materialises a score matrix per head; the fused
        # kernels do not, so this is the cautious reading.
        per_layer += batch * num_heads * length * length * ATTENTION_COPIES * bytes_per_param

    logits = batch * length * vocab_size * LOGIT_COPIES * bytes_per_param if vocab_size else 0
    return ((per_layer * num_layers) * checkpointing + logits) * SAFETY_MARGIN


def preflight(
    config: TrainingConfig,
    *,
    parameter_count: int,
    model_memory_mb: float | None = None,
    trainable_parameters: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    intermediate_size: int | None = None,
    vocab_size: int | None = None,
) -> PreflightResult:
    """Estimate whether this run fits, and suggest fixes when it does not.

    Pass the architecture when it is known: depth, vocabulary and MLP width
    dominate the memory a step needs, and none of them can be read off a
    parameter count.
    """
    from ai_studio.hardware.monitor import monitor

    snapshot = monitor.snapshot()
    gpu = snapshot.primary_gpu
    on_gpu = bool(gpu and snapshot.cuda_available)
    device = "cuda" if on_gpu else snapshot.accelerator

    if on_gpu and gpu is not None:
        available = gpu.free_memory_mb or gpu.total_memory_mb
    else:
        available = (snapshot.ram_available_gb or 0) * 1024 or None

    trainable = trainable_parameters if trainable_parameters is not None else parameter_count
    bytes_per_param = 2 if config.precision in {"fp16", "bf16"} else 4
    if config.quantization == "4bit":
        weight_bytes = parameter_count * 0.5
    elif config.quantization == "8bit":
        weight_bytes = parameter_count * 1.0
    else:
        weight_bytes = parameter_count * bytes_per_param

    moments = {"adamw": 2, "adam": 2, "adamw_8bit": 0.5, "sgd": 1, "adafactor": 0.5}.get(config.optimizer, 2)
    optimizer_bytes = trainable * 4 * moments
    gradient_bytes = trainable * bytes_per_param
    activation_bytes = activation_estimate(
        config,
        parameter_count=parameter_count,
        bytes_per_param=bytes_per_param,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
    )
    estimated = (weight_bytes + optimizer_bytes + gradient_bytes + activation_bytes) / 1024**2
    if model_memory_mb:
        estimated = max(estimated, model_memory_mb)

    warnings: list[str] = []
    blockers: list[str] = []
    suggestions: list[str] = []

    if not on_gpu:
        warnings.append(
            "No GPU detected — training will run on CPU. Models up to ~50M parameters are "
            "practical; anything larger will be extremely slow."
        )
        if parameter_count > 200_000_000:
            blockers.append(
                f"{parameter_count / 1e6:.0f}M parameters on CPU is not practical "
                "(expect days per epoch)."
            )
            suggestions.append("Choose a smaller model, or use LoRA on a small base.")
        if config.precision in {"fp16", "bf16"}:
            warnings.append(
                f"{config.precision.upper()} is unreliable on CPU — FP32 will be used instead."
            )

    if available:
        headroom = available * 0.9
        if estimated > available:
            blockers.append(
                f"Estimated {estimated / 1024:.1f} GB exceeds the {available / 1024:.1f} GB available."
            )
        elif estimated > headroom:
            warnings.append(
                f"Estimated {estimated / 1024:.1f} GB is close to the "
                f"{available / 1024:.1f} GB available — an OOM is likely."
            )

    if blockers or estimated > (available or float("inf")) * 0.9:
        if config.batch_size > 1:
            suggestions.append(f"Reduce batch size from {config.batch_size} to {max(1, config.batch_size // 2)}")
        suggestions.append("Increase gradient accumulation to keep the effective batch size")
        if not config.gradient_checkpointing:
            suggestions.append("Enable gradient checkpointing (~4x less activation memory)")
        if config.method == "finetune":
            suggestions.append("Switch to LoRA, or QLoRA for 4-bit base weights")
        if config.max_sequence_length > 512:
            suggestions.append(f"Lower max_sequence_length from {config.max_sequence_length} to 512")
        if config.optimizer == "adamw":
            suggestions.append("Use adafactor or adamw_8bit to shrink optimizer state")

    if config.flash_attention and not on_gpu:
        warnings.append("Flash Attention needs a CUDA GPU — it will be ignored.")
    if config.quantization != "none" and not on_gpu:
        blockers.append("8-bit/4-bit quantized training requires a CUDA GPU with bitsandbytes.")
        suggestions.append("Train in FP32 on CPU, or use a machine with an NVIDIA GPU.")

    return PreflightResult(
        ok=not blockers,
        device=device,
        estimated_mb=estimated,
        available_mb=available,
        warnings=warnings,
        blockers=blockers,
        suggestions=list(dict.fromkeys(suggestions)),
        details={
            "parameters": parameter_count,
            "trainable_parameters": trainable,
            "weights_mb": weight_bytes / 1024**2,
            "optimizer_mb": optimizer_bytes / 1024**2,
            "gradients_mb": gradient_bytes / 1024**2,
            "activations_mb": activation_bytes / 1024**2,
            "effective_batch_size": config.effective_batch_size,
        },
    )
