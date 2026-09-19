"""A real decoder-only transformer, implemented in PyTorch.

This is the architecture used by "train from scratch". It is a standard
pre-norm causal transformer with the options the Models screen exposes: RMSNorm
or LayerNorm, SwiGLU/GELU/ReLU feed-forward, rotary or learned positions,
grouped-query attention, and weight tying.

Nothing here is simulated — ``forward`` computes real logits and a real
cross-entropy loss, and the parameter counts reported to the UI come from the
instantiated module.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = ("swiglu", "gelu", "relu", "silu")
NORMS = ("rmsnorm", "layernorm")
POSITION_TYPES = ("rope", "learned", "none")


@dataclass
class TransformerConfig:
    """Architecture definition. Everything the UI exposes lives here."""

    vocab_size: int = 8192
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int | None = None          # None → multi-head (= num_heads)
    intermediate_size: int | None = None      # None → derived from hidden_size
    max_position_embeddings: int = 512
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    activation: str = "swiglu"
    position_embedding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    bias: bool = False
    model_type: str = "ai_studio_transformer"

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            if self.activation == "swiglu":
                # Keep SwiGLU parameter-comparable to a 4x GELU MLP.
                self.intermediate_size = int(8 * self.hidden_size / 3 / 64 + 0.5) * 64
            else:
                self.intermediate_size = 4 * self.hidden_size
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

    # ----------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return human-readable problems; empty list means the config is valid."""
        problems: list[str] = []
        if self.hidden_size % self.num_heads != 0:
            problems.append(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.num_kv_heads and self.num_heads % self.num_kv_heads != 0:
            problems.append(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        if self.activation not in ACTIVATIONS:
            problems.append(f"activation must be one of {ACTIVATIONS}")
        if self.norm_type not in NORMS:
            problems.append(f"norm_type must be one of {NORMS}")
        if self.position_embedding not in POSITION_TYPES:
            problems.append(f"position_embedding must be one of {POSITION_TYPES}")
        if self.vocab_size < 2:
            problems.append("vocab_size must be at least 2.")
        if self.num_layers < 1:
            problems.append("num_layers must be at least 1.")
        if not 0.0 <= self.dropout < 1.0:
            problems.append("dropout must be in [0, 1).")
        if self.head_dim % 2 != 0 and self.position_embedding == "rope":
            problems.append("RoPE needs an even head dimension (hidden_size / num_heads).")
        return problems

    # ------------------------------------------------------------ derived
    @property
    def head_dim(self) -> int:
        return self.hidden_size // max(1, self.num_heads)

    def parameter_count(self) -> dict[str, int]:
        """Analytic parameter count — matches the built module exactly."""
        h, v = self.hidden_size, self.vocab_size
        kv = self.num_kv_heads or self.num_heads
        head = self.head_dim
        inter = self.intermediate_size or 4 * h

        embeddings = v * h
        positions = self.max_position_embeddings * h if self.position_embedding == "learned" else 0

        q = h * h + (h if self.bias else 0)
        k = h * (kv * head) + (kv * head if self.bias else 0)
        value = h * (kv * head) + (kv * head if self.bias else 0)
        o = h * h + (h if self.bias else 0)
        attention = q + k + value + o

        if self.activation == "swiglu":
            mlp = 3 * h * inter + (2 * inter + h if self.bias else 0)
        else:
            mlp = 2 * h * inter + (inter + h if self.bias else 0)

        norms = 2 * h if self.norm_type == "rmsnorm" else 4 * h
        per_layer = attention + mlp + norms
        final_norm = h if self.norm_type == "rmsnorm" else 2 * h
        head_params = 0 if self.tie_word_embeddings else v * h

        total = embeddings + positions + self.num_layers * per_layer + final_norm + head_params
        return {
            "total": total,
            "embeddings": embeddings + positions,
            "per_layer": per_layer,
            "layers": self.num_layers * per_layer,
            "lm_head": head_params,
            "non_embedding": total - embeddings - positions,
        }

    def estimate_memory(
        self,
        *,
        batch_size: int = 1,
        sequence_length: int | None = None,
        precision: str = "fp32",
        optimizer: str = "adamw",
        gradient_checkpointing: bool = False,
    ) -> dict[str, float]:
        """Estimate training memory in MB. Deliberately conservative."""
        params = self.parameter_count()["total"]
        seq = sequence_length or self.max_position_embeddings
        bytes_per_param = 2 if precision in {"fp16", "bf16"} else 4

        weights = params * bytes_per_param
        gradients = params * bytes_per_param
        # AdamW keeps two fp32 moments (+ an fp32 master copy under mixed precision).
        moments = {"adamw": 2, "adam": 2, "adamw_8bit": 0.5, "sgd": 1, "adafactor": 0.5}.get(
            optimizer, 2
        )
        optimizer_state = params * 4 * moments
        master_weights = params * 4 if bytes_per_param == 2 else 0

        # Activations: a rough per-token, per-layer cost.
        per_token_layer = self.hidden_size * 12 + (self.intermediate_size or 0) * 2
        activations = batch_size * seq * self.num_layers * per_token_layer * bytes_per_param
        if gradient_checkpointing:
            activations *= 0.25
        logits = batch_size * seq * self.vocab_size * 4  # cross-entropy runs in fp32

        total = weights + gradients + optimizer_state + master_weights + activations + logits
        mb = 1024 * 1024
        return {
            "parameters": params,
            "weights_mb": weights / mb,
            "gradients_mb": gradients / mb,
            "optimizer_mb": (optimizer_state + master_weights) / mb,
            "activations_mb": activations / mb,
            "logits_mb": logits / mb,
            "total_mb": total / mb,
            "inference_mb": (weights + activations * 0.1) / mb,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> "TransformerConfig":
        data = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        return cls.from_dict(data)


# --------------------------------------------------------------------- parts
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


def build_norm(config: TransformerConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.hidden_size, config.norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.norm_eps)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE)."""

    def __init__(self, head_dim: int, max_positions: int, theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.scaling = max(scaling, 1e-6)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_positions)

    def _build_cache(self, positions: int) -> None:
        t = torch.arange(positions, dtype=torch.float32) / self.scaling
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_positions = positions

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0):
        needed = seq_len + offset
        if needed > self._cached_positions:
            self._build_cache(int(needed * 1.5))
        cos = self.cos_cached[offset : offset + seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head / grouped-query causal attention."""

    def __init__(self, config: TransformerConfig, rotary: RotaryEmbedding | None) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.head_dim
        self.repeats = self.num_heads // self.num_kv_heads
        self.rotary = rotary

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.bias)
        self.attention_dropout = config.attention_dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_key_value[0].shape[-2] if past_key_value is not None else 0
        if self.rotary is not None:
            cos, sin = self.rotary(seq, x.device, q.dtype, offset=offset)
            q, k = apply_rotary(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        if self.repeats > 1:  # grouped-query: broadcast kv heads to query heads
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        is_causal = attention_mask is None and offset == 0 and seq > 1
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.resid_dropout(self.o_proj(attn)), present


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        inter = config.intermediate_size or 4 * config.hidden_size
        self.activation = config.activation
        if self.activation == "swiglu":
            self.gate_proj = nn.Linear(config.hidden_size, inter, bias=config.bias)
            self.up_proj = nn.Linear(config.hidden_size, inter, bias=config.bias)
            self.down_proj = nn.Linear(inter, config.hidden_size, bias=config.bias)
        else:
            self.up_proj = nn.Linear(config.hidden_size, inter, bias=config.bias)
            self.down_proj = nn.Linear(inter, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        elif self.activation == "gelu":
            hidden = F.gelu(self.up_proj(x), approximate="tanh")
        elif self.activation == "silu":
            hidden = F.silu(self.up_proj(x))
        else:
            hidden = F.relu(self.up_proj(x))
        return self.dropout(self.down_proj(hidden))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, rotary: RotaryEmbedding | None) -> None:
        super().__init__()
        self.input_norm = build_norm(config)
        self.attention = CausalSelfAttention(config, rotary)
        self.post_attention_norm = build_norm(config)
        self.mlp = FeedForward(config)

    def forward(self, x, *, attention_mask=None, past_key_value=None, use_cache=False):
        attn_out, present = self.attention(
            self.input_norm(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_norm(x))
        return x, present


class TransformerLM(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        problems = config.validate()
        if problems:
            raise ValueError("Invalid model configuration: " + "; ".join(problems))

        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding == "learned"
            else None
        )
        rotary = (
            RotaryEmbedding(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.rope_scaling,
            )
            if config.position_embedding == "rope"
            else None
        )
        self.rotary = rotary
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, rotary) for _ in range(config.num_layers)])
        self.norm = build_norm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style) keeps deep stacks stable.
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    parameter, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.num_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------- helpers
    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        seen: set[int] = set()
        total = 0
        for parameter in params:  # tied weights must only count once
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    # ------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        batch, seq = input_ids.shape
        offset = past_key_values[0][0].shape[-2] if past_key_values else 0

        x = self.embed_tokens(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(offset, offset + seq, device=input_ids.device)
            x = x + self.position_embeddings(positions).unsqueeze(0)
        x = self.dropout(x)

        mask = None
        if attention_mask is not None:
            # (batch, seq) padding mask → additive (batch, 1, seq, total) causal mask
            total = offset + seq
            pad = attention_mask[:, None, None, :total].to(dtype=x.dtype)
            pad = (1.0 - pad) * torch.finfo(x.dtype).min
            causal = torch.full((seq, total), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
            causal = torch.triu(causal, diagonal=1 + offset)
            mask = pad + causal[None, None, :, :]

        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, present = torch.utils.checkpoint.checkpoint(
                    lambda inp, lyr=layer, m=mask: lyr(inp, attention_mask=m),
                    x,
                    use_reentrant=False,
                )
            else:
                x, present = layer(x, attention_mask=mask, past_key_value=past, use_cache=use_cache)
            if use_cache and present is not None:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "past_key_values": presents if use_cache else None}

    # ---------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        streamer: Any = None,
    ) -> torch.Tensor:
        self.eval()
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        generated = input_ids
        current = input_ids

        for _ in range(max_new_tokens):
            window = current[:, -self.config.max_position_embeddings :]
            outputs = self.forward(window, past_key_values=past, use_cache=True)
            past = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :].float()

            if repetition_penalty and repetition_penalty != 1.0:
                for token in set(generated[0].tolist()):
                    logits[0, token] /= repetition_penalty

            if temperature and temperature > 0:
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p and 0 < top_p < 1.0:
                    ordered, indices = torch.sort(logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
                    remove = cumulative - F.softmax(ordered, dim=-1) > top_p
                    ordered = ordered.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, ordered)
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            current = next_token
            if streamer is not None:
                streamer(next_token.item())
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    # ------------------------------------------------------- serialisation
    def save_pretrained(self, directory: Path | str, *, metadata: dict[str, Any] | None = None) -> Path:
        """Save weights (safetensors when available) plus config.json."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory)

        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        if self.config.tie_word_embeddings:
            state.pop("lm_head.weight", None)  # tied → restored on load

        try:
            from safetensors.torch import save_file

            save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
        except ImportError:
            torch.save(state, directory / "pytorch_model.bin")

        (directory / "generation_config.json").write_text(
            json.dumps(
                {
                    "max_new_tokens": 256,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if metadata:
            (directory / "ai_studio_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def from_pretrained(cls, directory: Path | str, *, device: str | torch.device = "cpu") -> "TransformerLM":
        directory = Path(directory)
        config = TransformerConfig.load(directory)
        model = cls(config)

        safetensors_path = directory / "model.safetensors"
        bin_path = directory / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path))
        elif bin_path.exists():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {directory}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not (config.tie_word_embeddings and k == "lm_head.weight")]
        if missing or unexpected:
            raise ValueError(f"Checkpoint mismatch — missing: {missing}, unexpected: {unexpected}")
        return model.to(device)


#: Ready-made sizes offered in the UI. Layer counts are tuned so the advertised
#: size matches the real parameter count (verified against the built module).
SIZE_PRESETS: dict[str, dict[str, Any]] = {
    "tiny-10m": dict(
        hidden_size=256, num_layers=10, num_heads=4, max_position_embeddings=512, vocab_size=8192
    ),
    "small-50m": dict(
        hidden_size=512, num_layers=13, num_heads=8, max_position_embeddings=1024, vocab_size=16000
    ),
    "base-100m": dict(
        hidden_size=768, num_layers=11, num_heads=12, max_position_embeddings=1024, vocab_size=32000
    ),
    "mid-200m": dict(
        hidden_size=896, num_layers=18, num_heads=14, max_position_embeddings=2048, vocab_size=32000
    ),
    "medium-300m": dict(
        hidden_size=1024, num_layers=21, num_heads=16, max_position_embeddings=2048, vocab_size=32000
    ),
    "large-500m": dict(
        hidden_size=1280, num_layers=23, num_heads=20, max_position_embeddings=2048, vocab_size=32000
    ),
    "huge-1b": dict(
        hidden_size=2048, num_layers=19, num_heads=16, max_position_embeddings=2048, vocab_size=32000
    ),
}

#: Smallest useful size for a smoke test / CPU experimentation.
SIZE_PRESETS["nano-1m"] = dict(
    hidden_size=128, num_layers=4, num_heads=4, max_position_embeddings=256, vocab_size=4096
)


def preset_config(preset: str, **overrides: Any) -> TransformerConfig:
    if preset not in SIZE_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(SIZE_PRESETS)}")
    return TransformerConfig(**{**SIZE_PRESETS[preset], **overrides})
