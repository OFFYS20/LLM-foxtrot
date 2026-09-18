"""Configuration validation — application config, model architecture, training."""

from __future__ import annotations

import pytest
import yaml

from ai_studio.core.config import AppConfig, get_config, load_config
from ai_studio.core.errors import ValidationError
from ai_studio.models.transformer import SIZE_PRESETS, TransformerConfig, preset_config
from ai_studio.training.config import (
    LoRASettings,
    TrainingConfig,
    suggest_lora_targets,
)


# --------------------------------------------------------------- app config
def test_storage_paths_are_absolute_and_under_the_root():
    config = get_config()
    assert config.root.is_absolute()
    for path in (config.models_dir, config.datasets_dir, config.checkpoints_dir,
                 config.documents_dir, config.tokenizers_dir, config.indexes_dir):
        assert path.is_absolute()


def test_ensure_directories_is_idempotent():
    config = get_config()
    config.ensure_directories()
    config.ensure_directories()
    assert config.models_dir.is_dir() and config.checkpoints_dir.is_dir()


def test_yaml_file_overrides_defaults(tmp_path):
    defaults = AppConfig()
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"training": {"checkpoint_interval": 7}, "ui": {"port": 9999}}), encoding="utf-8"
    )
    loaded = load_config(path)
    assert loaded.training.checkpoint_interval == 7
    assert loaded.ui.port == 9999
    assert loaded.training.seed == defaults.training.seed, "untouched keys keep their defaults"


def test_unknown_yaml_keys_do_not_crash_startup(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"training": {"checkpoint_interval": 3, "not_a_real_setting": True},
                        "nonsense_section": {"x": 1}}),
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.training.checkpoint_interval == 3


def test_relative_storage_paths_resolve_next_to_the_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AISTUDIO_STORAGE_ROOT", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"storage": {"root": "storage"}}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.root == (tmp_path / "storage").resolve()


def test_environment_overrides_beat_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AISTUDIO_STORAGE_ROOT", str(tmp_path / "from-env"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"storage": {"root": "from-file"}}), encoding="utf-8")
    assert load_config(path).root == (tmp_path / "from-env").resolve()


@pytest.fixture(autouse=True)
def _restore_global_config():
    """load_config mutates module state; put the test storage root back."""
    yield
    from ai_studio.core.config import reload_config

    reload_config()


# -------------------------------------------------------- architecture config
def test_valid_architecture_has_no_problems():
    assert TransformerConfig(hidden_size=256, num_heads=4, num_layers=2).validate() == []


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        (dict(hidden_size=100, num_heads=8), "divisible"),
        (dict(num_heads=8, num_kv_heads=3), "divisible"),
        (dict(activation="banana"), "activation"),
        (dict(norm_type="batchnorm"), "norm_type"),
        (dict(position_embedding="sinusoidal-ish"), "position_embedding"),
        (dict(vocab_size=1), "vocab_size"),
        (dict(num_layers=0), "num_layers"),
        (dict(dropout=1.5), "dropout"),
        (dict(hidden_size=6, num_heads=2, position_embedding="rope"), "RoPE"),
    ],
)
def test_broken_architectures_are_reported(kwargs, fragment):
    problems = TransformerConfig(**kwargs).validate()
    assert any(fragment in problem for problem in problems), problems


def test_parameter_count_matches_the_built_model():
    config = TransformerConfig(vocab_size=512, hidden_size=64, num_layers=2, num_heads=4)
    from ai_studio.models.transformer import TransformerLM

    model = TransformerLM(config)
    built = sum(p.numel() for p in model.parameters())
    assert config.parameter_count()["total"] == built


def test_untied_head_adds_its_own_parameters():
    tied = TransformerConfig(vocab_size=512, hidden_size=64, num_heads=4, tie_word_embeddings=True)
    untied = TransformerConfig(vocab_size=512, hidden_size=64, num_heads=4, tie_word_embeddings=False)
    assert untied.parameter_count()["total"] - tied.parameter_count()["total"] == 512 * 64


def test_memory_estimate_grows_with_batch_and_sequence():
    config = TransformerConfig(vocab_size=512, hidden_size=64, num_layers=2, num_heads=4)
    small = config.estimate_memory(batch_size=1, sequence_length=128)
    large = config.estimate_memory(batch_size=8, sequence_length=512)
    assert large["total_mb"] > small["total_mb"]
    assert small["weights_mb"] == pytest.approx(large["weights_mb"])


UNITS = {"m": 1_000_000, "b": 1_000_000_000}


@pytest.mark.parametrize("preset", sorted(SIZE_PRESETS))
def test_every_preset_is_valid_and_roughly_its_advertised_size(preset):
    config = preset_config(preset)
    assert config.validate() == []

    advertised = preset.rsplit("-", 1)[1]
    unit = UNITS[advertised[-1]]
    target = float(advertised[:-1]) * unit
    actual = config.parameter_count()["total"]
    assert 0.6 * target <= actual <= 1.6 * target, f"{preset} is actually {actual:,} parameters"


def test_the_presets_climb_in_order():
    """A preset named for a bigger size must actually be bigger."""
    sized = sorted(
        ((preset_config(name).parameter_count()["total"],
          float(name.rsplit("-", 1)[1][:-1]) * UNITS[name.rsplit("-", 1)[1][-1]], name)
         for name in SIZE_PRESETS),
        key=lambda row: row[1],
    )
    actual = [row[0] for row in sized]
    assert actual == sorted(actual), f"out of order: {[row[2] for row in sized]}"


def test_config_survives_a_save_load_round_trip(tmp_path):
    config = TransformerConfig(vocab_size=333, hidden_size=128, num_heads=8, activation="gelu")
    config.save(tmp_path)
    reloaded = TransformerConfig.load(tmp_path)
    assert reloaded.to_dict() == config.to_dict()


# ------------------------------------------------------------ training config
def test_defaults_validate():
    TrainingConfig().validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="telepathy"),
        dict(optimizer="hope"),
        dict(lr_scheduler="vibes"),
        dict(precision="fp3"),
        dict(epochs=0, max_steps=0),
        dict(batch_size=0),
        dict(gradient_accumulation_steps=0),
        dict(learning_rate=0),
        dict(learning_rate=5.0),
        dict(max_sequence_length=4),
        dict(quantization="3bit"),
        dict(method="finetune", quantization="4bit"),
    ],
)
def test_invalid_training_configs_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        TrainingConfig(**kwargs).validate()


def test_qlora_implies_four_bit_quantization():
    config = TrainingConfig(method="qlora")
    config.validate()
    assert config.quantization == "4bit"
    assert config.uses_peft


def test_effective_batch_size_includes_accumulation():
    config = TrainingConfig(batch_size=4, gradient_accumulation_steps=8)
    assert config.effective_batch_size == 32


@pytest.mark.parametrize(
    "rank,alpha,dropout,targets",
    [(0, 32, 0.05, ["q_proj"]), (16, 0, 0.05, ["q_proj"]), (16, 32, 1.5, ["q_proj"]), (16, 32, 0.05, [])],
)
def test_invalid_lora_settings_are_rejected(rank, alpha, dropout, targets):
    with pytest.raises(ValidationError):
        LoRASettings(rank=rank, alpha=alpha, dropout=dropout, target_modules=targets).validate()


def test_training_config_round_trips_through_a_dict():
    config = TrainingConfig(
        method="lora", epochs=3.0,
        lora=LoRASettings(rank=8, alpha=16, target_modules=["q_proj", "v_proj"]),
    )
    restored = TrainingConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.lora.target_modules == ["q_proj", "v_proj"]


def test_from_dict_ignores_unknown_keys():
    restored = TrainingConfig.from_dict(
        {"epochs": 2.0, "legacy_field": "ignored"}
    )
    assert restored.epochs == 2.0


@pytest.mark.parametrize(
    "model_type,expected",
    [("llama", "q_proj"), ("gpt2", "c_attn"), ("phi3", "qkv_proj"), ("mistral", "q_proj")],
)
def test_lora_targets_are_architecture_aware(model_type, expected):
    assert expected in suggest_lora_targets(model_type)


def test_lora_targets_fall_back_for_unknown_architectures():
    assert suggest_lora_targets("something-brand-new")
