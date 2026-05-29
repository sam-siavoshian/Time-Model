from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.qwen_time import (  # noqa: E402
    CHRONO_FORMULA,
    V15_TIMESCALES,
    QwenTimeConfig,
    _Chronometric,
    qwen_time_checkpoint_metadata,
    resolve_inject_layers,
)
import model.qwen_time_check as qwen_time_check  # noqa: E402
from model.qwen_time_check import load_trainable  # noqa: E402


def test_v15_config_defaults_are_canonical():
    cfg = QwenTimeConfig()
    assert cfg.timescales == V15_TIMESCALES
    assert len(cfg.timescales) == 15
    assert cfg.lora_rank == 8
    assert cfg.chunk_length == 512


def test_chronometric_formula_and_dimension():
    encoder = _Chronometric(V15_TIMESCALES)
    out = encoder(torch.tensor(float(V15_TIMESCALES[0])))
    assert encoder.out_dim == 31
    assert out.shape == (1, 31)
    assert CHRONO_FORMULA == "sin(tau / T), cos(tau / T), log1p(tau)"
    assert torch.isclose(out[0, 0], torch.tensor(math.sin(1.0)), atol=1e-6)
    assert torch.isclose(out[0, len(V15_TIMESCALES)], torch.tensor(math.cos(1.0)), atol=1e-6)
    assert torch.isclose(out[0, -1], torch.log1p(torch.tensor(float(V15_TIMESCALES[0]))), atol=1e-6)


def test_default_layer_policy_skips_final_layer():
    cfg = QwenTimeConfig()
    assert resolve_inject_layers(cfg, 36) == tuple(range(35))
    cfg.inject_layers = (0,)
    assert resolve_inject_layers(cfg, 36) == (0,)


class _DummyModel:
    def __init__(self):
        self.cfg = QwenTimeConfig()
        self._inject_layers = tuple(range(35))
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def named_parameters(self):
        return [("weight", self.weight)]

    def checkpoint_metadata(self):
        meta = qwen_time_checkpoint_metadata(self.cfg, self._inject_layers)
        meta["trainable_names"] = ["weight"]
        return meta


def test_checkpoint_config_mismatch_fails_loudly(tmp_path):
    model = _DummyModel()
    bad_cfg = QwenTimeConfig()
    bad_cfg.timescales = (2, 4)
    state = {
        "trainable_state": {"weight": torch.ones(1)},
        "config_metadata": qwen_time_checkpoint_metadata(bad_cfg, (0,)),
    }
    state["config_metadata"]["trainable_names"] = ["weight"]
    ckpt = tmp_path / "bad.pt"
    torch.save(state, ckpt)

    try:
        load_trainable(model, str(ckpt))
    except ValueError as exc:
        assert "Checkpoint config mismatch" in str(exc)
        assert "timescales" in str(exc)
    else:
        raise AssertionError("expected checkpoint config mismatch")


def test_unregistered_legacy_checkpoint_requires_explicit_override(tmp_path):
    model = _DummyModel()
    ckpt = tmp_path / "legacy.pt"
    torch.save({"trainable_state": {"weight": torch.ones(1)}}, ckpt)

    try:
        load_trainable(model, str(ckpt))
    except ValueError as exc:
        assert "no embedded config metadata" in str(exc)
    else:
        raise AssertionError("expected unregistered checkpoint rejection")

    load_trainable(model, str(ckpt), allow_unregistered_legacy_ckpt=True)
    assert torch.equal(model.weight.detach(), torch.ones(1))


def test_sidecar_registry_allows_historical_checkpoint(tmp_path, monkeypatch):
    model = _DummyModel()
    ckpt = tmp_path / "registered.pt"
    torch.save({"trainable_state": {"weight": torch.ones(1)}}, ckpt)
    registry = tmp_path / "metadata_registry.json"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "defaults": {
            "base_model_name": model.cfg.base_model_name,
            "timescales": list(model.cfg.timescales),
            "inject_layers": list(model.cfg.inject_layers),
            "lora_rank": model.cfg.lora_rank,
            "lora_layers": model.cfg.lora_layers,
            "lora_targets": list(model.cfg.lora_targets),
            "lora_lm_head": model.cfg.lora_lm_head,
            "unfreeze_base": model.cfg.unfreeze_base,
            "chunk_length": model.cfg.chunk_length,
            "injection_type": model.cfg.injection_type,
            "additive_beta_init": model.cfg.additive_beta_init,
            "use_ia3": model.cfg.use_ia3,
        },
        "entries": [{"path": str(ckpt), "status": "historical"}],
    }))
    monkeypatch.setattr(qwen_time_check, "CHECKPOINT_METADATA_REGISTRY", registry)

    load_trainable(model, str(ckpt))
    assert torch.equal(model.weight.detach(), torch.ones(1))
