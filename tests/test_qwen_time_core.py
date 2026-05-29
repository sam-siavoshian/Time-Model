from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.qwen_time import (
    V15_CHUNK_LENGTH,
    V15_LORA_RANK,
    V15_TIMESCALES,
    QwenTimeConfig,
    _Chronometric,
    build_qwen_time,
    qwen_time_checkpoint_metadata,
    qwen_time_config_dict,
)
from model.qwen_time_check import load_trainable


class _FakeAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)


class _FakeBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.self_attn = _FakeAttention(d_model)

    def forward(self, x):
        return x


class _FakeModelBody(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([_FakeBlock(d_model) for _ in range(n_layers)])


class _FakeCausalLM(nn.Module):
    def __init__(self, n_layers: int = 4, d_model: int = 6, vocab_size: int = 11):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=d_model, tie_word_embeddings=True)
        self.model = _FakeModelBody(n_layers, d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, return_dict=True):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return types.SimpleNamespace(logits=self.lm_head(h))


class _FakeTokenizer:
    pass


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _FakeTokenizer()


class _FakeAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _FakeCausalLM()


@pytest.fixture()
def fake_transformers(monkeypatch):
    fake_module = types.SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_chronometric_dimension_and_sin_tau_over_t_formula():
    enc = _Chronometric((2, 4))
    tau = torch.tensor(2.0)

    out = enc(tau)

    assert enc.out_dim == 5
    expected = torch.tensor([
        torch.sin(torch.tensor(1.0)),
        torch.sin(torch.tensor(0.5)),
        torch.cos(torch.tensor(1.0)),
        torch.cos(torch.tensor(0.5)),
        torch.log1p(torch.tensor(2.0)),
    ])
    torch.testing.assert_close(out.squeeze(0), expected)


def test_qwen_time_config_uses_canonical_v15_defaults():
    cfg = QwenTimeConfig()

    assert cfg.timescales == V15_TIMESCALES
    assert len(cfg.timescales) == 15
    assert 2 * len(cfg.timescales) + 1 == 31
    assert cfg.chunk_length == V15_CHUNK_LENGTH
    assert cfg.lora_rank == V15_LORA_RANK
    assert cfg.inject_layers == ()
    assert cfg.injection_type == "film"


def test_empty_inject_layers_resolves_to_all_but_final_layer(fake_transformers):
    model = build_qwen_time(QwenTimeConfig(base_model_name="fake/qwen"))

    assert model._inject_layers == (0, 1, 2)
    assert set(model.chrono_injectors.keys()) == {"0", "1", "2"}


def test_checkpoint_config_mismatch_is_rejected(fake_transformers, tmp_path):
    cfg = QwenTimeConfig(base_model_name="fake/qwen")
    model = build_qwen_time(cfg)

    mismatched_cfg = QwenTimeConfig(
        base_model_name="fake/qwen",
        timescales=V15_TIMESCALES[:-1],
    )
    ckpt = {
        "trainable_state": {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        },
        "cfg": qwen_time_config_dict(mismatched_cfg),
        "config_metadata": qwen_time_checkpoint_metadata(
            mismatched_cfg,
            tuple(model._inject_layers),
        ),
        "train_step": 1,
    }
    path = tmp_path / "mismatch.pt"
    torch.save(ckpt, path)

    with pytest.raises(ValueError, match="timescales"):
        load_trainable(model, str(path))

