from __future__ import annotations

import pytest
import torch

from llm_pretrain.config import ModelConfig
from llm_pretrain.model import CausalLM, RotaryEmbedding


def small_config(**overrides) -> ModelConfig:
    values = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "intermediate_size": 64,
        "activation_checkpointing": False,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_default_model_has_about_100m_parameters() -> None:
    with torch.device("meta"):
        model = CausalLM(ModelConfig())

    assert 99_000_000 <= model.num_parameters <= 102_000_000
    assert model.num_parameters == 100_289_280


def test_embedding_and_output_weights_are_tied() -> None:
    model = CausalLM(small_config())

    assert model.lm_head.weight is model.token_embedding.weight


def test_forward_returns_logits_and_shifted_loss() -> None:
    torch.manual_seed(7)
    model = CausalLM(small_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 8, model.config.vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_attention_is_causal() -> None:
    torch.manual_seed(11)
    model = CausalLM(small_config()).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    sequence = torch.tensor([[1, 2, 3, 4, 17, 23]])

    with torch.no_grad():
        prefix_logits = model(prefix).logits
        sequence_logits = model(sequence).logits[:, : prefix.size(1)]

    torch.testing.assert_close(prefix_logits, sequence_logits, rtol=1e-5, atol=1e-6)


def test_rotary_embedding_preserves_shape_and_vector_norm() -> None:
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=16, theta=10_000.0)
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 7, 8)

    rotated_query, rotated_key = rotary(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(
        rotated_query.float().norm(dim=-1),
        query.float().norm(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_activation_checkpointing_supports_backward() -> None:
    torch.manual_seed(19)
    model = CausalLM(small_config(activation_checkpointing=True)).train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_ignore_index_supports_response_only_loss() -> None:
    model = CausalLM(small_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])

    output = model(input_ids, labels=labels)

    expected = torch.nn.functional.cross_entropy(
        output.logits[:, 1:3].float().reshape(-1, model.config.vocab_size),
        torch.tensor([3, 4]),
    )
    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def test_all_ignored_labels_are_rejected() -> None:
    model = CausalLM(small_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with pytest.raises(ValueError, match="no trainable"):
        model(input_ids, labels=torch.full_like(input_ids, -100))
