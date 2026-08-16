"""A compact decoder-only transformer for the 100M pretraining project."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    """Output returned by :class:`CausalLM`."""

    logits: Tensor
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with an fp32 variance calculation."""

    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs: Tensor) -> Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalised = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalised * self.weight.to(inputs.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding frequencies shared by every attention layer."""

    cosine: Tensor
    sine: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        sequence_length = query.size(-2)
        cosine = self.cosine[:sequence_length].to(query.dtype)[None, None, :, :]
        sine = self.sine[:sequence_length].to(query.dtype)[None, None, :, :]
        return _apply_rotary(query, cosine, sine), _apply_rotary(key, cosine, sine)


def _apply_rotary(inputs: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


class CausalSelfAttention(nn.Module):
    """Bias-free multi-head attention backed by PyTorch SDPA."""

    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = rotary

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        qkv = self.qkv(inputs).view(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rotary(query, key)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.n_heads * self.head_dim)
        )
        return self.proj(attended)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(
            config.d_model,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, up = self.gate_up(inputs).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Pre-normalisation transformer block."""

    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, rotary)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.ffn_norm(inputs))


class CausalLM(nn.Module):
    """Decoder-only language model with tied token embedding and LM-head weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, rotary) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        """Compute causal logits and, when labels are supplied, next-token loss.

        ``labels`` has the same shape as ``input_ids``.  It is shifted internally;
        values equal to ``-100`` are ignored, which enables response-only SFT.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids sequence must not be empty")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))

        loss: Tensor | None = None
        if labels is not None:
            if labels.size(1) < 2:
                raise ValueError("at least two tokens are required to compute causal loss")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if not torch.compiler.is_compiling() and not bool(shift_labels.ne(-100).any()):
                raise ValueError("labels contain no trainable next-token targets")
            loss = F.cross_entropy(
                shift_logits.float().view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @property
    def num_parameters(self) -> int:
        """Number of unique trainable parameters (weight tying counted once)."""

        return sum(parameter.numel() for parameter in self.parameters())


# A concise alias for callers that prefer the architecture-oriented name.
TransformerLM = CausalLM


def estimate_model_flops_per_token(config: ModelConfig) -> int:
    """Return the conventional training FLOP approximation, ``6 * parameters``."""

    with torch.device("meta"):
        model = CausalLM(config)
    return 6 * model.num_parameters
