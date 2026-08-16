from __future__ import annotations

import pytest
import torch

from llm_pretrain.checkpoint import load_checkpoint, save_checkpoint
from llm_pretrain.config import ModelConfig, OptimizerConfig, RunState
from llm_pretrain.model import CausalLM
from llm_pretrain.optim import create_optimizers

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_muon_step_and_checkpoint_round_trip(tmp_path) -> None:
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    torch.manual_seed(1337)
    config = ModelConfig(
        vocab_size=128,
        max_seq_len=16,
        n_layers=2,
        d_model=64,
        n_heads=4,
        intermediate_size=128,
        activation_checkpointing=False,
    )
    model = CausalLM(config).cuda()
    optimizer = create_optimizers(model, OptimizerConfig())
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=input_ids)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    state = RunState(step=1, tokens_seen=input_ids.numel())
    checkpoint = save_checkpoint(
        tmp_path / "gpu-step.pt",
        model,
        optimizer=optimizer,
        run_state=state,
    )
    restored = CausalLM(config).cuda()
    restored_optimizer = create_optimizers(restored, OptimizerConfig())
    restored_state = RunState()
    load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        run_state=restored_state,
        map_location="cuda",
    )
    assert restored_state.step == 1
    torch.testing.assert_close(restored.token_embedding.weight, model.token_embedding.weight)
