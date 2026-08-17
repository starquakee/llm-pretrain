from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from llm_pretrain.web import (
    _INDEX_HTML,
    GenerationService,
    RequestError,
    _parse_generation_payload,
    serve_generation_ui,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.config = SimpleNamespace(max_seq_len=16)

    def forward(self, input_ids: Tensor) -> Tensor:
        logits = torch.full((*input_ids.shape, 8), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        return logits


class TinyTokenizer:
    eos_id = 3

    def encode(self, text: str, **kwargs) -> list[int]:
        return [1, 2] if text else []

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join("字" if token_id == 4 else "" for token_id in ids)


def test_generation_payload_is_strict_and_bounded() -> None:
    parsed = _parse_generation_payload(
        {"prompt": "  杭州是一座  ", "max_new_tokens": 12, "temperature": 0, "top_k": 0}
    )
    assert parsed["prompt"] == "杭州是一座"
    assert parsed["temperature"] == 0.0
    assert parsed["top_k"] is None

    with pytest.raises(RequestError, match="提示词"):
        _parse_generation_payload({"prompt": ""})
    with pytest.raises(RequestError, match=r"1.256"):
        _parse_generation_payload({"prompt": "测试", "max_new_tokens": 999})


def test_generation_service_reports_model_and_generates() -> None:
    service = GenerationService(
        TinyModel(), TinyTokenizer(), checkpoint="/tmp/best.pt", use_bf16=False
    )

    status = service.status()
    result = service.generate(
        {"prompt": "测试", "max_new_tokens": 3, "temperature": 0, "top_k": 50, "seed": 7}
    )

    assert status["ready"] is True
    assert status["parameters"] == 3
    assert status["context_length"] == 16
    assert result["completion"] == "字字字"
    assert result["generated_tokens"] == 3
    assert result["tokens_per_second"] > 0


def test_web_ui_is_local_only_and_self_contained() -> None:
    html = _INDEX_HTML.decode("utf-8")
    assert "壹亿字室" in html
    assert "/api/generate" in html
    assert "https://" not in html

    service = GenerationService(TinyModel(), TinyTokenizer(), checkpoint="best.pt")
    with pytest.raises(ValueError, match="local-only"):
        serve_generation_ui(service, host="0.0.0.0", port=7860)
