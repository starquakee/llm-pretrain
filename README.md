# llm-pretrain

一个面向学习与实验的单卡中文 LLM 预训练工程：在 RTX 4060 Ti 8GB 上训练约
100M 参数的 decoder-only 模型，并用受控 A/B 比较 Muon 与 AdamW。项目覆盖自训练
SentencePiece tokenizer、确定性数据打包、可恢复预训练、评测、COIG 指令微调和本地生成。

这个规模适合学习完整训练链路，不等于成熟聊天助手。不要期待可靠事实记忆、复杂推理或生产质量。

## 安全边界

- Git 只保存代码、配置、文档和极小合成测试数据。
- `data/`、tokenizer、shards、checkpoint、权重和 TensorBoard/JSONL 日志全部在仓库外。
- Ralph 只实现和验证代码；不会自动下载完整语料或启动最长 120 小时的正式训练。
- 正式训练前关闭壁纸、游戏启动器、GPU overlay、渲染和本地推理进程。

## WSL2 环境

推荐在 WSL2 Ubuntu 中运行。Windows NVIDIA 驱动会把 CUDA 映射进 WSL；不要在 WSL
里安装 `cuda-drivers` 或 Linux 显卡驱动。参见 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

```bash
git clone https://github.com/starquakee/llm-pretrain.git
cd llm-pretrain
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --frozen --extra dev
uv run llm-pretrain doctor --strict
```

如果只安装 PyTorch wheel，不需要额外安装完整 CUDA Toolkit。`doctor` 必须看到 CUDA、BF16、
SDPA、`torch.compile` 和 `torch.optim.Muon` 后才允许正式训练。

## 数据与许可

默认 token 配比为：

- 80% [OpenBMB Ultra-FineWeb 中文](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)
- 10% [中文 Wikimedia dump](https://dumps.wikimedia.org/legal.html)
- 10% [Hugging Face FineWeb 英文](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

SFT 使用 [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) 的确定性子集。配置记录数据集
revision、许可和来源，但网页数据集的整理许可不替代每个原始网页的权利。项目不重新分发语料或权重。

## 命令闭环

先查看每个子命令的参数：

```bash
uv run llm-pretrain --help
uv run llm-pretrain doctor --strict
uv run llm-pretrain data prepare --config configs/data_sources.yaml --allow-network
uv run llm-pretrain tokenizer train --config configs/tokenizer_24k.yaml
uv run llm-pretrain tokenizer eval --config configs/tokenizer_24k.yaml
uv run llm-pretrain data tokenize --config configs/data_sources.yaml
uv run llm-pretrain train ab --config configs/pretrain_100m_muon.yaml
uv run llm-pretrain evaluate --config configs/pretrain_100m_muon.yaml
uv run llm-pretrain train sft --config configs/sft_coig.yaml --allow-network
uv run llm-pretrain generate --config configs/pretrain_100m_muon.yaml --prompt "杭州是一座"
```

真实数据命令要求显式设置仓库外目录，例如：

```bash
export LLM_PRETRAIN_ARTIFACT_ROOT=/mnt/d/llm-pretrain-artifacts
```

数据准备和 COIG 下载默认离线；只有明确传入 `--allow-network` 才会访问远端。已有本地
JSONL 时，分别使用 `data prepare --input SOURCE=PATH` 和 `train sft --records PATH`，无需
联网开关。

### 正式预训练

只有 A/B gate 通过后才手动启动：

```bash
uv run llm-pretrain train pretrain \
  --config configs/pretrain_100m_muon.yaml \
  --gate-record "$LLM_PRETRAIN_ARTIFACT_ROOT/runs/ab/gate.json"
```

正式 token 数按端到端基准计算：

```text
min(2,000,000,000, measured_tokens_per_second × 120 hours × 0.9)
```

训练中断后使用同一配置与 `--resume` 指向 checkpoint。训练器不会在运行中静默修改
micro-batch 或优化器。

### 监控

```bash
uv run tensorboard --logdir "$LLM_PRETRAIN_ARTIFACT_ROOT/runs"
```

JSONL 和 TensorBoard 记录 loss、学习率、梯度范数、吞吐、显存、温度、评测与 checkpoint 事件。

## 模型与优化器

默认模型：12 层、宽度 768、12 头 MHA、SwiGLU 1920、RoPE、RMSNorm、24,576 tied
embedding、context 1024、无 bias/dropout，约 100.3M 参数。

Muon 只更新 Transformer block 内的二维 Linear 权重；embedding、共享输出头、RMSNorm 和其他
参数由 AdamW 更新。初始 A/B 各跑 100M tokens；Muon 必须保持有限、验证 loss 不高于 AdamW
`+0.05`、吞吐不低于 AdamW 的 80%。失败时只允许一次 `{0.01, 0.04}` LR 校准。

## 开发与 Ralph

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
uv run python scripts/audit_tracked_files.py
bash scripts/ralph/ralph.sh --agent codex --dry-run
```

Ralph 在 `ralph/100m-llm-pretrain` 分支按 `prd.json` 顺序一次完成一个故事，且本仓库明确
拒绝 `--unsafe`。外部 agent 不可用时，dry-run 会诚实失败，不会使用伪适配器绕过检查。

## License

代码使用 Apache License 2.0。数据集、数据内容和未来训练产物受各自许可与来源条款约束。
