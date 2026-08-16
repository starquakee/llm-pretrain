# PRD: 100M Chinese LLM Pretraining Lab

## Introduction

Build a readable, reproducible PyTorch project for training a roughly 100M-parameter
Chinese-first decoder model on one RTX 4060 Ti 8GB. The repository must cover tokenizer
training, deterministic data preparation, Muon versus AdamW validation, resumable pretraining,
evaluation, small-scale instruction tuning, and local generation without committing data or
weights.

## References

- `README.md`
- `configs/pretrain_100m_muon.yaml`
- PyTorch Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- Ultra-FineWeb: https://huggingface.co/datasets/openbmb/Ultra-FineWeb
- Wikimedia dumps: https://dumps.wikimedia.org/legal.html
- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- COIG: https://huggingface.co/datasets/BAAI/COIG

## Decisions

- Use WSL2 Ubuntu, Python 3.12, uv, PyTorch 2.13, BF16, SDPA, torch.compile where available,
  activation checkpointing, and one CUDA device.
- Use a clean decoder-only architecture: 12 layers, width 768, 12 MHA heads, SwiGLU 1920,
  RoPE, RMSNorm, tied 24,576-token embeddings, no bias, no dropout, context 1024.
- Apply Muon only to hidden Transformer Linear matrices. Use AdamW for embeddings, the tied
  output head, normalization gains, and all other parameters.
- Treat the markdown PRD as scope authority and `prd.json` only as Ralph execution state.
- Keep data, tokenizer artifacts, checkpoints, model weights, and run logs outside Git.
- Ralph implements and validates code but never starts the multi-day production run.
- If Muon fails the defined A/B gate after one bounded LR calibration, stop and diagnose;
  do not silently switch the formal run to AdamW.

## Goals

- Expose one `llm-pretrain` CLI for environment checks, data, tokenizer, training, evaluation,
  SFT, and generation.
- Make all sampling, splits, initialization, packing, and evaluation deterministic from config.
- Support atomic, complete checkpoints and exact-enough continuation of single-GPU runs.
- Provide CPU unit tests and opt-in CUDA smoke tests without network-dependent test cases.
- Publish code, configuration, documentation, and synthetic fixtures under Apache-2.0.

## Non-Goals

- No GQA, sliding-window attention, FP8, custom Triton kernels, multi-GPU training, or
  modded-nanoGPT speedrun features.
- No automatic WSL installation, system driver modification, process termination, or 120-hour
  training launch.
- No guarantee of production chat quality, factual reliability, or strong reasoning.
- No data, trained tokenizer, checkpoint, optimizer state, weight, or full log in Git.

## User Stories

### US-001: Bootstrap repository and environment doctor
**Description:** As an operator, I want safe packaging, configs, Git guards, and an environment
doctor so that unsupported machines fail before training.

**Acceptance Criteria:**
- [x] `llm-pretrain doctor` reports OS/WSL, Python, PyTorch, CUDA, BF16, Muon, SDPA, compile,
      memory, disk, and sequential I/O checks as structured output.
- [x] Strict mode returns non-zero when required training checks fail.
- [x] Apache-2.0, ignore rules, source package, CLI entry point, and synthetic fixture policy exist.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-002: Add validated configuration and 100M model
**Description:** As a researcher, I want a typed model configuration and readable decoder model
so that the experiment architecture is explicit.

**Acceptance Criteria:**
- [x] Config types load, validate, and serialize YAML without accepting unknown fields.
- [x] Default architecture has 99–102M trainable parameters and tied embedding/output weights.
- [x] CPU tests prove shape, causality, RoPE behavior, loss computation, and small-config backward.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-003: Train and evaluate a fixed-ID SentencePiece tokenizer
**Description:** As a researcher, I want a Chinese-first byte-fallback tokenizer so that raw text
maps reproducibly to the model vocabulary.

**Acceptance Criteria:**
- [x] SentencePiece BPE uses 24,576 pieces, byte fallback, and fixed IDs for seven special tokens.
- [x] Training reads only train-split documents and evaluation reports fertility, bytes/token,
      fallback rate, and normalized round-trip results.
- [x] Tests use local tiny fixtures and never download data.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-004: Prepare deterministic packed training data
**Description:** As a researcher, I want pinned source manifests and packed uint32 shards so that
training order and validation separation are reproducible.

**Acceptance Criteria:**
- [x] Source definitions encode 80% Ultra-FineWeb zh, 10% Chinese Wikimedia, 10% FineWeb en,
      with revision, license, and requested token counts.
- [x] Normalized SHA-256 deduplication precedes deterministic document-level train/val splitting.
- [x] EOS packing emits 1024-token uint32 shards, manifests, and resumable batch cursors.
- [x] Tests prove split disjointness, packing, shard reading, and cursor continuation.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-005: Implement Muon/AdamW training and scheduling
**Description:** As a researcher, I want explicit optimizer ownership and a stable accumulated
training step so that Muon can be compared fairly with AdamW.

**Acceptance Criteria:**
- [x] Optimizer partitioning is exhaustive/disjoint and excludes embeddings/head/norm from Muon.
- [x] Muon and AdamW defaults, warmup+cosine scheduling, BF16 accumulation, finite checks,
      clipping, and global-token accounting match the Decisions.
- [x] Missing `torch.optim.Muon` produces an actionable version error.
- [x] CPU focused tests cover partitioning, schedules, accumulation, and failure paths.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-006: Add atomic checkpoints and local monitoring
**Description:** As an operator, I want resumable checkpoints and local metrics so that a long run
can recover from interruption without hidden state loss.

**Acceptance Criteria:**
- [x] Atomic checkpoints include model, all optimizers/schedulers, RunState, sampler cursor, and RNG.
- [x] Retention keeps latest three plus best without deleting unrelated paths.
- [x] JSONL and optional TensorBoard record loss, LR, norms, throughput, memory, temperature,
      evaluation, and checkpoint events.
- [x] CPU resume equivalence and retention tests pass.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-007: Enforce the A/B gate and evaluation workflow
**Description:** As a researcher, I want a deterministic short-run comparison so that the formal
Muon command is emitted only with evidence.

**Acceptance Criteria:**
- [x] A/B runs share initialization, data order, 100M-token budget, global batch, and schedule.
- [x] Gate requires finite metrics, Muon mean validation loss ≤ AdamW + 0.05, and throughput ≥80%.
- [x] A failed first gate permits only 0.01/0.04 Muon LR calibration at 50M tokens before stopping.
- [x] Formal token target is `min(2B, measured_e2e_tokens_per_second * 120h * 0.9)` aligned to batch.
- [x] Tests cover pass, calibration, stop, and budget calculation.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-008: Add COIG SFT and deterministic generation
**Description:** As a learner, I want a small instruction-tuning and generation path so that the
base-to-chat workflow is demonstrable locally.

**Acceptance Criteria:**
- [x] COIG selection deterministically yields up to 50K train and 1K validation examples.
- [x] Chat formatting masks user/system tokens and trains only assistant response tokens.
- [x] SFT defaults to full-parameter AdamW, 5e-5, two epochs, max length 1024.
- [x] Generation supports seed, EOS, temperature, top-k, prompt mode, and interactive mode.
- [x] CPU tests cover selection, masking, evaluation, and deterministic generation.
- [x] `uv run pyright`, `uv run ruff check .`, and focused tests pass.

### US-009: Complete CLI, documentation, and residue guards
**Description:** As an operator, I want documented end-to-end commands and repository audits so
that I can run the lab safely without leaking artifacts.

**Acceptance Criteria:**
- [x] Every planned CLI command has help text, config validation, and actionable errors.
- [x] README documents WSL setup, data licensing caveats, dry run, A/B, manual long run, resume,
      TensorBoard, SFT, generation, and expected limitations.
- [x] Tracked-file audit fails on forbidden directories/extensions, secrets, or oversized files.
- [x] Ralph dry-run and the aggregate validation commands pass.
- [x] `uv run pyright`, `uv run ruff check .`, and all tests pass.

## Functional Requirements

- FR-1: All CLI commands must be non-interactive unless `generate --interactive` is explicit.
- FR-2: Commands that write large artifacts require a configured external artifact root.
- FR-3: Dataset tests must use synthetic local fixtures; real downloads are operator actions.
- FR-4: Validation split membership is decided before tokenization by stable document hash.
- FR-5: Packed training samples never cross train/validation boundaries.
- FR-6: The trainer must not change micro-batch or optimizer after a run starts.
- FR-7: OOM and non-finite values save diagnostic state when safe and exit non-zero.
- FR-8: The full training command requires a passing doctor and A/B gate record unless overridden
  by an explicit diagnostic-only flag that cannot be used for the formal run.

## Validation

- `uv sync --frozen --extra dev`
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `uv run pytest -q -m gpu` on the RTX 4060 Ti after CPU checks pass
- `bash scripts/ralph/ralph.sh --agent codex --dry-run`
- `uv run python scripts/audit_tracked_files.py`

## Risks

- PyTorch 2.13 wheels or official Muon may not yet be available in a selected package index;
  the doctor must report this rather than substituting an unreviewed optimizer.
- Windows desktop processes reduce the usable 8GB VRAM; auto-tuning must leave headroom and the
  operator must close GPU-heavy applications before the formal run.
- WSL-mounted NTFS can bottleneck many small reads; tokenized shards must be large sequential files
  and the doctor must expose measured throughput.
- Web-derived dataset licensing does not grant rights to every source document. The repository
  documents provenance but does not redistribute data or trained weights.
- A 100M model may produce poor subjective chat output despite a technically successful run.
