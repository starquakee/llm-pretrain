#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="llm-pretrain-pipeline"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_training_pipeline.sh --background ARTIFACT_ROOT
  scripts/run_training_pipeline.sh --foreground ARTIFACT_ROOT
  scripts/run_training_pipeline.sh --status ARTIFACT_ROOT

Runs data preparation, tokenizer training/evaluation, token packing, the Muon/AdamW
A/B gate, formal pretraining, evaluation, and COIG SFT. A failed stage stops the
pipeline; formal pretraining cannot start unless the A/B command writes a passing gate.
EOF
}

[[ $# -eq 2 ]] || { usage >&2; exit 2; }
MODE="$1"
ARTIFACT_ROOT="$(realpath -m "$2")"
LOG_DIR="$ARTIFACT_ROOT/logs"
STATE_DIR="$ARTIFACT_ROOT/pipeline-state"
SUPERVISOR_LOG="$LOG_DIR/pipeline.log"

show_status() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "session: running ($SESSION_NAME)"
  else
    echo "session: not running"
  fi
  if [[ -d "$STATE_DIR" ]]; then
    find "$STATE_DIR" -maxdepth 1 -type f -name '*.done' -printf 'complete: %f\n' | sort
  fi
  if [[ -f "$SUPERVISOR_LOG" ]]; then
    echo "--- pipeline.log (tail) ---"
    tail -40 "$SUPERVISOR_LOG"
  fi
}

case "$MODE" in
  --status)
    show_status
    exit 0
    ;;
  --background)
    mkdir -p "$LOG_DIR" "$STATE_DIR"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Pipeline is already running in tmux session $SESSION_NAME" >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" \
      "$(printf '%q ' "$SCRIPT_PATH" --foreground "$ARTIFACT_ROOT")"
    echo "Started tmux session: $SESSION_NAME"
    echo "Status: $SCRIPT_PATH --status $ARTIFACT_ROOT"
    exit 0
    ;;
  --foreground) ;;
  *) usage >&2; exit 2 ;;
esac

mkdir -p \
  "$LOG_DIR" \
  "$STATE_DIR" \
  "$ARTIFACT_ROOT/cache/huggingface" \
  "$ARTIFACT_ROOT/cache/xdg" \
  "$ARTIFACT_ROOT/cache/torchinductor"

exec 9>"$STATE_DIR/pipeline.lock"
flock -n 9 || { echo "Another pipeline process holds $STATE_DIR/pipeline.lock" >&2; exit 1; }

PYTHON="$PROJECT_ROOT/.venv-wsl/bin/python"
CLI="$PROJECT_ROOT/.venv-wsl/bin/llm-pretrain"
[[ -x "$PYTHON" && -x "$CLI" ]] || {
  echo "Missing WSL environment at $PROJECT_ROOT/.venv-wsl; run uv sync first" >&2
  exit 1
}

export LLM_PRETRAIN_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$ARTIFACT_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ARTIFACT_ROOT/cache/huggingface/datasets"
export XDG_CACHE_HOME="$ARTIFACT_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$ARTIFACT_ROOT/cache/torchinductor"
export PYTHONUNBUFFERED=1

CURRENT_STAGE="initialization"
stage_failed() {
  status=$?
  printf '%s stage=%s status=failed exit=%d\n' "$(date --iso-8601=seconds)" "$CURRENT_STAGE" "$status" \
    | tee -a "$SUPERVISOR_LOG" >&2
  exit "$status"
}
trap stage_failed ERR

run_stage() {
  local name="$1"
  shift
  local marker="$STATE_DIR/$name.done"
  local stage_log="$LOG_DIR/$name.log"
  CURRENT_STAGE="$name"
  if [[ -f "$marker" ]]; then
    printf '%s stage=%s status=skipped reason=complete\n' "$(date --iso-8601=seconds)" "$name" \
      | tee -a "$SUPERVISOR_LOG"
    return
  fi
  printf '%s stage=%s status=started\n' "$(date --iso-8601=seconds)" "$name" \
    | tee -a "$SUPERVISOR_LOG"
  "$@" 2>&1 | tee -a "$stage_log"
  touch "$marker"
  printf '%s stage=%s status=complete\n' "$(date --iso-8601=seconds)" "$name" \
    | tee -a "$SUPERVISOR_LOG"
}

cd "$PROJECT_ROOT"

run_stage doctor "$CLI" doctor --strict \
  --data-dir "$ARTIFACT_ROOT/data" \
  --min-free-disk-gb 100 \
  --min-free-ram-gb 4 \
  --min-free-vram-gb 0.7 \
  --io-test-mib 64

run_stage data-prepare "$CLI" data prepare \
  --config "$PROJECT_ROOT/configs/data_sources.yaml" \
  --allow-network

run_stage tokenizer-train "$CLI" tokenizer train \
  --config "$PROJECT_ROOT/configs/tokenizer_24k.yaml"

run_stage tokenizer-eval "$CLI" tokenizer eval \
  --config "$PROJECT_ROOT/configs/tokenizer_24k.yaml" \
  --output "$ARTIFACT_ROOT/data/tokenizer/metrics.json"

run_stage data-tokenize "$CLI" data tokenize \
  --config "$PROJECT_ROOT/configs/data_sources.yaml"

GATE_RECORD="$ARTIFACT_ROOT/runs/pretrain-muon/ab/gate.json"
run_stage train-ab "$CLI" train ab \
  --config "$PROJECT_ROOT/configs/pretrain_100m_muon.yaml" \
  --gate-record "$GATE_RECORD"

run_stage train-pretrain "$CLI" train pretrain \
  --config "$PROJECT_ROOT/configs/pretrain_100m_muon.yaml" \
  --gate-record "$GATE_RECORD"

run_stage evaluate "$CLI" evaluate \
  --config "$PROJECT_ROOT/configs/pretrain_100m_muon.yaml" \
  --output "$ARTIFACT_ROOT/runs/pretrain-muon/evaluation.json"

run_stage train-sft "$CLI" train sft \
  --config "$PROJECT_ROOT/configs/sft_coig.yaml" \
  --allow-network

CURRENT_STAGE="complete"
printf '%s pipeline=complete\n' "$(date --iso-8601=seconds)" | tee -a "$SUPERVISOR_LOG"
