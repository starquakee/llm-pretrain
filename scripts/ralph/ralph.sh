#!/usr/bin/env bash
# Safe Ralph runner for this repository. Unsafe execution is intentionally unsupported.
set -euo pipefail

AGENT="${RALPH_AGENT:-codex}"
AGENT_COMMAND="${RALPH_AGENT_COMMAND:-}"
MAX_ITERATIONS=10
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/ralph/ralph.sh [--agent codex|claude|amp|kimi] [--agent-command PATH] [--dry-run] [iterations]

This repository intentionally rejects --unsafe. A custom adapter receives scripts/ralph/AGENTS.md
as its first argument and must perform exactly one non-interactive agent turn.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|--tool) AGENT="${2:?missing agent}"; shift 2 ;;
    --agent=*|--tool=*) AGENT="${1#*=}"; shift ;;
    --agent-command) AGENT_COMMAND="${2:?missing adapter}"; shift 2 ;;
    --agent-command=*) AGENT_COMMAND="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --unsafe) echo "Error: --unsafe is forbidden by the project PRD." >&2; exit 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      [[ "$1" =~ ^[0-9]+$ ]] || { echo "Unknown argument: $1" >&2; usage >&2; exit 2; }
      MAX_ITERATIONS="$1"; shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$REPO_ROOT/prd.json"
PROGRESS_FILE="$REPO_ROOT/progress.txt"
ARCHIVE_DIR="$REPO_ROOT/archive"
PROMPT_FILE="$SCRIPT_DIR/AGENTS.md"

json_reader() {
  if command -v jq >/dev/null 2>&1; then echo jq
  elif command -v node >/dev/null 2>&1; then echo node
  elif command -v python3 >/dev/null 2>&1; then echo python3
  else return 1
  fi
}

read_branch() {
  case "$(json_reader)" in
    jq) jq -r '.branchName // empty' "$PRD_FILE" ;;
    node) node -e "const d=require(process.argv[1]); console.log(d.branchName || '')" "$PRD_FILE" ;;
    python3) python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('branchName',''))" "$PRD_FILE" ;;
  esac
}

all_pass() {
  case "$(json_reader)" in
    jq) jq -e '(.userStories|length)>0 and all(.userStories[];.passes==true)' "$PRD_FILE" >/dev/null ;;
    node) node -e "const d=require(process.argv[1]); process.exit(d.userStories?.length && d.userStories.every(x=>x.passes===true)?0:1)" "$PRD_FILE" ;;
    python3) python3 -c "import json,sys; s=json.load(open(sys.argv[1])).get('userStories',[]); raise SystemExit(0 if s and all(x.get('passes') is True for x in s) else 1)" "$PRD_FILE" ;;
  esac
}

validate_agent() {
  if [[ -n "$AGENT_COMMAND" ]]; then command -v "$AGENT_COMMAND" >/dev/null
  else
    case "$AGENT" in codex|claude|amp|kimi) command -v "$AGENT" >/dev/null ;;
      *) echo "Unknown agent: $AGENT" >&2; return 1 ;;
    esac
  fi
}

run_agent() {
  if [[ -n "$AGENT_COMMAND" ]]; then "$AGENT_COMMAND" "$PROMPT_FILE"; return; fi
  case "$AGENT" in
    codex) codex --sandbox workspace-write --ask-for-approval never exec - < "$PROMPT_FILE" ;;
    claude) claude --permission-mode acceptEdits --print < "$PROMPT_FILE" ;;
    amp) amp < "$PROMPT_FILE" ;;
    kimi) kimi -p "$(<"$PROMPT_FILE")" ;;
  esac
}

cd "$REPO_ROOT"
[[ -f "$PRD_FILE" ]] || { echo "Missing prd.json" >&2; exit 1; }
[[ -f "$PROGRESS_FILE" ]] || { echo "Missing progress.txt" >&2; exit 1; }
[[ -d "$ARCHIVE_DIR" ]] || { echo "Missing archive/" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "Missing AGENTS.md" >&2; exit 1; }
json_reader >/dev/null || { echo "jq, node, or python3 is required" >&2; exit 1; }
BRANCH="$(read_branch)"
[[ -n "$BRANCH" ]] || { echo "prd.json branchName is missing" >&2; exit 1; }
validate_agent || { echo "Agent '$AGENT' is unavailable" >&2; exit 1; }

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Ralph dry run OK\n  root: %s\n  branch: %s\n  agent: %s\n' "$REPO_ROOT" "$BRANCH" "$AGENT"
  exit 0
fi

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  echo "Ralph iteration $i/$MAX_ITERATIONS ($AGENT)"
  LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/ralph-output.XXXXXX")"
  set +e
  run_agent 2>&1 | tee "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
  set -e
  OUTPUT="$(<"$LOG_FILE")"
  rm -f -- "$LOG_FILE"
  [[ "$STATUS" -eq 0 ]] || { echo "Agent failed with status $STATUS" >&2; exit "$STATUS"; }
  if grep -qE '^[[:space:]]*<promise>COMPLETE</promise>[[:space:]]*$' <<<"$OUTPUT"; then
    all_pass && { echo "All Ralph stories complete"; exit 0; }
    echo "Ignoring premature completion token" >&2
  fi
done

echo "Ralph reached the iteration limit" >&2
exit 1

