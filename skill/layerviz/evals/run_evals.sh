#!/usr/bin/env bash
# Run each layerviz skill scenario on Sonnet and Opus with claude -p and save the transcripts.
#
# Usage: skill/layerviz/evals/run_evals.sh [scenario ...]
# Scenarios default to cnn_2d, spconv_remote, and training_snapshots.
# Each run gets a fresh temp project holding only the skill and the scenario's Setup files,
# so no CLAUDE.md from this repo leaks in. Transcripts and the final project land in
# evals/runs/<stamp>/<scenario>-<model>/. Grade each against its Expected outcome checklist.
# After each run, any layerviz serve process left in the temp project is stopped and the
# temp project is removed. The copy under evals/runs/ stays.
#
# Environment:
#   LAYERVIZ_REPO  layerviz checkout named in the prompts, default the repo holding this skill.
#   PYTHON         interpreter with torch whose folder goes first on PATH, default the repo venv.
#   CLAUDE_FLAGS   extra claude flags split on spaces, default accept edits and allow
#                  python, pip, and ls.
set -euo pipefail

EVALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$EVALS_DIR")"
LAYERVIZ_REPO="${LAYERVIZ_REPO:-$(cd "$SKILL_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-$LAYERVIZ_REPO/.venv/bin/python}"
CLAUDE_FLAGS="${CLAUDE_FLAGS:---permission-mode acceptEdits --allowedTools Bash(python:*) Bash(pip:*) Bash(ls:*)}"
read -r -a EXTRA_FLAGS <<< "$CLAUDE_FLAGS"
MODELS=(sonnet opus)
if [ "$#" -gt 0 ]; then
  SCENARIOS=("$@")
else
  SCENARIOS=(cnn_2d spconv_remote training_snapshots)
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
export PATH="$(dirname "$PYTHON"):$PATH"

# Print the lines under the heading "## $2" of file $1, up to the next level two heading.
section() {
  awk -v head="## $2" '$0 == head {on = 1; next} /^## / {on = 0} on' "$1"
}

# Create a temp project with the skill installed and the scenario's Setup files, print its path.
make_project() {
  local scenario_file="$1" work
  work="$(mktemp -d)"
  mkdir -p "$work/.claude/skills/layerviz"
  cp "$SKILL_DIR"/SKILL.md "$SKILL_DIR"/*.py "$work/.claude/skills/layerviz/"
  section "$scenario_file" Setup | sed '/^```/d' > "$work/.setup.sh"
  (cd "$work" && bash -e .setup.sh >&2)
  rm "$work/.setup.sh"
  printf '%s\n' "$work"
}

# Stop layerviz serve processes the run left behind in temp project $1, and only those.
stop_servers() {
  local work="$1" pid
  pkill -f "layerviz serve .*$work" || true
  # Catch servers started with a relative bundle path by matching their working folder.
  for pid in $(lsof -t -a -d cwd +D "$work" 2>/dev/null || true); do
    if ps -o command= -p "$pid" | grep -q "layerviz serve"; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

# Stop the servers of temp project $1 and delete it. An empty path is never removed.
remove_project() {
  local work="$1"
  if [ -z "$work" ] || [ ! -d "$work" ]; then
    return 0
  fi
  stop_servers "$work"
  rm -rf -- "$work"
}

# Remove the temp project of a run that an error or Ctrl-C interrupts.
CURRENT_WORK=""
trap 'remove_project "$CURRENT_WORK"' EXIT

# Run one scenario on one model and save its transcript and final project.
run_one() {
  local scenario="$1" model="$2" file run work prompt
  file="$EVALS_DIR/$scenario.md"
  run="$EVALS_DIR/runs/$STAMP/$scenario-$model"
  mkdir -p "$run"
  work="$(make_project "$file")"
  CURRENT_WORK="$work"
  prompt="$(section "$file" Prompt | sed "s|{{LAYERVIZ_REPO}}|$LAYERVIZ_REPO|g")"
  printf '%s\n' "$prompt" > "$run/prompt.md"
  echo "running $scenario on $model in $work"
  (cd "$work" && claude -p "$prompt" --model "$model" --output-format stream-json --verbose \
    "${EXTRA_FLAGS[@]}") > "$run/transcript.jsonl" || echo "claude exited non zero for $scenario on $model"
  cp -R "$work" "$run/project"
  remove_project "$work"
  CURRENT_WORK=""
}

for scenario in "${SCENARIOS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$scenario" "$model"
  done
done
echo "transcripts in $EVALS_DIR/runs/$STAMP"
