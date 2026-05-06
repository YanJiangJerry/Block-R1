#!/bin/bash
set -euo pipefail

DESIRED_MODEL="JetLM/SDAR-8B-Chat-b32"
# Escape characters that are special in sed replacement (/, &)
ESCAPED_MODEL=$(printf '%s' "$DESIRED_MODEL" | sed -e 's/[&\/]/\\&/g')
# Safe tag for filenames (no slashes/spaces)
SAFE_MODEL_TAG=$(printf '%s' "$DESIRED_MODEL" | tr '/ ' '--' | sed -E 's/[^A-Za-z0-9_.-]/-/g')
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_ALL="$ROOT_DIR/run_benchmark.sh"

# Extract default script list from run_benchmark.sh if no args are provided
get_default_scripts() {
  awk '/^SCRIPTS=\(/,/^\)/ {print}' "$START_ALL" \
    | grep '^[[:space:]]*"' \
    | grep -v '^[[:space:]]*#' \
    | sed -E 's/^[[:space:]]*"(.*)"[[:space:]]*,?[[:space:]]*$/\1/'
}

# Build list of scripts to run
SCRIPTS=()
if [ "$#" -gt 0 ]; then
  SCRIPTS=("$@")
else
  mapfile -t SCRIPTS < <(get_default_scripts)
fi

if [ ${#SCRIPTS[@]} -eq 0 ]; then
  echo "[ERROR] No scripts to run. Provide script paths as arguments or activate some in run_all.sh."
  exit 1
fi

temp_files=()
FAILED=()

for s in "${SCRIPTS[@]}"; do
  # Resolve to absolute path if relative
  if [[ "$s" != /* ]]; then
    s_abs="$ROOT_DIR/$s"
  else
    s_abs="$s"
  fi
  if [ ! -f "$s_abs" ]; then
    echo "ERROR: script not found: $s_abs"
    FAILED+=("$s (not found)")
    continue
  fi

  tmp_file=$(mktemp "/tmp/$(basename "$s_abs").XXXXXX.sh")
  cp "$s_abs" "$tmp_file"
  chmod +x "$tmp_file"

  # Replace active MODEL_NAME assignment lines (not commented), support optional leading 'export '
  # If no replacement occurred, we'll still export env var when running as fallback
  if sed -i -E "s/^[[:space:]]*(export[[:space:]]*)?MODEL_NAME=.*/MODEL_NAME=$ESCAPED_MODEL/" "$tmp_file"; then
    :
  fi

  # The training scripts now use SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-')
  # and output_dir=checkpoints/${RUN_NAME}/${SAFE_MODEL_NAME}, so no extra patching needed.

  echo
  echo "----------------------------------------------------------------"
  echo "Running: $s (override MODEL_NAME=$DESIRED_MODEL)"

  # Ensure logs directory exists if the script uses ./logs/%A.*
  mkdir -p "$ROOT_DIR/logs" 2>/dev/null || true

  # Run with env override as double-safety
  if MODEL_NAME="$DESIRED_MODEL" bash "$tmp_file"; then
    echo "Finished: $s"
  else
    rc=$?
    echo "Script $s exited with code $rc. Continuing to next script."
    FAILED+=("$s (exit $rc)")
  fi

  temp_files+=("$tmp_file")

done

# Cleanup temp files
for t in "${temp_files[@]}"; do
  rm -f "$t" 2>/dev/null || true

done

echo
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "The following scripts failed or were missing:"
  for f in "${FAILED[@]}"; do
    echo " - $f"
  done
  echo "Completed with errors. See above list."
  exit 1
else
  echo "All scripts finished successfully."
fi
