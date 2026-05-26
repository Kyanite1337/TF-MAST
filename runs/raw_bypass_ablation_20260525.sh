#!/usr/bin/env bash
set -uo pipefail

cd /home/coder/workspace/TF-MAST

DATE_TAG="20260525"
MAE_CKPT="runs/20260522-041335_mae_db5_full_window300_mae/best.pt"
TFC_CKPT="runs/20260523-114358_tfc_db5_full_window300_tfc_chunk2/best.pt"
PREFIX="runs/raw_bypass_ablation_${DATE_TAG}"
MANIFEST="${PREFIX}_manifest.csv"
RESULT_CSV="${PREFIX}_results.csv"
RESULT_XLSX="${PREFIX}_results.xlsx"
EVENTS="${PREFIX}_events.jsonl"
FAILURES="${PREFIX}_failures.log"
LOG_DIR="${PREFIX}_logs"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "$LOG_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  echo "experiment,phase,pretrain,head,bypass,init,extra_overrides" > "$MANIFEST"
fi

event() {
  local status="$1"
  local name="$2"
  local phase="$3"
  local message="${4:-}"
  printf '{"time":"%s","status":"%s","experiment":"%s","phase":"%s","message":"%s"}\n' \
    "$(date -Is)" "$status" "$name" "$phase" "$message" >> "$EVENTS"
}

is_done() {
  local name="$1"
  [[ -f "$RESULT_CSV" ]] && grep -q "^${name},done," "$RESULT_CSV"
}

run_ft() {
  local phase="$1"
  local name="$2"
  local pretrain="$3"
  local head="$4"
  local bypass="$5"
  local init="$6"
  shift 6
  local extra_overrides=("$@")
  local log_file="${LOG_DIR}/${name}.log"

  if is_done "$name"; then
    echo "[$(date -Is)] skip completed $name"
    event "skipped" "$name" "$phase" "already completed"
    return 0
  fi

  echo "$name,$phase,$pretrain,$head,$bypass,$init,${extra_overrides[*]}" >> "$MANIFEST"
  echo "[$(date -Is)] start $name pretrain=$pretrain head=$head bypass=$bypass init=$init"
  event "start" "$name" "$phase" "log=${log_file}"

  local started_at
  local started_sec
  started_at="$(date -Is)"
  started_sec="$(date +%s)"

  local cmd=(
    python -m tfmast.train
    --config configs/db5.yaml
    stage=finetune
    head="$head"
    experiment="$name"
    data.window_ms=300
    head.bypass="$bypass"
    train.num_workers=0
    wandb.mode=disabled
  )
  if [[ "$init" != "none" ]]; then
    cmd+=(init="$init")
  fi
  cmd+=("${extra_overrides[@]}")

  "${cmd[@]}" > "$log_file" 2>&1
  local exit_code=$?
  local finished_at
  local finished_sec
  finished_at="$(date -Is)"
  finished_sec="$(date +%s)"
  local duration=$((finished_sec - started_sec))

  if [[ "$exit_code" -ne 0 ]]; then
    echo "[$(date -Is)] failed $name exit_code=$exit_code log=$log_file"
    printf '[%s] failed %s exit_code=%s log=%s\n' "$(date -Is)" "$name" "$exit_code" "$log_file" >> "$FAILURES"
    tail -80 "$log_file" >> "$FAILURES"
    printf '\n' >> "$FAILURES"
    event "failed" "$name" "$phase" "exit_code=${exit_code};log=${log_file}"
    return "$exit_code"
  fi

  local run_dir
  run_dir="$(ls -td "runs/"*"_finetune_${name}" 2>/dev/null | head -1)"
  if [[ -z "$run_dir" ]]; then
    echo "[$(date -Is)] failed $name no run_dir found"
    printf '[%s] failed %s no run_dir found log=%s\n' "$(date -Is)" "$name" "$log_file" >> "$FAILURES"
    event "failed" "$name" "$phase" "no run_dir found"
    return 1
  fi

  python runs/append_raw_bypass_ablation_result.py \
    --run-dir "$run_dir" \
    --output "$RESULT_CSV" \
    --xlsx "$RESULT_XLSX" \
    --experiment "$name" \
    --pretrain "$pretrain" \
    --head "$head" \
    --bypass "$bypass" \
    --init-checkpoint "$init" \
    --status done \
    --started-at "$started_at" \
    --finished-at "$finished_at" \
    --duration-sec "$duration" \
    --git-commit "$GIT_COMMIT" \
    --error-log "$log_file"
  local append_exit=$?
  if [[ "$append_exit" -ne 0 ]]; then
    echo "[$(date -Is)] failed to append result for $name"
    printf '[%s] append failed %s exit_code=%s log=%s\n' "$(date -Is)" "$name" "$append_exit" "$log_file" >> "$FAILURES"
    event "failed" "$name" "$phase" "append_exit=${append_exit}"
    return "$append_exit"
  fi

  echo "[$(date -Is)] done $name duration=${duration}s"
  event "done" "$name" "$phase" "duration_sec=${duration};run_dir=${run_dir}"
  return 0
}

echo "[$(date -Is)] raw bypass ablation queue start git=$GIT_COMMIT"
event "queue_start" "raw_bypass_ablation_${DATE_TAG}" "queue" "git=${GIT_COMMIT}"

run_ft smoke rawb_smoke_mae_tfc_bimamba_bypass_true mae_tfc bimamba true "$TFC_CKPT" \
  train.finetune.epochs=1 train.max_batches=2 train.finetune.batch_size=64

for pretrain in none mae mae_tfc; do
  case "$pretrain" in
    none) init="none" ;;
    mae) init="$MAE_CKPT" ;;
    mae_tfc) init="$TFC_CKPT" ;;
    *) echo "unknown pretrain=$pretrain" >&2; exit 1 ;;
  esac

  for head in mlp mamba bimamba; do
    for bypass in true false; do
      run_ft phase1 "rawb_w300_${pretrain}_${head}_bypass_${bypass}" "$pretrain" "$head" "$bypass" "$init"
    done
  done
done

event "queue_done" "raw_bypass_ablation_${DATE_TAG}" "queue" "complete"
echo "[$(date -Is)] raw bypass ablation queue complete"
