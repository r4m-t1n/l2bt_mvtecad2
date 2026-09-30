#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
MODE="${1:-full}"
OBJECT="${OBJECT:-fabric}"
OUT_DIR="${OUT_DIR:-runs/${OBJECT}_tiled_l15_23}"
TILE="${TILE:-512}"
UPSCALE_ARG=""
[[ -n "${UPSCALE:-}" ]] && UPSCALE_ARG="--upscale $UPSCALE"
STEPS="${STEPS:-8000}"
BATCH="${BATCH:-8}"
TILE_BATCH="${TILE_BATCH:-8}"
SUBMISSION_ARG=""
[[ -n "${SUBMISSION:-}" ]] && SUBMISSION_ARG="--submission ${SUBMISSION}"

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

if [[ "$MODE" == "smoke" ]]; then
  OUT_DIR="${OUT_DIR}_smoke"
  STEPS="${SMOKE_STEPS:-40}"
  LIMIT="--limit 6"
  echo ">>> SMOKE MODE: $STEPS steps, 6 images per split, out=$OUT_DIR"
else
  LIMIT=""
fi

CKPT="$OUT_DIR/checkpoints/last.ckpt"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_all.log"
echo ">>> object=$OBJECT tile=$TILE upscale=${UPSCALE:-from OBJECTS}"
echo ">>> logging to $LOG"

banner() { echo; echo "=============== $* ==============="; }

if [[ "$MODE" != "score" ]]; then
  banner "1/5 train"
  "$PY" -m pipeline.train --object "$OBJECT" --steps "$STEPS" --batch-size "$BATCH" \
      --tile "$TILE" $UPSCALE_ARG --out-dir "$OUT_DIR" $LIMIT 2>&1 | tee -a "$LOG"

  banner "2/5 dump per-patch errors (train/val/test_public)"
  "$PY" -m pipeline.dump --object "$OBJECT" --ckpt "$CKPT" --tile "$TILE" $UPSCALE_ARG \
      --tile-batch "$TILE_BATCH" --out-dir "$OUT_DIR" $LIMIT 2>&1 | tee -a "$LOG"
fi

banner "3/5 sweep score aggregation"
"$PY" -m pipeline.sweep --object "$OBJECT" --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

banner "4/5 evaluate and calibrate thresholds"
"$PY" -m pipeline.evaluate --object "$OBJECT" --out-dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

banner "5/5 predict on private splits"
"$PY" -m pipeline.predict --object "$OBJECT" --ckpt "$CKPT" --tile "$TILE" \
    --tile-batch "$TILE_BATCH" --out-dir "$OUT_DIR" $SUBMISSION_ARG $LIMIT 2>&1 | tee -a "$LOG"

banner "done"
echo "metrics:    $OUT_DIR/results.json"
echo "chosen agg: $OUT_DIR/chosen_config.json"
echo "submission: ${SUBMISSION:-$OUT_DIR/submission}"
