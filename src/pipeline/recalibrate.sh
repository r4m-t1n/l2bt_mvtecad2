#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
SUBMISSION="${SUBMISSION:-runs/submission_all}"
OBJECTS="${OBJECTS:-can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts}"
PIXEL_RULE="${PIXEL_RULE:-joint}"
IMAGE_RULE="${IMAGE_RULE:-f1_public}"
MIN_PX="${MIN_PX:-32}"
GATE="${GATE:-}"

if [[ ! -d "${SUBMISSION}/anomaly_images" ]]; then
  echo "no ${SUBMISSION}/anomaly_images -- point SUBMISSION at the directory you uploaded" >&2
  exit 1
fi

for OBJ in ${OBJECTS}; do
  OUT="runs/${OBJ}_tiled_l15_23"

  if [[ ! -f "${OUT}/errors.npz" ]]; then
    echo "!! ${OBJ}: no ${OUT}/errors.npz, skipped" >&2
    continue
  fi

  echo ""
  echo "================ ${OBJ}  (${OUT}) ================"
  "${PY}" -m pipeline.evaluate --object "${OBJ}" --out-dir "${OUT}" \
      --image-rule "${IMAGE_RULE}" --pixel-rule "${PIXEL_RULE}" \
      --min-anomalous-px "${MIN_PX}" ${GATE} \
      2>&1 | tee "${OUT}/evaluate_gated.log"

  "${PY}" -m pipeline.restamp --object "${OBJ}" --out-dir "${OUT}" \
      --submission "${SUBMISSION}" \
      2>&1 | tee "${OUT}/restamp.log"
done

echo ""
echo "=================================================="
echo "done. masks rewritten under ${SUBMISSION}/anomaly_images_thresholded"
echo "the .tiff maps were not touched."
echo ""
echo "per-object summary (what the server will read as anomalous):"
grep -h "will read" runs/*/restamp.log 2>/dev/null || true
echo ""
echo "now verify with MVTec's pre-upload checker, wherever you unpacked it:"
echo "  python /path/to/MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py $(readlink -f "${SUBMISSION}")"
