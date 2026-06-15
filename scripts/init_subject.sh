#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subject_id>"
  exit 2
fi

SUBJECT_ID="$1"
SUBJECT_ROOT="$ROOT/data/subjects/$SUBJECT_ID"

mkdir -p \
  "$SUBJECT_ROOT/raw" \
  "$SUBJECT_ROOT/vhap" \
  "$SUBJECT_ROOT/gaussian_source" \
  "$SUBJECT_ROOT/gaussian_train" \
  "$SUBJECT_ROOT/final_asset"

cat <<EOF
[init_subject] created:
  $SUBJECT_ROOT

Next:
  1. Put raw videos/images into:
     $SUBJECT_ROOT/raw
  2. Run VHAP tracking:
     bash $ROOT/scripts/run_vhap_subject.sh $SUBJECT_ID --mode monocular --input \$RAW_VIDEO
  3. Export VHAP result:
     bash $ROOT/scripts/export_vhap_to_gaussian.sh $SUBJECT_ID
  4. Train Gaussian avatar:
     bash $ROOT/scripts/train_gaussian_subject.sh $SUBJECT_ID
  5. Build/register final asset:
     bash $ROOT/scripts/register_avatar_asset.sh $SUBJECT_ID <new_avatar_id>
EOF
