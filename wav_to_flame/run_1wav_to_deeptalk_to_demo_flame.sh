#!/usr/bin/env bash
set -euo pipefail

# =========================
# User paths: edit these
# =========================
DEEPTALK_REPO="/scratch/e1554543/DEEPTalk_runs/repos/DEEPTalk"
GAUSS_REPO="/scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars"
WAV="/scratch/e1554543/avatar_system_full/EmotiVoice_runs/repo/outputs/prompt_tts_open_source_joint/test_audio/audio/g_00140000/1.wav"
TEMPLATE_NPZ="${GAUSS_REPO}/media/306/flame_param.npz"
OUT_NPZ="${GAUSS_REPO}/media/306/flame_param_from_1wav_deeptalk.npz"

# After running DEEPTalk demo.py, set this to the actual motion file it produced.
# Examples you may see locally:
#   /scratch/.../DEEPTalk/outputs/params/test.npy
#   /scratch/.../DEEPTalk/outputs/params/1.npy
#   /scratch/.../DEEPTalk/outputs/<something>.npz
DEEPTALK_MOTION=""

cd "${DEEPTALK_REPO}/DEEPTalk"
python demo.py --audio_path "${WAV}"

echo
echo "DEEPTalk inference finished."
echo "Now inspect outputs and set DEEPTALK_MOTION to the real file:"
echo "  find ${DEEPTALK_REPO}/DEEPTalk/outputs -type f | sort"
echo

# Uncomment after you confirm the actual output motion path:
# python /scratch/e1554543/avatar_system_full/wav_to_flame/deeptalk_to_demo_flame_param.py \
#   --deeptalk_motion "${DEEPTALK_MOTION}" \
#   --template "${TEMPLATE_NPZ}" \
#   --out "${OUT_NPZ}" \
#   --expr-scale 1.0 \
#   --jaw-scale 1.0 \
#   --rot-scale 0.25 \
#   --neck-scale 0.25 \
#   --eyes-scale 0.25 \
#   --zero-translation
#
# cd "${GAUSS_REPO}"
# python local_viewer.py \
#   --point_path media/306/point_cloud.ply \
#   --motion_path media/306/flame_param_from_1wav_deeptalk.npz
