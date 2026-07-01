#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AVATAR_SYSTEM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

REPO="${AVATAR_RUNTIME_REPO:-1114531938/avatar-system-full}"
TAG="${AVATAR_RUNTIME_RELEASE_TAG:-runtime-assets-2026-07-01}"
PREFIX="${AVATAR_RUNTIME_ASSET_PREFIX:-avatar-system-full-runtime-assets}"
OUT_DIR="${AVATAR_RUNTIME_OUT_DIR:-$ROOT/runtime/release_assets/$TAG}"
SPLIT_SIZE="${AVATAR_RUNTIME_SPLIT_SIZE:-500M}"
ZSTD_LEVEL="${AVATAR_RUNTIME_ZSTD_LEVEL:-3}"
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
UPLOAD="${AVATAR_RUNTIME_UPLOAD:-auto}"
DRY_RUN="${AVATAR_RUNTIME_DRY_RUN:-0}"
SKIP_DU="${AVATAR_RUNTIME_SKIP_DU:-0}"

NORMAL_ASSET_PATHS=(
  "runtime/containers/gaussianav_jammy"
  "runtime/cache/bin/ffmpeg"
  "runtime/cache/bin/ffprobe"
  "runtime/cache/bin/ffmpeg.container-bin"
  "runtime/cache/bin/ffprobe.container-bin"
  "runtime/cache/xdg/whisper/small.pt"
  "runtime/cache/modelscope/models/iic/emotion2vec_plus_seed"
  "integrations/deeptalk/DEE/models/emo2vec/checkpoint/emotion2vec_base.pt"
  "integrations/avamerg/ckpt"
  "integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/g_00140000"
  "integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/do_00140000"
  "integrations/emotivoice/WangZeJun/simbert-base-chinese"
  "integrations/gaussian_avatar/media"
  "integrations/gaussian_avatar/flame_model/assets"
  "integrations/vhap/asset"
)

DEEPTALK_LINK_PATHS=(
  "integrations/deeptalk/DEE/checkpoint/DEE.pt"
  "integrations/deeptalk/DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth"
  "integrations/deeptalk/DEEPTalk/checkpoint/TH-VQVAE/TH-VQVAE.pth"
  "integrations/deeptalk/FER/checkpoint/FER.pth"
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd jq
need_cmd sha256sum
need_cmd split
need_cmd tar
need_cmd zstd

missing=0
for rel in "${NORMAL_ASSET_PATHS[@]}" "${DEEPTALK_LINK_PATHS[@]}"; do
  if [[ ! -e "$rel" ]]; then
    echo "Missing required asset: $rel" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/runtime_assets_manifest.txt"
SHA_FILE="$OUT_DIR/sha256sums.txt"
PART_PREFIX="$OUT_DIR/$PREFIX.tar.zst.part-"
STAGING_DIR="$OUT_DIR/staging"
trap 'rm -rf "$STAGING_DIR"' EXIT

{
  echo "repo=$REPO"
  echo "tag=$TAG"
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT"
  echo
  echo "[normal_paths]"
  printf '%s\n' "${NORMAL_ASSET_PATHS[@]}"
  echo
  echo "[deeptalk_symlinks_archived_as_real_files]"
  for rel in "${DEEPTALK_LINK_PATHS[@]}"; do
    printf '%s -> %s\n' "$rel" "$(readlink -f "$rel")"
  done
} >"$MANIFEST"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  echo "Dry run OK. Manifest written to $MANIFEST"
  exit 0
fi

if [[ -z "$TOKEN" && ( "$UPLOAD" == "1" || "$UPLOAD" == "true" || "$UPLOAD" == "yes" ) ]]; then
  echo "GH_TOKEN or GITHUB_TOKEN is required when AVATAR_RUNTIME_UPLOAD=$UPLOAD." >&2
  exit 1
fi

if [[ "$SKIP_DU" == "1" || "$SKIP_DU" == "true" || "$SKIP_DU" == "yes" ]]; then
  {
    echo
    echo "[sizes]"
    echo "skipped by AVATAR_RUNTIME_SKIP_DU=$SKIP_DU"
  } >>"$MANIFEST"
else
  {
    echo
    echo "[sizes]"
    du -sh "${NORMAL_ASSET_PATHS[@]}" "${DEEPTALK_LINK_PATHS[@]}" 2>/dev/null || true
  } >>"$MANIFEST"
fi

rm -f "$PART_PREFIX"* "$SHA_FILE"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

for rel in "${DEEPTALK_LINK_PATHS[@]}"; do
  mkdir -p "$STAGING_DIR/$(dirname "$rel")"
  cp -L "$rel" "$STAGING_DIR/$rel"
done

echo "Packing runtime assets to $OUT_DIR"
echo "Split size: $SPLIT_SIZE; zstd level: $ZSTD_LEVEL"
tar -cpf - \
  -C "$ROOT" "${NORMAL_ASSET_PATHS[@]}" \
  -C "$STAGING_DIR" "${DEEPTALK_LINK_PATHS[@]}" \
  | zstd -T0 "-$ZSTD_LEVEL" \
  | split -b "$SPLIT_SIZE" - "$PART_PREFIX"

(
  cd "$OUT_DIR"
  sha256sum "$PREFIX.tar.zst.part-"* runtime_assets_manifest.txt > sha256sums.txt
)

echo "Created assets:"
ls -lh "$OUT_DIR"/"$PREFIX.tar.zst.part-"* "$MANIFEST" "$SHA_FILE"

if [[ "$UPLOAD" == "0" || "$UPLOAD" == "false" || "$UPLOAD" == "no" ]]; then
  echo "Upload disabled by AVATAR_RUNTIME_UPLOAD=$UPLOAD"
  exit 0
fi

if [[ -z "$TOKEN" ]]; then
  echo "No GH_TOKEN/GITHUB_TOKEN found; local release files are ready in $OUT_DIR"
  echo "Set GH_TOKEN and rerun with AVATAR_RUNTIME_UPLOAD=1 to publish them."
  exit 0
fi

api="https://api.github.com/repos/$REPO/releases"
auth=(-H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28")

release_json="$(curl -fsS "${auth[@]}" "$api/tags/$TAG" || true)"
if [[ -z "$release_json" || "$(jq -r '.id // empty' <<<"$release_json")" == "" ]]; then
  body="Runtime assets for avatar_system_full. Download with scripts/download_runtime_assets.sh. Generated from $ROOT."
  release_json="$(jq -n \
    --arg tag "$TAG" \
    --arg name "$TAG" \
    --arg body "$body" \
    '{tag_name:$tag, name:$name, body:$body, draft:false, prerelease:false}')"
  release_json="$(curl -fsS "${auth[@]}" -X POST "$api" -d "$release_json")"
fi

upload_url="$(jq -r '.upload_url' <<<"$release_json" | sed 's/{?name,label}//')"
release_id="$(jq -r '.id' <<<"$release_json")"
if [[ -z "$upload_url" || "$upload_url" == "null" ]]; then
  echo "Could not determine GitHub release upload URL." >&2
  exit 1
fi

delete_existing_asset() {
  local name="$1"
  local asset_id
  asset_id="$(curl -fsS "${auth[@]}" "$api/$release_id/assets?per_page=100" \
    | jq -r --arg name "$name" '.[] | select(.name == $name) | .id' \
    | head -n 1)"
  if [[ -n "$asset_id" ]]; then
    echo "Deleting existing release asset $name"
    curl -fsS "${auth[@]}" -X DELETE "$api/assets/$asset_id" >/dev/null
  fi
}

upload_one() {
  local file="$1"
  local name
  name="$(basename "$file")"
  delete_existing_asset "$name"
  echo "Uploading $name"
  curl -fsS "${auth[@]}" \
    -H "Content-Type: application/octet-stream" \
    -X POST \
    --upload-file "$file" \
    "$upload_url?name=$name" >/dev/null
}

upload_one "$MANIFEST"
upload_one "$SHA_FILE"
for file in "$OUT_DIR"/"$PREFIX.tar.zst.part-"*; do
  upload_one "$file"
done

echo "Published runtime assets to https://github.com/$REPO/releases/tag/$TAG"
