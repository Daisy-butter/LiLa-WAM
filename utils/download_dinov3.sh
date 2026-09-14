#!/usr/bin/env bash
# Download DINOv3 ViT-L/16 to SSD for LiLa-WAM.
#
# Prerequisites (gated repo):
#   1. Accept the model license at:
#      https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#   2. Create a HF token with access to public gated repos, then either:
#        export HF_TOKEN=hf_xxx
#      or:  huggingface-cli login
#
# Usage:
#   export HF_ENDPOINT=https://hf-mirror.com
#   export HF_TOKEN=hf_xxx   # required for gated DINOv3
#   bash utils/download_dinov3.sh

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/SSD_DISK/users/wuruihan/hf_cache}"

DEST="${1:-/SSD_DISK/users/wuruihan/models/dinov3-vitl16-pretrain-lvd1689m}"
REPO_ID="facebook/dinov3-vitl16-pretrain-lvd1689m"
PY="${PYTHON:-/SSD_DISK/users/wuruihan/conda_envs/lila/bin/python}"

mkdir -p "$HF_HOME" "$DEST"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "ERROR: DINOv3 is a gated model. Set HF_TOKEN after accepting the license on HuggingFace."
  echo "  export HF_ENDPOINT=https://hf-mirror.com"
  echo "  export HF_TOKEN=hf_xxx"
  echo "  bash utils/download_dinov3.sh"
  exit 1
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "DEST=$DEST"

DEST="$DEST" REPO_ID="$REPO_ID" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

dest = os.environ["DEST"]
path = snapshot_download(
    repo_id=os.environ["REPO_ID"],
    local_dir=dest,
)
print("OK downloaded to", path)
PY

# Convenience symlink into the repo-expected relative path
REPO_LINK="/home/wuruihan/LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m"
mkdir -p /home/wuruihan/LiLa-WAM/dinov3_pretrain
if [[ -L "$REPO_LINK" || ! -e "$REPO_LINK" ]]; then
  ln -sfn "$DEST" "$REPO_LINK"
  echo "Linked $REPO_LINK -> $DEST"
fi

ls -lh "$DEST" | head -20
echo "Done."
