#!/usr/bin/env bash
# Download the SAM 2.1 checkpoints into models/ (CLIP downloads itself on first run).
set -e
cd "$(dirname "$0")"
mkdir -p models
for m in tiny base_plus; do
  f="models/sam2.1_hiera_$m.pt"
  [ -f "$f" ] || curl -L -o "$f" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$m.pt"
done
echo "models ready"
