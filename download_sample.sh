#!/usr/bin/env bash
# ==============================================================================
# Script: download_sample.sh
# Description: Download and extract sample dataset images using uvx gdown.
#
# Usage:
#   ./download_sample.sh
#   ./download_sample.sh <GDRIVE_URL>
#   KEEP_ZIP=true ./download_sample.sh
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")"

GDRIVE_URL="${1:-https://drive.google.com/file/d/18EMET1QDfQgVrJeC1VWbuUK0U7b7cuWT/view?usp=drive_link}"
DEST_DIR="${DEST_DIR:-Dataset}"
ZIP_PATH="${DEST_DIR}/sample.zip"
KEEP_ZIP="${KEEP_ZIP:-false}"

echo "================================================================"
echo "  PIAA Dataset Downloader & Extractor"
echo "  Target directory : ${DEST_DIR}"
echo "  Keep ZIP file    : ${KEEP_ZIP}"
echo "================================================================"

mkdir -p "${DEST_DIR}"

# 1. Determine download tool
if command -v uvx &> /dev/null; then
    GDOWN_CMD="uvx gdown"
elif command -v gdown &> /dev/null; then
    GDOWN_CMD="gdown"
else
    echo "[ERROR] Neither 'uvx' nor 'gdown' was found on your system."
    echo "Please install uv (https://docs.astral.sh/uv/) or run: pip install gdown"
    exit 1
fi

# 2. Determine python command for extraction & verification
if command -v uv &> /dev/null; then
    PY_CMD="uv run python"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    PY_CMD="python"
fi

# 3. Download dataset archive
if [ -f "${ZIP_PATH}" ]; then
    echo "[INFO] Found existing archive at ${ZIP_PATH}, skipping download."
else
    echo "[INFO] Downloading dataset archive via ${GDOWN_CMD}..."
    ${GDOWN_CMD} "${GDRIVE_URL}" -O "${ZIP_PATH}"
fi

# 4. Extract archive cleanly using Python (skips __MACOSX and redundant nested zips)
echo "[INFO] Extracting archive to ${DEST_DIR}/..."
${PY_CMD} - "${ZIP_PATH}" "${DEST_DIR}" <<'PY'
import sys
import os
import zipfile

zip_path = sys.argv[1]
dest_dir = sys.argv[2]

if not os.path.exists(zip_path):
    raise SystemExit(f"[ERROR] Zip file not found: {zip_path}")

print(f"Reading {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    members = zf.infolist()
    # Filter out macOS metadata and nested zip files
    valid_members = [
        m for m in members
        if not m.filename.startswith("__MACOSX")
        and "/__MACOSX" not in m.filename
        and not m.filename.endswith(".zip")
    ]
    total = len(valid_members)
    print(f"Extracting {total} files/directories (excluding __MACOSX and nested zips)...")
    
    extracted = 0
    for idx, member in enumerate(valid_members):
        zf.extract(member, dest_dir)
        extracted += 1
        if idx % 1000 == 0 or idx == total - 1:
            pct = int((idx + 1) / total * 100)
            print(f"  Progress: {pct}% ({idx + 1}/{total})", flush=True)

print("[INFO] Extraction completed successfully.")
PY

# 5. Clean up zip archive if KEEP_ZIP is false
if [ "${KEEP_ZIP}" != "true" ]; then
    echo "[INFO] Removing ${ZIP_PATH} (set KEEP_ZIP=true to retain)..."
    rm -f "${ZIP_PATH}"
fi

# 6. Verify extracted dataset
echo "[INFO] Verifying dataset contents..."
${PY_CMD} - "${DEST_DIR}/sample" <<'PY'
import sys
import os
from pathlib import Path

sample_dir = Path(sys.argv[1])
if not sample_dir.exists():
    print(f"[WARN] Expected sample directory at {sample_dir} not found.")
    sys.exit(0)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
images = [p for p in sample_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and "__MACOSX" not in p.parts]

print(f"================================================================")
print(f"  Total valid images found : {len(images)}")
print(f"  Sample location          : {sample_dir.resolve()}")
print(f"================================================================")
if len(images) >= 6526:
    print("[SUCCESS] Dataset verification passed! All expected images are present.")
else:
    print(f"[WARN] Expected at least 6,526 images, but found {len(images)}.")
PY

echo "[SUCCESS] Done!"
