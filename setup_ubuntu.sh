#!/usr/bin/env bash
# Baccarat Vision AI — Ubuntu setup script (tested on 22.04 / 24.04).
#
# Installs system packages required for OpenCV+FFmpeg decoding, animated WebP,
# and CUDA/NVDEC (optional). Creates a Python venv and installs requirements.
#
# Usage:
#     bash setup_ubuntu.sh
#     bash setup_ubuntu.sh --with-cuda     # also installs NVIDIA codec headers
#     bash setup_ubuntu.sh --no-system     # skip apt, only venv + pip

set -euo pipefail

WITH_CUDA=0
NO_SYSTEM=0
for arg in "$@"; do
    case "$arg" in
        --with-cuda) WITH_CUDA=1 ;;
        --no-system) NO_SYSTEM=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [ "$NO_SYSTEM" -eq 0 ]; then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "error: apt-get not found — this script targets Ubuntu/Debian."
        exit 1
    fi
    echo "==> installing apt packages (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y \
        python3 python3-venv python3-pip python3-dev \
        ffmpeg libavcodec-extra \
        libsm6 libxext6 libgl1 libglib2.0-0 \
        libwebp-dev webp \
        build-essential pkg-config

    if [ "$WITH_CUDA" -eq 1 ]; then
        echo "==> installing NVIDIA codec headers for NVDEC..."
        sudo apt-get install -y nv-codec-headers || \
            echo "warn: nv-codec-headers not in apt — install from https://github.com/FFmpeg/nv-codec-headers if needed"
    fi
fi

echo "==> creating venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "==> installing python requirements..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo
echo "==> sanity checks..."
python -c "import cv2; print('opencv:', cv2.__version__)"
python -c "from PIL import Image; print('pillow:', Image.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())" || true
ffmpeg -hide_banner -version | head -n1

echo
echo "==> running unit tests..."
cd "${PROJECT_ROOT}"
for t in tests/test_*.py; do
    printf "%-40s " "${t}"
    python "${t}" 2>&1 | tail -1
done

echo
echo "==> done. activate the venv with:"
echo "    source ${VENV_DIR}/bin/activate"
echo
echo "==> run on a webp clip:"
echo "    python run.py --source path/to/clip.webp --max-frames 200 --log-level DEBUG"
