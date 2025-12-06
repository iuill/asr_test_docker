#!/bin/bash
# Build base images for ReazonSpeech Docker
# Usage: ./scripts/build-base-images.sh [gpu|cpu|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Enable BuildKit
export DOCKER_BUILDKIT=1

build_gpu() {
    echo "=========================================="
    echo "Building GPU base images..."
    echo "=========================================="

    echo "[1/3] Building common-base:gpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.common-gpu" \
        -t reazonspeech-common-base:gpu \
        "$PROJECT_ROOT"

    echo "[2/3] Building k2-v2-base:gpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.k2-v2-gpu" \
        -t reazonspeech-k2-v2-base:gpu \
        "$PROJECT_ROOT"

    echo "[3/3] Building espnet-v2-base:gpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.espnet-v2-gpu" \
        -t reazonspeech-espnet-v2-base:gpu \
        "$PROJECT_ROOT"

    echo "GPU base images built successfully!"
}

build_cpu() {
    echo "=========================================="
    echo "Building CPU base images..."
    echo "=========================================="

    echo "[1/3] Building common-base:cpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.common-cpu" \
        -t reazonspeech-common-base:cpu \
        "$PROJECT_ROOT"

    echo "[2/3] Building k2-v2-base:cpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.k2-v2-cpu" \
        -t reazonspeech-k2-v2-base:cpu \
        "$PROJECT_ROOT"

    echo "[3/3] Building espnet-v2-base:cpu..."
    docker build \
        -f "$PROJECT_ROOT/services/base/Dockerfile.espnet-v2-cpu" \
        -t reazonspeech-espnet-v2-base:cpu \
        "$PROJECT_ROOT"

    echo "CPU base images built successfully!"
}

case "${1:-all}" in
    gpu)
        build_gpu
        ;;
    cpu)
        build_cpu
        ;;
    all)
        build_gpu
        build_cpu
        ;;
    *)
        echo "Usage: $0 [gpu|cpu|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Base images ready. You can now run:"
echo "  docker compose build"
echo "=========================================="
