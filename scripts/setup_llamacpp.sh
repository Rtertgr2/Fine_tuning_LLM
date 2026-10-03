#!/usr/bin/env bash
# Build llama.cpp (Vulkan) ลง tools/llama.cpp — pin commit เสมอ (ไม่ checkout branch)
# Pin: tag b11371 = 99b95488cac0f00ce3f05af113a8c1e287753f87 (เมื่อ 2026-10-03)
set -euo pipefail

REPO="https://github.com/ggml-org/llama.cpp"
TAG="b11371"
PIN="99b95488cac0f00ce3f05af113a8c1e287753f87"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/tools/llama.cpp"

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "ERROR: vulkaninfo not found. Install a Vulkan driver (e.g. vulkan-intel) and re-run." >&2
  exit 1
fi

if [ -d "$DIR/.git" ]; then
  echo "==> Reusing existing checkout: $DIR"
  git -C "$DIR" fetch --depth 1 origin "$PIN"
  git -C "$DIR" checkout --detach "$PIN"
else
  echo "==> Cloning llama.cpp $TAG (shallow) -> $DIR"
  mkdir -p "$(dirname "$DIR")"
  git clone --depth 1 --branch "$TAG" "$REPO" "$DIR"
fi

actual="$(git -C "$DIR" rev-parse HEAD)"
if [ "$actual" != "$PIN" ]; then
  echo "ERROR: pin mismatch — expected $PIN, got $actual (tag $TAG moved?)" >&2
  exit 1
fi

echo "==> Building (Vulkan, Release) ..."
# LLAMA_UI_GZIP=OFF: CMake 4.4.3 file(ARCHIVE_CREATE) พังบน symlinked path
# (worktree tools/ -> main) — input ถูก relativize จาก cwd แล้ว lstat ผิดที่
# UI ยังทำงานปกติ (embed ไม่บีบ gzip — ไม่กระทบ llama-server API/health)
cmake -S "$DIR" -B "$DIR/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_UI_GZIP=OFF
cmake --build "$DIR/build" -j"$(nproc)"

echo "==> Verifying expected binaries ..."
for b in llama-server llama-quantize llama-bench llama-cli; do
  if [ ! -x "$DIR/build/bin/$b" ]; then
    echo "ERROR: expected binary missing: $DIR/build/bin/$b" >&2
    exit 1
  fi
done
if [ ! -f "$DIR/convert_hf_to_gguf.py" ]; then
  echo "ERROR: convert_hf_to_gguf.py missing at $DIR" >&2
  exit 1
fi

echo "==> Done. Binaries: $DIR/build/bin/ (pin $TAG)"
