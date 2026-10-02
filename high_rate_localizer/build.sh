#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${FLS_LOCALIZER_BUILD_DIR:-${script_dir}/build}"

cmake \
  -S "${script_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLS_LOCALIZER_BUILD_TESTS=OFF \
  -DFLS_LOCALIZER_ENABLE_LIBCAMERA=ON

cmake --build "${build_dir}" --config Release --target fls_localizer --parallel
