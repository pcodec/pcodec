#!/usr/bin/env bash
# Regenerates the public-api.txt snapshot of each published library crate.
# Requires cargo-public-api and a nightly toolchain.
set -euo pipefail
cd "$(dirname "$0")/.."

snapshot() {
  local package=$1 dir=$2
  shift 2
  cargo +nightly public-api -p "$package" -ss "$@" > "$dir/public-api.txt"
}

# pco's bench feature exposes internals that aren't part of its public API
snapshot pco pco
snapshot better_io better_io
snapshot dtype_dispatch dtype_dispatch
