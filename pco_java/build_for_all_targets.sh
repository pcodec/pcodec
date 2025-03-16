#!/bin/bash
set -eoux pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
GROUP_ID="io/github/pcodec"

build_for_target() {
  rustup target add "$RUST_TARGET"
  cargo build --release -p pco_java --target "$RUST_TARGET"
  dest_dir="src/main/resources/$GROUP_ID/$JNI_TARGET"
  mkdir -p "$dest_dir"
  cp "../target/$RUST_TARGET/release/${PREFIX}pco_java.$SUFFIX" $dest_dir
}

# MAC
PREFIX=lib
SUFFIX=dylib
RUST_TARGET=x86_64-apple-darwin JNI_TARGET="darwin-x86-64" build_for_target
RUST_TARGET=aarch64-apple-darwin JNI_TARGET="darwin-aarch64" build_for_target

# LINUX
PREFIX=lib
SUFFIX=so
RUST_TARGET=x86_64-unknown-linux-gnu JNI_TARGET="linux-x86-64" build_for_target
RUST_TARGET=aarch64-unknown-linux-gnu JNI_TARGET="linux-aarch64" build_for_target

# WINDOWS
PREFIX=
SUFFIX=dll
RUST_TARGET=x86_64-pc-windows-gnu JNI_TARGET="win32-x86-64" build_for_target
RUST_TARGET=x86_64-pc-windows-msvc JNI_TARGET="win32-x86-64" build_for_target
