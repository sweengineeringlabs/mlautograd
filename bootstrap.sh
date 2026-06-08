#!/usr/bin/env bash
set -euo pipefail

# Install the pinned toolchain and build the crate.
rustup toolchain install 1.82
rustup override set 1.82
cargo build
