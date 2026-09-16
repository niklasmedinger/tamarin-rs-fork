#!/usr/bin/env bash
# Downloads, builds, and installs `bliss` -- the external graph canonical-
# labeling / automorphism-group tool `tamarin_theory::bliss_proc` drives as a
# one-shot subprocess (see that module's own doc comments for the exact
# input/output format this integration depends on).
#
# Usage:
#   scripts/install_bliss.sh [--force]
#
#   --force   rebuild/reinstall even if a working `bliss` is already on PATH
#
# Environment:
#   BLISS_VERSION   version to install (default: 0.77 -- the version
#                    bliss_proc.rs's own doc comments were validated against)
#   PREFIX          install prefix; the binary lands at $PREFIX/bin/bliss
#                    (default: $HOME/.local)
#
# Downloads bliss's source zip from the upstream download page
# (https://users.aalto.fi/~tjunttil/bliss/download.html), builds it with
# `make -f Makefile-manual` (no GMP, no cmake -- the simplest of the page's
# documented build paths, and the one this project's bliss integration was
# built and tested against), and installs just the `bliss` executable to
# $PREFIX/bin. Does NOT install the static/shared libraries: bliss_proc.rs
# drives the tool over stdin/stdout as a subprocess, never links against it.
#
# Idempotent: exits early (printing the found version) if a working `bliss`
# is already on PATH, unless --force is given. Set TAM_ALLOW_NO_BLISS=1 in
# your own shell instead of running this script at all if you'd rather skip
# bliss-backed tests than install the tool (see bliss_proc::bliss_available).

set -euo pipefail

BLISS_VERSION="${BLISS_VERSION:-0.77}"
PREFIX="${PREFIX:-$HOME/.local}"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE=1
            ;;
        -h | --help)
            sed -n '2,26p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg (see --help)" >&2
            exit 2
            ;;
    esac
done

if [ "$FORCE" -eq 0 ] && command -v bliss >/dev/null 2>&1 && bliss -version >/dev/null 2>&1; then
    echo "bliss already installed: $(bliss -version 2>&1) ($(command -v bliss))"
    echo "pass --force to rebuild/reinstall anyway"
    exit 0
fi

missing=()
for tool in make unzip; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    missing+=("a C++ compiler (g++/clang++/c++)")
fi
if [ "${#missing[@]}" -gt 0 ]; then
    echo "error: missing required tool(s): ${missing[*]}" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    downloader() { curl -fsSL -o "$1" "$2"; }
elif command -v wget >/dev/null 2>&1; then
    downloader() { wget -q -O "$1" "$2"; }
else
    echo "error: need either curl or wget to download bliss" >&2
    exit 1
fi

url="https://users.aalto.fi/~tjunttil/bliss/downloads/bliss-${BLISS_VERSION}.zip"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "Downloading bliss ${BLISS_VERSION} from ${url} ..."
downloader "$workdir/bliss.zip" "$url"

echo "Extracting ..."
unzip -q "$workdir/bliss.zip" -d "$workdir"

src_dir="$workdir/bliss-${BLISS_VERSION}"
if [ ! -d "$src_dir" ]; then
    # Some releases nest differently -- fall back to whatever single
    # directory the zip actually extracted.
    src_dir="$(find "$workdir" -mindepth 1 -maxdepth 1 -type d | head -n1)"
fi
if [ -z "$src_dir" ] || [ ! -f "$src_dir/Makefile-manual" ]; then
    echo "error: could not find Makefile-manual after extracting -- unexpected zip layout (bliss version bump?)" >&2
    exit 1
fi

echo "Building (make -f Makefile-manual) in $src_dir ..."
make -C "$src_dir" -f Makefile-manual

if [ ! -x "$src_dir/bliss" ]; then
    echo "error: build finished but $src_dir/bliss was not produced" >&2
    exit 1
fi

mkdir -p "$PREFIX/bin"
install -m 755 "$src_dir/bliss" "$PREFIX/bin/bliss"
echo "Installed to $PREFIX/bin/bliss"

if command -v bliss >/dev/null 2>&1; then
    echo "$(bliss -version 2>&1) ($(command -v bliss))"
else
    echo "warning: $PREFIX/bin is not on your PATH -- add it, e.g.:" >&2
    echo "  export PATH=\"$PREFIX/bin:\$PATH\"" >&2
    "$PREFIX/bin/bliss" -version
fi
