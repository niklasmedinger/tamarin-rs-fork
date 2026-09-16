#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THEORY="$SCRIPT_DIR/../../tamarin-prover/examples/Tutorial.spthy"

for name in tutorial_client_khu_system tutorial_khu_client_system; do
    json="$SCRIPT_DIR/examples/$name.json"
    svg="$SCRIPT_DIR/examples/$name.svg"
    cargo run -p tamarin-theory --example system_json_to_graphviz -- \
        --theory "$THEORY" "$json" \
        | dot -Tsvg -o "$svg"
done
