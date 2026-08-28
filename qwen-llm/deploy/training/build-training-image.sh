#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /usr/bin/podman build \
    --tag localhost/gta-ai-ms-swift:4.5.2 \
    --file "$script_dir/Containerfile" \
    "$script_dir"
