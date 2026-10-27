#!/bin/bash
# Install the Google Colab CLI (google-colab-cli) so `colab` is on PATH.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  pip install --quiet --break-system-packages uv 2>/dev/null || pip install --quiet uv
fi

if ! command -v colab >/dev/null 2>&1; then
  uv tool install --quiet google-colab-cli
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
