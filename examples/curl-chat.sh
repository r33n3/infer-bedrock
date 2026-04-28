#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL to your InferBedrock endpoint, e.g. https://abc123.execute-api.us-east-1.amazonaws.com}"
: "${ADAPTER_API_KEY:?Set ADAPTER_API_KEY}"

curl -fsSL -X POST "${API_URL}/v1/chat/completions" \
  -H "content-type: application/json" \
  -H "x-api-key: ${ADAPTER_API_KEY}" \
  -d @"$(dirname "$0")/request.json" | jq .
