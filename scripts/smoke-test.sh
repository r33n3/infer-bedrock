#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL to your InferBedrock endpoint}"
: "${ADAPTER_API_KEY:?Set ADAPTER_API_KEY}"

echo "==> Smoke test: GET /health"
HEALTH=$(curl -fsSL "${API_URL}/health")
echo "${HEALTH}" | jq .
STATUS=$(echo "${HEALTH}" | jq -r '.status')
if [[ "${STATUS}" != "ok" ]]; then
  echo "FAIL: expected status=ok, got ${STATUS}"
  exit 1
fi
echo "PASS: /health"

echo ""
echo "==> Smoke test: POST /v1/chat/completions"
RESPONSE=$(curl -fsSL -X POST "${API_URL}/v1/chat/completions" \
  -H "content-type: application/json" \
  -H "x-api-key: ${ADAPTER_API_KEY}" \
  -d '{
    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "messages": [{"role": "user", "content": "Reply with only the word: pong"}],
    "max_tokens": 20
  }')
echo "${RESPONSE}" | jq .

CONTENT=$(echo "${RESPONSE}" | jq -r '.choices[0].message.content')
if [[ -z "${CONTENT}" ]]; then
  echo "FAIL: empty content in response"
  exit 1
fi
echo "PASS: /v1/chat/completions — response: ${CONTENT}"
