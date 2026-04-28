#!/usr/bin/env bash
# Syncs all InferBedrock-available Bedrock models into your LiteLLM proxy.
# Run after deploy or whenever you want to refresh the model list.
#
# Requirements:
#   - InferBedrock endpoint live (INFER_BEDROCK_URL)
#   - InferBedrock API key (INFER_BEDROCK_API_KEY)
#   - LiteLLM proxy running locally (LITELLM_URL, default http://localhost:4000)
#   - LiteLLM master key (LITELLM_MASTER_KEY)
set -euo pipefail

: "${INFER_BEDROCK_URL:=https://exhot3ztm9.execute-api.us-east-1.amazonaws.com}"
: "${INFER_BEDROCK_API_KEY:?Set INFER_BEDROCK_API_KEY to your adapter key}"
: "${LITELLM_URL:=http://localhost:4000}"
: "${LITELLM_MASTER_KEY:?Set LITELLM_MASTER_KEY to your LiteLLM master key}"

echo "==> Fetching available models from InferBedrock..."
MODELS_JSON=$(curl -fsSL \
  -H "x-api-key: ${INFER_BEDROCK_API_KEY}" \
  "${INFER_BEDROCK_URL}/v1/models")

MODEL_COUNT=$(echo "${MODELS_JSON}" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
echo "    Found ${MODEL_COUNT} models available on Bedrock."

echo ""
echo "==> Registering models in LiteLLM at ${LITELLM_URL}..."

ADDED=0
FAILED=0

echo "${MODELS_JSON}" | python3 - <<'PYEOF'
import sys, json, os, urllib.request, urllib.error

data = json.load(sys.stdin)
litellm_url = os.environ["LITELLM_URL"].rstrip("/")
master_key = os.environ["LITELLM_MASTER_KEY"]
infer_url = os.environ["INFER_BEDROCK_URL"]
api_key = os.environ["INFER_BEDROCK_API_KEY"]

added = 0
failed = 0

for model in data["data"]:
    model_id = model["id"]
    display_name = model.get("display_name", model_id)
    # Use provider/model-id as the LiteLLM model name for clarity
    provider = model.get("owned_by", "bedrock").lower().replace(" ", "-")
    litellm_model_name = f"{provider}/{model_id}"

    payload = json.dumps({
        "model_name": litellm_model_name,
        "litellm_params": {
            "model": f"openai/{model_id}",
            "api_base": infer_url,
            "api_key": api_key,
        },
        "model_info": {
            "description": f"{display_name} via InferBedrock",
        }
    }).encode()

    req = urllib.request.Request(
        f"{litellm_url}/model/new",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {master_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            added += 1
            print(f"  + {litellm_model_name}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "already exists" in body.lower() or e.code == 409:
            print(f"  = {litellm_model_name}  (already registered)")
        else:
            print(f"  ! {litellm_model_name}  FAILED {e.code}: {body[:80]}")
            failed += 1
    except Exception as e:
        print(f"  ! {litellm_model_name}  ERROR: {e}")
        failed += 1

print(f"\nDone. Added: {added}  Failed: {failed}")
PYEOF
