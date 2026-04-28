import json
import logging
import os
import time
import uuid

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "2000"))
ADAPTER_API_KEY_SECRET_ARN = os.environ.get("ADAPTER_API_KEY_SECRET_ARN", "")
_ADAPTER_API_KEY_ENV = os.environ.get("ADAPTER_API_KEY", "")  # local/test fallback, read at import
LOG_PROMPTS = os.environ.get("LOG_PROMPTS", "false").lower() == "true"
CORS_ENABLED = os.environ.get("CORS_ENABLED", "false").lower() == "true"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MAX_BODY_BYTES = 64 * 1024  # 64 KB hard ceiling

_bedrock = None
_bedrock_mgmt = None
_api_key_cache: dict = {"value": None, "fetched_at": 0.0}
_KEY_CACHE_TTL = 300  # seconds; allows key rotation without redeployment


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock


def _get_bedrock_mgmt():
    global _bedrock_mgmt
    if _bedrock_mgmt is None:
        _bedrock_mgmt = boto3.client("bedrock", region_name=BEDROCK_REGION)
    return _bedrock_mgmt


def _get_adapter_api_key() -> str:
    """Return the adapter API key, fetching from Secrets Manager when configured.

    Falls back to ADAPTER_API_KEY env var for local/test use when no ARN is set.
    Cached for _KEY_CACHE_TTL seconds so rotation takes effect within 5 minutes.
    """
    if not ADAPTER_API_KEY_SECRET_ARN:
        return _ADAPTER_API_KEY_ENV

    now = time.time()
    if _api_key_cache["value"] is not None and (now - _api_key_cache["fetched_at"]) < _KEY_CACHE_TTL:
        return _api_key_cache["value"]  # type: ignore[return-value]

    try:
        sm = boto3.client("secretsmanager", region_name=BEDROCK_REGION)
        resp = sm.get_secret_value(SecretId=ADAPTER_API_KEY_SECRET_ARN)
        key = resp.get("SecretString", "")
        _api_key_cache["value"] = key
        _api_key_cache["fetched_at"] = now
        return key
    except ClientError as exc:
        logger.error("secretsmanager_error: %s", exc)
        # Return stale cache rather than breaking all requests if SM is briefly unavailable
        if _api_key_cache["value"] is not None:
            return _api_key_cache["value"]  # type: ignore[return-value]
        return _ADAPTER_API_KEY_ENV


def _base_headers():
    headers = {"content-type": "application/json"}
    if CORS_ENABLED:
        headers["access-control-allow-origin"] = "*"
    return headers


def _error(status: int, code: str, message: str) -> dict:
    return {
        "statusCode": status,
        "headers": _base_headers(),
        "body": json.dumps({"error": {"code": code, "message": message}}),
    }


def _ok(body: dict) -> dict:
    return {
        "statusCode": 200,
        "headers": _base_headers(),
        "body": json.dumps(body),
    }


def _check_api_key(event: dict):
    """Return an error response if the API key is wrong, else None.

    Accepts the key from either:
      x-api-key: <key>                  (curl, direct calls)
      Authorization: Bearer <key>       (LiteLLM, OpenAI SDK)
    """
    required_key = _get_adapter_api_key()
    if not required_key:
        return None
    headers = event.get("headers") or {}

    provided = (
        headers.get("x-api-key")
        or headers.get("X-Api-Key")
        or ""
    )

    if not provided:
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    if provided != required_key:
        return _error(401, "unauthorized", "Missing or invalid API key")
    return None


def convert_messages(messages: list) -> tuple:
    """
    Split OpenAI-style messages into (system_blocks, bedrock_messages).
    system_blocks: list of {"text": str} for Bedrock system param
    bedrock_messages: list of {"role": str, "content": [{"text": str}]}
    Raises ValueError on bad input.
    """
    system_blocks = []
    bedrock_messages = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if not isinstance(content, str):
            raise ValueError(
                f"messages[{i}].content must be a string, got {type(content).__name__}"
            )

        if role == "system":
            system_blocks.append({"text": content})
        elif role in ("user", "assistant"):
            bedrock_messages.append(
                {"role": role, "content": [{"text": content}]}
            )
        else:
            raise ValueError(f"messages[{i}].role '{role}' is not supported")

    return system_blocks, bedrock_messages


def _handle_health(_event: dict) -> dict:
    return _ok({"status": "ok", "service": "infer-bedrock"})


def _handle_chat_completions(event: dict) -> dict:
    auth_err = _check_api_key(event)
    if auth_err:
        return auth_err

    raw_body = event.get("body") or ""
    if len(raw_body.encode("utf-8")) > MAX_BODY_BYTES:
        return _error(413, "request_too_large", "Request body exceeds 64 KB limit")

    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        return _error(400, "invalid_json", "Request body is not valid JSON")

    if not isinstance(body, dict):
        return _error(400, "invalid_body", "Request body must be a JSON object")

    model = body.get("model")
    if not model:
        return _error(400, "missing_field", "model is required")
    if ALLOWED_MODELS and model not in ALLOWED_MODELS:
        return _error(400, "disallowed_model", f"Model '{model}' is not in the allowed list")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return _error(400, "missing_field", "messages must be a non-empty array")

    max_tokens = body.get("max_tokens", 500)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        return _error(400, "invalid_field", "max_tokens must be a positive integer")
    if max_tokens > MAX_TOKENS_LIMIT:
        return _error(
            400,
            "max_tokens_exceeded",
            f"max_tokens {max_tokens} exceeds server limit {MAX_TOKENS_LIMIT}",
        )

    temperature = body.get("temperature", 0.7)

    try:
        system_blocks, bedrock_messages = convert_messages(messages)
    except ValueError as exc:
        return _error(400, "invalid_messages", str(exc))

    if not bedrock_messages:
        return _error(400, "no_messages", "At least one user or assistant message is required")

    if LOG_PROMPTS:
        first = str(bedrock_messages[0])[:120] if bedrock_messages else ""
        logger.info(
            "chat_completions model=%s messages=%d first_truncated=%s",
            model,
            len(messages),
            first,
        )

    converse_kwargs: dict = {
        "modelId": model,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": float(temperature),
        },
    }
    if system_blocks:
        converse_kwargs["system"] = system_blocks

    try:
        resp = _get_bedrock().converse(**converse_kwargs)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        logger.error("bedrock_error code=%s message=%s", code, msg)
        if code == "ValidationException":
            return _error(400, "bedrock_validation", msg)
        if code in ("AccessDeniedException", "ResourceNotFoundException"):
            return _error(502, "bedrock_access", f"Bedrock returned {code}")
        return _error(502, "bedrock_error", f"Bedrock returned {code}")

    output_msg = resp.get("output", {}).get("message", {})
    content_blocks = output_msg.get("content", [])
    text = "".join(b.get("text", "") for b in content_blocks if "text" in b)

    stop_reason = resp.get("stopReason", "end_turn")
    finish_reason = "stop" if stop_reason in ("end_turn", "stop_sequence") else stop_reason

    usage = resp.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)

    return _ok(
        {
            "id": f"infer-bedrock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    )


def _handle_models(event: dict) -> dict:
    """GET /v1/models — returns all Bedrock foundation models in OpenAI format."""
    auth_err = _check_api_key(event)
    if auth_err:
        return auth_err

    try:
        resp = _get_bedrock_mgmt().list_foundation_models(
            byOutputModality="TEXT",
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        logger.error("list_foundation_models error code=%s message=%s", code, msg)
        return _error(502, "bedrock_error", f"Could not list models: {code}")

    models = []
    for m in resp.get("modelSummaries", []):
        model_id = m.get("modelId", "")
        if not model_id:
            continue
        # Only include models that support on-demand inference (no provisioned throughput required)
        inference_types = m.get("inferenceTypesSupported", [])
        if "ON_DEMAND" not in inference_types:
            continue
        models.append({
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": m.get("providerName", "bedrock"),
            "display_name": m.get("modelName", model_id),
            "input_modalities": m.get("inputModalities", []),
            "output_modalities": m.get("outputModalities", []),
        })

    return _ok({"object": "list", "data": models})


def lambda_handler(event: dict, _context) -> dict:
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "")
    path = event.get("rawPath", "")

    if path == "/health" and method == "GET":
        return _handle_health(event)

    if path == "/v1/models" and method == "GET":
        return _handle_models(event)

    if path == "/v1/chat/completions" and method == "POST":
        return _handle_chat_completions(event)

    return _error(404, "not_found", f"{method} {path} not found")
