# InferBedrock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployable AWS Bedrock inference adapter (API Gateway HTTP API + Lambda) that exposes OpenAI-compatible `/v1/chat/completions` and `/health` endpoints for local tooling without requiring AWS credentials on the client.

**Architecture:** Lambda proxy receives requests from API Gateway HTTP API, validates API key and request shape, converts OpenAI-style messages to Bedrock Converse format, calls Bedrock Runtime, and returns an OpenAI-compatible response. GitHub Actions uses OIDC to assume an AWS IAM role and deploy via CloudFormation.

**Tech Stack:** Python 3.12, boto3, AWS Lambda, API Gateway HTTP API (v2), Amazon Bedrock Converse API, CloudFormation, GitHub Actions OIDC

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/app.py` | Lambda handler: routing, auth, message conversion, Bedrock call, response shaping |
| `infra/template.yaml` | CloudFormation: IAM role, Lambda, log group, HTTP API, routes, stage, Lambda permission |
| `infra/parameters.example.json` | Example CloudFormation parameter overrides for manual deploy reference |
| `.github/workflows/deploy.yml` | GitHub Actions: OIDC auth, zip Lambda, upload to S3, `cloudformation deploy`, output endpoint |
| `tests/test_app.py` | Unit tests: health, auth, invalid JSON, disallowed model, token limit, message conversion, happy path |
| `examples/request.json` | Example request body for curl testing |
| `examples/curl-chat.sh` | Curl invocation script using env vars |
| `scripts/deploy-local.sh` | Manual deploy via AWS CLI for local testing without GitHub Actions |
| `scripts/smoke-test.sh` | Post-deploy smoke test: hits /health then /v1/chat/completions |
| `docs/SECURITY.md` | Security model: API key, model allowlist, IAM, no prompt logging |
| `docs/CONFIGURATION.md` | All env vars and CloudFormation parameters with defaults and descriptions |
| `docs/ARCHITECTURE.md` | Architecture diagram, request flow, design decisions |
| `README.md` | What this is, bootstrap, deploy, usage, LiteLLM/WTK integration, cost |
| `.gitignore` | Python + Lambda + CDK artifacts |

---

## Task 1: Repo skeleton + .gitignore

**Files:**
- Create: `.gitignore`
- Create: all directories (`src/`, `infra/`, `tests/`, `examples/`, `scripts/`, `docs/`)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src infra tests examples scripts docs
```

- [ ] **Step 2: Write .gitignore**

```text
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
*.egg

# Lambda packaging
lambda.zip
packaged-template.yaml
*.zip

# AWS SAM / CDK
.aws-sam/
cdk.out/

# env / secrets
.env
*.env.local

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Commit**

```bash
git init
git add .gitignore
git commit -m "chore: initial repo skeleton"
```

---

## Task 2: Lambda handler — src/app.py

**Files:**
- Create: `src/app.py`

- [ ] **Step 1: Write src/app.py**

```python
import json
import logging
import os
import uuid

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "2000"))
ADAPTER_API_KEY = os.environ.get("ADAPTER_API_KEY", "")
LOG_PROMPTS = os.environ.get("LOG_PROMPTS", "false").lower() == "true"
CORS_ENABLED = os.environ.get("CORS_ENABLED", "false").lower() == "true"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MAX_BODY_BYTES = 64 * 1024  # 64 KB hard ceiling

_bedrock = None


def _get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock


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
    """Return an error response if the API key is wrong, else None."""
    if not ADAPTER_API_KEY:
        return None
    headers = event.get("headers") or {}
    provided = headers.get("x-api-key") or headers.get("X-Api-Key") or ""
    if provided != ADAPTER_API_KEY:
        return _error(401, "unauthorized", "Missing or invalid x-api-key header")
    return None


def convert_messages(messages: list) -> tuple[list, list]:
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
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    )


def lambda_handler(event: dict, _context) -> dict:
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "")
    path = event.get("rawPath", "")

    if path == "/health" and method == "GET":
        return _handle_health(event)

    if path == "/v1/chat/completions" and method == "POST":
        return _handle_chat_completions(event)

    return _error(404, "not_found", f"{method} {path} not found")
```

- [ ] **Step 2: Commit**

```bash
git add src/app.py
git commit -m "feat: Lambda handler with Bedrock Converse integration"
```

---

## Task 3: Unit tests — tests/test_app.py

**Files:**
- Create: `tests/test_app.py`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Write tests/test_app.py**

```python
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_event(method="GET", path="/health", body=None, api_key=None):
    headers = {}
    if api_key is not None:
        headers["x-api-key"] = api_key
    return {
        "requestContext": {"http": {"method": method}},
        "rawPath": path,
        "headers": headers,
        "body": json.dumps(body) if body is not None else "",
    }


def _load_app(env_overrides=None):
    """Import src.app with patched env vars and a mocked boto3."""
    env = {
        "ALLOWED_MODELS": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "MAX_TOKENS_LIMIT": "2000",
        "ADAPTER_API_KEY": "test-key",
        "LOG_PROMPTS": "false",
        "BEDROCK_REGION": "us-east-1",
        "CORS_ENABLED": "false",
    }
    if env_overrides:
        env.update(env_overrides)

    # Remove cached module so env vars are re-read on import
    for key in list(sys.modules.keys()):
        if "app" in key:
            del sys.modules[key]

    mock_boto3 = MagicMock()
    with patch.dict(os.environ, env, clear=False):
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            # Insert src onto path temporarily
            src_path = os.path.join(os.path.dirname(__file__), "..", "src")
            sys.path.insert(0, os.path.abspath(src_path))
            import app
            sys.path.pop(0)
    return app, mock_boto3


class TestHealthRoute(unittest.TestCase):
    def test_health_returns_ok(self):
        app, _ = _load_app()
        event = _make_event("GET", "/health")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "infer-bedrock")

    def test_unknown_route_returns_404(self):
        app, _ = _load_app()
        event = _make_event("GET", "/unknown")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 404)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "not_found")


class TestApiKeyAuth(unittest.TestCase):
    def test_missing_api_key_returns_401(self):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body={"model": "x", "messages": []})
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 401)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "unauthorized")

    def test_wrong_api_key_returns_401(self):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body={}, api_key="wrong")
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 401)

    def test_no_key_required_when_env_empty(self):
        app, mock_boto3 = _load_app({"ADAPTER_API_KEY": ""})
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2},
        }
        body = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        }
        event = _make_event("POST", "/v1/chat/completions", body=body)
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)


class TestChatCompletionsValidation(unittest.TestCase):
    def _post(self, body, api_key="test-key"):
        app, _ = _load_app()
        event = _make_event("POST", "/v1/chat/completions", body=body, api_key=api_key)
        return app.lambda_handler(event, None)

    def test_invalid_json_returns_400(self):
        app, _ = _load_app()
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/v1/chat/completions",
            "headers": {"x-api-key": "test-key"},
            "body": "not-json{{{",
        }
        resp = app.lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "invalid_json")

    def test_disallowed_model_returns_400(self):
        resp = self._post({"model": "openai.gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "disallowed_model")

    def test_max_tokens_exceeded_returns_400(self):
        resp = self._post({
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9999,
        })
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "max_tokens_exceeded")

    def test_missing_messages_returns_400(self):
        resp = self._post({"model": "anthropic.claude-3-5-sonnet-20240620-v1:0"})
        self.assertEqual(resp["statusCode"], 400)

    def test_non_string_content_returns_400(self):
        resp = self._post({
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [{"role": "user", "content": ["list", "content"]}],
        })
        self.assertEqual(resp["statusCode"], 400)
        body = json.loads(resp["body"])
        self.assertEqual(body["error"]["code"], "invalid_messages")


class TestMessageConversion(unittest.TestCase):
    def setUp(self):
        app, _ = _load_app()
        self.convert = app.convert_messages

    def test_user_message(self):
        sys_blocks, msgs = self.convert([{"role": "user", "content": "Hello"}])
        self.assertEqual(sys_blocks, [])
        self.assertEqual(msgs, [{"role": "user", "content": [{"text": "Hello"}]}])

    def test_system_message_extracted(self):
        sys_blocks, msgs = self.convert([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ])
        self.assertEqual(sys_blocks, [{"text": "Be concise."}])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_assistant_message(self):
        _, msgs = self.convert([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"},
        ])
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"][0]["text"], "Hello there")

    def test_non_string_content_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.convert([{"role": "user", "content": 42}])
        self.assertIn("must be a string", str(ctx.exception))

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.convert([{"role": "tool", "content": "result"}])
        self.assertIn("not supported", str(ctx.exception))


class TestHappyPath(unittest.TestCase):
    def test_successful_converse_call(self):
        app, mock_boto3 = _load_app()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello, world!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        body = {
            "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hello."},
            ],
            "max_tokens": 100,
            "temperature": 0.2,
        }
        event = _make_event("POST", "/v1/chat/completions", body=body, api_key="test-key")
        resp = app.lambda_handler(event, None)

        self.assertEqual(resp["statusCode"], 200)
        result = json.loads(resp["body"])
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello, world!")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 5)
        self.assertEqual(result["usage"]["total_tokens"], 15)

        # Verify system block was passed correctly
        call_kwargs = mock_client.converse.call_args.kwargs
        self.assertIn("system", call_kwargs)
        self.assertEqual(call_kwargs["system"], [{"text": "Be concise."}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Create tests/__init__.py**

```bash
touch tests/__init__.py
```

- [ ] **Step 3: Run tests**

```bash
cd /mnt/c/Users/bradj/Development/InferBedrock
python -m pytest tests/test_app.py -v
```

Expected: all tests pass (no real AWS calls — boto3 is mocked).

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: unit tests for Lambda handler"
```

---

## Task 4: CloudFormation — infra/template.yaml

**Files:**
- Create: `infra/template.yaml`

- [ ] **Step 1: Write infra/template.yaml**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: InferBedrock - Lightweight Bedrock inference adapter

Parameters:
  ProjectName:
    Type: String
    Default: infer-bedrock
  BedrockRegion:
    Type: String
    Default: us-east-1
  AllowedModels:
    Type: String
    Default: anthropic.claude-3-5-sonnet-20240620-v1:0
    Description: Comma-separated list of allowed Bedrock model IDs
  DefaultModel:
    Type: String
    Default: anthropic.claude-3-5-sonnet-20240620-v1:0
  MaxTokensLimit:
    Type: Number
    Default: 2000
  AdapterApiKey:
    Type: String
    NoEcho: true
    Default: ''
    Description: API key required in x-api-key header. Leave empty to disable auth (local/test only).
  LogPrompts:
    Type: String
    Default: 'false'
    AllowedValues: ['true', 'false']
  CorsEnabled:
    Type: String
    Default: 'false'
    AllowedValues: ['true', 'false']
  LambdaS3Bucket:
    Type: String
    Description: S3 bucket containing the Lambda deployment zip
  LambdaS3Key:
    Type: String
    Description: S3 key of the Lambda deployment zip (e.g. infer-bedrock/abc123/lambda.zip)

Resources:

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${ProjectName}-lambda-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: BedrockConverse
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - bedrock:InvokeModel
                  - bedrock:Converse
                # Foundation model ARNs follow arn:aws:bedrock:REGION::foundation-model/MODEL_ID.
                # Wildcard on model ID is intentional: restricting per-model here would require
                # maintaining a parallel ARN list. The ALLOWED_MODELS env var is the enforcement layer.
                Resource: !Sub 'arn:aws:bedrock:${BedrockRegion}::foundation-model/*'

  LambdaLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/aws/lambda/${ProjectName}-adapter'
      RetentionInDays: 7

  LambdaFunction:
    Type: AWS::Lambda::Function
    DependsOn: LambdaLogGroup
    Properties:
      FunctionName: !Sub '${ProjectName}-adapter'
      Runtime: python3.12
      Handler: app.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Timeout: 30
      MemorySize: 512
      Code:
        S3Bucket: !Ref LambdaS3Bucket
        S3Key: !Ref LambdaS3Key
      Environment:
        Variables:
          ALLOWED_MODELS: !Ref AllowedModels
          MAX_TOKENS_LIMIT: !Ref MaxTokensLimit
          ADAPTER_API_KEY: !Ref AdapterApiKey
          LOG_PROMPTS: !Ref LogPrompts
          BEDROCK_REGION: !Ref BedrockRegion
          CORS_ENABLED: !Ref CorsEnabled

  HttpApi:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: !Sub '${ProjectName}-api'
      ProtocolType: HTTP
      Description: InferBedrock HTTP API

  HttpApiIntegration:
    Type: AWS::ApiGatewayV2::Integration
    Properties:
      ApiId: !Ref HttpApi
      IntegrationType: AWS_PROXY
      IntegrationUri: !Sub
        - 'arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${FnArn}/invocations'
        - FnArn: !GetAtt LambdaFunction.Arn
      PayloadFormatVersion: '2.0'

  HealthRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref HttpApi
      RouteKey: 'GET /health'
      Target: !Sub 'integrations/${HttpApiIntegration}'

  ChatRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref HttpApi
      RouteKey: 'POST /v1/chat/completions'
      Target: !Sub 'integrations/${HttpApiIntegration}'

  HttpApiStage:
    Type: AWS::ApiGatewayV2::Stage
    Properties:
      ApiId: !Ref HttpApi
      StageName: '$default'
      AutoDeploy: true

  LambdaInvokePermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref LambdaFunction
      Action: lambda:InvokeFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub 'arn:aws:execute-api:${AWS::Region}:${AWS::AccountId}:${HttpApi}/*'

Outputs:
  ApiEndpoint:
    Description: InferBedrock API endpoint
    Value: !Sub 'https://${HttpApi}.execute-api.${AWS::Region}.amazonaws.com'
    Export:
      Name: !Sub '${ProjectName}-api-endpoint'
  FunctionName:
    Description: Lambda function name
    Value: !Ref LambdaFunction
    Export:
      Name: !Sub '${ProjectName}-function-name'
  LambdaRoleArn:
    Description: Lambda execution role ARN
    Value: !GetAtt LambdaExecutionRole.Arn
    Export:
      Name: !Sub '${ProjectName}-lambda-role-arn'
```

- [ ] **Step 2: Commit**

```bash
git add infra/template.yaml
git commit -m "feat: CloudFormation template for API Gateway + Lambda + IAM"
```

---

## Task 5: infra/parameters.example.json

**Files:**
- Create: `infra/parameters.example.json`

- [ ] **Step 1: Write infra/parameters.example.json**

```json
[
  { "ParameterKey": "ProjectName",      "ParameterValue": "infer-bedrock" },
  { "ParameterKey": "BedrockRegion",    "ParameterValue": "us-east-1" },
  { "ParameterKey": "AllowedModels",    "ParameterValue": "anthropic.claude-3-5-sonnet-20240620-v1:0" },
  { "ParameterKey": "DefaultModel",     "ParameterValue": "anthropic.claude-3-5-sonnet-20240620-v1:0" },
  { "ParameterKey": "MaxTokensLimit",   "ParameterValue": "2000" },
  { "ParameterKey": "AdapterApiKey",    "ParameterValue": "REPLACE_WITH_SECRET" },
  { "ParameterKey": "LogPrompts",       "ParameterValue": "false" },
  { "ParameterKey": "CorsEnabled",      "ParameterValue": "false" },
  { "ParameterKey": "LambdaS3Bucket",   "ParameterValue": "REPLACE_WITH_DEPLOY_BUCKET" },
  { "ParameterKey": "LambdaS3Key",      "ParameterValue": "infer-bedrock/latest/lambda.zip" }
]
```

- [ ] **Step 2: Commit**

```bash
git add infra/parameters.example.json
git commit -m "chore: example CloudFormation parameters"
```

---

## Task 6: GitHub Actions — .github/workflows/deploy.yml

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Write .github/workflows/deploy.yml**

```yaml
name: Deploy InferBedrock

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    name: Deploy to AWS
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: ${{ vars.AWS_REGION }}
          role-session-name: infer-bedrock-deploy

      - name: Package Lambda zip
        run: |
          cd src
          zip -j ../lambda.zip app.py
          cd ..
          echo "Lambda zip size: $(wc -c < lambda.zip) bytes"

      - name: Upload Lambda zip to S3
        run: |
          S3_KEY="infer-bedrock/${{ github.sha }}/lambda.zip"
          aws s3 cp lambda.zip "s3://${{ vars.DEPLOY_BUCKET }}/${S3_KEY}"
          echo "S3_KEY=${S3_KEY}" >> "$GITHUB_ENV"

      - name: Deploy CloudFormation stack
        run: |
          aws cloudformation deploy \
            --template-file infra/template.yaml \
            --stack-name "${{ vars.STACK_NAME }}" \
            --capabilities CAPABILITY_NAMED_IAM \
            --parameter-overrides \
              ProjectName=infer-bedrock \
              BedrockRegion="${{ vars.BEDROCK_REGION }}" \
              AllowedModels="${{ vars.ALLOWED_MODELS }}" \
              DefaultModel="${{ vars.DEFAULT_MODEL }}" \
              MaxTokensLimit="${{ vars.MAX_TOKENS_LIMIT }}" \
              AdapterApiKey="${{ secrets.ADAPTER_API_KEY }}" \
              LambdaS3Bucket="${{ vars.DEPLOY_BUCKET }}" \
              LambdaS3Key="${{ env.S3_KEY }}" \
            --no-fail-on-empty-changeset

      - name: Update Lambda code
        run: |
          FUNCTION_NAME=$(aws cloudformation describe-stacks \
            --stack-name "${{ vars.STACK_NAME }}" \
            --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue" \
            --output text)
          aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --s3-bucket "${{ vars.DEPLOY_BUCKET }}" \
            --s3-key "${{ env.S3_KEY }}" \
            --publish

      - name: Wait for Lambda update
        run: |
          FUNCTION_NAME=$(aws cloudformation describe-stacks \
            --stack-name "${{ vars.STACK_NAME }}" \
            --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue" \
            --output text)
          aws lambda wait function-updated --function-name "$FUNCTION_NAME"

      - name: Show stack outputs
        run: |
          aws cloudformation describe-stacks \
            --stack-name "${{ vars.STACK_NAME }}" \
            --query "Stacks[0].Outputs" \
            --output table
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: GitHub Actions OIDC deploy workflow"
```

---

## Task 7: Examples

**Files:**
- Create: `examples/request.json`
- Create: `examples/curl-chat.sh`

- [ ] **Step 1: Write examples/request.json**

```json
{
  "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "messages": [
    {"role": "system", "content": "You are concise. Answer in one sentence."},
    {"role": "user", "content": "Say hello and tell me what you are."}
  ],
  "max_tokens": 200,
  "temperature": 0.2
}
```

- [ ] **Step 2: Write examples/curl-chat.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL to your InferBedrock endpoint, e.g. https://abc123.execute-api.us-east-1.amazonaws.com}"
: "${ADAPTER_API_KEY:?Set ADAPTER_API_KEY}"

curl -fsSL -X POST "${API_URL}/v1/chat/completions" \
  -H "content-type: application/json" \
  -H "x-api-key: ${ADAPTER_API_KEY}" \
  -d @"$(dirname "$0")/request.json" | jq .
```

- [ ] **Step 3: Make script executable**

```bash
chmod +x examples/curl-chat.sh
```

- [ ] **Step 4: Commit**

```bash
git add examples/
git commit -m "chore: example request and curl script"
```

---

## Task 8: Scripts

**Files:**
- Create: `scripts/deploy-local.sh`
- Create: `scripts/smoke-test.sh`

- [ ] **Step 1: Write scripts/deploy-local.sh**

```bash
#!/usr/bin/env bash
# Manual deploy without GitHub Actions. Requires AWS CLI configured with deploy permissions.
set -euo pipefail

: "${DEPLOY_BUCKET:?Set DEPLOY_BUCKET to your S3 artifacts bucket}"
: "${STACK_NAME:=infer-bedrock}"
: "${AWS_REGION:=us-east-1}"
: "${BEDROCK_REGION:=us-east-1}"
: "${ALLOWED_MODELS:=anthropic.claude-3-5-sonnet-20240620-v1:0}"
: "${MAX_TOKENS_LIMIT:=2000}"
: "${ADAPTER_API_KEY:?Set ADAPTER_API_KEY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_KEY="infer-bedrock/local-$(date +%Y%m%d%H%M%S)/lambda.zip"

echo "==> Packaging Lambda..."
cd "${ROOT}/src"
zip -j "${ROOT}/lambda.zip" app.py
cd "${ROOT}"

echo "==> Uploading to s3://${DEPLOY_BUCKET}/${S3_KEY}"
aws s3 cp lambda.zip "s3://${DEPLOY_BUCKET}/${S3_KEY}" --region "${AWS_REGION}"

echo "==> Deploying CloudFormation stack: ${STACK_NAME}"
aws cloudformation deploy \
  --template-file "${ROOT}/infra/template.yaml" \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=infer-bedrock \
    BedrockRegion="${BEDROCK_REGION}" \
    AllowedModels="${ALLOWED_MODELS}" \
    MaxTokensLimit="${MAX_TOKENS_LIMIT}" \
    AdapterApiKey="${ADAPTER_API_KEY}" \
    LambdaS3Bucket="${DEPLOY_BUCKET}" \
    LambdaS3Key="${S3_KEY}" \
  --no-fail-on-empty-changeset

FUNCTION_NAME=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue" \
  --output text)

echo "==> Updating Lambda code..."
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --s3-bucket "${DEPLOY_BUCKET}" \
  --s3-key "${S3_KEY}" \
  --region "${AWS_REGION}" \
  --publish

aws lambda wait function-updated --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"

echo "==> Stack outputs:"
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs" --output table

rm -f "${ROOT}/lambda.zip"
echo "==> Done."
```

- [ ] **Step 2: Write scripts/smoke-test.sh**

```bash
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
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/deploy-local.sh scripts/smoke-test.sh
```

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "chore: deploy-local and smoke-test scripts"
```

---

## Task 9: Docs

**Files:**
- Create: `docs/SECURITY.md`
- Create: `docs/CONFIGURATION.md`
- Create: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Write docs/SECURITY.md**

```markdown
# Security Model

## Authentication

InferBedrock uses a static API key enforced inside Lambda (not at API Gateway level — HTTP APIs do not support native API key/usage plans).

- Header: `x-api-key`
- Configured via: `ADAPTER_API_KEY` environment variable / `AdapterApiKey` CloudFormation parameter
- If `ADAPTER_API_KEY` is empty, all requests are accepted without a key. **Never deploy without a key in production.**

## Model Allowlist

Requests are rejected with HTTP 400 if the requested model is not in `ALLOWED_MODELS`. This prevents calls to unintended or expensive models.

## Token Ceiling

`MAX_TOKENS_LIMIT` caps the `max_tokens` value per request. Requests above the ceiling are rejected with HTTP 400.

## IAM Least Privilege

The Lambda execution role is granted only:
- `bedrock:InvokeModel` and `bedrock:Converse` scoped to `arn:aws:bedrock:REGION::foundation-model/*`
- CloudWatch Logs write access (via `AWSLambdaBasicExecutionRole`)

No S3, DynamoDB, network, or other AWS permissions are granted.

## Prompt Logging

Prompt content is never logged by default. Set `LOG_PROMPTS=true` to enable truncated/redacted logging for debugging only. Never enable in production.

## CORS

CORS is disabled by default. Enable only if you need browser-based access by setting `CORS_ENABLED=true`. When enabled, `access-control-allow-origin: *` is added to responses — restrict this further if needed.

## GitHub Actions

GitHub Actions deploys using short-lived OIDC credentials. No long-lived AWS keys are stored in GitHub.

The IAM role trusted by GitHub Actions is scoped to:
- Repository: `r33n3/infer-bedrock`
- Branch: `refs/heads/main`

## Request Size

Requests are hard-capped at 64 KB in Lambda code regardless of API Gateway settings.
```

- [ ] **Step 2: Write docs/CONFIGURATION.md**

```markdown
# Configuration Reference

## CloudFormation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ProjectName` | `infer-bedrock` | Resource name prefix |
| `BedrockRegion` | `us-east-1` | AWS region for Bedrock API calls |
| `AllowedModels` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Comma-separated allowed model IDs |
| `DefaultModel` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Documentation reference only |
| `MaxTokensLimit` | `2000` | Hard ceiling on max_tokens per request |
| `AdapterApiKey` | _(empty)_ | API key for x-api-key header auth. Empty = no auth. |
| `LogPrompts` | `false` | Enable truncated prompt logging (`true`/`false`) |
| `CorsEnabled` | `false` | Add CORS headers to responses (`true`/`false`) |
| `LambdaS3Bucket` | _(required)_ | S3 bucket for Lambda deployment zip |
| `LambdaS3Key` | _(required)_ | S3 key for Lambda deployment zip |

## Lambda Environment Variables

These map 1:1 to CloudFormation parameters and are set automatically during deploy.

| Variable | Description |
|----------|-------------|
| `ALLOWED_MODELS` | Comma-separated model allowlist |
| `MAX_TOKENS_LIMIT` | Token ceiling |
| `ADAPTER_API_KEY` | API key value |
| `LOG_PROMPTS` | `true`/`false` |
| `BEDROCK_REGION` | Bedrock client region |
| `CORS_ENABLED` | `true`/`false` |

## GitHub Actions Variables and Secrets

### Variables (non-sensitive, set in repo Settings > Variables > Actions)

| Variable | Example | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | Region for CloudFormation deploy |
| `STACK_NAME` | `infer-bedrock` | CloudFormation stack name |
| `BEDROCK_REGION` | `us-east-1` | Bedrock region |
| `ALLOWED_MODELS` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Allowed models |
| `DEFAULT_MODEL` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Default model |
| `MAX_TOKENS_LIMIT` | `2000` | Token ceiling |
| `DEPLOY_BUCKET` | `infer-bedrock-deploy-us-east-1` | S3 bucket for Lambda artifacts |

### Secrets (sensitive, set in repo Settings > Secrets > Actions)

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_TO_ASSUME` | ARN of the GitHub Actions IAM role |
| `ADAPTER_API_KEY` | API key for the endpoint |
```

- [ ] **Step 3: Write docs/ARCHITECTURE.md**

```markdown
# Architecture

## Overview

```
Local WTK / LiteLLM / agents
         ↓  HTTPS + x-api-key
  InferBedrock API Gateway (HTTP API)
         ↓  Lambda proxy integration
  infer-bedrock-adapter (Lambda, Python 3.12)
         ↓  boto3 / IAM execution role
  Amazon Bedrock Runtime (Converse API)
         ↓
  Foundation Model (Claude, etc.)
```

## Components

### API Gateway HTTP API
- Protocol: HTTP (v2), not REST
- Routes: `GET /health`, `POST /v1/chat/completions`
- Integration: Lambda proxy, payload format version 2.0
- Auth: none at gateway level — delegated to Lambda

### Lambda Function
- Runtime: Python 3.12
- Handler: `app.lambda_handler`
- Memory: 512 MB, Timeout: 30s
- Responsibilities: API key check, request validation, message conversion, Bedrock call, response shaping

### IAM Role
- Least-privilege: `bedrock:Converse` + `bedrock:InvokeModel` on `foundation-model/*`
- CloudWatch Logs write access only

### Amazon Bedrock Converse API
- Used for all chat-style requests
- Provides a consistent interface across supported models
- `InvokeModel` permission is included for future compatibility but not currently called

## Request Flow

1. Client sends `POST /v1/chat/completions` with `x-api-key` header
2. API Gateway forwards full event to Lambda (payload format 2.0)
3. Lambda validates API key, JSON body, model, messages, and token ceiling
4. Lambda converts OpenAI-style messages to Bedrock Converse format (system blocks separated)
5. Lambda calls `bedrock:Converse` using its IAM execution role
6. Lambda maps Bedrock response to OpenAI-compatible shape and returns to API Gateway
7. API Gateway returns Lambda's response directly to client

## Design Decisions

**HTTP API over REST API:** HTTP APIs are lighter, cheaper (~70% cost reduction vs REST API), and natively support Lambda proxy with v2 payload. The trade-off is no native API key/usage plans — API key enforcement is in Lambda code instead.

**Converse over InvokeModel:** Converse provides a model-agnostic chat interface. InvokeModel requires model-specific request/response shapes. We keep `bedrock:InvokeModel` in IAM for future extensibility.

**No VPC:** Lambda runs without a VPC. Bedrock Runtime is accessed via public endpoint using the Lambda execution role. A VPC would add NAT Gateway cost and complexity without security benefit for this use case.

**S3 for Lambda artifacts:** Lambda code is packaged as a zip and stored in S3. CloudFormation references the S3 location. This is required for code > 4 KB (the ZipFile inline limit).

## Dark Factory / PeaRL Positioning

InferBedrock is a **provider bridge**, not a governance layer.

```
WTK / local agent tooling
         ↓
  PeaRL / local policy gateway  ← governance lives here
         ↓
  InferBedrock  ← provider bridge
         ↓
  AWS Bedrock
```

PeaRL or a local gateway should decide which agents are permitted to call InferBedrock. InferBedrock enforces only: valid API key, allowed model, token ceiling.
```

- [ ] **Step 4: Commit**

```bash
git add docs/SECURITY.md docs/CONFIGURATION.md docs/ARCHITECTURE.md
git commit -m "docs: security, configuration, and architecture documentation"
```

---

## Task 10: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md** (see full content in Task 10 execution)

Key sections:
- What this is / what this is not
- Architecture diagram
- One-time AWS bootstrap (OIDC + IAM role + S3 bucket)
- GitHub Actions setup (variables + secrets)
- Deploy
- Local smoke test
- LiteLLM integration
- Cost notes
- Troubleshooting

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with bootstrap and usage instructions"
```

---

## Task 11: GitHub repo creation and push

**Files:** none

- [ ] **Step 1: Create GitHub repo**

```bash
gh repo create r33n3/infer-bedrock \
  --public \
  --description "Lightweight AWS Bedrock inference adapter — HTTPS endpoint for local AI tooling" \
  --source . \
  --remote origin \
  --push
```

- [ ] **Step 2: Verify default branch is main**

```bash
gh repo view r33n3/infer-bedrock --json defaultBranchRef --jq '.defaultBranchRef.name'
```

Expected: `main`

- [ ] **Step 3: Protect main branch (optional but recommended)**

```bash
gh api repos/r33n3/infer-bedrock/branches/main/protection \
  --method PUT \
  --field required_status_checks=null \
  --field enforce_admins=false \
  --field required_pull_request_reviews=null \
  --field restrictions=null
```

---

## AWS Prerequisites Checklist

Before the first GitHub Actions deploy will succeed, the following must exist in AWS:

### 1. GitHub OIDC Provider
In IAM > Identity Providers, if not already present:
- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

### 2. GitHub Actions IAM Role
Create an IAM role with this trust policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:r33n3/infer-bedrock:ref:refs/heads/main"}
    }
  }]
}
```

Attach a policy allowing:
- `cloudformation:*` on stack `arn:aws:cloudformation:REGION:ACCOUNT:stack/infer-bedrock/*`
- `lambda:*` on `arn:aws:lambda:REGION:ACCOUNT:function:infer-bedrock-*`
- `apigateway:*` on the API Gateway resources
- `iam:CreateRole`, `iam:DeleteRole`, `iam:AttachRolePolicy`, `iam:DetachRolePolicy`, `iam:PutRolePolicy`, `iam:DeleteRolePolicy`, `iam:GetRole`, `iam:PassRole` on `arn:aws:iam::ACCOUNT:role/infer-bedrock-*`
- `logs:CreateLogGroup`, `logs:PutRetentionPolicy`, `logs:DeleteLogGroup` on the log group
- `s3:GetObject`, `s3:PutObject` on `arn:aws:s3:::DEPLOY_BUCKET/*`
- `s3:GetBucketLocation` on `arn:aws:s3:::DEPLOY_BUCKET`

### 3. S3 Deploy Bucket
Create an S3 bucket in the same region as your deploy:
```bash
aws s3 mb s3://infer-bedrock-deploy-ACCOUNT_ID --region us-east-1
```

### 4. Bedrock Model Access
In the Bedrock console (us-east-1), request access to the models in your `ALLOWED_MODELS` list if not already enabled.

### 5. GitHub Repo Variables and Secrets
Variables (Settings > Secrets and variables > Actions > Variables):
- `AWS_REGION` = `us-east-1`
- `STACK_NAME` = `infer-bedrock`
- `BEDROCK_REGION` = `us-east-1`
- `ALLOWED_MODELS` = `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `DEFAULT_MODEL` = `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `MAX_TOKENS_LIMIT` = `2000`
- `DEPLOY_BUCKET` = `infer-bedrock-deploy-ACCOUNT_ID`

Secrets (Settings > Secrets and variables > Actions > Secrets):
- `AWS_ROLE_TO_ASSUME` = `arn:aws:iam::ACCOUNT_ID:role/YOUR_GITHUB_ACTIONS_ROLE`
- `ADAPTER_API_KEY` = your chosen API key string
