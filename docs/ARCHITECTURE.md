# Architecture

## Overview

```
Local tooling / LiteLLM / AI agents
         |  HTTPS + x-api-key
         v
  InferBedrock API Gateway (HTTP API)
         |  Lambda proxy integration (payload v2.0)
         v
  infer-bedrock-adapter  (Lambda, Python 3.12)
         |  boto3 / IAM execution role
         v
  Amazon Bedrock Runtime  (Converse API)
         |
         v
  Foundation Model (Claude, Llama, Titan, etc.)
```

## Components

### API Gateway HTTP API
- Protocol: HTTP v2 (not REST API)
- Routes: `GET /health`, `POST /v1/chat/completions`
- Integration: Lambda proxy, payload format version 2.0
- Auth: none at gateway level — delegated to Lambda
- Stage: `$default` with AutoDeploy enabled

### Lambda Function (`infer-bedrock-adapter`)
- Runtime: Python 3.12, Memory: 512 MB, Timeout: 30s
- Handler: `app.lambda_handler`
- Responsibilities: API key validation, request validation, message conversion, Bedrock call, response shaping

### IAM Role (`infer-bedrock-lambda-role`)
- `bedrock:InvokeModel` + `bedrock:Converse` on `arn:aws:bedrock:REGION::foundation-model/*`
- CloudWatch Logs write (via `AWSLambdaBasicExecutionRole`)
- No other permissions

### Amazon Bedrock Converse API
- Used for all chat requests
- Provides a model-agnostic interface across supported models
- `InvokeModel` IAM permission included for future compatibility

## Request Flow

```
1. Client → POST /v1/chat/completions + x-api-key header
2. API Gateway → Lambda (full event, payload format 2.0)
3. Lambda: validate API key
4. Lambda: validate JSON body, model allowlist, token ceiling
5. Lambda: convert OpenAI messages → Bedrock Converse format
           (system messages extracted into system[] param)
6. Lambda → bedrock:Converse (using IAM execution role, no client credentials)
7. Bedrock → response with output message + usage
8. Lambda: map response → OpenAI-compatible shape
9. Lambda → API Gateway → Client
```

## Design Decisions

**HTTP API over REST API**
HTTP APIs are ~70% cheaper than REST APIs and natively support Lambda proxy with v2 payload format. The trade-off is no native API key/usage plans (a REST API feature). API key enforcement is in Lambda instead — equivalent security, simpler setup.

**Converse over InvokeModel**
Converse provides a model-agnostic chat interface. InvokeModel requires model-specific request/response shapes for each provider. Converse is the right default; `bedrock:InvokeModel` is included in IAM for future use.

**No VPC**
Lambda runs without a VPC. Bedrock Runtime is accessed via public endpoint using the Lambda IAM execution role. A VPC would add NAT Gateway cost (~$32/month) and complexity without security benefit for this use case.

**S3 for Lambda artifacts**
Lambda code exceeds the 4 KB inline ZipFile limit. The deployment zip is stored in S3; CloudFormation references the S3 location. GitHub Actions uploads a new zip keyed by `github.sha` on each deploy.

**Model allowlist in Lambda, not IAM**
Foundation model ARNs would require a parallel, synchronized list to scope IAM per-model. The Lambda-level `ALLOWED_MODELS` env var is the enforcement layer — IAM scopes to the `foundation-model/*` resource class.

## Governance Positioning

InferBedrock is a **provider bridge**, not a governance layer. It enforces: valid API key, allowed model list, token ceiling. If you need to control which callers or agents may reach this endpoint, add a policy gateway in front of it.
