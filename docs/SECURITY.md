# Security Model

## Authentication

InferBedrock uses a static API key enforced inside Lambda. HTTP APIs do not support native API key/usage plans (that is a REST API feature), so enforcement happens in code.

- Header: `x-api-key`
- Configured via: `ADAPTER_API_KEY` environment variable / `AdapterApiKey` CloudFormation parameter
- If `ADAPTER_API_KEY` is empty, all requests are accepted without a key. **Never deploy without a key in production.**

## Model Allowlist

Requests are rejected with HTTP 400 if the requested model is not in `ALLOWED_MODELS`. This prevents calls to unintended or expensive models regardless of what the client sends.

## Token Ceiling

`MAX_TOKENS_LIMIT` caps the `max_tokens` value per request. Requests above the ceiling are rejected with HTTP 400 before any Bedrock call is made.

## IAM Least Privilege

The Lambda execution role is granted only:
- `bedrock:InvokeModel` and `bedrock:Converse` scoped to `arn:aws:bedrock:REGION::foundation-model/*`
- CloudWatch Logs write access (via `AWSLambdaBasicExecutionRole`)

No S3, DynamoDB, VPC, network, or other AWS permissions are granted.

Bedrock model ARN wildcard rationale: foundation model ARNs require a wildcard on the model ID because maintaining a per-model ARN list would require synchronized IAM updates for every model allowlist change. The `ALLOWED_MODELS` env var is the effective enforcement layer; IAM scopes the action to foundation models only.

## Prompt Logging

Prompt content is never logged by default. Set `LOG_PROMPTS=true` to enable truncated logging (first 120 characters of the first message) for debugging. Never enable in production environments where prompt content is sensitive.

## CORS

CORS is disabled by default. Enable only if you need browser-based access by setting `CORS_ENABLED=true`. When enabled, `access-control-allow-origin: *` is added — restrict this further if your deployment requires it.

## Request Size

Requests are hard-capped at 64 KB in Lambda code regardless of API Gateway limits.

## GitHub Actions

GitHub Actions deploys using short-lived OIDC credentials. No long-lived AWS keys are stored in GitHub.

The IAM role trusted by GitHub Actions is scoped to:
- Repository: `r33n3/infer-bedrock`
- Branch: `refs/heads/main`

Pushes from other branches or forks cannot assume the deploy role.
