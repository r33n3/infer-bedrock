# Configuration Reference

## CloudFormation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ProjectName` | `infer-bedrock` | Resource name prefix used for all AWS resource names |
| `BedrockRegion` | `us-east-1` | AWS region for Bedrock API calls |
| `AllowedModels` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Comma-separated list of allowed Bedrock model IDs |
| `DefaultModel` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Documentation reference; not enforced separately |
| `MaxTokensLimit` | `2000` | Hard ceiling on `max_tokens` per request |
| `AdapterApiKey` | *(empty)* | API key for `x-api-key` header. Empty = no auth. |
| `LogPrompts` | `false` | Enable truncated prompt logging (`true`/`false`) |
| `CorsEnabled` | `false` | Add `access-control-allow-origin: *` to responses |
| `LambdaS3Bucket` | *(required)* | S3 bucket containing the Lambda deployment zip |
| `LambdaS3Key` | *(required)* | S3 key of the Lambda zip |

## Lambda Environment Variables

Set automatically by CloudFormation. These map 1:1 to parameters.

| Variable | Description |
|----------|-------------|
| `ALLOWED_MODELS` | Comma-separated allowlist |
| `MAX_TOKENS_LIMIT` | Token ceiling (integer) |
| `ADAPTER_API_KEY` | API key value |
| `LOG_PROMPTS` | `true` or `false` |
| `BEDROCK_REGION` | Bedrock client region |
| `CORS_ENABLED` | `true` or `false` |

## GitHub Actions Variables

Set in: **Settings > Secrets and variables > Actions > Variables** (non-sensitive)

| Variable | Example | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | Region for CloudFormation deploy |
| `STACK_NAME` | `infer-bedrock` | CloudFormation stack name |
| `BEDROCK_REGION` | `us-east-1` | Bedrock region passed to Lambda |
| `ALLOWED_MODELS` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Allowed model IDs |
| `DEFAULT_MODEL` | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Default model |
| `MAX_TOKENS_LIMIT` | `2000` | Token ceiling |
| `DEPLOY_BUCKET` | `infer-bedrock-deploy-123456789` | S3 bucket for Lambda artifacts |

## GitHub Actions Secrets

Set in: **Settings > Secrets and variables > Actions > Secrets** (sensitive)

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_TO_ASSUME` | ARN of the GitHub Actions IAM role (e.g. `arn:aws:iam::123456789:role/infer-bedrock-github-deploy`) |
| `ADAPTER_API_KEY` | The API key clients must send in `x-api-key` |
