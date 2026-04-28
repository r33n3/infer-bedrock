# InferBedrock

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Lightweight AWS-hosted Bedrock inference adapter. Exposes a narrow HTTPS endpoint for local AI tooling without requiring AWS credentials on the client.

## What This Is

InferBedrock is a **provider bridge**: it accepts OpenAI-compatible chat requests over HTTPS and forwards them to Amazon Bedrock using Lambda's IAM execution role. Your local tools never touch AWS credentials.

```
Local tooling / LiteLLM / AI agents
         |  HTTPS + x-api-key
         v
  InferBedrock (API Gateway HTTP API)
         |
         v
  Lambda (Python 3.12)
         |  IAM role
         v
  Amazon Bedrock Runtime
```

## What This Is Not

- Not a governance layer. Add a policy gateway in front of it if you need to control which callers may reach this endpoint.
- Not a full OpenAI proxy. The response shape is OpenAI-compatible but only covers the fields InferBedrock maps from Bedrock.
- Not LiteLLM, not ECS, not a VPC deployment. It is intentionally minimal.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — no auth required |
| `POST` | `/v1/chat/completions` | Chat inference via Bedrock Converse |

## One-Time AWS Bootstrap

Do this once per AWS account before the first deploy.

### 1. GitHub OIDC Provider

In IAM > Identity Providers, create if not already present:
- Provider type: OpenID Connect
- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

### 2. GitHub Actions IAM Role

Create an IAM role with this trust policy (replace `ACCOUNT_ID`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main"
      }
    }
  }]
}
```

Attach an inline policy allowing (replace `ACCOUNT_ID`, `REGION`, `DEPLOY_BUCKET`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["cloudformation:*"],
      "Resource": "arn:aws:cloudformation:REGION:ACCOUNT_ID:stack/infer-bedrock/*"
    },
    {
      "Effect": "Allow",
      "Action": ["cloudformation:ListStacks", "cloudformation:ValidateTemplate"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["lambda:*"],
      "Resource": "arn:aws:lambda:REGION:ACCOUNT_ID:function:infer-bedrock-*"
    },
    {
      "Effect": "Allow",
      "Action": ["apigateway:*"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole", "iam:DeleteRole",
        "iam:AttachRolePolicy", "iam:DetachRolePolicy",
        "iam:PutRolePolicy", "iam:DeleteRolePolicy",
        "iam:GetRole", "iam:GetRolePolicy", "iam:PassRole",
        "iam:ListRolePolicies", "iam:ListAttachedRolePolicies"
      ],
      "Resource": "arn:aws:iam::ACCOUNT_ID:role/infer-bedrock-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup", "logs:DeleteLogGroup",
        "logs:PutRetentionPolicy", "logs:DescribeLogGroups"
      ],
      "Resource": "arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/lambda/infer-bedrock-*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::DEPLOY_BUCKET/infer-bedrock/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
      "Resource": "arn:aws:s3:::DEPLOY_BUCKET"
    }
  ]
}
```

### 3. S3 Deploy Bucket

Create an S3 bucket in the same region as your deploy:

```bash
aws s3 mb s3://infer-bedrock-deploy-ACCOUNT_ID --region us-east-1
```

### 4. Bedrock Model Access

In the Bedrock console, go to **Model access** and request access to the models you plan to use. For Claude models this requires accepting the terms of service.

## GitHub Actions Setup

### Variables (Settings > Secrets and variables > Actions > Variables)

| Variable | Example value |
|----------|---------------|
| `AWS_REGION` | `us-east-1` |
| `STACK_NAME` | `infer-bedrock` |
| `BEDROCK_REGION` | `us-east-1` |
| `ALLOWED_MODELS` | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| `DEFAULT_MODEL` | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| `MAX_TOKENS_LIMIT` | `2000` |
| `DEPLOY_BUCKET` | `infer-bedrock-deploy-123456789012` |

### Secrets (Settings > Secrets and variables > Actions > Secrets)

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_TO_ASSUME` | ARN of the GitHub Actions role you created above |
| `ADAPTER_API_KEY` | API key clients must send in `x-api-key` header |

## Deploy

Push to `main` — GitHub Actions deploys automatically.

To deploy manually:

```bash
gh workflow run deploy.yml
```

Or use the local script:

```bash
export DEPLOY_BUCKET=infer-bedrock-deploy-123456789012
export ADAPTER_API_KEY=your-secret-key
bash scripts/deploy-local.sh
```

After deploy, find your endpoint in the CloudFormation stack outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name infer-bedrock \
  --query "Stacks[0].Outputs" \
  --output table
```

## Usage

### Health check

```bash
curl https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/health
```

Response:
```json
{"status": "ok", "service": "infer-bedrock"}
```

### Chat completion

```bash
export API_URL=https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com
export ADAPTER_API_KEY=your-secret-key

curl -X POST "${API_URL}/v1/chat/completions" \
  -H "content-type: application/json" \
  -H "x-api-key: ${ADAPTER_API_KEY}" \
  -d @examples/request.json | jq .
```

Or use the example script:

```bash
API_URL=https://... ADAPTER_API_KEY=... bash examples/curl-chat.sh
```

### Smoke test after deploy

```bash
API_URL=https://... ADAPTER_API_KEY=... bash scripts/smoke-test.sh
```

## LiteLLM Integration

Add InferBedrock as a custom OpenAI-compatible provider in your LiteLLM config:

```yaml
model_list:
  - model_name: infer-bedrock/claude-3-5-sonnet
    litellm_params:
      model: openai/anthropic.claude-3-5-sonnet-20240620-v1:0
      api_base: https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com
      api_key: your-adapter-api-key
```

InferBedrock does not use the `api_key` for AWS auth — it is the adapter key for InferBedrock itself. LiteLLM sends it as `Authorization: Bearer`, which InferBedrock accepts (it checks both `x-api-key` and `Authorization: Bearer`).

For direct header control:

```python
import openai

client = openai.OpenAI(
    base_url="https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com",
    api_key="unused",
    default_headers={"x-api-key": "your-adapter-api-key"},
)
response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

## Cost Notes

Approximate costs for light usage (us-east-1, 2025 pricing):

| Component | Cost |
|-----------|------|
| API Gateway HTTP API | $1.00 / million requests |
| Lambda | ~$0 for low volume (512 MB, 30s max) |
| Bedrock Claude 3.5 Sonnet | ~$3 / million input tokens, ~$15 / million output tokens |
| CloudWatch Logs (7-day retention) | Minimal |
| S3 (deploy artifacts) | Negligible |

There is no always-on compute cost. You pay per request.

## Troubleshooting

**401 Unauthorized** — Set `x-api-key` header to your `ADAPTER_API_KEY` value.

**400 disallowed_model** — The model ID you sent is not in `ALLOWED_MODELS`. Check the CloudFormation parameter or Lambda env var.

**400 max_tokens_exceeded** — Lower `max_tokens` or increase `MAX_TOKENS_LIMIT` in the stack parameters.

**502 bedrock_access** — Lambda cannot call Bedrock. Check: (1) model access is enabled in the Bedrock console, (2) `BEDROCK_REGION` matches the region where you enabled model access, (3) the Lambda IAM role has `bedrock:Converse` on `foundation-model/*`.

**GitHub Actions fails: "Could not assume role"** — The OIDC provider is missing, or the trust policy does not match the repo/branch. Verify the `sub` condition matches `repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main`.

**CloudFormation fails: "already exists"** — A stack named `infer-bedrock` exists in a bad state. Check the CloudFormation console for the failure reason and delete the stack before redeploying.

## Security

See [docs/SECURITY.md](docs/SECURITY.md).

## Configuration Reference

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

[MIT](LICENSE)
