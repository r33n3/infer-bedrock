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
