#!/usr/bin/env bash
# One-time AWS bootstrap for InferBedrock.
# Run this once with credentials that have IAM + S3 create permissions.
# Prerequisites: AWS CLI configured (aws configure or env vars), jq installed.
set -euo pipefail

ACCOUNT_ID="111526027101"
REGION="us-east-1"
ROLE_NAME="infer-bedrock-github-deploy"
DEPLOY_BUCKET="infer-bedrock-deploy-${ACCOUNT_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Bootstrap: InferBedrock AWS setup"
echo "    Account : ${ACCOUNT_ID}"
echo "    Region  : ${REGION}"
echo "    Role    : ${ROLE_NAME}"
echo "    Bucket  : ${DEPLOY_BUCKET}"
echo ""

# ── IAM Role ────────────────────────────────────────────────────────────────

echo "==> Creating IAM role: ${ROLE_NAME}"
ROLE_ARN=$(aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document "file://${ROOT}/infra/iam/github-actions-trust-policy.json" \
  --description "GitHub Actions OIDC deploy role for InferBedrock" \
  --query "Role.Arn" \
  --output text 2>&1) || {
    if echo "${ROLE_ARN}" | grep -q "EntityAlreadyExists"; then
      echo "    Role already exists — fetching ARN"
      ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query "Role.Arn" --output text)
    else
      echo "ERROR: ${ROLE_ARN}"
      exit 1
    fi
  }
echo "    Role ARN: ${ROLE_ARN}"

echo "==> Attaching deploy policy to role"
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "InferBedrockDeploy" \
  --policy-document "file://${ROOT}/infra/iam/github-actions-deploy-policy.json"
echo "    Policy attached."

# ── S3 Deploy Bucket ─────────────────────────────────────────────────────────

echo "==> Creating S3 deploy bucket: ${DEPLOY_BUCKET}"
aws s3 mb "s3://${DEPLOY_BUCKET}" --region "${REGION}" 2>&1 || {
  echo "    Bucket may already exist — continuing"
}

echo "==> Blocking public access on deploy bucket"
aws s3api put-public-access-block \
  --bucket "${DEPLOY_BUCKET}" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
echo "    Public access blocked."

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo " Bootstrap complete. Set these in GitHub:"
echo "=========================================="
echo ""
echo "  Secret  AWS_ROLE_TO_ASSUME = ${ROLE_ARN}"
echo "  Variable DEPLOY_BUCKET     = ${DEPLOY_BUCKET}"
echo "  Variable AWS_REGION        = ${REGION}"
echo "  Variable STACK_NAME        = infer-bedrock"
echo "  Variable BEDROCK_REGION    = ${REGION}"
echo "  Variable ALLOWED_MODELS    = anthropic.claude-3-5-sonnet-20240620-v1:0"
echo "  Variable DEFAULT_MODEL     = anthropic.claude-3-5-sonnet-20240620-v1:0"
echo "  Variable MAX_TOKENS_LIMIT  = 2000"
echo "  Secret  ADAPTER_API_KEY    = <choose a strong random string>"
echo ""
