#!/usr/bin/env bash
# Creates the infer-bedrock-cli-admin IAM user and generates access keys.
# Run this once with root or AdministratorAccess credentials.
set -euo pipefail

ACCOUNT_ID="111526027101"
USER_NAME="infer-bedrock-cli-admin"
POLICY_NAME="InferBedrockCliAdmin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Creating IAM user: ${USER_NAME}"
aws iam create-user --user-name "${USER_NAME}" 2>&1 || {
  echo "    User may already exist — continuing"
}

echo "==> Creating managed policy: ${POLICY_NAME}"
POLICY_ARN=$(aws iam create-policy \
  --policy-name "${POLICY_NAME}" \
  --policy-document "file://${ROOT}/infra/iam/cli-admin-policy.json" \
  --description "InferBedrock CLI admin: deploy, bootstrap, and debug" \
  --query "Policy.Arn" --output text 2>&1) || {
    echo "    Policy may already exist — fetching ARN"
    POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
  }
echo "    Policy ARN: ${POLICY_ARN}"

echo "==> Attaching managed policy to user"
aws iam attach-user-policy \
  --user-name "${USER_NAME}" \
  --policy-arn "${POLICY_ARN}"
echo "    Policy attached."

echo "==> Creating access key"
KEYS=$(aws iam create-access-key --user-name "${USER_NAME}")
ACCESS_KEY_ID=$(echo "${KEYS}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['AccessKey']['AccessKeyId'])")
SECRET_KEY=$(echo "${KEYS}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['AccessKey']['SecretAccessKey'])")

echo ""
echo "======================================================="
echo " IAM user created. Configure AWS CLI with these keys:"
echo "======================================================="
echo ""
echo "  aws configure --profile infer-bedrock"
echo ""
echo "  AWS Access Key ID     : ${ACCESS_KEY_ID}"
echo "  AWS Secret Access Key : ${SECRET_KEY}"
echo "  Default region        : us-east-1"
echo "  Default output format : json"
echo ""
echo "  Or set env vars:"
echo "  export AWS_ACCESS_KEY_ID=${ACCESS_KEY_ID}"
echo "  export AWS_SECRET_ACCESS_KEY=${SECRET_KEY}"
echo "  export AWS_DEFAULT_REGION=us-east-1"
echo ""
echo "  IMPORTANT: Save the Secret Access Key now — it is not retrievable again."
echo ""
