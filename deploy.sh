#!/bin/bash
set -e

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Warehouse Optimizer Backend Deployment ===${NC}"

# 環境変数
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID="904368995258"
ECR_REPOSITORY="warehouse-optimizer-backend"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
IMAGE_TAG="latest"

# Docker が起動しているか確認
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Starting Docker Desktop..."
    open -a Docker
    echo "Waiting for Docker to start..."
    sleep 10
    
    # 再確認
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Failed to start Docker${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# buildx が使用可能か確認（マルチアーキテクチャビルド用）
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${RED}Error: docker buildx is not available${NC}"
    exit 1
fi

# ビルダーインスタンスの確認・作成
BUILDER_NAME="multi"
if ! docker buildx inspect ${BUILDER_NAME} > /dev/null 2>&1; then
    echo -e "${YELLOW}Creating buildx builder instance...${NC}"
    docker buildx create --name ${BUILDER_NAME} --use --platform linux/amd64,linux/arm64
else
    echo -e "${GREEN}✓ Using existing buildx builder: ${BUILDER_NAME}${NC}"
    docker buildx use ${BUILDER_NAME}
fi

# ECR にログイン
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_URI}

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to login to ECR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Logged in to ECR${NC}"

# マルチアーキテクチャビルド＆プッシュ
echo -e "${YELLOW}Building and pushing multi-architecture Docker image...${NC}"
echo -e "${YELLOW}Platforms: linux/amd64, linux/arm64${NC}"

cd backend

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag ${ECR_URI}:${IMAGE_TAG} \
    --tag ${ECR_URI}:$(date +%Y%m%d-%H%M%S) \
    --push \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Docker build and push failed${NC}"
    exit 1
fi

cd ..

echo -e "${GREEN}✓ Multi-architecture image pushed to ECR${NC}"
echo -e "${GREEN}Image: ${ECR_URI}:${IMAGE_TAG}${NC}"

# App Runner のデプロイ状況確認
echo -e "${YELLOW}Checking App Runner deployment status...${NC}"
APP_RUNNER_ARN="arn:aws:apprunner:${AWS_REGION}:${AWS_ACCOUNT_ID}:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be"

STATUS=$(aws apprunner describe-service \
    --service-arn ${APP_RUNNER_ARN} \
    --region ${AWS_REGION} \
    --query 'Service.Status' \
    --output text 2>/dev/null || echo "UNKNOWN")

echo -e "${GREEN}App Runner Status: ${STATUS}${NC}"

if [ "$STATUS" != "UNKNOWN" ]; then
    echo -e "${YELLOW}App Runner will automatically deploy the new image in 3-5 minutes.${NC}"
    echo ""
    echo "Monitor deployment:"
    echo "  aws apprunner list-operations \\"
    echo "    --service-arn ${APP_RUNNER_ARN} \\"
    echo "    --region ${AWS_REGION} \\"
    echo "    --max-results 3"
else
    echo -e "${YELLOW}Note: Could not retrieve App Runner status. Check AWS Console.${NC}"
fi

# ECS Celery ワーカーの再デプロイ
echo ""
echo -e "${YELLOW}Forcing ECS Celery worker redeployment...${NC}"
ECS_CLUSTER="warehouse-optimizer-celery"
ECS_SERVICE="warehouse-optimizer-celery-worker"

ECS_UPDATE=$(aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION} \
    --query 'service.status' \
    --output text 2>/dev/null || echo "FAILED")

if [ "$ECS_UPDATE" = "ACTIVE" ]; then
    echo -e "${GREEN}✓ ECS Celery worker redeployment triggered${NC}"
else
    echo -e "${YELLOW}Warning: ECS Celery worker redeployment returned: ${ECS_UPDATE}${NC}"
fi

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Wait 3-5 minutes for App Runner to deploy"
echo "  2. Check health: curl https://api.wh-optim.mnyk.me/health"
echo "  3. Deploy frontend: cd frontend && npx vercel --prod --yes"
