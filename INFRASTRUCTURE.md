# Warehouse Optimizer Infrastructure Documentation

最終更新: 2026年1月21日

## 概要

倉庫ロケーション最適化システムのAWSインフラストラクチャ構成。

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Internet                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐  ┌───────────────┐  ┌──────────────────┐
│     Vercel      │  │  App Runner   │  │  App Runner      │
│   (Frontend)    │  │   (Backend)   │  │  Auto Deploy     │
│ Next.js 15.5.9  │  │  FastAPI      │  │  from ECR        │
└────────┬────────┘  └───────┬───────┘  └──────────────────┘
         │                   │
         │                   │ VPC Connector
         │                   ▼
         │     ┌─────────────────────────────────────────┐
         │     │           VPC (10.0.0.0/16)             │
         │     │  ┌─────────────────────────────────┐   │
         │     │  │      Private Subnet              │   │
         │     │  │                                  │   │
         │     │  │  ┌─────────────┐  ┌───────────┐ │   │
         └─────┼──┼─▶│ ElastiCache │  │   RDS     │ │   │
               │  │  │   Redis     │  │  Aurora   │ │   │
               │  │  │  (Broker)   │  │PostgreSQL │ │   │
               │  │  └──────┬──────┘  └─────┬─────┘ │   │
               │  │         │               │       │   │
               │  │         ▼               │       │   │
               │  │  ┌─────────────┐        │       │   │
               │  │  │ECS Fargate  │◀───────┘       │   │
               │  │  │Celery Worker│                │   │
               │  │  └─────────────┘                │   │
               │  └─────────────────────────────────┘   │
               └─────────────────────────────────────────┘
```

---

## AWS リソース詳細

### 1. App Runner (Backend API)

| 項目 | 値 |
|------|-----|
| Service Name | `warehouse-optimizer-backend` |
| Service URL | `https://cdcdwjbjms.ap-northeast-1.awsapprunner.com` |
| Region | `ap-northeast-1` (Tokyo) |
| Image Source | ECR (Auto Deploy enabled) |
| Image | `904368995258.dkr.ecr.ap-northeast-1.amazonaws.com/warehouse-optimizer-backend:latest` |
| CPU | 1024 (1 vCPU) |
| Memory | 2048 MB |
| Port | 8000 |

**Health Check:**
| 項目 | 値 |
|------|-----|
| Protocol | HTTP |
| Path | `/health` |
| Interval | 10 seconds |
| Timeout | 5 seconds |
| Healthy Threshold | 1 |
| Unhealthy Threshold | 5 |

**環境変数:**
```
DATABASE_URL=postgresql+psycopg2://appuser:***@warehouse-optimizer-backend-pg-cluster.cluster-clgcso6ya7jn.ap-northeast-1.rds.amazonaws.com:5432/warehouse
CELERY_BROKER_URL=redis://warehouse-optimizer-redis.lioy0n.0001.apne1.cache.amazonaws.com:6379/0
CELERY_RESULT_BACKEND=redis://warehouse-optimizer-redis.lioy0n.0001.apne1.cache.amazonaws.com:6379/1
REDIS_URL=redis://warehouse-optimizer-redis.lioy0n.0001.apne1.cache.amazonaws.com:6379/0
ENV=production
```

---

### 2. ECS Fargate (Celery Worker)

| 項目 | 値 |
|------|-----|
| Cluster | `warehouse-optimizer-celery` |
| Service | `warehouse-optimizer-celery-worker` |
| Task Definition | `warehouse-optimizer-celery-worker:2` |
| Launch Type | FARGATE |
| Platform Version | LATEST (1.4.0) |
| Desired Count | 1 |
| Running Count | 1 |
| CPU | 1024 (1 vCPU) |
| Memory | 2048 MB |

**Celery Worker Command:**
```bash
celery -A app.core.celery_app worker -Q relocation --loglevel=info --concurrency=2
```

**タスクのタイムアウト設定:**
- Hard Timeout: 10分 (600秒) - 強制終了
- Soft Timeout: 9分 (540秒) - 例外発生

**環境変数:** App Runnerと同一

---

### 3. ElastiCache (Redis)

| 項目 | 値 |
|------|-----|
| Cluster ID | `warehouse-optimizer-redis` |
| Engine | Redis |
| Engine Version | 7.1.0 |
| Node Type | `cache.t3.micro` |
| Number of Nodes | 1 |
| Endpoint | `warehouse-optimizer-redis.lioy0n.0001.apne1.cache.amazonaws.com:6379` |

**Redis DB 使い分け:**
- DB 0: Celery Broker (タスクキュー)
- DB 1: Celery Result Backend (タスク結果)

**カスタムキー:**
- `relocation:status:{trace_id}` - タスクステータス (TTL: 1時間)
- `relocation:result:{trace_id}` - タスク結果 (TTL: 2時間)
- `relocation:active_task` - 現在アクティブなタスクID (TTL: 30分)

---

### 4. RDS Aurora PostgreSQL

| 項目 | 値 |
|------|-----|
| Cluster ID | `warehouse-optimizer-backend-pg-cluster` |
| Engine | Aurora PostgreSQL |
| Engine Version | 15.13 |
| Endpoint | `warehouse-optimizer-backend-pg-cluster.cluster-clgcso6ya7jn.ap-northeast-1.rds.amazonaws.com` |
| Port | 5432 |
| Database Name | `warehouse` |
| Username | `appuser` |

---

### 5. ECR (Container Registry)

| 項目 | 値 |
|------|-----|
| Repository Name | `warehouse-optimizer-backend` |
| Repository URI | `904368995258.dkr.ecr.ap-northeast-1.amazonaws.com/warehouse-optimizer-backend` |

---

### 6. VPC & Networking

| 項目 | 値 |
|------|-----|
| VPC ID | `vpc-0e28bbd2a069a17e5` |
| VPC Name | `warehouse-optimizer-backend-vpc` |
| CIDR Block | `10.0.0.0/16` |

**Security Groups:**
- App Runner VPC Connector
- ECS Tasks
- ElastiCache
- RDS

---

### 7. CloudWatch Logs

| Log Group | 用途 |
|-----------|------|
| `/ecs/warehouse-optimizer-celery` | Celery Worker ログ |
| `/aws/apprunner/warehouse-optimizer-backend/*/application` | App Runner アプリケーションログ |
| `/aws/apprunner/warehouse-optimizer-backend/*/service` | App Runner サービスログ |

---

## Frontend (Vercel)

| 項目 | 値 |
|------|-----|
| Framework | Next.js 15.5.9 |
| Domain | `app.warehouse-optimizer.net` |
| API Base | `https://cdcdwjbjms.ap-northeast-1.awsapprunner.com` |

---

## デプロイ手順

### Backend (App Runner + ECS)

```bash
# 1. ECRにログイン
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin 904368995258.dkr.ecr.ap-northeast-1.amazonaws.com

# 2. Dockerイメージをビルド
cd backend
docker build --platform linux/amd64 -t warehouse-optimizer-backend .

# 3. ECRにプッシュ
docker tag warehouse-optimizer-backend:latest \
  904368995258.dkr.ecr.ap-northeast-1.amazonaws.com/warehouse-optimizer-backend:latest
docker push 904368995258.dkr.ecr.ap-northeast-1.amazonaws.com/warehouse-optimizer-backend:latest

# 4. App Runnerは自動デプロイ（AutoDeploymentsEnabled: true）

# 5. ECS Celery Workerを再デプロイ
aws ecs update-service \
  --cluster warehouse-optimizer-celery \
  --service warehouse-optimizer-celery-worker \
  --force-new-deployment \
  --region ap-northeast-1
```

### Frontend (Vercel)

```bash
# Git pushで自動デプロイ
git push origin main
```

---

## 運用コマンド

### ログ確認

```bash
# Celery Worker ログ (最新30分)
aws logs tail /ecs/warehouse-optimizer-celery --since 30m --region ap-northeast-1

# App Runner ログ
aws logs tail "/aws/apprunner/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be/application" \
  --since 30m --region ap-northeast-1
```

### サービス状態確認

```bash
# ECS Service
aws ecs describe-services \
  --cluster warehouse-optimizer-celery \
  --services warehouse-optimizer-celery-worker \
  --region ap-northeast-1 | jq '.services[0] | {status, runningCount, desiredCount}'

# App Runner
aws apprunner describe-service \
  --service-arn "arn:aws:apprunner:ap-northeast-1:904368995258:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be" \
  --region ap-northeast-1 | jq '.Service.Status'
```

### Celeryキューのパージ

```bash
curl -X POST https://cdcdwjbjms.ap-northeast-1.awsapprunner.com/v1/debug/purge-celery-queue
```

### ECS Worker 再起動

```bash
aws ecs update-service \
  --cluster warehouse-optimizer-celery \
  --service warehouse-optimizer-celery-worker \
  --force-new-deployment \
  --region ap-northeast-1
```

---

## 費用見積もり (月額概算)

| サービス | 見積もり |
|----------|----------|
| App Runner (1 vCPU, 2GB) | ~$30-50 |
| ECS Fargate (1 vCPU, 2GB, 24h) | ~$30-40 |
| ElastiCache (cache.t3.micro) | ~$12-15 |
| RDS Aurora (Serverless v2 or db.t3.micro) | ~$30-50 |
| ECR | ~$1-5 |
| Data Transfer | ~$5-10 |
| **合計** | **~$110-170/月** |

---

## トラブルシューティング

### タスクがハングする場合

1. Celery Workerのログを確認
2. キューをパージ: `POST /v1/debug/purge-celery-queue`
3. ECS Workerを再起動

### タイムアウトエラー

- Celeryタスクは9分でソフトタイムアウト、10分でハードタイムアウト
- データ量を減らすか `max_moves` を小さく設定

### 前のタスクが終わらない

- 新しいリクエストを送信すると前のタスクは自動キャンセルされる
- `relocation:active_task` キーで管理

---

## 関連ドキュメント

- [README.md](README.md) - プロジェクト概要
- [DEPLOY.md](DEPLOY.md) - デプロイ詳細手順
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - 開発ガイド
