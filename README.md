# Warehouse Optimizer

AI を用いた倉庫ロケーション最適化システム。

## 1. リポジトリ概要
- Backend: FastAPI + SQLModel (`backend/`)
- Frontend: Next.js (`frontend/`)
- Infra IaC: Terraform (App Runner / ECR) `infra/apprunner/`

## 2. ローカル開発

バックエンド仮想環境 & 依存インストール:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

フロントエンド:
```bash
cd frontend
npm install
npm run dev  # http://localhost:3000
```

## 3. AWS へのデプロイ (App Runner + ECR)

### 3.1 全体構成
| コンポーネント | サービス | 備考 |
|----------------|----------|------|
| Backend API    | AWS App Runner | Docker イメージを ECR から自動デプロイ |
| Container Image| Amazon ECR     | latest タグを App Runner がウォッチ |
| DB (将来)       | Amazon RDS (PostgreSQL) | 初期は SQLite, 本番は RDS 推奨 |
| オブジェクト保存(将来) | Amazon S3 | アップロードCSV、ログなど |
| Frontend(将来) | S3 + CloudFront or Vercel | 静的/SSR 要件次第 |

### 3.2 ネットワーク/ドメインの順序について
最初に *アプリをインターネットで公開できる最小構成* を作り、その後データ永続化（S3/RDS）へ拡張するのが効率的です。従って順序は以下推奨:
1. ECR と App Runner (コンテナ公開) ← いま実装済み
2. カスタムドメイン (必要なら) – App Runner のカスタムドメイン設定 or CloudFront 経由
3. RDS (PostgreSQL) – 環境変数 `DATABASE_URL` を差し替え
4. S3 (CSV 永続化) – `AWS_ACCESS_KEY_ID` 等を追加し、保存処理を差し替え
5. ログ/監視 (CloudWatch, OpenTelemetry 等)

RDS や S3 を **先に** 作る必要はありません。DB を切り替える時点で環境変数を更新し、再デプロイすれば OK です。

### 3.3 Terraform 初期化 & 適用
```bash
cd infra/apprunner
terraform init
terraform apply -auto-approve
```
出力例:
- `ecr_repository_url` … ECR の URI (例: xxxxx.dkr.ecr.ap-northeast-1.amazonaws.com/warehouse-optimizer-backend)
- `apprunner_service_url` … 公開URL (例: https://xxxxxxxx.apprunner.amazonaws.com)

### 3.4 GitHub Actions による自動デプロイ
Workflow: `.github/workflows/deploy-backend.yml`

OIDC ロール作成後、Secrets に以下を追加:
- `AWS_OIDC_ROLE_ARN` : GitHub Actions から AssumeRole する IAM Role ARN

main ブランチに push:
1. Docker build (`backend/Dockerfile`)
2. ECR push (`latest`)
3. App Runner が自動デプロイ (auto_deployments_enabled=true)

### 3.5 カスタムドメイン (任意)
App Runner > Custom domains で ACM 証明書を発行→検証→紐付け。
DNS(Route53等)に CNAME を作成し適用。完了後 HTTPS で利用可能。

### 3.6 環境変数一覧 (例)
| 変数 | 目的 | 例 |
|------|------|----|
| `ENV` | 実行環境種別 | `production` |
| `DATABASE_URL` | DB接続文字列 | `postgresql+psycopg2://user:pass@host:5432/db` |
| `FRONTEND_ORIGINS` | CORS許可 | `https://app.example.com` |
| `OPENAI_API_KEY` | AI機能 | `sk-...` |
| `S3_BUCKET` | CSV 永続化用 | `warehouse-optimizer-data` |

### 3.7 RDS への移行 (概要)
1. RDS PostgreSQL を作成 (Public か VPC 内か要件次第)
2. セキュリティグループで App Runner からの 5432 を許可
3. `DATABASE_URL` を Secrets or App Runner コンソールで差し替え
4. 再デプロイ後、初回マイグレーション (Alembic) を実行（別タスク化推奨）

### 3.8 S3 導入 (概要)
1. バケット作成 `warehouse-optimizer-data`
2. IAM ロールに `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` 権限追加
3. アップロード処理でローカル `/tmp` 保存 → S3 へ転送へ変更

## 4. トラブルシューティング
| 症状 | 原因 | 対策 |
|------|------|------|
| App Runner 502 | 起動遅延 or health失敗 | `HEALTHCHECK` path 正しいか、CPU/メモリ増量 |
| CSV アップロード失敗 | メモリ不足 | インスタンスメモリ2GB→4GBへ拡張 |
| CORS エラー | オリジン未許可 | `FRONTEND_ORIGINS` 更新 |
| AI 動作しない | APIキー未設定 | `OPENAI_API_KEY` を設定 |

## 5. 今後の拡張案
- CloudWatch Logs への構造化 JSON 出力
- OpenTelemetry によるトレース
- 自動 Alembic マイグレーションジョブ
- Pre-signed URL による巨大ファイル直S3アップロード

## 6. ライセンス / 注意
内部利用を想定。外部公開時は認証/権限管理(例えば Cognito + JWT)を追加してください。

***
最小構成が稼働したら次は RDS/S3 を段階的に追加するのが推奨です。