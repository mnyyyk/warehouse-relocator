# デプロイ手順

## 概要

このプロジェクトは以下の構成でデプロイされています：
- **バックエンド**: AWS App Runner (Docker)
- **フロントエンド**: Vercel
- **データベース**: AWS RDS PostgreSQL

---

## バックエンドのデプロイ

### 前提条件
- Docker Desktop が起動していること
- AWS CLI が設定済みであること
- ECR リポジトリが作成済みであること

### デプロイ手順

1. **変更をコミット**
```bash
cd /Users/kounoyousuke/warehouse-optimizer
git add -A
git commit -m "your commit message"
git push
```

2. **Dockerビルド＆デプロイ実行**
```bash
cd /Users/kounoyousuke/warehouse-optimizer
bash deploy.sh
```

`deploy.sh`スクリプトは以下を自動実行します：
- Dockerイメージのビルド（AMD64アーキテクチャ）
- ECRへのログイン
- イメージのタグ付け
- ECRへのプッシュ
- App Runnerが自動的に新しいイメージをデプロイ（3-5分）

3. **デプロイ完了確認**
```bash
# App Runnerのステータス確認
aws apprunner describe-service \
  --service-arn "arn:aws:apprunner:ap-northeast-1:904368995258:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be" \
  --region ap-northeast-1 \
  --query 'Service.Status' \
  --output text

# デプロイ履歴確認
aws apprunner list-operations \
  --service-arn "arn:aws:apprunner:ap-northeast-1:904368995258:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be" \
  --region ap-northeast-1 \
  --max-results 3 \
  --query 'OperationSummaryList[].[Type,Status,StartedAt,EndedAt]' \
  --output table
```

### トラブルシューティング

#### Docker not running
```bash
open -a Docker  # Docker Desktopを起動
sleep 10        # 起動を待つ
```

#### ポート8000が使用中（ローカル開発時）
```bash
lsof -ti tcp:8000 | xargs kill -9
```

---

## フロントエンドのデプロイ

### 前提条件
- Vercel CLI がインストール済みであること
- Vercelプロジェクトにログイン済みであること

### デプロイ手順

1. **変更をコミット**
```bash
cd /Users/kounoyousuke/warehouse-optimizer
git add -A
git commit -m "your commit message"
git push
```

2. **Vercelにデプロイ**
```bash
cd /Users/kounoyousuke/warehouse-optimizer/frontend
npx vercel --prod --yes
```

3. **デプロイ完了確認**
```bash
# 最新のデプロイ一覧
npx vercel ls

# 本番環境で確認
curl -s https://app.warehouse-optimizer.net/upload | grep -o "種類：SKU"
```

### 強制デプロイ（キャッシュクリア）
```bash
cd /Users/kounoyousuke/warehouse-optimizer/frontend
npx vercel --prod --force --yes
```

### トラブルシューティング

#### ビルドエラー（TypeScript/ESLint）
`next.config.ts`で一時的に無効化されています：
```typescript
eslint: {
  ignoreDuringBuilds: true,
},
typescript: {
  ignoreBuildErrors: true,
}
```

#### 変更が反映されない
1. ブラウザのキャッシュをクリア（Cmd+Shift+R / Ctrl+Shift+R）
2. シークレットモードで確認
3. 最新のVercel URLで直接確認

---

## 完全デプロイ（バックエンド＋フロントエンド）

### 手順

```bash
# 1. バックエンドのデプロイ
cd /Users/kounoyousuke/warehouse-optimizer
git add -A
git commit -m "feat: your changes"
git push
bash deploy.sh

# 2. フロントエンドのデプロイ
cd frontend
npx vercel --prod --yes

# 3. 確認
# バックエンド
curl -s https://api.wh-optim.mnyk.me/health || \
  curl -s https://cdcdwjbjms.ap-northeast-1.awsapprunner.com/health

# フロントエンド
curl -I https://app.warehouse-optimizer.net
```

---

## 環境変数

### バックエンド（App Runner）
- `DATABASE_URL`: PostgreSQL接続文字列
- `CORS_ORIGINS`: 許可するオリジン（本番フロントエンドURL含む）

App Runnerコンソールで設定：
https://console.aws.amazon.com/apprunner

### フロントエンド（Vercel）
- `NEXT_PUBLIC_API_BASE`: バックエンドAPIのURL

Vercelダッシュボードで設定：
https://vercel.com/mnyys-projects/warehouse-optimizer/settings/environment-variables

---

## デプロイ後の確認項目

### バックエンド
- [ ] App RunnerステータスがRUNNING
- [ ] ヘルスチェックが200 OK
- [ ] `/docs`でAPI仕様が確認できる
- [ ] データベース接続が正常

### フロントエンド
- [ ] Vercelビルドが成功
- [ ] カスタムドメインでアクセス可能
- [ ] アップロードページに注釈が表示される
- [ ] API呼び出しが正常（CORS設定）

---

## ロールバック

### バックエンド
```bash
# 以前のイメージタグを確認
aws ecr describe-images \
  --repository-name warehouse-optimizer-backend \
  --region ap-northeast-1 \
  --query 'sort_by(imageDetails,& imagePushedAt)[-5:].[imageTags[0],imagePushedAt]' \
  --output table

# App Runnerコンソールから以前のデプロイに戻す
# または以前のコミットをデプロイ
git checkout <previous-commit>
bash deploy.sh
```

### フロントエンド
```bash
# Vercelダッシュボードから以前のデプロイをプロモート
# https://vercel.com/mnyys-projects/warehouse-optimizer

# または以前のコミットをデプロイ
git checkout <previous-commit>
cd frontend
npx vercel --prod --yes
```

---

## 監視とログ

### バックエンドログ
```bash
# App Runnerのログを確認
aws logs tail /aws/apprunner/warehouse-optimizer-backend/application \
  --follow \
  --region ap-northeast-1
```

### フロントエンドログ
Vercelダッシュボード → Deployments → ビルドログ確認
https://vercel.com/mnyys-projects/warehouse-optimizer

---

## 緊急連絡先・リソース

- **App Runner**: https://console.aws.amazon.com/apprunner
- **ECR**: https://console.aws.amazon.com/ecr
- **RDS**: https://console.aws.amazon.com/rds
- **Vercel**: https://vercel.com/mnyys-projects/warehouse-optimizer
- **GitHub**: https://github.com/mnyyyk/warehouse-relocator
