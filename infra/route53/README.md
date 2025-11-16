# Route53 + Custom Domain Setup

このディレクトリは `warehouse-optimizer.com` のRoute53設定とカスタムドメイン設定を管理します。

## 前提条件

1. **ドメインの購入**: `warehouse-optimizer.com` をRoute53または他のレジストラで購入済み
2. **App Runner デプロイ済み**: `infra/apprunner` でバックエンドがデプロイ済み

## セットアップ手順

### 1. Terraform変数の設定

```bash
cd infra/route53
cp terraform.tfvars.example terraform.tfvars
```

`terraform.tfvars` を編集:
```hcl
domain_name              = "warehouse-optimizer.com"
apprunner_service_arn    = "arn:aws:apprunner:ap-northeast-1:904368995258:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be"
apprunner_default_domain = "cdcdwjbjms.ap-northeast-1.awsapprunner.com"
```

App Runner の値は以下で取得:
```bash
cd ../apprunner
terraform output apprunner_service_arn
terraform output apprunner_service_url
```

### 2. Route53 ホストゾーン作成

```bash
terraform init
terraform plan
terraform apply
```

### 3. ネームサーバーをドメインレジストラに設定

Terraform apply 後、ネームサーバーが出力されます:
```bash
terraform output hosted_zone_name_servers
```

これらのNSレコードをドメインレジストラ(お名前.com、Route53、GoDaddyなど)に設定します。

**例: Route53でドメインを購入した場合**
- Route53コンソール → Registered domains → warehouse-optimizer.com
- Name servers セクションで、上記のNSレコードに変更

**例: 他のレジストラの場合**
- レジストラの管理画面でカスタムネームサーバーを設定
- 4つのNSレコードすべてを追加

### 4. DNS伝播待機 (15分〜48時間)

```bash
# DNS伝播確認
dig NS warehouse-optimizer.com
dig api.warehouse-optimizer.com
```

### 5. App Runner カスタムドメイン検証確認

```bash
terraform output apprunner_custom_domain_status
# "active" になるまで待機 (通常5〜15分)
```

App Runner コンソールでも確認可能:
- App Runner → Services → warehouse-optimizer-backend → Custom domains

### 6. API動作確認

```bash
curl https://api.warehouse-optimizer.com/health
# {"status":"ok"} が返ればOK
```

## Vercel フロントエンド設定

### 7. Vercel プロジェクト作成

```bash
cd ../../frontend
npx vercel login
npx vercel
```

プロンプトに従って:
- Link to existing project? → No
- Project name? → warehouse-optimizer-frontend
- Directory? → ./
- Detected framework: Next.js

### 8. 環境変数設定 (Vercel)

Vercel ダッシュボードまたはCLI:
```bash
npx vercel env add NEXT_PUBLIC_API_BASE production
# 値: https://api.warehouse-optimizer.com
```

または Vercel ダッシュボード:
- Settings → Environment Variables
- `NEXT_PUBLIC_API_BASE` = `https://api.warehouse-optimizer.com`

### 9. カスタムドメイン追加 (Vercel)

**Option A: Vercel CLI**
```bash
npx vercel domains add app.warehouse-optimizer.com
```

**Option B: Vercel ダッシュボード**
- Settings → Domains
- Add domain: `app.warehouse-optimizer.com`
- Vercel が提供するDNSレコードを確認

### 10. Route53にVercel DNSレコード追加

Vercel から提供されたCNAMEレコードをRoute53に追加:

```bash
# Vercel から取得したCNAME値を確認
# 通常: cname.vercel-dns.com または 76.76.21.21 (A record)

# Route53にレコード追加 (手動またはTerraform)
aws route53 change-resource-record-sets \
  --hosted-zone-id $(terraform output -raw hosted_zone_id) \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "app.warehouse-optimizer.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [{"Value": "cname.vercel-dns.com"}]
      }
    }]
  }'
```

### 11. 最終確認

```bash
# API確認
curl https://api.warehouse-optimizer.com/health

# フロントエンド確認
open https://app.warehouse-optimizer.com
```

## トラブルシューティング

### App Runner カスタムドメインが "pending" のまま
- Route53のDNS伝播が完了していない → 最大48時間待機
- 証明書検証レコードが正しくない → `terraform output certificate_validation_records` で確認

### Vercel ドメインが検証できない
- Route53でCNAMEレコードが作成されているか確認
- TTLが低い値(300秒)になっているか確認
- `dig app.warehouse-optimizer.com` でDNS解決確認

### CORS エラー
- backend/app/main.py で `https://app.warehouse-optimizer.com` を許可リストに追加
- App Runner を再デプロイ

## クリーンアップ

```bash
# Route53リソース削除
terraform destroy

# Vercel プロジェクト削除
npx vercel remove warehouse-optimizer-frontend
```
