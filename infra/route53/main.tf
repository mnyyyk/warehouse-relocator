terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ===== Use existing Route53 Hosted Zone (created during domain registration) =====
data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

# Note: If you want Terraform to manage the hosted zone, use:
# resource "aws_route53_zone" "main" {
#   name = var.domain_name
#   tags = {
#     Name        = var.domain_name
#     Environment = "production"
#     Project     = "warehouse-optimizer"
#   }
# }
# Then run: terraform import aws_route53_zone.main Z091798314YXSBTOLVJ18

# ===== App Runner Custom Domain Association =====
resource "aws_apprunner_custom_domain_association" "api" {
  domain_name = "api.${var.domain_name}"
  service_arn = var.apprunner_service_arn

  lifecycle {
    create_before_destroy = true
  }
}

# ===== Route53 Records for App Runner (api subdomain) =====
# App Runner custom domain validation records are created after
# the custom domain association is initiated.

# Note: This requires a two-step apply:
# 1. First apply creates the custom domain association
# 2. Second apply creates the validation records and CNAME

# For now, we'll comment out the validation records and CNAME
# and create them after the custom domain association is created

# Uncomment after first apply:
# resource "aws_route53_record" "api_validation" {
#   for_each = {
#     for dvo in aws_apprunner_custom_domain_association.api.certificate_validation_records : dvo.name => {
#       name   = dvo.name
#       record = dvo.value
#       type   = dvo.type
#     }
#   }
#
#   zone_id         = data.aws_route53_zone.main.zone_id
#   name            = each.value.name
#   type            = each.value.type
#   records         = [each.value.record]
#   ttl             = 300
#   allow_overwrite = true
# }

# Uncomment after validation records are created:
# resource "aws_route53_record" "api" {
#   zone_id = data.aws_route53_zone.main.zone_id
#   name    = "api.${var.domain_name}"
#   type    = "CNAME"
#   ttl     = 300
#   records = [var.apprunner_default_domain]
#
#   depends_on = [
#     aws_apprunner_custom_domain_association.api,
#     aws_route53_record.api_validation
#   ]
# }

# ===== Placeholder A record for app subdomain (Vercel will provide IP) =====
# This will be configured in Vercel dashboard or via Vercel CLI
# Vercel typically uses CNAME pointing to cname.vercel-dns.com

# For documentation purposes, we'll add a comment here:
# Manual step required:
# 1. In Vercel dashboard, add custom domain: app.warehouse-optimizer.com
# 2. Vercel will provide DNS records (CNAME or A)
# 3. Add those records to Route53 manually or via Vercel's API

# If using Vercel DNS integration, you can create the record here:
# resource "aws_route53_record" "app" {
#   zone_id = aws_route53_zone.main.zone_id
#   name    = "app.${var.domain_name}"
#   type    = "CNAME"
#   ttl     = 300
#   records = ["cname.vercel-dns.com"]
# }

# ===== Root domain redirect (optional) =====
# Redirect warehouse-optimizer.com -> app.warehouse-optimizer.com
# This requires CloudFront + S3 or a simple redirect service
# For now, we'll leave this as a placeholder
