terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  app_name       = var.app_name
  ecr_repo_name  = var.ecr_repo_name
}

resource "aws_ecr_repository" "backend" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# App RunnerがECRへアクセスするためのサービスロール
resource "aws_iam_role" "apprunner_ecr_access" {
  name               = "${local.app_name}-apprunner-ecr-role"
  assume_role_policy = data.aws_iam_policy_document.apprunner_assume.json
}

data "aws_iam_policy_document" "apprunner_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com", "tasks.apprunner.amazonaws.com", "apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# App Runner - ECR に latest イメージプッシュ後に作成可能
resource "aws_apprunner_service" "backend" {
  service_name = local.app_name

  source_configuration {
    auto_deployments_enabled = true  # ECR latest更新時に自動デプロイ
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.backend.repository_url}:latest"
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          ENV = "production"
          DATABASE_URL = var.enable_private_rds ? "postgresql+psycopg2://${var.rds_master_username}:${var.rds_master_password}@${aws_rds_cluster.postgres[0].endpoint}:5432/${var.rds_db_name}" : ""
        }
      }
    }
  }

  instance_configuration {
    cpu    = var.instance_cpu    # e.g. 1 vCPU → "1024"
    memory = var.instance_memory # e.g. 2 GB   → "2048"
  }

  health_check_configuration {
    healthy_threshold   = 1
    interval            = 10
    path                = "/health"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 5
  }

  # B案(Private RDS)のときのみ VPC Connector を有効化
  dynamic "network_configuration" {
    for_each = var.enable_private_rds ? [1] : []
    content {
      egress_configuration {
        egress_type       = "VPC"
        vpc_connector_arn = aws_apprunner_vpc_connector.this[0].arn
      }
    }
  }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.backend.repository_url
}

output "apprunner_service_arn" {
  value = aws_apprunner_service.backend.arn
}

output "apprunner_service_url" {
  value = aws_apprunner_service.backend.service_url
}

output "apprunner_vpc_connector_arn" {
  value = var.enable_private_rds ? aws_apprunner_vpc_connector.this[0].arn : null
}
