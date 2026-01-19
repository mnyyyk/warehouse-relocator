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

resource "aws_apprunner_service" "backend" {
  service_name = local.app_name

  source_configuration {
    auto_deployments_enabled = true
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

  # VPC Connector を使用してプライベートRDSへ接続
  network_configuration {
    egress_type        = "VPC"
    vpc_connector_arn  = aws_apprunner_vpc_connector.this.arn
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
  value = aws_apprunner_vpc_connector.this.arn
}
