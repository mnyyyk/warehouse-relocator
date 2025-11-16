variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "domain_name" {
  description = "Root domain name (e.g., warehouse-optimizer.net)"
  type        = string
  default     = "warehouse-optimizer.net"
}

variable "apprunner_service_arn" {
  description = "ARN of the App Runner service for API backend"
  type        = string
  # Get this from the apprunner module output
  # Example: arn:aws:apprunner:ap-northeast-1:904368995258:service/warehouse-optimizer-backend/c75ca38ef67148e0bced484f9a2569be
}

variable "apprunner_default_domain" {
  description = "Default App Runner domain (e.g., cdcdwjbjms.ap-northeast-1.awsapprunner.com)"
  type        = string
  # Get this from the apprunner module output
}
