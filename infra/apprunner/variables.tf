variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-northeast-1"
}

variable "app_name" {
  description = "App Runner service name"
  type        = string
  default     = "warehouse-optimizer-backend"
}

variable "ecr_repo_name" {
  description = "ECR repository name"
  type        = string
  default     = "warehouse-optimizer-backend"
}

variable "instance_cpu" {
  description = "App Runner instance CPU in MB (1024=1 vCPU)"
  type        = string
  default     = "1024"
}

variable "instance_memory" {
  description = "App Runner instance memory in MB (2048=2GB)"
  type        = string
  default     = "2048"
}

# -----------------------------
# Networking & RDS (B案用)
# -----------------------------
variable "vpc_cidr" {
  description = "VPC CIDR"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr_a" {
  description = "Public subnet CIDR (AZ-A)"
  type        = string
  default     = "10.0.0.0/24"
}

variable "private_subnet_cidr_a" {
  description = "Private subnet CIDR (AZ-A)"
  type        = string
  default     = "10.0.1.0/24"
}

variable "az_a" {
  description = "Availability Zone for 'A'"
  type        = string
  default     = "ap-northeast-1a"
}

variable "rds_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "Aurora PostgreSQL instance class"
  type        = string
  default     = "db.serverless" # or db.r6g.large etc.
}

variable "rds_db_name" {
  description = "Database name"
  type        = string
  default     = "warehouse"
}

variable "rds_master_username" {
  description = "DB master username"
  type        = string
  default     = "appuser"
}

variable "rds_master_password" {
  description = "DB master password (use tfvars or env vars in CI)"
  type        = string
  sensitive   = true
}
