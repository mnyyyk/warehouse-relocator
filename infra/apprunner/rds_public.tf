# Option A: Simple public RDS PostgreSQL (for early staging/MVP)

resource "aws_db_instance" "public_postgres" {
  count                        = var.enable_public_rds ? 1 : 0
  identifier                   = "${var.app_name}-public-pg"
  engine                       = "postgres"
  engine_version               = "16"
  instance_class               = "db.t4g.micro"
  allocated_storage            = 20
  max_allocated_storage        = 100
  db_name                      = var.rds_db_name
  username                     = var.rds_master_username
  password                     = var.rds_master_password
  publicly_accessible          = true
  storage_encrypted            = true
  skip_final_snapshot          = true
  deletion_protection          = false
  auto_minor_version_upgrade   = true
  backup_retention_period      = 7
  performance_insights_enabled = false
  apply_immediately            = true

  # Default VPC & subnet group auto-selection (omit subnet group for simplicity)
  # For stricter network control later, move to custom VPC and set publicly_accessible=false.

  lifecycle {
    ignore_changes = [password]
  }
}

output "public_rds_endpoint" {
  value = var.enable_public_rds ? aws_db_instance.public_postgres[0].address : null
}
output "public_rds_port" {
  value = var.enable_public_rds ? aws_db_instance.public_postgres[0].port : null
}