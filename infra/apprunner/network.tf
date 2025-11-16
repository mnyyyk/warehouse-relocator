###############################
# VPC & Networking (Private RDS + App Runner VPC Connector)
###############################

resource "aws_vpc" "main" {
  count               = var.enable_private_rds ? 1 : 0
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.app_name}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  count  = var.enable_private_rds ? 1 : 0
  vpc_id = aws_vpc.main[0].id
  tags = { Name = "${var.app_name}-igw" }
}

resource "aws_subnet" "public_a" {
  count                   = var.enable_private_rds ? 1 : 0
  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = var.public_subnet_cidr_a
  availability_zone       = var.az_a
  map_public_ip_on_launch = true
  tags = { Name = "${var.app_name}-public-a" }
}

resource "aws_subnet" "private_a" {
  count             = var.enable_private_rds ? 1 : 0
  vpc_id            = aws_vpc.main[0].id
  cidr_block        = var.private_subnet_cidr_a
  availability_zone = var.az_a
  tags = { Name = "${var.app_name}-private-a" }
}

# 2nd AZ for Aurora RDS requirement
resource "aws_subnet" "private_c" {
  count             = var.enable_private_rds ? 1 : 0
  vpc_id            = aws_vpc.main[0].id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "ap-northeast-1c"
  tags = { Name = "${var.app_name}-private-c" }
}

resource "aws_route_table" "public" {
  count  = var.enable_private_rds ? 1 : 0
  vpc_id = aws_vpc.main[0].id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw[0].id
  }
  
  tags = { Name = "${var.app_name}-public-rt" }
}

resource "aws_route_table_association" "public_a" {
  count          = var.enable_private_rds ? 1 : 0
  subnet_id      = aws_subnet.public_a[0].id
  route_table_id = aws_route_table.public[0].id
}

# NAT (単AZ / 開発向け) --------------------------------------------------
resource "aws_eip" "nat_eip" {
  count  = var.enable_private_rds ? 1 : 0
  domain = "vpc"
  tags   = { Name = "${var.app_name}-nat-eip" }
}

resource "aws_nat_gateway" "nat" {
  count         = var.enable_private_rds ? 1 : 0
  allocation_id = aws_eip.nat_eip[0].id
  subnet_id     = aws_subnet.public_a[0].id
  tags = { Name = "${var.app_name}-nat" }
  depends_on    = [aws_internet_gateway.igw]
}

resource "aws_route_table" "private" {
  count  = var.enable_private_rds ? 1 : 0
  vpc_id = aws_vpc.main[0].id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat[0].id
  }
  
  tags = { Name = "${var.app_name}-private-rt" }
}

resource "aws_route_table_association" "private_a" {
  count          = var.enable_private_rds ? 1 : 0
  subnet_id      = aws_subnet.private_a[0].id
  route_table_id = aws_route_table.private[0].id
}

###############################
# Security Groups
###############################

resource "aws_security_group" "rds" {
  count       = var.enable_private_rds ? 1 : 0
  name        = "${var.app_name}-rds-sg"
  description = "RDS access from App Runner"
  vpc_id      = aws_vpc.main[0].id
  tags = { Name = "${var.app_name}-rds-sg" }

  ingress {
    description      = "PostgreSQL from VPC"
    from_port        = 5432
    to_port          = 5432
    protocol         = "tcp"
    cidr_blocks      = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

###############################
# RDS PostgreSQL (Private)
###############################

resource "aws_db_subnet_group" "rds" {
  count      = var.enable_private_rds ? 1 : 0
  name       = "${var.app_name}-rds-subnet-group"
  subnet_ids = [aws_subnet.private_a[0].id, aws_subnet.private_c[0].id]
  tags = { Name = "${var.app_name}-rds-subnet-group" }
}

resource "aws_rds_cluster" "postgres" {
  count                    = var.enable_private_rds ? 1 : 0
  cluster_identifier      = "${var.app_name}-pg-cluster"
  engine                  = "aurora-postgresql"
  engine_version          = var.rds_engine_version
  database_name           = var.rds_db_name
  master_username         = var.rds_master_username
  master_password         = var.rds_master_password
  skip_final_snapshot     = true
  db_subnet_group_name    = aws_db_subnet_group.rds[0].name
  vpc_security_group_ids  = [aws_security_group.rds[0].id]
  deletion_protection     = false
  apply_immediately       = true

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 1.0
  }
}

resource "aws_rds_cluster_instance" "postgres_writer" {
  count              = var.enable_private_rds ? 1 : 0
  identifier         = "${var.app_name}-pg-writer"
  cluster_identifier = aws_rds_cluster.postgres[0].id
  instance_class     = var.rds_instance_class
  engine             = aws_rds_cluster.postgres[0].engine
  engine_version     = aws_rds_cluster.postgres[0].engine_version
  publicly_accessible = false
}

###############################
# App Runner VPC Connector
###############################

resource "aws_apprunner_vpc_connector" "this" {
  count              = var.enable_private_rds ? 1 : 0
  vpc_connector_name = "${var.app_name}-connector"
  subnets            = [aws_subnet.private_a[0].id]
  security_groups    = [aws_security_group.rds[0].id]
  tags = { Name = "${var.app_name}-vpc-connector" }
}

###############################
# Outputs
###############################

output "vpc_id" { value = var.enable_private_rds ? aws_vpc.main[0].id : null }
output "private_subnet_id" { value = var.enable_private_rds ? aws_subnet.private_a[0].id : null }
output "rds_endpoint" { value = var.enable_private_rds ? aws_rds_cluster.postgres[0].endpoint : null }
output "rds_reader_endpoint" { value = var.enable_private_rds ? aws_rds_cluster.postgres[0].reader_endpoint : null }