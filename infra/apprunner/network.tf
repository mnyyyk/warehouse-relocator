###############################
# VPC & Networking (Private RDS + App Runner VPC Connector)
###############################

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.app_name}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags = { Name = "${var.app_name}-igw" }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidr_a
  availability_zone       = var.az_a
  map_public_ip_on_launch = true
  tags = { Name = "${var.app_name}-public-a" }
}

resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidr_a
  availability_zone = var.az_a
  tags = { Name = "${var.app_name}-private-a" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0" gateway_id = aws_internet_gateway.igw.id }
  tags = { Name = "${var.app_name}-public-rt" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

# NAT (単AZ / 開発向け) --------------------------------------------------
resource "aws_eip" "nat_eip" {
  vpc = true
  tags = { Name = "${var.app_name}-nat-eip" }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_a.id
  tags = { Name = "${var.app_name}-nat" }
  depends_on    = [aws_internet_gateway.igw]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0" nat_gateway_id = aws_nat_gateway.nat.id }
  tags = { Name = "${var.app_name}-private-rt" }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

###############################
# Security Groups
###############################

resource "aws_security_group" "rds" {
  name        = "${var.app_name}-rds-sg"
  description = "RDS access from App Runner"
  vpc_id      = aws_vpc.main.id
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
  name       = "${var.app_name}-rds-subnet-group"
  subnet_ids = [aws_subnet.private_a.id]
  tags = { Name = "${var.app_name}-rds-subnet-group" }
}

resource "aws_rds_cluster" "postgres" {
  cluster_identifier      = "${var.app_name}-pg-cluster"
  engine                  = "aurora-postgresql"
  engine_version          = var.rds_engine_version
  database_name           = var.rds_db_name
  master_username         = var.rds_master_username
  master_password         = var.rds_master_password
  skip_final_snapshot     = true
  db_subnet_group_name    = aws_db_subnet_group.rds.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  deletion_protection     = false
  apply_immediately       = true
}

resource "aws_rds_cluster_instance" "postgres_writer" {
  identifier         = "${var.app_name}-pg-writer"
  cluster_identifier = aws_rds_cluster.postgres.id
  instance_class     = var.rds_instance_class
  engine             = aws_rds_cluster.postgres.engine
  engine_version     = aws_rds_cluster.postgres.engine_version
  publicly_accessible = false
}

###############################
# App Runner VPC Connector
###############################

resource "aws_apprunner_vpc_connector" "this" {
  vpc_connector_name = "${var.app_name}-connector"
  subnets            = [aws_subnet.private_a.id]
  security_groups    = [aws_security_group.rds.id]
  tags = { Name = "${var.app_name}-vpc-connector" }
}

###############################
# Outputs
###############################

output "vpc_id" { value = aws_vpc.main.id }
output "private_subnet_id" { value = aws_subnet.private_a.id }
output "rds_endpoint" { value = aws_rds_cluster.postgres.endpoint }
output "rds_reader_endpoint" { value = aws_rds_cluster.postgres.reader_endpoint }