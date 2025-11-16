output "hosted_zone_id" {
  description = "Route53 hosted zone ID"
  value       = data.aws_route53_zone.main.zone_id
}

output "hosted_zone_name_servers" {
  description = "Name servers for the hosted zone"
  value       = data.aws_route53_zone.main.name_servers
}

output "api_domain" {
  description = "API custom domain"
  value       = "api.${var.domain_name}"
}

output "app_domain" {
  description = "Frontend app domain (to be configured in Vercel)"
  value       = "app.${var.domain_name}"
}

output "apprunner_custom_domain_status" {
  description = "Status of App Runner custom domain association"
  value       = aws_apprunner_custom_domain_association.api.status
}

output "certificate_validation_records" {
  description = "Certificate validation records for App Runner"
  value = [
    for record in aws_apprunner_custom_domain_association.api.certificate_validation_records : {
      name  = record.name
      type  = record.type
      value = record.value
    }
  ]
}
