# Task: Replace Cloud Foundation Fabric PostgreSQL with Native Resources

## Status: In Progress

## Problem
- Cloud Foundation Fabric v44.1.0 cloudsql-instance module failing to enable connectivity
- Module configured with public IP despite private network setup
- Need to switch to native Google Cloud SQL resources

## Solution Steps
1. ✅ **Analyze current configuration** - Cloud Foundation Fabric module at lines 163-215 in data-warehouse.tf
2. ✅ **Replace with native google_sql_database_instance resource**
3. ✅ **Configure private IP only with proper VPC integration**
4. ✅ **Maintain all existing settings** (PostgreSQL 15, HA, backups, performance flags)
5. ✅ **Update outputs to reference new resource**
6. ✅ **Test with terragrunt plan** - Plan successful, will create 1 resource
7. ⏳ **Apply if validation succeeds**

## Current Configuration Analysis
- PostgreSQL 15 with db-n1-standard-2 tier
- High availability (REGIONAL)
- 100GB PD_SSD storage
- Automated backups with 7-day retention
- Point-in-time recovery enabled
- **Issue**: Public IP enabled, authorized networks configured instead of private-only

## Required Changes
- Remove Cloud Foundation Fabric module block
- Add native google_sql_database_instance resource
- Configure ip_configuration for private network only
- Ensure proper dependency on google_service_networking_connection