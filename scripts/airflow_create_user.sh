#!/bin/bash
# Run after airflow-init completes:
#   docker exec airflow-webserver bash scripts/airflow_create_user.sh
docker exec airflow-webserver airflow users create \
  --username admin \
  --firstname Healthcare \
  --lastname Admin \
  --role Admin \
  --email admin@healthcare.local \
  --password admin_secret_2026 2>/dev/null || echo "User already exists"
