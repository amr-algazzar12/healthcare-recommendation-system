.PHONY: up fix-airflow down restart logs ps verify \
        shell-namenode shell-clickhouse shell-airflow shell-spark \
        generate-data init-hdfs load-hdfs clean-hdfs init_clickhouse load-clickhouse clean-clickhouse create-airflow-user \
        init-everything pipeline clean

# ── Correct startup sequence ──────────────────────────────────────────────────
# 1. postgres must be healthy before airflow-init runs
# 2. airflow-init must complete before webserver/scheduler start
# Docker Compose handles this via depends_on conditions in docker-compose.yml

up:
	docker compose up -d --build
	@echo ""
	@echo "==> Waiting for airflow-init to complete..."
	@timeout 120 bash -c 'until docker inspect airflow-init --format "{{.State.Status}}" 2>/dev/null | grep -q "exited"; do sleep 3; printf "."; done'
	@echo ""
	@EXIT=$$(docker inspect airflow-init --format "{{.State.ExitCode}}" 2>/dev/null); \
	  if [ "$$EXIT" = "0" ]; then \
	    echo "==> airflow-init succeeded. Creating admin user..."; \
	    docker exec airflow-webserver airflow users create \
	      --username admin --firstname Healthcare --lastname Admin \
	      --role Admin --email admin@healthcare.local \
	      --password admin_secret_2026 2>/dev/null || true; \
	  else \
	    echo "ERROR: airflow-init failed (exit $$EXIT). Check: docker logs airflow-init"; \
	  fi
	sudo chown -R 50000:0 ./logs/airflow
	docker compose restart airflow-init airflow-webserver airflow-scheduler

fix-airflow:
	sudo chown -R 50000:0 ./logs/airflow
	sudo chmod -R 775 ./logs/airflow
	docker compose restart airflow-init airflow-webserver airflow-scheduler

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

# ── Shell access ──────────────────────────────────────────────────────────────
shell-namenode:
	docker exec -it namenode bash

shell-clickhouse:
	docker exec -it clickhouse clickhouse-client \
	  --user healthcare_user --password ch_secret_2026 --database healthcare

shell-airflow:
	docker exec -it airflow-webserver bash

shell-spark:
	docker exec -it spark-master bash

# ── Data ──────────────────────────────────────────────────────────────────────
generate-data:
	bash data/synthea/generate_data.sh

init-hdfs:
	docker exec namenode hdfs dfs -mkdir -p hdfs://namenode:9001/data/raw
	docker exec namenode hdfs dfs -mkdir -p hdfs://namenode:9001/data/processed
	docker exec namenode hdfs dfs -mkdir -p hdfs://namenode:9001/data/features
	docker exec namenode hdfs dfs -chmod -R 777 hdfs://namenode:9001/data

load-hdfs:
	docker exec namenode sh -c "hdfs dfs -put /mnt/data/raw/* hdfs://namenode:9001/data/raw/"
	docker exec namenode hdfs dfs -ls hdfs://namenode:9001/data/raw/
# 	docker cp ./data/raw/ namenode:/tmp/data_raw/
# 	docker exec namenode sh -c "hdfs dfs -moveFromLocal /tmp/data_raw/*.csv hdfs://namenode:9001/data/raw/"
# 	docker exec namenode rm -rf /tmp/data_raw/
# 	docker exec namenode hdfs dfs -ls hdfs://namenode:9001/data/raw/

clean-hdfs:
	docker exec namenode sh -c "hdfs dfs -rm -r hdfs://namenode:9001/data/raw/*"	

init_clickhouse:
	docker exec -i clickhouse clickhouse-client --multiquery < docker/clickhouse/init/01_create_schema.sql

load-clickhouse:
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.patients;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.conditions;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.medications;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.observations;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.recommendations;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.patient_features;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.encounters;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.procedures;"
	bash scripts/load_clickhouse.sh

clean-clickhouse:
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.patients;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.conditions;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.medications;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.observations;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.recommendations;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.patient_features;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.encounters;"
	docker exec clickhouse clickhouse-client --query "TRUNCATE TABLE IF EXISTS healthcare.procedures;"

create-airflow-user:
	docker exec airflow-webserver airflow users create \
	  --username admin --firstname Healthcare --lastname Admin \
	  --role Admin --email admin@healthcare.local \
	  --password admin_secret_2026 2>/dev/null || echo "User already exists"

init-everything: init-hdfs init_clickhouse create-airflow-user	

# ── Verify ────────────────────────────────────────────────────────────────────
verify:
	@echo "=== Container status ==="
	@docker compose ps
	@echo ""
	@echo "=== ClickHouse ==="
	@docker exec clickhouse wget -qO- \
	  'http://healthcare_user:ch_secret_2026@localhost:8123/?query=SHOW%20TABLES%20FROM%20healthcare' \
	  2>/dev/null && echo " OK" || echo " NOT READY"
	@echo "=== Airflow ==="
	@curl -sf http://localhost:8081/health | python3 -m json.tool 2>/dev/null || echo " NOT READY"
	@echo "=== MLflow ==="
	@curl -sf http://localhost:5001/ -o /dev/null && echo " OK" || echo " NOT READY"
	@echo "=== Spark Master ==="
	@curl -sf http://localhost:8080/ -o /dev/null && echo " OK" || echo " NOT READY"
	@echo "=== HDFS ==="
	@curl -sf http://localhost:9870/ -o /dev/null && echo " OK" || echo " NOT READY"

pipeline:
	docker exec airflow-webserver airflow dags trigger dag_ingest_to_hdfs

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	docker compose down -v --remove-orphans
	sudo rm -rf data/processed/* data/features/* data/clickhouse/* models/* logs/*
