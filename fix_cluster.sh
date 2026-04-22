#!/usr/bin/env bash
# =============================================================================
# fix_cluster.sh — Fix all crashing containers in-place
# Run from inside your project root:  bash fix_cluster.sh
# =============================================================================
set -euo pipefail

echo "==> Stopping cluster..."
docker compose down 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — docker-compose.yml  (all service issues)
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Rewriting docker-compose.yml..."

cat > docker-compose.yml << 'COMPOSEEOF'
version: "3.8"

# ── Airflow shared anchor ─────────────────────────────────────────────────────
x-airflow-common: &airflow-common
  image: apache/airflow:2.8.1-python3.9
  env_file: .env
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow_secret_2026@postgres:5432/airflow
    AIRFLOW__CORE__FERNET_KEY: 46BKJoQYlPPOexq0OhDZnIlNepKFf87WFwLt0nfdW0M=
    AIRFLOW__WEBSERVER__SECRET_KEY: healthcare_webserver_secret_2026
    AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
    AIRFLOW_HOME: /opt/airflow
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "true"
    PYTHONPATH: /opt/airflow/src
    CLICKHOUSE_HOST: clickhouse
    CLICKHOUSE_HTTP_PORT: "8123"
    CLICKHOUSE_DB: healthcare
    CLICKHOUSE_USER: healthcare_user
    CLICKHOUSE_PASSWORD: ch_secret_2026
    HDFS_NAMENODE_HOST: namenode
    HDFS_NAMENODE_PORT: "9001"
    MLFLOW_TRACKING_URI: http://mlflow:5001
    _AIRFLOW_WWW_USER_CREATE: "true"
    _AIRFLOW_WWW_USER_USERNAME: admin
    _AIRFLOW_WWW_USER_PASSWORD: admin_secret_2026
    _AIRFLOW_WWW_USER_FIRSTNAME: Healthcare
    _AIRFLOW_WWW_USER_LASTNAME: Admin
    _AIRFLOW_WWW_USER_EMAIL: admin@healthcare.local
    _AIRFLOW_WWW_USER_ROLE: Admin
  volumes:
    - ./dags:/opt/airflow/dags
    - ./src:/opt/airflow/src
    - ./data:/opt/airflow/data
    - ./models:/opt/airflow/models
    - ./logs/airflow:/opt/airflow/logs
    - mlflow_artifacts:/mlflow/artifacts
  user: "50000:0"
  networks:
    - healthcare_net
  depends_on:
    postgres:
      condition: service_healthy

# ── Hadoop shared anchor ──────────────────────────────────────────────────────
x-hadoop-common: &hadoop-common
  env_file:
    - hadoop.env
    - .env
  networks:
    - healthcare_net
  restart: unless-stopped

services:

  # ── PostgreSQL ──────────────────────────────────────────────────────────────
  postgres:
    image: postgres:15.3-alpine
    container_name: postgres
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow_secret_2026
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/postgres/init-mlflow-db.sql:/docker-entrypoint-initdb.d/init-mlflow-db.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 5s
      timeout: 5s
      retries: 10
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── pgAdmin ─────────────────────────────────────────────────────────────────
  pgadmin:
    image: dpage/pgadmin4:7.8
    container_name: pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@healthcare.local
      PGADMIN_DEFAULT_PASSWORD: pgadmin_secret_2026
    ports:
      - "5055:80"
    depends_on:
      - postgres
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── ClickHouse ──────────────────────────────────────────────────────────────
  # Healthcheck uses wget (built-in) — clickhouse-client is NOT in PATH by default
  clickhouse:
    image: clickhouse/clickhouse-server:24.3
    container_name: clickhouse
    environment:
      CLICKHOUSE_DB: healthcare
      CLICKHOUSE_USER: healthcare_user
      CLICKHOUSE_PASSWORD: ch_secret_2026
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: "1"
    volumes:
      - ./data/clickhouse:/var/lib/clickhouse
      - ./docker/clickhouse/config.xml:/etc/clickhouse-server/config.d/custom.xml:ro
      - ./docker/clickhouse/users.xml:/etc/clickhouse-server/users.d/healthcare_user.xml:ro
      - ./docker/clickhouse/init:/docker-entrypoint-initdb.d:ro
      - ./data/features:/mnt/features:ro
    ports:
      - "8123:8123"
      - "9900:9000"
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- 'http://healthcare_user:ch_secret_2026@localhost:8123/?query=SELECT%201' || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── HDFS Namenode ───────────────────────────────────────────────────────────
  namenode:
    image: bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
    container_name: namenode
    <<: *hadoop-common
    environment:
      CLUSTER_NAME: healthcare-cluster
      CORE_CONF_fs_defaultFS: hdfs://namenode:9001
    volumes:
      - namenode_data:/hadoop/dfs/name
      - ./data/raw:/mnt/data/raw
      - ./data/processed:/mnt/data/processed
      - ./data/features:/mnt/data/features
      - ./src:/mnt/src
      - ./models:/mnt/models
      - mlflow_artifacts:/mlflow/artifacts
    ports:
      - "9870:9870"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9870"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  # ── HDFS Datanode ───────────────────────────────────────────────────────────
  datanode:
    image: bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8
    container_name: datanode
    <<: *hadoop-common
    environment:
      SERVICE_PRECONDITION: "namenode:9870"
      CORE_CONF_fs_defaultFS: hdfs://namenode:9001
    volumes:
      - datanode_data:/hadoop/dfs/data
    ports:
      - "9864:9864"
    depends_on:
      - namenode

  # ── YARN ResourceManager ────────────────────────────────────────────────────
  resourcemanager:
    image: bde2020/hadoop-resourcemanager:2.0.0-hadoop3.2.1-java8
    container_name: resourcemanager
    <<: *hadoop-common
    environment:
      SERVICE_PRECONDITION: "namenode:9001 namenode:9870 datanode:9864"
    ports:
      - "8088:8088"
    depends_on:
      - namenode
      - datanode

  # ── YARN NodeManager ────────────────────────────────────────────────────────
  nodemanager:
    image: bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8
    container_name: nodemanager
    <<: *hadoop-common
    environment:
      SERVICE_PRECONDITION: "namenode:9001 namenode:9870 datanode:9864 resourcemanager:8088"
    ports:
      - "8042:8042"
    depends_on:
      - resourcemanager

  # ── Spark History Server ────────────────────────────────────────────────────
  historyserver:
    image: bde2020/hadoop-historyserver:2.0.0-hadoop3.2.1-java8
    container_name: historyserver
    <<: *hadoop-common
    environment:
      SERVICE_PRECONDITION: "namenode:9001 namenode:9870 datanode:9864 resourcemanager:8088"
    volumes:
      - historyserver_data:/hadoop/yarn/timeline
    ports:
      - "8188:8188"
    depends_on:
      - resourcemanager

  # ── Spark Master ────────────────────────────────────────────────────────────
  # SPARK_NO_DAEMONIZE keeps the process in the foreground so Docker doesn't
  # think the container exited.
  spark-master:
    image: apache/spark:3.5.3
    container_name: spark-master
    user: root
    environment:
      - SPARK_NO_DAEMONIZE=true
      - SPARK_MASTER_HOST=spark-master
      - SPARK_MASTER_PORT=7077
      - SPARK_MASTER_WEBUI_PORT=8080
    entrypoint: ["/opt/spark/sbin/start-master.sh"]
    ports:
      - "8080:8080"
      - "7077:7077"
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── Spark Worker ────────────────────────────────────────────────────────────
  spark-worker:
    image: apache/spark:3.5.3
    container_name: spark-worker
    user: root
    environment:
      - SPARK_NO_DAEMONIZE=true
      - SPARK_WORKER_MEMORY=4g
      - SPARK_WORKER_CORES=4
      - SPARK_WORKER_WEBUI_PORT=8081
    entrypoint: ["/opt/spark/sbin/start-worker.sh", "spark://spark-master:7077"]
    ports:
      - "8082:8081"
    depends_on:
      - spark-master
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── Airflow Init ────────────────────────────────────────────────────────────
  # restart: "no" is critical — this is a one-shot init job, not a service.
  # It uses the _AIRFLOW_WWW_USER_* env vars from the anchor to create the
  # admin user automatically during `db migrate`.
  airflow-init:
    <<: *airflow-common
    container_name: airflow-init
    command: ["db", "migrate"]
    restart: "no"
    # Override the depends_on from the anchor — init only needs postgres
    depends_on:
      postgres:
        condition: service_healthy

  # ── Airflow Webserver ───────────────────────────────────────────────────────
  airflow-webserver:
    <<: *airflow-common
    container_name: airflow-webserver
    command: webserver
    ports:
      - "8081:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    depends_on:
      postgres:
        condition: service_healthy
      airflow-init:
        condition: service_completed_successfully
    restart: unless-stopped

  # ── Airflow Scheduler ───────────────────────────────────────────────────────
  airflow-scheduler:
    <<: *airflow-common
    container_name: airflow-scheduler
    command: scheduler
    depends_on:
      postgres:
        condition: service_healthy
      airflow-init:
        condition: service_completed_successfully
    restart: unless-stopped

  # ── MLflow ──────────────────────────────────────────────────────────────────
  mlflow:
    build:
      context: ./docker/mlflow
      dockerfile: Dockerfile
    container_name: mlflow
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql+psycopg2://airflow:airflow_secret_2026@postgres:5432/mlflow
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    ports:
      - "5001:5001"
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - healthcare_net
    restart: unless-stopped

  # ── App (Flask API + Streamlit) ─────────────────────────────────────────────
  # Runs a health endpoint so the container stays alive during development
  # even before the full API is implemented.
  app:
    build:
      context: ./docker/app
      dockerfile: Dockerfile
    container_name: app
    environment:
      - CLICKHOUSE_HOST=clickhouse
      - CLICKHOUSE_HTTP_PORT=8123
      - CLICKHOUSE_DB=healthcare
      - CLICKHOUSE_USER=healthcare_user
      - CLICKHOUSE_PASSWORD=ch_secret_2026
      - MLFLOW_TRACKING_URI=http://mlflow:5001
      - FLASK_ENV=development
    volumes:
      - ./src:/app/src
      - ./models:/app/models
      - mlflow_artifacts:/mlflow/artifacts
    ports:
      - "5050:5050"
      - "8501:8501"
    depends_on:
      clickhouse:
        condition: service_healthy
      mlflow:
        condition: service_started
    networks:
      - healthcare_net
    restart: unless-stopped

networks:
  healthcare_net:
    driver: bridge
    name: healthcare_net

volumes:
  postgres_data:
  namenode_data:
  datanode_data:
  historyserver_data:
  mlflow_artifacts:
COMPOSEEOF

echo "    docker-compose.yml: OK"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — ClickHouse healthcheck: users.xml must allow default user too
# The wget healthcheck hits localhost without auth first; ensure anonymous
# ping works OR use the explicit user:pass URL (already done above).
# Also regenerate config.xml with correct log path that exists in the image.
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Fixing ClickHouse config..."

cat > docker/clickhouse/config.xml << 'CHEOF'
<clickhouse>
  <listen_host>0.0.0.0</listen_host>
  <http_port>8123</http_port>
  <tcp_port>9000</tcp_port>
  <timezone>UTC</timezone>
  <logger>
    <level>warning</level>
    <console>1</console>
  </logger>
  <user_files_path>/mnt/features/</user_files_path>
</clickhouse>
CHEOF

echo "    clickhouse/config.xml: OK"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — App Dockerfile: stub Flask app so container doesn't crash on start
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Fixing app Dockerfile and stub app..."

cat > docker/app/Dockerfile << 'DOCKEREOF'
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy stub entrypoint — real src/ is volume-mounted at runtime
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5050 8501

ENTRYPOINT ["/entrypoint.sh"]
DOCKEREOF

cat > docker/app/entrypoint.sh << 'ENTRYEOF'
#!/bin/bash
set -e

# If the real Flask app exists, use it; otherwise run the stub
FLASK_APP_PATH="/app/src/api/app.py"
STREAMLIT_APP_PATH="/app/src/dashboard/app.py"

if [ -f "$FLASK_APP_PATH" ] && grep -q "Flask\|flask" "$FLASK_APP_PATH" 2>/dev/null; then
  echo "Starting Flask API..."
  python "$FLASK_APP_PATH" &
else
  echo "Flask stub: real app not implemented yet, running health endpoint..."
  python -c "
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'stub — implement src/api/app.py'})

@app.route('/recommend', methods=['POST'])
def recommend():
    return jsonify({'status': 'stub', 'message': 'Model serving not yet implemented'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
" &
fi

if [ -f "$STREAMLIT_APP_PATH" ] && grep -q "streamlit\|st\." "$STREAMLIT_APP_PATH" 2>/dev/null; then
  echo "Starting Streamlit dashboard..."
  streamlit run "$STREAMLIT_APP_PATH" --server.port 8501 --server.address 0.0.0.0
else
  echo "Streamlit stub: real dashboard not implemented yet..."
  python -c "
import streamlit as st
st.title('Healthcare Recommendation System')
st.info('Dashboard stub — implement src/dashboard/app.py')
st.markdown('### Services')
st.write('Flask API: http://localhost:5050')
st.write('MLflow: http://localhost:5001')
" &
  # Keep container alive
  tail -f /dev/null
fi
ENTRYEOF

chmod +x docker/app/entrypoint.sh

echo "    app/: OK"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — Airflow: fix log directory permissions so uid 50000 can write
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Fixing Airflow log directory permissions..."
mkdir -p logs/airflow
chmod -R 777 logs/airflow
mkdir -p dags src models data/raw data/processed data/features
chmod -R 777 dags src models

echo "    log permissions: OK"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — Create admin user separately AFTER db migrate completes
# The _AIRFLOW_WWW_USER_* vars only work with `airflow db init`, not
# `airflow db migrate` in 2.8. We add an explicit users create step.
# ─────────────────────────────────────────────────────────────────────────────
cat > scripts/airflow_create_user.sh << 'AUEOF'
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
AUEOF
chmod +x scripts/airflow_create_user.sh

# ─────────────────────────────────────────────────────────────────────────────
# Update Makefile with corrected startup sequence
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Updating Makefile..."

cat > Makefile << 'MAKEEOF'
.PHONY: up down restart logs ps verify \
        shell-namenode shell-clickhouse shell-airflow shell-spark \
        generate-data init-hdfs load-hdfs load-clickhouse create-airflow-user \
        pipeline clean

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
	docker exec namenode hdfs dfs -put -f /mnt/data/raw/*.csv hdfs://namenode:9001/data/raw/
	docker exec namenode hdfs dfs -ls hdfs://namenode:9001/data/raw/

load-clickhouse:
	bash scripts/load_clickhouse.sh

create-airflow-user:
	docker exec airflow-webserver airflow users create \
	  --username admin --firstname Healthcare --lastname Admin \
	  --role Admin --email admin@healthcare.local \
	  --password admin_secret_2026 2>/dev/null || echo "User already exists"

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
MAKEEOF

echo "    Makefile: OK"

echo ""
echo "============================================================"
echo " All fixes applied. Start the cluster with:"
echo ""
echo "   make up"
echo ""
echo " Then verify with:"
echo "   make verify"
echo ""
echo " If airflow-init still fails, check:"
echo "   docker logs airflow-init"
echo "   docker logs postgres"
echo "============================================================"