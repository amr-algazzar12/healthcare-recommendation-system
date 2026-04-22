# Airflow Configuration Notes

All Airflow configuration is done via environment variables (AIRFLOW__ prefix)
in the `.env` file. No `airflow.cfg` edits are needed.

Key settings:
- AIRFLOW__CORE__EXECUTOR = LocalExecutor
- AIRFLOW__DATABASE__SQL_ALCHEMY_CONN → points to postgres:5432/airflow
- AIRFLOW__CORE__FERNET_KEY → rotate before production
- Webserver at localhost:8081 (host port; container is 8080)
- Admin credentials: admin / admin_secret_2026

## ClickHouse connection in Airflow DAGs

Instead of SSHOperator → Beeline (old Hive approach), DAGs use:
- clickhouse-connect Python SDK via PythonOperator
- Or direct HTTP calls via BashOperator:
    curl -s "http://clickhouse:8123/?query=SELECT+count()+FROM+healthcare.patients" \
         -u healthcare_user:ch_secret_2026

## DAG → ClickHouse pattern
```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host=os.environ['CLICKHOUSE_HOST'],
    port=int(os.environ['CLICKHOUSE_HTTP_PORT']),
    database=os.environ['CLICKHOUSE_DB'],
    username=os.environ['CLICKHOUSE_USER'],
    password=os.environ['CLICKHOUSE_PASSWORD'],
)
result = client.query("SELECT count() FROM patients")
```
