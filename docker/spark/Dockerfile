FROM apache/spark:3.5.3

USER root

# Install build essentials for C-extensions (needed for XGBoost/Pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for the default Python 3.8
RUN python3 -m pip install --upgrade pip

# Install Python 3.8 compatible versions
# Note: clickhouse-connect must be <= 0.5.20 to avoid the IPv6Address error
RUN python3 -m pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    xgboost==1.7.6 \
    mlflow==2.3.2 \
    pyarrow==11.0.0 \
    scipy==1.10.1 \
    clickhouse-connect==0.5.20