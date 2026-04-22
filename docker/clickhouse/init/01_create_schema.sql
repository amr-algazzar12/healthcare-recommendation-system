-- ClickHouse bootstrap DDL
-- Runs on first container start via /docker-entrypoint-initdb.d

CREATE DATABASE IF NOT EXISTS healthcare;

-- Raw patients table (populated by Airflow ingestion DAG)
CREATE TABLE IF NOT EXISTS healthcare.patients
(
    patient_id      String,
    birthdate       Date,
    deathdate       Nullable(Date),
    ssn             String,
    first           String,
    last            String,
    gender          FixedString(1),
    race            String,
    ethnicity       String,
    city            String,
    state           String,
    zip             String,
    lat             Nullable(Float64),
    lon             Nullable(Float64),
    healthcare_expenses Float64,
    healthcare_coverage Float64
)
ENGINE = MergeTree()
ORDER BY patient_id;

-- Conditions
CREATE TABLE IF NOT EXISTS healthcare.conditions
(
    start_date      Date,
    stop_date       Nullable(Date),
    patient_id      String,
    encounter_id    String,
    code            String,
    description     String
)
ENGINE = MergeTree()
ORDER BY (patient_id, start_date);

-- Medications
CREATE TABLE IF NOT EXISTS healthcare.medications
(
    start_date      Date,
    stop_date       Nullable(Date),
    patient_id      String,
    encounter_id    String,
    code            String,
    description     String,
    reasoncode      Nullable(String),
    reasondescription Nullable(String)
)
ENGINE = MergeTree()
ORDER BY (patient_id, start_date);

-- Observations
CREATE TABLE IF NOT EXISTS healthcare.observations
(
    date            DateTime,
    patient_id      String,
    encounter_id    String,
    code            String,
    description     String,
    value           String,
    units           Nullable(String),
    type            String
)
ENGINE = MergeTree()
ORDER BY (patient_id, date);

-- Encounters
CREATE TABLE IF NOT EXISTS healthcare.encounters
(
    encounter_id    String,
    start_dt        DateTime,
    stop_dt         Nullable(DateTime),
    patient_id      String,
    encounterclass  String,
    code            String,
    description     String,
    reasoncode      Nullable(String),
    reasondescription Nullable(String),
    base_encounter_cost Float64,
    total_claim_cost Float64,
    payer_coverage  Float64
)
ENGINE = MergeTree()
ORDER BY (patient_id, encounter_id);

-- Procedures
CREATE TABLE IF NOT EXISTS healthcare.procedures
(
    date            Date,
    patient_id      String,
    encounter_id    String,
    code            String,
    description     String,
    reasoncode      Nullable(String),
    reasondescription Nullable(String),
    base_cost       Float64
)
ENGINE = MergeTree()
ORDER BY (patient_id, date);

-- ML feature store — written by Spark, queried for model serving
CREATE TABLE IF NOT EXISTS healthcare.patient_features
(
    patient_id              String,
    age                     Int32,
    gender_encoded          Int8,
    race_encoded            Int8,
    num_conditions          Int32,
    num_medications         Int32,
    num_encounters          Int32,
    has_diabetes            UInt8,
    has_hypertension        UInt8,
    has_asthma              UInt8,
    has_hyperlipidemia      UInt8,
    has_coronary_disease    UInt8,
    condition_vector        Array(Float32),
    medication_history_flags Array(UInt8),
    feature_version         String,
    created_at              DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY patient_id;

-- Recommendations output table
CREATE TABLE IF NOT EXISTS healthcare.recommendations
(
    recommendation_id   UUID DEFAULT generateUUIDv4(),
    patient_id          String,
    model_version       String,
    rank                Int32,
    treatment_code      String,
    treatment_name      String,
    score               Float64,
    explanation         String,
    created_at          DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (patient_id, created_at);
