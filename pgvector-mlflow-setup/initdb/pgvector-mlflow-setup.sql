-- mlflow-postgres: initdb script for pgvector, mlflow, and checkpointer setup
-- Runs once, on first DB init, connected to the default database (lc_vector_db).

-- Enable pgvector on the primary lc_vector_db (created from POSTGRES_* env).
CREATE EXTENSION IF NOT EXISTS vector;

-- mlflow user + database (backend store for the MLflow tracking server).
CREATE USER mlflow WITH PASSWORD 'mlflow!';
CREATE DATABASE mlflow_db OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;


-- Langchain CheckPointer database
CREATE DATABASE lc_checkpointer_db OWNER langchain;
GRANT ALL PRIVILEGES ON DATABASE lc_checkpointer_db TO langchain;

