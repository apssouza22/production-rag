-- Created only on first Postgres volume init (docker-entrypoint-initdb.d).
-- Application data lives in rag_db (POSTGRES_DB). Airflow metadata uses airflow_db.
CREATE DATABASE airflow_db OWNER rag_user;