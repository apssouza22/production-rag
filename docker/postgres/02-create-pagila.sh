#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE pagila' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pagila')\gexec
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname pagila -f /docker-entrypoint-initdb.d/pagila.sql