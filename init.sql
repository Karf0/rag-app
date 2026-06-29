-- Runs once on first db init (mounted into /docker-entrypoint-initdb.d).
-- Creates the pgvector extension so the api can create vector columns.
CREATE EXTENSION IF NOT EXISTS vector;
