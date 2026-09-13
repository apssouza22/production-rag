# Airflow Configuration

This directory contains Apache Airflow configuration and DAGs for the arXiv Paper Curator project.

## Production-Ready DAGs
- **hello_world_dag.py**: Basic health check DAG for Week 1
- **arxiv_paper_ingestion.py**: Main production DAG for automated arXiv paper fetching and processing

### Production Pipeline Features
- **Daily arXiv ingestion**: Automated fetching of CS.AI papers
- **PDF processing**: Download and parse papers using Docling
- **Database storage**: Store complete paper metadata and content in PostgreSQL
- **Error handling**: Comprehensive retry logic and error reporting
- **Cross-platform compatibility**: Works on macOS, Linux, WSL, and Ubuntu
