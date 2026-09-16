from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from src.domain.arxiv_ingestion import (
    download_papers,
    generate_daily_report,
    index_papers_hybrid,
    parse_pdfs,
    setup_environment,
)

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

# Create the DAG
dag = DAG(
    "arxiv_paper_ingestion",
    default_args=default_args,
    description="Daily arXiv CS.AI paper pipeline: download → parse → store → chunk & embed → hybrid OpenSearch indexing",
    schedule="0 6 * * 1-5",  # Monday-Friday at 6 AM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search", "embeddings", "chunks"],
)

# Task definitions
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

download_task = PythonOperator(
    task_id="download_papers",
    python_callable=download_papers,
    dag=dag,
)

parse_task = PythonOperator(
    task_id="parse_pdfs",
    python_callable=parse_pdfs,
    dag=dag,
)

index_hybrid_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_papers_hybrid,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_daily_report,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="""
    echo "Cleaning up old cached PDFs..."
    # Remove PDFs older than 30 days to manage disk space (host ./data/arxiv_pdfs)
    find /opt/airflow/data/arxiv_pdfs -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Task dependencies
setup_task >> download_task >> parse_task >> index_hybrid_task >> report_task >> cleanup_task
