# Contexto da sessão — discovery-001

## Warm-up (rag-pdf-markdown-export)


> Contexto aquecido para **uma tarefa**. O `/eng.start` ou o `/prod.spec*` que usá-lo copia este conteúdo para o `context.md` da sessão e apaga a pasta (`rules-on-demand/warm-up-consume-rules.md`).

## Escopo

| Repositório | Papel | Stack | Branch alvo |
|-------------|-------|-------|-------------|
| production-rag | API de RAG agêntico (busca híbrida, agentes, text-to-SQL) + ingestão de artigos do arXiv | python (>=3.12) / fastapi | main |

**Especialidades carregadas:** backend: `eng-backend` (base) + `eng-backend-python-fastapi` · **Fora do escopo:** frontend, qa, data, automation, platform

**Tarefa prevista:** exportação de Markdown (com figuras PNG e tabelas CSV) no `DoclingParser` / `PDFParserService`, em `production-rag/src/domain/pdf_parser/` (`docling.py`, `parser.py`, `models.py`, `factory.py`). O parser é consumido pela ingestão (`src/domain/arxiv_ingestion/fetching.py` → `parse_pdfs`, chamado por `src/ingest.py`).

## Central docs

Ligado (`filesystem`, `./central-docs`), mas ainda sem consolidação do squad: não existe `CORE/index.md`. Nada lido. Para gerar a primeira carga: `/eng.spec-consolidate --inicial`.

## Entregas recentes

Sem merges na `main` (histórico linear, sem merge commits): usados os 5 últimos commits.

| Repositório | Commit | Título | Arquivos tocados |
|-------------|--------|--------|------------------|
| production-rag | f030fe1, 2026-10-25 | Add ingest python file | `src/ingest.py` |
| production-rag | bbe5887, 2026-10-08 | Decouple airflow from domain code | `airflow/dags/`, `src/domain/arxiv_ingestion/{fetching,indexing,reporting,service}.py` |
| production-rag | ed621df, 2026-10-05 | Remove cache | `src/platform/cache/` (removido), `src/api/hybrid_search.py`, `src/config.py`, `src/dependencies.py`, `src/main.py`, testes de cache |
| production-rag | 36dade9, 2026-10-05 | Remove bifrost | `tests/unit/platform/llm/`, testes de agentes, `tests/unit/platform/bifrost/` (removido) |
| production-rag | 002cd66, 2026-09-29 | Add langfuse experiment integration | `evals/rag_evals/langfuse_experiment.py`, `tests/evals/` |

Tendência recente: simplificação (remoção de cache e do bifrost) e desacoplamento do Airflow em relação ao código de domínio. A ingestão agora tem um ponto de entrada próprio (`src/ingest.py`) que chama o domínio `arxiv_ingestion`, que por sua vez usa o `pdf_parser`. Mudanças no `pdf_parser` devem manter esse fluxo funcionando sem depender do Airflow.

## Ferramentas (context7)

| Ferramenta | Versão no projeto (`uv.lock`) | Versão atual | Observação |
|------------|-------------------------------|--------------|------------|
| fastapi | 0.116.1 | não confirmada | O Context7 não informa a versão; os padrões `lifespan` e `Annotated[..., Depends]` usados no projeto seguem a recomendação atual |
| langgraph | 1.2.9 | não confirmada | O Context7 indexa até a 1.0.8, atrás da versão do projeto |
| sqlalchemy | 2.0.43 | série 2.0 | O Context7 indexa a documentação 2.0 |
| opensearch-py | 3.0.0 | não confirmada | O Context7 não informa a versão |
| pytest | 8.4.2 | 9.1.1 (mais recente indexada) | Uma major atrás; não afeta a tarefa |

Relevante para a tarefa (fora das 5 do mapa): `docling` 2.52.0 travado no `uv.lock`.
