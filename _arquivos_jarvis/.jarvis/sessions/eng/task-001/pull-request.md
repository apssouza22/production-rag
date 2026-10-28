# TASK-001: Exportação de Markdown no pdf_parser

## Descrição

Acrescenta ao parser de PDF uma exportação **opcional** do artigo para Markdown, com as figuras (PNG) e as tabelas (CSV) gravadas ao lado, para revisão e depuração do que o Docling extraiu. A exportação é ativada pelo parâmetro novo `markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`; sem ele, o comportamento é o de hoje e a ingestão não muda.

## Tarefas Relacionadas

- [x] TASK-001 (controle próprio, sem board)

## Mudanças Propostas

- **Exportação** (`src/domain/pdf_parser/markdown_export.py`, novo): monta `{pdf_stem}.md` numa única passada por `doc.iterate_items()` e grava `{pdf_stem}_assets/` com `figure_NNN.png` e `table_NNN.csv` (contadores separados a partir de 001, na ordem de visita).
  - Tabelas (`TABLE`/`DOCUMENT_INDEX`) ficam inline em GFM, sem link para o CSV; o CSV (UTF-8, vírgula, `QUOTE_MINIMAL`) espelha o grid e só tem cabeçalho quando o grid tem a linha separadora.
  - Figuras (`PICTURE`/`CHART`): o `alt` vem de uma legenda "Figure", cada legenda usada uma vez — primeiro a próxima legenda livre na seção, depois as legendas que o Docling ligou à figura, depois a legenda livre anterior na seção; senão `Figure NNN`. O `alt` é escapado e o link é codificado para URL.
  - Figura sem imagem vira `_{alt} (image unavailable)_`, sem link quebrado.
  - Reexportar no mesmo diretório remove o `.md` e os assets da exportação anterior.
- **Parser** (`docling.py`, `parser.py`): parâmetro `markdown_output_dir` (repassado pelo serviço); exportação depois de `extract_sections_from_document`; erro de gravação vira `PDFParsingException` com a mensagem original; `generate_picture_images=True` sempre; `accelerator_device` opcional no `DoclingParser` (padrão: comportamento atual, que respeita `DOCLING_DEVICE`).
- **Modelo** (`models.py`): `PdfContent.markdown_path` e `PdfContent.artifacts_dir` (`Optional[Path]`, padrão `None`).
- **Testes**: unitários com documentos falsos e com `DoclingDocument` real em memória; teste de integração com o Docling real no PDF `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` (15 páginas, com README de origem e licença).
- **Fora do escopo original (pedido durante a tarefa):**
  - a coleta de **toda** a suíte estava quebrada na `main` (`pytest_plugins` em `tests/unit/domain/rerank/conftest.py`, conftest não raiz); as fixtures compartilhadas foram para `tests/unit/domain/conftest.py`;
  - imports ordenados (ruff `I001`) em 5 arquivos sem relação com a feature;
  - `requires-python = ">=3.12,<3.13"` (o `uv` escolhia o Python 3.15, onde o `torch` 2.8.0 não instala); `uv.lock` regenerado sem mudar nenhuma versão;
  - documentação: `AGENTS.md`, `README.md`, `docs/onboarding.md` e `docs/ARTICLE.md`.

## Testes Realizados

- `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py`: 76 passaram; `markdown_export.py` com 100% das linhas cobertas.
- `uv run pytest tests/integration/pdf_parser`: 6 passaram (Docling real em CPU; 7 figuras, todas com o `alt` da legenda certa, conferido nos PNGs; 11 tabelas; CSV igual à tabela inline).
- `make lint`: ruff e mypy sem erros.
- `make test`: 290 passaram, 3 falharam e 7 erros, **todos anteriores a esta branch** (veja Observações).

Para ver a saída:

```python
from pathlib import Path
from src.domain.pdf_parser.docling import DoclingParser

DoclingParser(30, 20, accelerator_device="cpu").parse_pdf(
    Path("tests/fixtures/arxiv_pdfs/2609.17429v1.pdf"), markdown_output_dir=Path("/tmp/out")
)
# abrir /tmp/out/2609.17429v1.md
```

## Observações Específicas

- **Falhas anteriores à branch no `make test`:** `tests/api/*` (1 falha e 7 erros) precisam de PostgreSQL em `localhost:5412` e OpenSearch (`make start`); 2 testes de `tests/unit/platform/graph/test_graph_builder.py` falham também na `main` (o hook `after` do middleware não é chamado) e merecem uma tarefa própria.
- **Custo de `generate_picture_images=True`:** no fixture (15 páginas, CPU), 113,9 s e 99,2 s sem imagens contra 101,9 s e 97,5 s com, sem diferença mensurável; medido com um PDF só, não com um lote.
- **`make test` mais lento:** o teste de integração leva de 1 a 2 minutos e baixa os modelos do Docling na primeira execução.
- **GPU antiga:** GPUs não suportadas pelo `torch` instalado (ex.: compute capability 6.1) falham em CUDA; o teste de integração força CPU, e pela API/Airflow use `DOCLING_DEVICE=cpu` (nota em `docs/onboarding.md`).
- **Pipe escapado numa célula:** o `.md` mantém a forma escapada e o CSV grava a barra vertical sem escape (mesmo conteúdo, grafias diferentes).
- **Só o `alt` é escapado:** títulos, parágrafos e listas saem com o texto do PDF como está.
- **Melhoria futura:** rótulos de texto de diagramas vetoriais saem como parágrafos soltos no `.md`.

## Checklist de Revisão

- [x] As mudanças estão alinhadas com os requisitos da tarefa (24 de 24 critérios de aceitação).
- [x] Os testes unitários/funcionais propostos foram executados e passaram.
- [x] O código segue as convenções de estilo do projeto (ruff).
- [x] A documentação foi atualizada.
- [x] A solução não muda a ingestão nem outras áreas do sistema (o parâmetro é opcional).
- [x] Não há erros ou advertências nas análises de qualidade de código (`make lint`).
- [ ] A solicitação de merge foi revisada por pelo menos um revisor.
