# Gate checklist — TASK-001

> _Recriado em 2026-10-26 a partir do histórico da sessão: versão final da 3ª rodada (decisões preenchidas), antes de o `gate-checklist.md` ser atualizado pela 4ª rodada._

> Branch: `TASK-001-exportacao-markdown-pdf-parser` → `main` · Data: 2026-10-26 · **3ª rodada**
> **Semáforo:** 🟢 pronto para PR

## Resumo

A branch acrescenta a exportação opcional de Markdown ao parser de PDF (`markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`; módulo `src/domain/pdf_parser/markdown_export.py`), com testes unitários e de integração no PDF real, e corrige, a pedido do usuário, a coleta quebrada da suíte, a ordem de imports e a versão do Python. As 27 lacunas da 1ª rodada e as 13 da 2ª foram corrigidas (fases 6 a 11 do `plan.md`) e os quatro agentes confirmaram todas. Nesta 3ª rodada, **24 de 24 critérios de aceitação estão atendidos** e restam **5 lacunas, todas ⚪ menores (0 🔴, 0 🟡)**: uma otimização de custo raro e quatro ajustes de texto (dois na documentação do repositório, dois em documentos da sessão). O PR Brief abaixo foi reescrito com o comportamento atual (resolve a lacuna N1 apontada nesta rodada).

## Lacunas

### Produto e documentação

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| N2 | A Q4 ainda mostra o texto antigo `(imagem indisponível)` na opção escolhida e na "Decisão adotada"; só a nota abaixo diz que mudou para `(image unavailable)` | `.jarvis/sessions/eng/task-001/architecture.md:518`, `:539` | Quem lê só a opção escolhida vê o texto antigo | Trocar o texto nas duas linhas e manter a nota como histórico | ⚪ Menor | Corrigir | sem commit (`architecture.md` da sessão) |
| N3 | Fases antigas do plano citam `(imagem indisponível)` | `.jarvis/sessions/eng/task-001/plan.md:233`, `:267` | Só confunde quem lê as fases em ordem (é histórico) | Nota "substituído na FASE 10" | ⚪ Menor | Corrigir | sem commit (`plan.md` da sessão) |

### Código

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| C12 | `take_text` percorre todas as legendas a cada chamada para achar as posições com o mesmo texto | `markdown_export.py:231` | O(figuras × legendas) no passo 2 do `alt`; só pesa em PDFs com centenas de figuras em que a busca adiante falha muito | Montar `by_text: dict[str, list[int]]` no `__post_init__` do `_CaptionIndex` | ⚪ Menor | Corrigir | [`646abcb`](https://github.com/phelipperibeiro/production-rag/commit/646abcbd73a47944b56d56bb68c06464140526af) |

### Testes

Nenhuma lacuna encontrada. T6–T9 resolvidas; os regex do teste de integração continuam válidos com o link codificado e o `alt` escapado.

### Não funcionais

Nenhuma lacuna nova.

### Interface e UX

Nenhuma lacuna encontrada (a branch não tem interface).

### Segurança

Nenhuma lacuna encontrada. Varredura de segredos no diff: 0 ocorrências; nenhum `.env`, `.pem` ou `.key` no diff.

### Documentação do projeto

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| D10 | O bullet do Docling não diz que reexportar na mesma pasta substitui o `.md` e os assets anteriores, nem que figura sem imagem vira texto em itálico | `docs/onboarding.md:41` | Quem inspeciona a saída pode estranhar arquivos sumindo ou a figura sem link | Acrescentar: "reexportar o mesmo PDF na mesma pasta substitui o `.md` e os `figure_*.png`/`table_*.csv` anteriores; figuras sem imagem aparecem como `_<legenda> (image unavailable)_`" | ⚪ Menor | Corrigir | [`646abcb`](https://github.com/phelipperibeiro/production-rag/commit/646abcbd73a47944b56d56bb68c06464140526af) |
| D11 | A docstring do módulo não cita o fallback de figura sem imagem (só a de `_render_picture`) | `src/domain/pdf_parser/markdown_export.py:1-6` | Visão geral do módulo incompleta | "pictures are saved as PNG and linked from the Markdown (a picture without an image is rendered as italic alt text followed by `(image unavailable)`)" | ⚪ Menor | Corrigir | [`646abcb`](https://github.com/phelipperibeiro/production-rag/commit/646abcbd73a47944b56d56bb68c06464140526af) |

## Evidências

- `git diff main...HEAD`: `src` +422 −5; `tests` +849 −61; `uv.lock` +19 −1153 (só remoção de wheels/marcadores de Python 3.13+, nenhuma versão alterada); PDF fixture (652 KB).
- `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py -q`: 76 passaram.
- `uv run pytest tests/integration/pdf_parser -q`: 6 passaram (Docling real, CPU; 7 figuras com o `alt` da própria legenda, 11 tabelas).
- `make lint`: ruff "All checks passed!"; mypy "Success: no issues found in 132 source files" (`ignore_errors = true`).
- `make test`: 290 passaram, 3 falharam, 7 erros, todos anteriores à branch (`tests/api/*` sem PostgreSQL/OpenSearch; 2 de `tests/unit/platform/graph/test_graph_builder.py`, que falham também na `main`).
- Cobertura (unitários, linhas + ramos): `markdown_export.py` 99% (100% das linhas; 2 ramos parciais de `_split_gfm_row` aceitos), `models.py`/`exceptions.py`/`factory.py` 100%, `parser.py` 97%, `docling.py` 78% (linhas não cobertas anteriores à branch); pacote 93%.
- Varredura de segredos no diff: 0 ocorrências.
- Agentes (3ª rodada, só leitura): `prod.pm-checker` (24/24; P10, P11, D9 resolvidas), `eng.dev-code-reviewer` (C8–C11 corrigidas; 🟢; bordas de `bisect`, `quote` com não ASCII, `unlink` e escape testadas sem bug), `eng.qa.test-planner` (T6–T9 resolvidas; sem lacuna nova), `eng.docs-writer` (D7–D9 resolvidas).

### Rodadas anteriores (histórico)

- **1ª rodada:** 27 lacunas (0 🔴, 14 🟡, 13 ⚪), todas corrigidas nas fases 6 a 9.
- **2ª rodada:** 13 lacunas ⚪ (C8–C11, T6–T9, P10, P11, D7–D9), todas corrigidas nas fases 10 e 11 (commits `a56b7b2`, `8cf700c`); D9 trocou o texto do fallback para `(image unavailable)` por decisão do usuário.

## PR Brief

- **Resumo do que mudou:** `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf` aceitam `markdown_output_dir` (opcional). Com ele, o parser grava `{pdf_stem}.md`, montado numa passada por `doc.iterate_items()`, e `{pdf_stem}_assets/` com `figure_NNN.png` e `table_NNN.csv` (contadores separados a partir de 001). Tabelas ficam inline em GFM, sem link para o CSV, e o CSV espelha o grid; figuras (`PICTURE`/`CHART`) e tabelas (`TABLE`/`DOCUMENT_INDEX`) são reconhecidas pelos dois rótulos do Docling. O `alt` de cada figura vem de uma legenda "Figure", cada legenda usada uma vez: primeiro a próxima legenda livre na seção, depois cada legenda que o Docling ligou à figura (resolvida separadamente), depois a legenda livre anterior na seção; senão `Figure NNN`. O `alt` escapa `\ ` `` ` `` `* _ [ ] < &` e o link é codificado com `urllib.parse.quote`; figura sem imagem vira `_{alt} (image unavailable)_`. Reexportar remove o `.md` e os assets da exportação anterior. `PdfContent` ganha `markdown_path` e `artifacts_dir`; `DoclingParser` ganha `accelerator_device` opcional (padrão: comportamento atual, que respeita `DOCLING_DEVICE`). `generate_picture_images=True` passa a valer sempre. A ingestão não muda.
- **Fora do escopo original (a pedido):** a coleta de toda a suíte estava quebrada na `main` (`pytest_plugins` em `tests/unit/domain/rerank/conftest.py`); as fixtures compartilhadas foram para `tests/unit/domain/conftest.py`. Imports ordenados (ruff `I001`) em 5 arquivos sem relação com a feature. `requires-python` passou a `>=3.12,<3.13` (o `uv` escolhia o 3.15, onde o `torch` 2.8.0 não instala); o `uv.lock` foi regenerado sem mudar nenhuma versão.
- **Como testar:** `uv sync`; `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py`; `uv run pytest tests/integration/pdf_parser` (~1-2 min em CPU; baixa os modelos do Docling na primeira vez). Para ver a saída: `DoclingParser(30, 20, accelerator_device="cpu").parse_pdf(Path("tests/fixtures/arxiv_pdfs/2609.17429v1.pdf"), markdown_output_dir=Path("/tmp/out"))` e abrir `/tmp/out/2609.17429v1.md` (7 figuras, todas com a legenda certa; 11 tabelas).
- **Riscos / trade-offs:**
  - `generate_picture_images=True` em toda conversão: no fixture (15 páginas, CPU), 113,9 s e 99,2 s sem imagens contra 101,9 s e 97,5 s com, sem diferença mensurável; medido com um PDF só, não com um lote.
  - `make test` fica 1 a 2 min mais lento por causa do teste de integração.
  - `make test` tem falhas **anteriores a esta branch**: `tests/api/*` (1 falha e 7 erros; precisam de PostgreSQL em `localhost:5412` e OpenSearch, via `make start`) e 2 testes de `tests/unit/platform/graph/test_graph_builder.py` (o hook `after` do middleware não é chamado; falham também na `main`; merecem uma tarefa própria).
  - Pipe escapado dentro de uma célula: o `.md` mantém a forma escapada e o CSV grava a barra vertical sem escape (mesmo conteúdo, grafias diferentes).
  - Só o `alt` é escapado; títulos, parágrafos e itens de lista saem com o texto do PDF como está (decisão de design).
  - `accelerator_device` só vale ao instanciar o `DoclingParser`; pela API, `PDFParserService` e Airflow, usar `DOCLING_DEVICE=cpu` em GPUs não suportadas.
  - O fixture `2609.17429v1.pdf` foi commitado com a licença conferida pelo responsável (registro em `tests/fixtures/arxiv_pdfs/README.md`).
  - Melhoria futura registrada na issue: rótulos de texto de diagramas vetoriais saem como parágrafos soltos no `.md`.
