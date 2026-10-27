# Gate checklist — TASK-001

> _Recriado em 2026-10-26 a partir do histórico da sessão: versão final da 2ª rodada (decisões preenchidas), antes de o `gate-checklist.md` ser sobrescrito pela 3ª rodada._

> Branch: `TASK-001-exportacao-markdown-pdf-parser` → `main` · Data: 2026-10-26 · **2ª rodada**
> **Semáforo:** 🟢 pronto para PR

## Resumo

A branch acrescenta a exportação opcional de Markdown ao parser de PDF (`markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`; módulo `src/domain/pdf_parser/markdown_export.py`), com testes unitários e de integração no PDF real, e corrige, a pedido do usuário, a coleta quebrada da suíte, a ordem de imports e a versão do Python. Na 1ª rodada foram 27 lacunas (0 🔴, 14 🟡, 13 ⚪), todas marcadas "Corrigir" e corrigidas nas fases 6 a 9 do `plan.md` (commits `a2078d8`, `f177b4a`, `c2e03da`, `fa4255b`); os quatro agentes confirmaram a correção de todas. Nesta 2ª rodada, **24 de 24 critérios de aceitação estão atendidos** e restam **13 lacunas, todas ⚪ menores (0 🔴, 0 🟡)**: casos de borda raros no `alt` e no escape, testes extras de borda e ajustes de texto. Nenhuma afeta o PDF do arXiv usado nos testes.

## Lacunas

### Produto e documentação

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| P10 | A issue diz "passada única", mas a regra do `alt` monta antes um índice de legendas e seções sobre a lista de itens | `markdown_export.py:339-340`; issue, critério de Montagem | Leitor literal da issue pode ver divergência (o código está correto) | Uma linha em "Divergências" da issue | ⚪ Menor | Corrigir | sem commit (issue TASK-001, "Divergências") |
| P11 | Pipe escapado: o `.md` mantém a forma escapada e o CSV grava a barra vertical sem escape | `markdown_export.py:73-96`; issue, critério "conteúdo das células igual" | Já está no PR Brief, mas não na issue | Uma frase em "Divergências" da issue | ⚪ Menor | Corrigir | sem commit (issue TASK-001, "Divergências") |

### Código

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| C8 | Figura com várias legendas ligadas: o `caption_text` do Docling concatena todas sem separador (`"Figure 1: AFigure 1b: B"`), não casa com nenhuma legenda visitada e vira o `alt`; a legenda visitada continua livre e pode ser usada por outra figura. Duas figuras com o mesmo `caption_text` não visitado recebem o mesmo `alt` | `markdown_export.py:226-228` (`take_text`); `docling_core` `document.py:1320-1325` | `alt` colado ou repetido (reproduzido pelo revisor com `DoclingDocument`); raro em artigos | Iterar `picture.captions` e chamar `take_text` por legenda; registrar também os textos devolvidos sem posição | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| C9 | A limpeza de assets só casa nomes de 3 dígitos (`figure_1000.png` não é limpo); e, se a exportação falhar no meio, o `.md` antigo fica apontando para assets já apagados | `markdown_export.py:31`, `:308-312` | Órfãos acima de 999 figuras/tabelas; estado inconsistente após erro | `\d{3,}` na regex; gravar o `.md` num arquivo temporário e trocar no fim (ou documentar) | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| C10 | Escape incompleto: `pdf_stem` com `%`, `#`, `?` ou `\` gera link quebrado; `&` e `<` no `alt` são interpretados como entidade/HTML | `markdown_export.py:273-282` | Link ou `alt` errado para nomes de arquivo fora do padrão arXiv | `urllib.parse.quote(link, safe="/")`; escapar `&` e `<` no `alt` | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| C11 | `take_next`/`take_previous` percorrem todas as legendas a cada figura e `take_previous` cria uma lista a cada chamada | `markdown_export.py:202`, `:213` | Custo O(figuras × legendas); aceitável no tamanho dos artigos | Lista ordenada de posições + `bisect` | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |

### Testes

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| T6 | Escape do `alt` sem teste para `\`, crase e `]` isolado | `tests/unit/domain/pdf_parser/test_markdown_export.py:423-428` | `]` fechando o `alt` é o que quebraria o link; o escape de `\` precisa vir antes dos outros | Teste com `alt` contendo os três | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| T7 | Troca de `<` e `>` por `%3C`/`%3E` no link nunca é exercitada | `markdown_export.py:281` | Ramo sem teste | Teste com `pdf_stem` contendo `<` ou `>` | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| T8 | Não há teste de que a limpeza poupa nomes parecidos (`figure_1.png`) e pastas com nome de asset | `markdown_export.py:306-312` | Garantia de não apagar arquivos do usuário | Criar `figure_1.png` e uma pasta `table_001.csv/` e verificar que sobrevivem | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |
| T9 | `accelerator_device` inválido (ex.: `"gpu"`) sem teste | `src/domain/pdf_parser/docling.py` | Fixar o comportamento (erro no construtor) | Teste esperando o erro de validação | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) |

### Não funcionais

Nenhuma lacuna nova (medição registrada no PR Brief).

### Interface e UX

Nenhuma lacuna encontrada (a branch não tem interface).

### Segurança

Nenhuma lacuna encontrada. Varredura de segredos no diff: 0 ocorrências; nenhum `.env`, `.pem` ou `.key` no diff.

### Documentação do projeto

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| D7 | Ainda aparece "3.12+" | `docs/onboarding.md:54`; badge `Python-3.12+` em `README.md:5` | Sugere que 3.13 serve, e o `pyproject` agora recusa | "Python 3.12 (`>=3.12,<3.13`)" e badge `Python-3.12` | ⚪ Menor | Corrigir | [`8cf700c`](https://github.com/phelipperibeiro/production-rag/commit/8cf700c995ffe03a76affdfcb91a26cc6ce2616b) |
| D8 | `accelerator_device` só existe no `DoclingParser`: `PDFParserService` e `PDFParserSettings` não o repassam | `docs/onboarding.md:480`; `parser.py`; docstring de `DoclingParser` | Pela API e pelo Airflow, só `DOCLING_DEVICE=cpu` ou `CUDA_VISIBLE_DEVICES=""` funcionam | Dizer isso na linha de troubleshooting (ou repassar o parâmetro no serviço) | ⚪ Menor | Corrigir | [`8cf700c`](https://github.com/phelipperibeiro/production-rag/commit/8cf700c995ffe03a76affdfcb91a26cc6ce2616b) |
| D9 | O fallback `_{alt} (imagem indisponível)_` está em português num `.md` exportado e num código em inglês | `markdown_export.py:302` | Mistura de idioma no artefato; o texto foi fixado pela Q4 | Confirmar o texto da Q4 ou trocar por `(image unavailable)` | ⚪ Menor | Corrigir | [`a56b7b2`](https://github.com/phelipperibeiro/production-rag/commit/a56b7b2975565820ac7d6614f3d48ceb7fb40b97) (código e testes); issue e `architecture.md` sem commit |

## Evidências

- `git diff main...HEAD`: `src` +402 −5; `tests` +773 −61; `uv.lock` +19 −1153 (só remoção de wheels/marcadores de Python 3.13+, nenhuma versão alterada); PDF fixture (652 KB).
- `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py -q`: 68 passaram.
- `uv run pytest tests/integration/pdf_parser -q`: 6 passaram em 116 s (Docling real, CPU; 7 figuras com o `alt` da própria legenda, 11 tabelas).
- `make lint`: ruff "All checks passed!"; mypy "Success: no issues found in 132 source files" (`ignore_errors = true`).
- `make test`: 282 passaram, 3 falharam, 7 erros, todos anteriores à branch (`tests/api/*` sem PostgreSQL/OpenSearch; 2 de `tests/unit/platform/graph/test_graph_builder.py`, que falham também na `main`).
- Cobertura (unitários, linhas + ramos): `markdown_export.py` 99% (só os 3 ramos parciais aceitos antes), `models.py`/`exceptions.py`/`factory.py` 100%, `parser.py` 97%, `docling.py` 78% (linhas não cobertas anteriores à branch; o código novo está coberto).
- Ambiente: `uv sync --frozen` num ambiente novo, sem `--python`, escolheu o CPython 3.12.15.
- Varredura de segredos no diff: 0 ocorrências.
- Agentes (2ª rodada, só leitura): `prod.pm-checker` (24/24 critérios; P1–P9 resolvidas), `eng.dev-code-reviewer` (C1–C7 corrigidas; 🟢), `eng.qa.test-planner` (T1–T5 resolvidas), `eng.docs-writer` (D1–D6 resolvidas).

### 1ª rodada (histórico)

27 lacunas (0 🔴, 14 🟡, 13 ⚪), todas marcadas "Corrigir" pelo usuário e corrigidas nas fases 6 a 9 do `plan.md`: C1–C7 (rótulos `CHART`/`DOCUMENT_INDEX`, escape de `alt` e link, limpeza de assets, `_CaptionIndex`, custo, testes sem atributos privados, cerca de código), T1–T5 (testes da Q5, precedência, CSV no PDF real, legenda mais próxima, limite por `TITLE`), P1–P9 (issue com "Divergências" e status, `architecture.md` pós-Q5, notas do PR, licença do fixture, melhoria futura) e D1–D6 (Python 3.12 no `pyproject`, documentação do repositório e da raiz).

## PR Brief

- **Resumo do que mudou:** `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf` aceitam `markdown_output_dir` (opcional). Com ele, o parser grava `{pdf_stem}.md`, montado numa passada por `doc.iterate_items()`, e `{pdf_stem}_assets/` com `figure_NNN.png` e `table_NNN.csv` (contadores separados a partir de 001). Tabelas ficam inline em GFM, sem link para o CSV, e o CSV espelha o grid; figuras (`PICTURE`/`CHART`) e tabelas (`TABLE`/`DOCUMENT_INDEX`) são reconhecidas pelos dois rótulos do Docling. O `alt` vem da legenda "Figure" (cada legenda usada uma vez: busca adiante, `caption_text`, busca para trás na mesma seção; senão `Figure NNN`), com escape de Markdown; figura sem imagem vira o `alt` em itálico. Reexportar remove os assets antigos. `PdfContent` ganha `markdown_path` e `artifacts_dir`; `DoclingParser` ganha `accelerator_device` opcional (padrão: comportamento atual). `generate_picture_images=True` passa a valer sempre. A ingestão não muda.
- **Fora do escopo original (a pedido):** a coleta de toda a suíte estava quebrada na `main` (`pytest_plugins` em `tests/unit/domain/rerank/conftest.py`); as fixtures compartilhadas foram para `tests/unit/domain/conftest.py`. Imports ordenados (ruff `I001`) em 5 arquivos sem relação com a feature. `requires-python` passou a `>=3.12,<3.13` (o `uv` escolhia o 3.15, onde o `torch` 2.8.0 não instala); o `uv.lock` foi regenerado sem mudar nenhuma versão.
- **Como testar:** `uv sync`; `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py`; `uv run pytest tests/integration/pdf_parser` (~2 min em CPU; baixa os modelos do Docling na primeira vez). Para ver a saída: `DoclingParser(30, 20, accelerator_device="cpu").parse_pdf(Path("tests/fixtures/arxiv_pdfs/2609.17429v1.pdf"), markdown_output_dir=Path("/tmp/out"))` e abrir `/tmp/out/2609.17429v1.md` (7 figuras, todas com a legenda certa; 11 tabelas).
- **Riscos / trade-offs:**
  - `generate_picture_images=True` em toda conversão: no fixture (15 páginas, CPU), 113,9 s e 99,2 s sem imagens contra 101,9 s e 97,5 s com, sem diferença mensurável; medido com um PDF só, não com um lote.
  - `make test` fica cerca de 2 min mais lento por causa do teste de integração.
  - `make test` tem falhas **anteriores a esta branch**: `tests/api/*` (1 falha e 7 erros; precisam de PostgreSQL em `localhost:5412` e OpenSearch, via `make start`) e 2 testes de `tests/unit/platform/graph/test_graph_builder.py` (o hook `after` do middleware não é chamado; falham também na `main`; merecem uma tarefa própria).
  - Pipe escapado dentro de uma célula: o `.md` mantém a forma escapada e o CSV grava a barra vertical sem escape (mesmo conteúdo, grafias diferentes).
  - O fixture `2609.17429v1.pdf` foi commitado com a licença conferida pelo responsável (registro em `tests/fixtures/arxiv_pdfs/README.md`).
  - Melhoria futura registrada na issue: rótulos de texto de diagramas vetoriais saem como parágrafos soltos no `.md`.
  - Lacunas ⚪ da 2ª rodada conforme as decisões deste relatório.
