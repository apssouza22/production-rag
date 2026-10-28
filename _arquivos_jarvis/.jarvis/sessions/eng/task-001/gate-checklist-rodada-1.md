# Gate checklist — TASK-001

> _Recriado em 2026-10-26 a partir do histórico da sessão: versão final da 1ª rodada (decisões preenchidas e PR Brief da FASE 9), antes de o `gate-checklist.md` ser sobrescrito pela 2ª rodada._

> Branch: `TASK-001-exportacao-markdown-pdf-parser` → `main` · Data: 2026-10-26
> **Semáforo:** 🟡 pode abrir com notas

## Resumo

A branch acrescenta a exportação opcional de Markdown ao parser de PDF (`markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`; módulo novo `src/domain/pdf_parser/markdown_export.py`), com testes unitários e um teste de integração com o Docling real num PDF do arXiv, e corrige, a pedido do usuário, a coleta quebrada da suíte e a ordem de imports (commit `81ef422`). 22 dos 23 critérios de aceitação estão atendidos; o que falta (`make test` 100% verde) depende de falhas anteriores à branch. Foram encontradas **27 lacunas: 0 🔴 bloqueadoras, 14 🟡 importantes e 13 ⚪ menores**. O principal risco é de conteúdo: tabelas rotuladas `DOCUMENT_INDEX` (e figuras `CHART`) somem do `.md` sem aviso, e legendas com `[`, `]` ou `_` podem quebrar a sintaxe do Markdown gerado.

## Lacunas

### Produto e documentação

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| P1 | `make test` não passa por inteiro: 269 passando, 3 falhas e 7 erros, todos anteriores à branch | `make test`; issue, critério "make lint e make test passam"; `architecture.md` 1.4 | O critério de aceitação não fica verde. Causas: `tests/api/*` precisam de PostgreSQL (`localhost:5412`) e OpenSearch; `tests/unit/platform/graph/test_graph_builder.py` (2 testes) falha também na `main` | Declarar no PR, com a evidência da `main`; rodar com `make start` para os testes de API; abrir tarefa separada para o `graph_builder` | 🟡 Importante | Corrigir | sem commit (PR Brief do `gate-checklist.md`) |
| P2 | A issue não reflete a Q5 (regra do `alt` mudou durante a implementação) | `docs/jarvis/product/issues/task-001-exportacao-markdown-pdf-parser.md` (critérios de Figuras e caso extremo 6) | Quem ler só a issue vai achar a implementação errada | Atualizar os critérios do `alt` e criar a seção "Divergências" (regra de `prod.spec-rules.md`); o caso extremo 6 foi resolvido (legenda é filha da figura) | 🟡 Importante | Corrigir | sem commit (issue TASK-001, fora do git) |
| P3 | `architecture.md` com trechos desatualizados depois da Q5 | `.jarvis/sessions/eng/task-001/architecture.md`: R7 (seção 5), "Decisão adotada" da Q1 e Q2, seção 10; `plan.md` (pré-requisitos "Q1–Q4") | O documento de arquitetura diz duas coisas diferentes sobre a regra vigente | Marcar Q1/Q2 como substituídas pela Q5; atualizar R7 (mitigado), seção 10 e o checklist do plano | 🟡 Importante | Corrigir | sem commit (`architecture.md` e `plan.md` da sessão) |
| P4 | Mudanças fora do escopo original da issue | commit `81ef422` (conftests e 6 `I001` em `src/domain/arxiv/client.py`, `src/domain/indexing/factory.py`, `src/domain/opensearch/client.py`, `tests/evals/*`) | O PR mistura feature com correção de teste e lint; só o `plan.md` registra | Parágrafo "Fora do escopo original" na descrição do PR (o commit já é isolado) | 🟡 Importante | Corrigir | sem commit (PR Brief do `gate-checklist.md`) |
| P5 | Medição de desempenho feita com um PDF só, em CPU | comentários da FASE 5 do `plan.md`; issue pede "um lote" | O número existe, mas é limitado | Levar os tempos ao PR com essa limitação (113,9 s / 99,2 s sem imagens; 101,9 s / 97,5 s com) | 🟡 Importante | Corrigir | sem commit (PR Brief do `gate-checklist.md`) |
| P6 | Status e checklists não atualizados | issue `status: backlog`; `plan.md` "Checklist Final" (código revisado, documentação, pronto para PR) | A documentação de produto deve acompanhar o estado real (`prod-rules.md`) | Passar a issue para `in_progress` (ou `in_production` depois do merge) e marcar o checklist | 🟡 Importante | Corrigir | sem commit (issue TASK-001 e `plan.md`) |
| P7 | Pipe escapado numa célula: o `.md` mantém a forma escapada (barra invertida + barra vertical) e o CSV grava a barra vertical sem escape | `src/domain/pdf_parser/markdown_export.py:54-77` | Critério "mesmo conteúdo" é atendido no sentido, não literalmente | Nota no PR | ⚪ Menor | Corrigir | sem commit (PR Brief do `gate-checklist.md`) |
| P8 | Licença do fixture conferida, mas sem registro no repositório | `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` | Quem vier depois não sabe que a redistribuição foi verificada | Mencionar a licença no PR ou num README em `tests/fixtures/` | ⚪ Menor | Corrigir | [`fa4255b`](https://github.com/phelipperibeiro/production-rag/commit/fa4255ba3d198e9cc4d95209c02cade2c971068a) |
| P9 | Rótulos de diagramas vetoriais viram parágrafos soltos no `.md` | saída do fixture (ex.: "live sensor", "dead sensor") | Ruído na leitura; fora dos critérios | Registrar como melhoria futura (issue nova) | ⚪ Menor | Corrigir | sem commit (issue TASK-001, "Fora do escopo") |

### Código

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| C1 | Tabelas `DOCUMENT_INDEX` e figuras `CHART` são descartadas sem aviso | `markdown_export.py:253` e `:258`; Docling cria tabelas com o rótulo do layout (`readingorder_model.py:212`, `LayoutModel.TABLE_LABELS = [TABLE, DOCUMENT_INDEX]`); `PictureItem.label` aceita `PICTURE \| CHART` | Conteúdo some do `.md` (sem bloco, sem CSV, sem log); o teste de integração não pega, porque compara a contagem de tabelas inline com a de CSVs | Decidir por tipo (`isinstance(item, TableItem/PictureItem)`) em vez do rótulo, e testar com `label=DOCUMENT_INDEX`. A issue fala em `DocItemLabel.TABLE`/`PICTURE`: mudar é decisão sua | 🟡 Importante | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| C2 | `alt` e link entram no Markdown sem escape | `markdown_export.py:220`, `:223`, `:257` | Legenda com `[0, 1)` quebra a imagem; `_` (ex.: `F_1`) quebra o itálico do fallback; `pdf_stem` com espaço ou parêntese gera link inválido (reproduzido pelo revisor com um `DoclingDocument` real) | Escapar `\ [ ] * _` e crase no `alt`; usar destino `<...>` ou `urllib.parse.quote`; testar com `alt="Figure 1: [0, 1) F_1"` e `stem="a b (v2)"` | 🟡 Importante | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| C3 | Reexportar no mesmo diretório deixa arquivos antigos na pasta de assets | `markdown_export.py:245` (`mkdir(exist_ok=True)`) | Se o número de figuras ou tabelas mudar, ficam `figure_*.png`/`table_*.csv` órfãos; o glob de quem lê a pasta vê arquivos fantasmas | Limpar `figure_*.png` e `table_*.csv` da pasta no início, ou documentar na docstring | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| C4 | `_resolve_alt` mistura controle por posição e por texto; `used_captions=None` desliga a regra "uma vez só" sem aviso | `markdown_export.py:151`, `:172-173`, `:187-189` | Legendas de texto repetido podem escapar do controle; a assinatura engana quem chamar sem o conjunto | Tornar `used_captions` obrigatório; pré-calcular as posições das legendas "Figure" e usar um único critério | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| C5 | Custo O(figuras × itens) em documentos sem cabeçalhos | `markdown_export.py:173`, `:175`, `:189`, `:192` | Documento sem `SECTION_HEADER` vira uma seção só e cada figura varre tudo; aceitável com `max_pages` 30 | Resolvido junto com C4 (busca só nas posições de legenda) | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| C6 | Testes acoplados a atributos privados | `tests/integration/pdf_parser/test_markdown_export.py:25` (`parser._converter.format_to_options`); fixture de `TestDoclingParserMarkdownExport` em `tests/unit/domain/test_pdf_parser.py` (`_validate_pdf`, `_converter`) | Refatorar o construtor quebra os testes sem mudar comportamento; a troca para CPU depende do pipeline ser montado só no `convert` | Aceitar `accelerator_options`/`device` no construtor do `DoclingParser`; nos unitários, patch de `DocumentConverter` no módulo | ⚪ Menor | Corrigir | [`f177b4a`](https://github.com/phelipperibeiro/production-rag/commit/f177b4ac52b54ea4394bf1bd509a22bfa43de32e) |
| C7 | Bloco de código quebra se o texto tiver três crases | `markdown_export.py:50` | Artigos sobre LLMs mostram prompts com crases; o resto do `.md` sai errado | Cerca maior que a maior sequência de crases do texto | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |

### Testes

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| T1 | O teste de integração não trava a Q5 | `tests/integration/pdf_parser/test_markdown_export.py:60-64` | Se uma figura voltar ao fallback `Figure 001`, o teste ainda passa (o fallback também é único) | Afirmar que nenhum `alt` é `^Figure \d{3}$`, ou que cada legenda "Figure N:" do `.md` aparece como `alt` exatamente uma vez | 🟡 Importante | Corrigir | [`f177b4a`](https://github.com/phelipperibeiro/production-rag/commit/f177b4ac52b54ea4394bf1bd509a22bfa43de32e) |
| T2 | Precedência "busca adiante antes de `caption_text`" sem teste | `markdown_export.py:175-190` | Uma figura com legenda própria seguida da legenda visitada da figura seguinte pega a da seguinte; inverter os passos não quebraria nenhum teste | Teste com duas figuras (a primeira com `caption_text="Figure 1: A"`, a segunda com "Figure 2: B" visitada logo depois), fixando o resultado da Q5 | 🟡 Importante | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| T3 | CSV x tabela inline não comparados no PDF real | `tests/integration/pdf_parser/test_markdown_export.py` | O critério "CSV igual à tabela inline" só é testado com documentos falsos e sintéticos | Ler o primeiro grid GFM do `.md` e comparar linha a linha com `csv.reader` de `table_001.csv` | 🟡 Importante | Corrigir | [`f177b4a`](https://github.com/phelipperibeiro/production-rag/commit/f177b4ac52b54ea4394bf1bd509a22bfa43de32e) |
| T4 | Busca para trás testada só com um candidato | `markdown_export.py:192-199` | A regra é "a mais próxima"; trocar a ordem não quebraria nenhum teste | Duas legendas livres antes da figura, nenhuma depois, esperando a mais próxima | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |
| T5 | Limite de seção testado só com `SECTION_HEADER` | `markdown_export.py:177`, `:194` | `TITLE` também interrompe as buscas e não tem teste | Repetir o teste de limite com `DocItemLabel.TITLE` | ⚪ Menor | Corrigir | [`a2078d8`](https://github.com/phelipperibeiro/production-rag/commit/a2078d881f31e56b679fec4cfdce5b482e925102) |

### Não funcionais

Nenhuma lacuna além de P5 (medição com um PDF só). `generate_picture_images=True` não mostrou diferença mensurável de tempo no fixture.

### Interface e UX

Nenhuma lacuna encontrada (a branch não tem interface).

### Segurança

Nenhuma lacuna encontrada. Sem auth, endpoint, sessão ou input externo de usuário; o parâmetro novo não é exposto pela API. Varredura de segredos no diff sem ocorrências; nenhum arquivo `.env`, `.pem` ou `.key` no diff.

### Documentação do projeto

| # | Lacuna | Onde | Por que importa | Sugestão | Severidade | Decisão | Corrigido em |
|---|---|---|---|---|---|---|---|
| D1 | `make test` agora inclui um teste de cerca de 2 min, e as fixtures de domínio mudaram de lugar | `production-rag/AGENTS.md`, seção Testing (~l. 124-131) | Quem roda `make test` não espera a demora; a orientação de fixtures está incompleta | Citar `tests/integration/pdf_parser/` (~2 min, CPU, `uv run pytest tests/integration/pdf_parser`) e `tests/unit/domain/conftest.py` | 🟡 Importante | Corrigir | [`c2e03da`](https://github.com/phelipperibeiro/production-rag/commit/c2e03daf1e1bb293e51dcfb76e67d5deae3bcb8a) |
| D2 | Versão do Python: `AGENTS.md` diz `<3.13`, mas `pyproject.toml` diz só `>=3.12` | `production-rag/AGENTS.md:110`, `pyproject.toml:5`, `README.md` (~l. 38), `docs/onboarding.md` (~l. 54, 60) | `uv sync` (e `make setup`) escolhe o CPython 3.15 e falha no `torch` 2.8.0 | Correção de raiz: `requires-python = ">=3.12,<3.13"` ou `.python-version` com 3.12; senão, documentar `uv sync --python 3.12` | 🟡 Importante | Corrigir | [`c2e03da`](https://github.com/phelipperibeiro/production-rag/commit/c2e03daf1e1bb293e51dcfb76e67d5deae3bcb8a) |
| D3 | Sem nota para GPU não suportada | `production-rag/docs/onboarding.md` (sem seção de troubleshooting) | GPUs antigas (ex.: compute capability 6.1) fazem o Docling falhar em CUDA sem explicação | Nota de troubleshooting: forçar CPU (`CUDA_VISIBLE_DEVICES=""` antes de iniciar o processo) | 🟡 Importante | Corrigir | [`c2e03da`](https://github.com/phelipperibeiro/production-rag/commit/c2e03daf1e1bb293e51dcfb76e67d5deae3bcb8a) |
| D4 | Árvore do projeto não cita o módulo novo | `production-rag/docs/onboarding.md` (~l. 348) | É como alguém descobre o recurso | `pdf_parser/  # Docling wrapper + exportação opcional para Markdown (markdown_export.py)` | ⚪ Menor | Corrigir | [`c2e03da`](https://github.com/phelipperibeiro/production-rag/commit/c2e03daf1e1bb293e51dcfb76e67d5deae3bcb8a) |
| D5 | Documentação do parser não menciona a exportação | `production-rag/docs/onboarding.md` (~l. 41); `docs/ARTICLE.md` §ingestão (~l. 110-116) | Recurso opcional fica invisível | Uma frase sobre `parse_pdf(..., markdown_output_dir=...)` e os arquivos gerados | ⚪ Menor | Corrigir | [`c2e03da`](https://github.com/phelipperibeiro/production-rag/commit/c2e03daf1e1bb293e51dcfb76e67d5deae3bcb8a) |
| D6 | `AGENTS.md`/`CLAUDE.md` da raiz do workspace com o mesmo problema de D1 e D2 | `/AGENTS.md` (Comandos de Setup, Instruções de Teste) | Fora do repositório, mas é o que a IA lê na sessão | Acrescentar `--python 3.12` e o aviso do teste lento | ⚪ Menor | Corrigir | sem commit (`AGENTS.md`/`CLAUDE.md` da raiz, fora do git) |

## Evidências

- `git diff --stat main...HEAD`: 17 arquivos, +908 −65 em `.py` (`src` +298 −4; `tests` +610 −61), mais o PDF fixture (652 KB). Código de produção na faixa de 100-500 linhas; a maior parte do diff é teste.
- `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py -q`: 56 passaram.
- `uv run pytest tests/integration/pdf_parser -q`: 5 passaram em 117 s (Docling real, CPU).
- `make lint`: `ruff check --fix` "All checks passed!"; `mypy src/` "Success: no issues found in 132 source files" (com `ignore_errors = true` no `pyproject.toml`).
- `make test`: 269 passaram, 3 falharam, 7 erros. Na `main`: `tests/api` com 1 falha e 7 erros iguais, e os 2 testes de `test_graph_builder.py` também falham. Na `main`, a coleta da suíte inteira era interrompida pelo `pytest_plugins` de `tests/unit/domain/rerank/conftest.py` (corrigido em `81ef422`).
- Cobertura (`--cov=src/domain/pdf_parser`, unitários): `markdown_export.py` 100% (ramos 98%), `models.py` 100%, `parser.py` 96%, `docling.py` 74% (linhas não cobertas anteriores à branch).
- Medição (`generate_picture_images`, fixture de 15 páginas, CPU, primeira execução descartada): `False` 113,9 s e 99,2 s; `True` 101,9 s e 97,5 s.
- Varredura de segredos no diff (`api_key|secret|password|token|BEGIN RSA/OPENSSH|sk-…`): 0 ocorrências.
- Agentes: `prod.pm-checker` (22/23 critérios), `eng.dev-code-reviewer` (achados C1 e C2 reproduzidos com um `DoclingDocument` real), `eng.qa.test-planner`, `eng.docs-writer` (só leitura; nenhum arquivo alterado).

## PR Brief

> Atualizado depois das correções das fases 6 a 9 do `plan.md` (todas as 27 lacunas marcadas "Corrigir").

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
