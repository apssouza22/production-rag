---
# Modelo de IA recomendado para trabalhar neste documento. Se a sessão estiver em outro,
# o comando avisa e segue (você decide se troca). Opções: opus | sonnet | haiku
recommended_model: opus

# Card ou número de controle da tarefa (TASK_MANAGER do ENV.md; freelance = número de controle).
task_key: TASK-001

# URL do card no task manager (padrão da eng.integrations-rules.md). Vazio no modo freelance.
card_url:

# Pull/Merge Request aberto: número e URL. Vazio se o PR ainda não foi aberto (o daily.md é gerado mesmo assim).
pr: "#1 https://github.com/phelipperibeiro/production-rag/pull/1"

# Branch de trabalho e branch base (Branch base: do architecture.md).
branch: TASK-001-exportacao-markdown-pdf-parser
base: main

# Dias efetivamente trabalhados no período eng.start → registro (descontados fim de semana não
# trabalhado, feriados e outros dias não trabalhados informados). Ex.: "3 (2026-10-13, 2026-10-14, 2026-10-15)"
worked_days: 1 (2026-10-26)

# A descrição usa o formato post mortem? Opções: yes | no
# yes = bugfix/hotfix, mais de 1 dia trabalhado, ou investigação longa confirmada pelo usuário
post_mortem: no

# Data da última atualização deste arquivo (YYYY-MM-DD).
updated_at: 2026-10-26
---

# Daily — TASK-001

> Registro pessoal do esforço: narrativa auditável e linha pronta para o timesheet. Fica só na sessão, nunca é publicado. Uma seção por data (`## AAAA-MM-DD`), no formato abaixo, para o `/eng.daily-report` juntar os registros por dia.

## 2026-10-26

### Descrição detalhada

Implementei no `production-rag` a exportação opcional de Markdown no parser de PDF (`src/domain/pdf_parser/`), para revisão e depuração do que o Docling extrai de cada artigo do arXiv. Com o parâmetro novo `markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`, o parser grava `{pdf_stem}.md`, montado numa única passada por `doc.iterate_items()`, e a pasta `{pdf_stem}_assets/` com as figuras em PNG e as tabelas em CSV; sem o parâmetro, nada muda e a ingestão continua igual. As tabelas ficam inline em GFM e o CSV espelha o mesmo grid; o `alt` de cada figura vem da legenda "Figure", com cada legenda usada por uma figura só. Essa regra foi decidida com base no PDF real de teste (`2609.17429v1`): a regra original acertava 5 de 7 figuras, porque o Docling agrupa legendas depois de várias figuras ou as coloca antes da figura, e a regra final acerta as 7, conferido nas imagens. A entrega passou por quatro rodadas de revisão pré-PR, com as lacunas corrigidas em cada uma (rótulos `CHART`/`DOCUMENT_INDEX`, escape do `alt` e do link, limpeza da exportação anterior, testes sem atributos privados, documentação). No caminho, corrigi também a coleta da suíte de testes, que estava quebrada na `main` por um `pytest_plugins` num conftest não raiz, alinhei o `requires-python` a `>=3.12,<3.13` (o `uv` escolhia o Python 3.15, incompatível com o `torch` travado) e registrei que GPUs antigas exigem rodar o Docling em CPU. Ficaram 76 testes unitários e 6 de integração passando, `make lint` sem erros e `make test` com 290 passando; as falhas restantes já existiam na `main` (testes de API sem PostgreSQL/OpenSearch e 2 testes de `test_graph_builder.py`).

**Referências:** PR [#1](https://github.com/phelipperibeiro/production-rag/pull/1) · issue [TASK-001](../../../../docs/jarvis/product/issues/task-001-exportacao-markdown-pdf-parser.md) · commits `921a049`, `f2627cd`, `9ad0706`, `848281b`, `81ef422`, `377db54`, `a2078d8`, `f177b4a`, `c2e03da`, `fa4255b`, `a56b7b2`, `8cf700c`, `646abcb`

### Timesheet

`TASK-001 — Exportação opcional de Markdown (figuras PNG e tabelas CSV) no parser de PDF, com testes no PDF real e correção da coleta da suíte de testes`
