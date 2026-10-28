# Análise de Cobertura de Testes da Branch — TASK-001

## Informações da Branch
- Branch: TASK-001-exportacao-markdown-pdf-parser
- Base: main (diff `main...HEAD` mais working tree)
- Arquivos de produção alterados: 4 (`markdown_export.py`, `docling.py`, `parser.py`, `models.py`)
- Arquivos com lacunas relevantes: 1 (`docling.py`, lacunas menores)

## Resumo Executivo
50 testes passam. Cobertura medida do pacote `src/domain/pdf_parser`: 90% total.
`markdown_export.py` 100%, `models.py` 100%, `parser.py` 96% (linha 44 é o ramo `except Exception` anterior à tarefa), `docling.py` 74% (linhas 54, 127-130, 142-152, 209, 214-240 são validação e tratamento de erro anteriores à tarefa; o código novo de `parse_pdf` está coberto). Não há lacuna de alta prioridade. Restam lacunas baixas de critério de aceitação e casos de borda.

## Lacunas (priorizadas)

### Média-baixa
1. `tests/unit/domain/test_pdf_parser.py` (`test_with_output_dir_exports_and_returns_paths`) / `docling.py:~183-186`. Critério "com `markdown_output_dir`, `sections` e `raw_text` saem iguais aos da chamada sem o parâmetro" não é verificado. Sugestão: parametrizar com e sem `markdown_output_dir` e comparar `sections` e `raw_text`.
2. `markdown_export.py:131-134` (`_render_table`). Tabela cujo `export_to_markdown` não devolve nenhuma linha `|` (tabela vazia) gera CSV vazio e bloco vazio (descartado do `.md`), mas o contador avança. Sem teste. Sugestão: tabela fake com markdown vazio e verificar `table_001.csv` vazio e `.md` sem bloco, com a tabela seguinte virando `table_002.csv`.

### Baixa
3. `markdown_export.py:210-235`. Só o `OSError` de `write_text` é testado; falhas em `mkdir`, `image.save` e escrita de CSV usam o mesmo `except`, então o risco é pequeno. Sugestão: um teste parametrizado com `Path.mkdir` levantando `OSError`.
4. `markdown_export.py:104` (`_parse_gfm_table`). Grid sem cabeçalho cuja segunda linha tem só células como `---` ou `-` seria tomada por separador, e a primeira linha viraria header. Improvável em artigos, mas sem teste. Sugestão: documentar o comportamento com um teste de caracterização.
5. Critério "chamada sem `markdown_output_dir` não grava nada" é verificado só por `export_markdown` não ter sido chamado (mock). Sugestão: afirmar também que `tmp_path` continua vazio.

## Fora do escopo (conforme pedido)
Teste de integração com o PDF real e medição de performance.

## Recomendações
Nenhuma lacuna bloqueia o PR. Os itens 1 e 2 valem um teste rápido; os demais são opcionais.
