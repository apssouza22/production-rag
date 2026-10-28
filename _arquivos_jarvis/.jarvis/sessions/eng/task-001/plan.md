# Exportação de Markdown no pdf_parser

> 📋 **Plano de Execução** | Chave: TASK-001 (freelance, sem board) · Issue: [TASK-001](../../../../docs/jarvis/product/issues/task-001-exportacao-markdown-pdf-parser.md) · Arquitetura: [architecture.md](architecture.md)
>
> Se você está trabalhando nesta feature, **atualize este arquivo** conforme progride.
>
> **Execução:** todas de uma vez (fases 1–5: 18 de 18; fases 6–9: 18 de 18; fases 10–11: 9 de 9; fase 12: 5 de 5) · **Commits:** um por fase · **Depois do work:** rodar /eng.pre-pr _(preenchido pelo `eng.work`, Fase 1.3)_

---

## Resumo

`DoclingParser.parse_pdf` e `PDFParserService.parse_pdf` ganham o parâmetro opcional `markdown_output_dir`. Quando ele vem, um módulo novo (`src/domain/pdf_parser/markdown_export.py`) monta o `{pdf_stem}.md` numa passada única por `doc.iterate_items()` e grava `figure_NNN.png` e `table_NNN.csv` em `{pdf_stem}_assets/`; `PdfContent` devolve os caminhos. Sem o parâmetro, nada muda além de `generate_picture_images=True` na conversão.

Todos os caminhos de código abaixo são relativos a `production-rag/`. Comandos rodam dentro de `production-rag/`.

---

## Pré-requisitos

- [x] `architecture.md` revisado e aprovado (Q1–Q4 decididas; Q5 levantada e decidida no `eng.work`)
- [ ] Ambiente de desenvolvimento configurado (`make setup`; hoje não há `.venv`): tarefa 1.1
- [ ] Dependências instaladas (Docling 2.52.0 / docling-core 2.48.1 do `uv.lock`)
- [x] Branch base atualizada (`main`) e branch `TASK-001-exportacao-markdown-pdf-parser` criada
- [ ] PDF fixture `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` fornecido pelo usuário (bloqueia a FASE 5)

---

## FASE 1: Ambiente, modelo e pipeline options [Completada ✅]

> ⏱️ **Tempo estimado**: ~40 minutos
>
> 📝 **Objetivo**: Ambiente pronto e as mudanças compatíveis que não dependem da exportação: campos novos em `PdfContent` e `generate_picture_images=True`.

### 1.1 Preparar o ambiente local [Completada ✅]

**O que fazer:**
Instalar as dependências do projeto e confirmar que a suíte atual passa antes de qualquer mudança (linha de base).

**Onde fazer:**
- `production-rag/` (sem alterar arquivos)

**Como fazer:**
1. `make setup` (`uv sync`, grupos `dev` e `evals`)
2. `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py tests/unit/domain/arxiv_ingestion -q` para registrar o estado atual
3. Anotar nos comentários da fase os testes que já falham antes da mudança (se houver)

**Como verificar:**
- `.venv` criado e `uv run python -c "import docling, docling_core"` sem erro
- Linha de base dos testes do domínio registrada

### 1.2 Campos novos em `PdfContent` [Completada ✅]

**O que fazer:**
Adicionar `markdown_path` e `artifacts_dir` ao modelo (decisão 3.3 do `architecture.md`).

**Onde fazer:**
- `src/domain/pdf_parser/models.py` (classe `PdfContent`, l. 36-45)

**Como fazer:**
1. `from pathlib import Path`
2. `markdown_path: Optional[Path] = Field(default=None, description="Exported Markdown file, when markdown_output_dir is set")`
3. `artifacts_dir: Optional[Path] = Field(default=None, description="Assets folder (figures and tables) of the exported Markdown")`

**Como verificar:**
- `PdfContent(raw_text="x", parser_used=ParserType.DOCLING)` tem os dois campos `None`
- `tests/unit/domain/arxiv_ingestion/test_metadata_fetcher.py` continua passando (instancia `PdfContent`)

### 1.3 Ligar `generate_picture_images` [Completada ✅]

**O que fazer:**
Ligar a geração de imagens das figuras na conversão (TASK-001, requisito 5; sempre ligado, decisão do discovery).

**Onde fazer:**
- `src/domain/pdf_parser/docling.py` (`DoclingParser.__init__`, l. 94-97)

**Como fazer:**
1. Acrescentar `generate_picture_images=True` aos argumentos de `PdfPipelineOptions(...)`

**Como verificar:**
- `DoclingParser(...)._converter` é criado sem erro
- Teste da 1.T confirma a opção

### 1.T Testes da Fase 1 [Completada ✅]

> 🧪 **Obrigatório**: Toda fase deve incluir testes para o código implementado.

**O que testar:**
- `PdfContent` sem os campos novos tem `markdown_path` e `artifacts_dir` iguais a `None` (1.2)
- `DoclingParser` monta as pipeline options com `generate_picture_images=True` (1.3), patchando `PdfFormatOption`/`DocumentConverter` em `src.domain.pdf_parser.docling` para inspecionar os argumentos sem carregar modelos

**Tipos de teste:**
- [x] Unitário: `PdfContent`, `DoclingParser.__init__`
- [ ] Integração: não se aplica nesta fase
- [ ] E2E: não se aplica

**Onde criar:**
- `tests/unit/domain/test_pdf_parser.py` (classe `TestDoclingParser` existente)

**Cobertura mínima:** todo código novo da fase coberto (architecture.md 6.5)

**Como verificar:**
- `uv run pytest tests/unit/domain -q` passando
- `uv run ruff check src tests` sem erros novos

### Comentários da Fase 1:

- **Mudança**: `uv sync` sem `--python` escolheu o CPython 3.15 (o `requires-python = ">=3.12"` é aberto) e falhou no `torch` 2.8.0 do lock, que só tem wheels até cp313. Ambiente criado com `uv sync --python 3.12` (3.12.15), alinhado ao `AGENTS.md` do repositório; nenhum arquivo do projeto foi alterado.
- **Aprendizado**: linha de base antes da mudança: 19 testes passando em `tests/unit/domain/pdf_parser`, `test_pdf_parser.py` e `arxiv_ingestion`. Depois: 21 (2 novos).
- **Atenção**: `pytest tests/unit/domain` (pasta inteira) falha na coleta por `pytest_plugins` em `tests/unit/domain/rerank/conftest.py` (conftest não raiz). Já existia antes desta tarefa; `tests/unit/domain/rerank` sozinho passa. Conferir o comportamento do `make test` (raiz) na FASE 5.
- **Atenção**: 7 erros `I001` de ruff já existentes (ordenação de imports), um deles em `src/domain/pdf_parser/parser.py`, que será corrigido na FASE 4 por estar no escopo. Os outros 6 arquivos ficam fora: `make lint` roda `ruff check --fix`, que os alteraria; validar com `ruff check` sem `--fix` nos arquivos tocados.
- **Decisão**: o `eng.qa.test-planner` roda uma vez sobre o diff de código completo ao fim da FASE 4, em vez de a cada fase (fases 1 a 3 são incrementais no mesmo módulo).

---

## FASE 2: `markdown_export.py`: texto e tabelas [Completada ✅]

> ⏱️ **Tempo estimado**: ~60 minutos
>
> 📝 **Objetivo**: Funções puras de renderização de itens de texto e de tabela, incluindo o CSV derivado do GFM (Q3, opção 1). Ainda sem figuras e sem ligação com o parser.
>
> ⚠️ **Dependência**: Requer FASE 1 completada (ambiente)

### 2.1 Renderização dos itens de texto [Completada ✅]

**O que fazer:**
Criar o módulo e a função que converte um item de texto em Markdown, conforme a suposição aprovada no discovery e nos critérios de "Montagem" da TASK-001.

**Onde fazer:**
- `src/domain/pdf_parser/markdown_export.py` (novo)

**Como fazer:**
1. `logger = logging.getLogger(__name__)`; importar `DocItemLabel` de `docling_core.types.doc` (mesmo import de `docling.py:9`)
2. `_render_text_item(item, depth) -> str | None`: texto via `getattr(item, "text", "").strip()`; vazio → `None`
   - `TITLE` → `# {texto}`; `SECTION_HEADER` → `"#" * (nível + 1)` com nível de `getattr(item, "level", None) or max(depth, 1)` (mesma regra de `extract_sections_from_document`, `docling.py:58-62`), limitado a 6
   - `LIST_ITEM` → `- {texto}`
   - `CODE` → bloco cercado com ```` ``` ````
   - demais textos (`TEXT`, `PARAGRAPH`, `CAPTION`, `FORMULA`, …) → parágrafo
3. Blocos separados por linha em branco na montagem final

**Como verificar:**
- Testes 2.T de cada label

### 2.2 Renderização de tabelas e CSV [Completada ✅]

**O que fazer:**
GFM inline vindo de `TableItem.export_to_markdown(doc)` e CSV lido desse mesmo texto (Q3, opção 1).

**Onde fazer:**
- `src/domain/pdf_parser/markdown_export.py`

**Como fazer:**
1. `_parse_gfm_table(markdown: str) -> tuple[list[str] | None, list[list[str]]]`: separa as linhas `| … |`; se a segunda linha for a separadora (só `|`, `-`, `:` e espaços), a primeira é o header e a separadora é descartada; senão, header `None` e todas são dados
2. Separar células respeitando `\|` escapado (não dividir nele; gravar `|` no CSV) e aplicar `.strip()` em cada célula
3. `_write_table_csv(path, header, rows)`: `open(path, "w", encoding="utf-8", newline="")` + `csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)`; escreve o header só se existir
4. `_render_table(item, doc, csv_path) -> str`: `markdown = item.export_to_markdown(doc)`; grava o CSV a partir dele; devolve o mesmo `markdown` para o `.md` (sem link para o CSV)

**Como verificar:**
- Testes 2.T com e sem header e com `\|` escapado

### 2.T Testes da Fase 2 [Completada ✅]

> 🧪 **Obrigatório**: Toda fase deve incluir testes para o código implementado.

**O que testar:**
- Cada label de texto gera o Markdown esperado; texto vazio é ignorado; nível de `SECTION_HEADER` (2.1)
- `_parse_gfm_table` com separadora → header + dados; sem separadora → só dados (2.2)
- CSV em UTF-8 (célula com acento), vírgula, `QUOTE_MINIMAL` (célula com vírgula sai entre aspas, as demais não) (2.2)
- `\|` escapado vira `|` dentro de uma célula do CSV (2.2)
- `_render_table` devolve exatamente o texto de `export_to_markdown` e não contém link para `.csv` (2.2)

**Tipos de teste:**
- [x] Unitário: `_render_text_item`, `_parse_gfm_table`, `_write_table_csv`, `_render_table` (tabela falsa com `export_to_markdown` via `SimpleNamespace`; CSV em `tmp_path`)
- [ ] Integração: não se aplica
- [ ] E2E: não se aplica

**Onde criar:**
- `tests/unit/domain/pdf_parser/test_markdown_export.py` (novo; padrão de `test_docling_sections.py`)

**Cobertura mínima:** todos os critérios de "Montagem" e "Tabelas" da TASK-001 com pelo menos um teste

**Como verificar:**
- `uv run pytest tests/unit/domain/pdf_parser -q` passando
- `uv run ruff check src tests` sem erros novos

### Comentários da Fase 2:

- **Aprendizado**: no `docling-core` 2.48.1, `TableItem.export_to_markdown(doc)` devolve **legenda + linha em branco + grid** (`MarkdownTableSerializer`, `docling_core/transforms/serializer/markdown.py`). A legenda da tabela também é visitada como `CAPTION` pelo `iterate_items()` (é filha da tabela, `docling/models/readingorder_model.py:212-223`).
- **Decisão**: o bloco inline e o CSV usam só as linhas `|…|` da saída (`_extract_gfm_table_lines`); a legenda aparece uma vez, como parágrafo, quando o `iterate_items()` a visita. Mantém a Q3 (fonte = `export_to_markdown` do `TableItem`) sem duplicar texto.
- **Aprendizado**: o serializer sempre usa a primeira linha como header (`tabulate(rows[1:], headers=rows[0])`), então o grid sem separador é raro nesta versão; o caminho headerless continua implementado e testado conforme a TASK-001.
- **Aprendizado**: `tabulate` não escapa `|` dentro das células; `_split_gfm_row` trata `\|`, mas uma célula com `|` sem escape quebraria as colunas (limitação do Docling, sem caso no fixture até agora).
- **Atenção**: o CSV usa o terminador padrão do módulo `csv` (`\r\n`); os testes leem os bytes para não depender da tradução de fim de linha.

---

## FASE 3: `markdown_export.py`: figuras, `alt` e montagem completa [Completada ✅]

> ⏱️ **Tempo estimado**: ~60 minutos
>
> 📝 **Objetivo**: Figuras (Q4), regra do `alt` (Q1 + Q2) e a função pública `export_markdown` com a passada única.
>
> ⚠️ **Dependência**: Requer FASE 2 completada

### 3.1 Regra do `alt` [Completada ✅]

**O que fazer:**
Resolver o `alt` de cada figura (Q1, opção 3; Q2, opção 4).

**Onde fazer:**
- `src/domain/pdf_parser/markdown_export.py`

**Como fazer:**
1. `_resolve_alt(items, index, picture, doc, number) -> str` sobre a lista `items = list(doc.iterate_items())`
2. Procurar adiante a partir de `index + 1`; parar no primeiro `TITLE`/`SECTION_HEADER` (fim da seção, Q2); sem parar em outras figuras ou tabelas
3. Primeiro `CAPTION` cujo texto (`.strip()`) começa com `"Figure"` → `alt`
4. Sem ele: `getattr(picture, "caption_text", None)`; se existir, `picture.caption_text(doc).strip()` começando com `"Figure"` → `alt` (Q1)
5. Senão: `f"Figure {number:03d}"`

**Como verificar:**
- Testes 3.T dos quatro caminhos

### 3.2 Renderização de figuras [Completada ✅]

**O que fazer:**
Gravar o PNG e gerar a linha de imagem; tratar `get_image` sem resultado (Q4, opção 2).

**Onde fazer:**
- `src/domain/pdf_parser/markdown_export.py`

**Como fazer:**
1. `_render_picture(picture, doc, alt, png_path, rel_path) -> str`
2. `image = picture.get_image(doc)`; com imagem: `image.save(png_path, "PNG")` e devolve `f"![{alt}]({rel_path})"`, com `rel_path = f"{pdf_stem}_assets/figure_NNN.png"`
3. Sem imagem: `logger.warning("Picture %s has no image; skipping %s", number, png_path.name)` e devolve `f"_{alt} (imagem indisponível)_"` (o contador já avançou) _(texto substituído na FASE 10 por `(image unavailable)`)_

**Como verificar:**
- Testes 3.T com e sem imagem

### 3.3 `export_markdown` (passada única e gravação) [Completada ✅]

**O que fazer:**
Função pública que monta e grava o `.md` e os assets.

**Onde fazer:**
- `src/domain/pdf_parser/markdown_export.py`

**Como fazer:**
1. `export_markdown(doc, pdf_stem: str, output_dir: Path) -> tuple[Path, Path]`
2. `assets_dir = output_dir / f"{pdf_stem}_assets"`; `assets_dir.mkdir(parents=True, exist_ok=True)` (pasta criada mesmo vazia; TASK-001, casos 2 e 3)
3. `items = list(doc.iterate_items())`; `figure_counter = table_counter = 0`; laço único por índice: `PICTURE` → 3.1 + 3.2; `TABLE` → 2.2 (`table_NNN.csv`); demais → 2.1
4. `markdown_path = output_dir / f"{pdf_stem}.md"`; gravar `"\n\n".join(blocos) + "\n"` em UTF-8 (sobrescreve; caso 4)
5. `OSError` na gravação → `logger.error(...)` e `raise PDFParsingException(f"Failed to export Markdown to {output_dir}: {e}")` (caso 5)
6. `logger.info("Exported %s (%d figures, %d tables)", markdown_path, figure_counter, table_counter)`
7. Não chamar `doc.export_to_markdown()` em nenhum ponto (requisito 1)

**Como verificar:**
- Testes 3.T de montagem e contadores

### 3.T Testes da Fase 3 [Completada ✅]

> 🧪 **Obrigatório**: Toda fase deve incluir testes para o código implementado.

**O que testar:**
- `alt` pelo próximo `CAPTION` "Figure" após texto e tabela na mesma seção (3.1)
- `CAPTION` "Figure" depois de um `SECTION_HEADER` não é usado; cai em `caption_text` ou `Figure NNN` (3.1, Q2)
- `caption_text` "Figure …" usado quando não há `CAPTION` visitado; `caption_text` sem "Figure" → `Figure NNN` (3.1, Q1)
- Figura sem legenda seguida de figura com legenda na mesma seção recebe a legenda da seguinte (comportamento aceito, R7) (3.1)
- `get_image` com imagem grava `figure_001.png` e linha `![alt](stem_assets/figure_001.png)`; sem imagem não grava arquivo, mostra `_alt (imagem indisponível)_` e o contador avança para a próxima figura ser `figure_002` (3.2) _(texto substituído na FASE 10 por `(image unavailable)`)_
- Documento misto (título, parágrafo, figura, tabela, figura, tabela): ordem dos blocos igual à visita, contadores separados `figure_001/002` e `table_001/002`, `.md` e pasta criados, retorno com os dois caminhos (3.3)
- Documento sem figuras e sem tabelas cria a pasta de assets vazia (3.3)
- Falha de gravação (diretório só leitura ou `patch` de `Path.write_text` com `OSError`) levanta `PDFParsingException` (3.3)
- `doc.export_to_markdown` nunca é chamado (documento falso que levanta erro se for chamado) (3.3)

**Tipos de teste:**
- [x] Unitário: `_resolve_alt`, `_render_picture`, `export_markdown` com documentos falsos (`SimpleNamespace`; imagem falsa com `save`)
- [ ] Integração: na FASE 5
- [ ] E2E: não se aplica

**Onde criar:**
- `tests/unit/domain/pdf_parser/test_markdown_export.py`

**Cobertura mínima:** todos os critérios de "Figuras" e "Montagem" da TASK-001 e os casos extremos 2 a 5 com pelo menos um teste

**Como verificar:**
- `uv run pytest tests/unit/domain/pdf_parser -q` passando
- `uv run ruff check src tests` sem erros novos

### Comentários da Fase 3:

- **Aprendizado (R1 praticamente resolvido)**: no pipeline de PDF, a legenda é criada como filha da figura (`docling/models/readingorder_model.py:243-252`, `_add_caption_or_footnote` com `parent=pic`) e o `iterate_items()` visita os filhos listados em `picture.captions` mesmo com `traverse_pictures=False` (`docling_core/types/doc/document.py:4094-4109`). A legenda aparece logo depois da figura, então a regra "próximo `CAPTION`" casa; o fallback `caption_text` da Q1 fica como rede de segurança.
- **Decisão**: o `alt` normaliza espaços e quebras de linha da legenda (`" ".join(text.split())`), para não quebrar a sintaxe `![alt](...)`.
- **Decisão**: o `PDFParsingException` da exportação cobre também a falha ao salvar o PNG (`image.save` dentro do mesmo `try`).
- **Atenção**: só `DocItemLabel.TABLE` vira tabela/CSV (requisito da TASK-001). Tabelas que o Docling rotula como `DOCUMENT_INDEX` (sumário) não geram CSV nem bloco no `.md`, pois não têm `text`.
- **Atenção**: o link da figura usa `{pdf_stem}_assets/...` sem escapar espaços; nomes do arXiv não têm espaço.

---

## FASE 4: Ligar a exportação ao parser e ao serviço [Completada ✅]

> ⏱️ **Tempo estimado**: ~40 minutos
>
> 📝 **Objetivo**: Parâmetro `markdown_output_dir` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`, com a exportação depois do fluxo existente.
>
> ⚠️ **Dependência**: Requer FASE 3 completada

### 4.1 `DoclingParser.parse_pdf` [Completada ✅]

**O que fazer:**
Aceitar o parâmetro e chamar `export_markdown` depois de `extract_sections_from_document`.

**Onde fazer:**
- `src/domain/pdf_parser/docling.py` (`parse_pdf`, l. 158-226)

**Como fazer:**
1. Assinatura: `def parse_pdf(self, pdf_path: Path, markdown_output_dir: Path | None = None) -> Optional[PdfContent]` e docstring com o parâmetro
2. Depois de `sections = extract_sections_from_document(doc)` (l. 178): se `markdown_output_dir` vier, `markdown_path, artifacts_dir = export_markdown(doc, pdf_path.stem, markdown_output_dir)`; senão, ambos `None`
3. Passar `markdown_path` e `artifacts_dir` no `PdfContent(...)` (l. 181)
4. Acrescentar `except PDFParsingException: raise` antes do `except Exception` genérico (l. 200), para a mensagem da exportação não ser reembrulhada pelos testes de string do bloco genérico
5. Import: `from .markdown_export import export_markdown`

**Como verificar:**
- Testes 4.T

### 4.2 `PDFParserService.parse_pdf` [Completada ✅]

**O que fazer:**
Aceitar o parâmetro e repassá-lo.

**Onde fazer:**
- `src/domain/pdf_parser/parser.py` (l. 22 e 33)

**Como fazer:**
1. `async def parse_pdf(self, pdf_path: Path, markdown_output_dir: Path | None = None) -> Optional[PdfContent]`
2. `await asyncio.to_thread(self.docling_parser.parse_pdf, pdf_path, markdown_output_dir)`
3. Docstring com o parâmetro

**Como verificar:**
- Testes 4.T; `src/domain/arxiv_ingestion/service.py:441` continua chamando `parse_pdf(pdf_path)` sem mudança

### 4.T Testes da Fase 4 [Completada ✅]

> 🧪 **Obrigatório**: Toda fase deve incluir testes para o código implementado.

**O que testar:**
- `PDFParserService.parse_pdf(path, markdown_output_dir=dir)` chama `DoclingParser.parse_pdf(path, dir)`; sem o parâmetro, chama com `None` (4.2)
- `DoclingParser.parse_pdf` sem parâmetro não chama `export_markdown` e devolve campos `None`, com `_validate_pdf` e `_converter.convert` mockados e um documento falso (4.1)
- Com parâmetro, chama `export_markdown(doc, stem, dir)` depois das seções e devolve os caminhos no `PdfContent` (4.1)
- `PDFParsingException` levantada pela exportação chega ao chamador com a mensagem original (4.1)
- Testes existentes de `test_pdf_parser.py` e `arxiv_ingestion` continuam passando

**Tipos de teste:**
- [x] Unitário: `DoclingParser.parse_pdf`, `PDFParserService.parse_pdf` (mocks com `patch`, padrão do arquivo)
- [ ] Integração: na FASE 5
- [ ] E2E: não se aplica

**Onde criar:**
- `tests/unit/domain/test_pdf_parser.py`

**Cobertura mínima:** todos os critérios de "Parâmetro e compatibilidade" da TASK-001 (exceto os que dependem do Docling real, cobertos na FASE 5)

**Como verificar:**
- `uv run pytest tests/unit -q` passando
- `uv run ruff check src tests` sem erros novos

### Comentários da Fase 4:

- **Mudança**: o `except PDFParsingException: raise` ficou **depois** do `except PDFValidationError`, e não antes como no plano: `PDFValidationError` é subclasse de `PDFParsingException` (`src/domain/pdf_parser/exceptions.py:9`), e colocá-lo antes faria os limites de páginas/tamanho levantarem erro em vez de devolver `None`. Teste de regressão: `test_page_limit_still_returns_none`.
- **Decisão**: corrigido o `I001` de `src/domain/pdf_parser/parser.py` (arquivo no escopo). Os outros 6 `I001` antigos ficam fora.
- **Mudança**: as duas asserções antigas `assert_called_once_with(valid_pdf_path)` do `PDFParserService` passaram a esperar `(valid_pdf_path, None)` (repasse do parâmetro).
- **Aprendizado (test-planner)**: cobertura do código da tarefa: `markdown_export.py` 100%, `models.py` 100%, `parser.py` 96%, `docling.py` 74% (linhas não cobertas são o tratamento de erro antigo). Relatório: `docs/jarvis/engineering/qa/task-001-test_coverage_branch_report.md`. Lacunas cobertas: `sections`/`raw_text` iguais com e sem exportação; sem parâmetro não grava nada no disco; tabela sem linhas `|` (CSV vazio, contador avança). Não cobertas, por baixo valor: falhas de `mkdir`/`image.save` (mesmo `except OSError` já testado) e grid sem header cuja segunda linha seja só `---` (seria lido como separador; improvável em artigos).
- **Atenção**: 57 testes passando em `test_pdf_parser.py`, `pdf_parser/` e `arxiv_ingestion/`.

---

## FASE 5: Integração com o PDF real e validação final [Completada ✅]

> ⏱️ **Tempo estimado**: ~50 minutos (fora o download dos modelos do Docling na primeira execução)
>
> 📝 **Objetivo**: Validar a exportação com o Docling real no PDF fixture, medir o custo de `generate_picture_images` e rodar a suíte completa.
>
> ⚠️ **Dependência**: Requer FASE 4 completada **e** o PDF fixture fornecido pelo usuário (sem ele, a fase fica 🚫 Bloqueada)

### 5.1 Conferir o PDF fixture [Completada ✅]

**O que fazer:**
Confirmar que o arquivo fornecido está no lugar e dentro do limite de páginas.

**Onde fazer:**
- `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf`

**Como fazer:**
1. Confirmar que o arquivo existe e começa com `%PDF-`
2. Contar páginas com `pypdfium2` (mesma checagem de `_validate_pdf`); precisa ser ≤ 30 (`PDF_PARSER__MAX_PAGES` padrão)
3. Confirmar com o usuário que a licença foi conferida (R5)

**Como verificar:**
- Arquivo presente, ≤ 30 páginas, licença confirmada; senão, marcar a fase como 🚫 e parar

### 5.2 Teste de integração com o Docling real [Completada ✅]

**O que fazer:**
Criar o teste que roda `parse_pdf` real com o fixture (decisão 3.4).

**Onde fazer:**
- `tests/integration/pdf_parser/__init__.py` (novo)
- `tests/integration/pdf_parser/test_markdown_export.py` (novo)

**Como fazer:**
1. Fixture de módulo com `DoclingParser(max_pages=30, max_file_size_mb=20)` e um único `parse_pdf(FIXTURE, markdown_output_dir=tmp_path_factory.mktemp(...))` reaproveitado pelos asserts (evita converter o PDF várias vezes)
2. Asserts: `markdown_path == out / "2609.17429v1.md"` existe; `artifacts_dir == out / "2609.17429v1_assets"` existe; se houver figuras, `figure_001.png` existe e cada `figure_*.png` tem link `![…](2609.17429v1_assets/figure_NNN.png)` no `.md`; se houver tabelas, `table_001.csv` existe e o número de CSVs é igual ao de tabelas inline; nenhum link para `.csv` no `.md`
3. Assert de R1: pelo menos um `alt` começa com "Figure" e não é só `Figure NNN` (detecta legenda não encontrada)
4. `sections` e `raw_text` não vazios

**Como verificar:**
- `uv run pytest tests/integration/pdf_parser -q` passando

### 5.3 Medir o custo de `generate_picture_images` [Completada ✅]

**O que fazer:**
Registrar o impacto de R4 sem mudar código: tempo de conversão do fixture com a opção ligada e desligada.

**Onde fazer:**
- Script descartável fora do repositório (scratchpad da sessão); resultado nos comentários desta fase

**Como fazer:**
1. Converter o fixture com `DocumentConverter` + `PdfPipelineOptions(generate_picture_images=False)` e `=True`, medindo o tempo de cada (descartar a primeira execução, que carrega modelos)
2. Anotar os tempos nos comentários da fase e no PR

**Como verificar:**
- Dois tempos registrados nos comentários da FASE 5

### 5.T Testes da Fase 5 [Completada ✅]

> 🧪 **Obrigatório**: Toda fase deve incluir testes para o código implementado.

**O que testar:**
- Teste de integração da 5.2
- Suíte completa sem regressões

**Tipos de teste:**
- [ ] Unitário: suíte completa (regressão)
- [x] Integração: `tests/integration/pdf_parser/test_markdown_export.py` com o Docling real
- [ ] E2E: não se aplica

**Onde criar:**
- `tests/integration/pdf_parser/`

**Cobertura mínima:** todos os critérios de "Testes" da TASK-001

**Como verificar:**
- `make lint` (`ruff check --fix` + `mypy src/`) e `make test` passando
- Inspeção manual do `.md` gerado (abrir e conferir texto, tabelas e figuras na ordem do PDF), registrada nos comentários

### Comentários da Fase 5:

- **Bloqueio (26/10/2026)**: `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` não existe no repositório; o fixture é fornecido pelo usuário (decisão no `eng.start`). Retomar com `/eng.work TASK-001` depois de colocar o arquivo (licença conferida, até 30 páginas).
- **Desbloqueio**: fixture fornecido pelo usuário: `%PDF-1.7`, 15 páginas (limite 30), licença conferida pelo usuário.
- **Mudança (pedido do usuário)**: corrigidos dois problemas antigos fora do escopo original, em commit separado (`81ef422`): (1) `pytest_plugins` em `tests/unit/domain/rerank/conftest.py` interrompia a coleta de **toda** a suíte (`make test` quebrado na `main`); as fixtures compartilhadas foram para `tests/unit/domain/conftest.py`; (2) os 6 `I001` restantes do ruff.
- **Questão nova (Q5, levantada no `eng.work`)**: no PDF real a regra da Q1 + Q2 acertou 5/7 `alt`s (legendas "Figure 1" e "Figure 2" agrupadas depois das duas figuras; "Figure 7" antes da figura). Usuário escolheu "uma vez só + olhar para trás": 7/7 no fixture. Registrada na seção 8 do `architecture.md`. Freelance: sem comentário no card.
- **Aprendizado (ambiente)**: a GPU desta máquina (GTX 1050, compute capability 6.1) não é suportada pelo `torch` 2.8.0 instalado (`CUDA error: no kernel image`). `DOCLING_DEVICE` não funciona depois do import (o `AcceleratorOptions` padrão é criado na importação, `docling/datamodel/pipeline_options.py:259`); o teste de integração fixa CPU nas opções do conversor antes do `convert`. O código de produção usa `device="auto"` e, nesta máquina, falharia na GPU: problema de ambiente, fora do escopo.
- **Medição (5.3, CPU, fixture de 15 páginas, primeira execução descartada)**: `generate_picture_images=False`: 113,9 s e 99,2 s; `=True`: 101,9 s e 97,5 s. Sem diferença mensurável de tempo (a variação entre execuções é maior). Pico de memória do processo: 2,7 GB (não separado por configuração).
- **Resultado no fixture**: 7 figuras (`figure_001` a `007`, todas com o `alt` da legenda certa, conferido nos PNGs) e 11 tabelas (`table_001` a `011`), 32 seções.
- **Atenção (inspeção manual)**: rótulos de texto de diagramas vetoriais (ex.: "live sensor", "dead sensor") aparecem como parágrafos soltos no `.md`; o Docling os emite como itens de texto no corpo. Não está nos critérios da TASK-001; registrado para melhoria futura.
- **Testes**: `make lint` ok (ruff + mypy). `make test`: 269 passando; 3 falhas e 7 erros **anteriores a esta tarefa**: `tests/api/*` (precisam de PostgreSQL em `localhost:5412` e OpenSearch, via `make start`) e 2 em `tests/unit/platform/graph/test_graph_builder.py` (hook `after` do middleware não chamado; falham também na `main`). Integração nova: 5 testes passando (cerca de 2 min).
- **Commit**: `377db54`.

---

## FASE 6: Robustez do `markdown_export` (pre-pr: C1–C5, C7, T2, T4, T5) [Completada ✅]

> ⏱️ **Tempo estimado**: ~60 minutos
>
> 📝 **Objetivo**: Corrigir as lacunas de código do `gate-checklist.md` em `src/domain/pdf_parser/markdown_export.py`.
>
> ⚠️ **Dependência**: FASE 5 completada; decisões "Corrigir" do `gate-checklist.md`

### 6.1 Tabelas e figuras pelo tipo (C1) [Completada ✅]

**O que fazer:** decidir por `isinstance(item, TableItem)` / `isinstance(item, PictureItem)`, para tabelas `DOCUMENT_INDEX` e figuras `CHART` não sumirem.
**Onde fazer:** `src/domain/pdf_parser/markdown_export.py` (`export_markdown`)
**Como verificar:** teste com `doc.add_table(..., label=DocItemLabel.DOCUMENT_INDEX)` gera bloco inline e `table_001.csv`.

### 6.2 Escape do `alt` e do link (C2) [Completada ✅]

**O que fazer:** escapar `\ [ ] * _` e crase no `alt` (imagem e fallback em itálico) e usar destino `<...>` no link.
**Onde fazer:** `markdown_export.py` (`_render_picture`)
**Como verificar:** teste com `alt="Figure 1: [0, 1) F_1"` e `pdf_stem="a b (v2)"`.

### 6.3 Limpeza de assets antigos (C3) e cerca de código (C7) [Completada ✅]

**O que fazer:** remover `figure_*.png` e `table_*.csv` da pasta `{stem}_assets` antes de exportar; cerca de código maior que a maior sequência de crases do texto.
**Onde fazer:** `markdown_export.py` (`export_markdown`, `_render_text_item`)
**Como verificar:** teste de reexportação com menos figuras não deixa órfãos; teste de `CODE` com três crases.

### 6.4 `_resolve_alt` com critério único (C4, C5) [Completada ✅]

**O que fazer:** pré-calcular as posições das legendas "Figure" uma vez; `used_captions` obrigatório; um só critério (posição) nos três passos; buscas percorrem só as posições de legenda e de seção.
**Onde fazer:** `markdown_export.py` (`_resolve_alt`, `export_markdown`)
**Como verificar:** todos os testes de `alt` existentes continuam passando.

### 6.T Testes da Fase 6 (T2, T4, T5) [Completada ✅]

**O que testar:** C1, C2, C3, C7; precedência busca adiante x `caption_text` (T2); busca para trás pega a mais próxima (T4); limite de seção com `TITLE` (T5).
**Onde criar:** `tests/unit/domain/pdf_parser/test_markdown_export.py`
**Como verificar:** `uv run pytest tests/unit/domain/pdf_parser tests/unit/domain/test_pdf_parser.py`; `ruff check`/`ruff format --check`.

### Comentários da Fase 6:

- **Decisão (C1)**: em vez de `isinstance`, conjuntos de rótulos `{PICTURE, CHART}` e `{TABLE, DOCUMENT_INDEX}`: é exatamente o tipo dos campos `label` de `PictureItem`/`TableItem` no `docling-core` 2.48.1, e mantém os testes com documentos falsos. Amplia o critério da issue (que citava só `TABLE`/`PICTURE`): vai para "Divergências" na FASE 9.
- **Decisão (C2)**: o `alt` escapa `\ ` `` ` `` `* _ [ ]`; o destino do link só vai entre `<...>` quando tem espaço, parêntese ou `<`/`>`, para a saída comum (stems do arXiv) não mudar.
- **Decisão (C3)**: só arquivos com o padrão `figure_NNN.png`/`table_NNN.csv` são removidos da pasta de assets antes de exportar; outros arquivos ficam (teste com `notes.txt`).
- **Mudança (C4, C5)**: `_resolve_alt` passou a usar `_CaptionIndex` (posições das legendas "Figure" e dos inícios de seção, montado uma vez por documento, com `bisect`). Um só critério (posição) nos três passos; o conjunto de legendas usadas é obrigatório (faz parte do índice). A busca percorre só as posições de legenda.
- **Aprendizado (T2)**: pela ordem da Q5, uma figura cuja legenda própria não foi visitada pode pegar a legenda visitada da figura seguinte; o teste documenta o comportamento. No pipeline do Docling a legenda própria é visitada logo depois da figura, então o caso é raro.
- **Testes**: 66 passando (`pdf_parser/` e `test_pdf_parser.py`); `markdown_export.py` 100% das linhas e 99% dos ramos.
- **Atenção**: `ruff format --check` acusa 3 arquivos antigos em `tests/unit/domain/agents/` (não tocados nesta branch; `make lint` não roda `ruff format`).

---

## FASE 7: Testes do parser e da integração (pre-pr: C6, T1, T3) [Completada ✅]

> ⏱️ **Tempo estimado**: ~45 minutos (o teste de integração leva ~2 min)
>
> 📝 **Objetivo**: Testes sem atributos privados e integração travando a Q5 e o CSV.
>
> ⚠️ **Dependência**: Requer FASE 6 completada

### 7.1 Dispositivo no construtor do `DoclingParser` (C6) [Completada ✅]

**O que fazer:** parâmetro opcional `accelerator_device` (padrão `auto`, comportamento atual) repassado às `PdfPipelineOptions`; teste de integração usa `accelerator_device="cpu"` em vez de `parser._converter`.
**Onde fazer:** `src/domain/pdf_parser/docling.py`, `tests/integration/pdf_parser/test_markdown_export.py`
**Como verificar:** teste unitário confere o dispositivo nas pipeline options; integração passa.

### 7.2 Testes unitários do parser sem atributos privados (C6) [Completada ✅]

**O que fazer:** `TestDoclingParserMarkdownExport` faz patch de `DocumentConverter` e de `pdfium` no módulo, em vez de trocar `_converter`/`_validate_pdf` na instância.
**Onde fazer:** `tests/unit/domain/test_pdf_parser.py`

### 7.3 Integração trava a Q5 e o CSV (T1, T3) [Completada ✅]

**O que fazer:** nenhum `alt` no fallback `Figure NNN` no fixture; cada legenda "Figure N:" do `.md` aparece como `alt` uma vez; primeiro grid GFM igual a `table_001.csv`.
**Onde fazer:** `tests/integration/pdf_parser/test_markdown_export.py`

### 7.T Testes da Fase 7 [Completada ✅]

**Como verificar:** `uv run pytest tests/unit/domain/test_pdf_parser.py tests/integration/pdf_parser`.

### Comentários da Fase 7:

- **Decisão (C6)**: `DoclingParser(..., accelerator_device=None)`. Com `None` (padrão), nada muda: o Docling segue com o `AcceleratorOptions` padrão, que respeita `DOCLING_DEVICE`; passar `AcceleratorOptions(device="auto")` sempre quebraria essa variável em produção. A factory não usa o parâmetro.
- **Mudança (C6)**: `TestDoclingParserMarkdownExport` agora faz patch de `DocumentConverter` e `pdfium.PdfDocument` no módulo; a validação real roda com um arquivo `%PDF-` temporário, e o teste de limite usa 40 páginas de verdade pelo `_validate_pdf`. Testes novos do dispositivo (com e sem `accelerator_device`).
- **Mudança (T1, T3)**: a integração exige que nenhum `alt` seja `Figure NNN` e que os `alt`s sejam exatamente as legendas "Figure N:" do `.md` (sem repetição), e compara `table_001.csv` com a primeira tabela inline (via `_parse_gfm_table`). O regex do `alt` aceita os escapes da FASE 6.
- **Testes**: 68 unitários passando (`test_pdf_parser.py` e `pdf_parser/`); integração 6 passando em 116 s.

---

## FASE 8: Ambiente e documentação do repositório (pre-pr: D1–D6) [Completada ✅]

> ⏱️ **Tempo estimado**: ~40 minutos
>
> 📝 **Objetivo**: `uv sync` funcionar sem `--python` e documentação refletir a mudança.
>
> ⚠️ **Dependência**: Requer FASE 7 completada

### 8.1 Versão do Python (D2) [Completada ✅]

**O que fazer:** `requires-python = ">=3.12,<3.13"` no `pyproject.toml` e `uv lock`; conferir que `uv sync` (sem `--python`) escolhe o 3.12.
**Onde fazer:** `pyproject.toml`, `uv.lock`

### 8.2 Documentação do repositório (D1, D3, D4, D5) [Completada ✅]

**O que fazer:** AGENTS.md (seção Testing: teste de integração lento e `tests/unit/domain/conftest.py`), onboarding (troubleshooting de GPU, árvore do `pdf_parser`, frase sobre a exportação), ARTICLE.md (frase na ingestão).
**Onde fazer:** `AGENTS.md`, `docs/onboarding.md`, `docs/ARTICLE.md`

### 8.3 `AGENTS.md`/`CLAUDE.md` da raiz do workspace (D6) [Completada ✅]

**O que fazer:** aviso do teste lento em "Instruções de Teste" e nota da versão do Python (fora do repositório; sem commit).
**Onde fazer:** `/AGENTS.md`, `/CLAUDE.md`

### 8.T Testes da Fase 8 [Completada ✅]

**Como verificar:** `uv sync` sem `--python` resolve o 3.12; `make lint`; `make test` (mesmos resultados da FASE 5).

### Comentários da Fase 8:

- **Mudança (D2)**: `requires-python = ">=3.12,<3.13"` e `uv lock`: nenhuma versão de pacote mudou (0 linhas `version =` alteradas); saíram as wheels/marcadores de 3.13+ e o formato do lock subiu de `revision = 3` para `5` (uv 0.13.1). Num ambiente novo, `uv sync --frozen` sem `--python` escolheu o CPython 3.12.15. A linha `requires-python` do `AGENTS.md` do repositório passou a estar correta sem edição.
- **Mudança (D1, D3, D4, D5)**: em inglês, como o restante da documentação do repositório: `AGENTS.md` (Testing: conftest compartilhado, aviso sobre `pytest_plugins` e o teste de integração lento), `README.md` (Python 3.12), `docs/onboarding.md` (Docling com a exportação, árvore do `pdf_parser`, linha de Common Issues para o erro de CUDA), `docs/ARTICLE.md` (uma frase na seção de ingestão).
- **Mudança (D6)**: `AGENTS.md` e `CLAUDE.md` da raiz do workspace (fora do repositório, sem commit): versão do Python e aviso do teste lento no `make test`.
- **Testes**: `make lint` ok; `make test` 282 passando (13 a mais que na FASE 5), com as mesmas 3 falhas e 7 erros anteriores à branch.

---

## FASE 9: Documentos de produto, sessão e notas do PR (pre-pr: P1–P9) [Completada ✅]

> ⏱️ **Tempo estimado**: ~30 minutos
>
> 📝 **Objetivo**: Documentos de produto e da sessão coerentes com o que foi implementado.
>
> ⚠️ **Dependência**: Requer FASE 8 completada

### 9.1 Issue TASK-001 (P2, P6, P9) [Completada ✅]

**O que fazer:** seção "Divergências" (Q5, tabelas/figuras por tipo, escape, limpeza de assets, dispositivo no construtor, versão do Python); caso extremo 6 resolvido; `status: in_progress`; melhoria futura dos rótulos de diagramas no "Fora do escopo".
**Onde fazer:** `docs/jarvis/product/issues/task-001-exportacao-markdown-pdf-parser.md`

### 9.2 `architecture.md` e `plan.md` (P3) [Completada ✅]

**O que fazer:** Q1/Q2 marcadas como substituídas pela Q5; R7 mitigado; seção 10 e pré-requisitos do plano atualizados.

### 9.3 Registro da licença do fixture (P8) [Completada ✅]

**O que fazer:** `tests/fixtures/arxiv_pdfs/README.md` com origem do arquivo e a conferência de licença (sem inventar o tipo da licença).

### 9.4 Notas do PR (P1, P4, P5, P7) [Completada ✅]

**O que fazer:** PR Brief do `gate-checklist.md` com falhas anteriores à branch, "fora do escopo original", medição com um PDF só e a diferença do pipe escapado.

### 9.T Verificação da Fase 9 [Completada ✅]

**Como verificar:** links dos documentos abrem; nenhum código alterado nesta fase além do README do fixture.

### Comentários da Fase 9:

- **Mudança (P2, P6, P9)**: issue TASK-001 com os critérios de Figuras e Tabelas atualizados (Q5, `CHART`/`DOCUMENT_INDEX`, escape, link com `<...>`), casos extremos 4, 6 e 7 resolvidos, `status: in_progress` e a seção "Divergências" no fim; a melhoria futura dos rótulos de diagramas entrou no "Fora do escopo". O discovery-001 não foi alterado (regra "Discovery como insumo").
- **Mudança (P3)**: `architecture.md` com Q1/Q2 marcadas como substituída e incorporada à Q5, R7 mitigado, seção 10 e histórico atualizados; `plan.md` com pré-requisitos e riscos atualizados.
- **Decisão (P8)**: `tests/fixtures/arxiv_pdfs/README.md` registra origem, páginas e a conferência da licença pelo responsável, sem citar o tipo de licença (não informado).
- **Mudança (P1, P4, P5, P7)**: PR Brief do `gate-checklist.md` reescrito com as falhas anteriores à branch, o que está fora do escopo original, a limitação da medição e a diferença do pipe escapado.
- **Verificação**: links issue → discovery e architecture → issue/discovery abrem.

---

## FASE 10: Bordas do `alt`, do escape e da limpeza (pre-pr 2ª rodada: C8–C11, T6–T9, D9) [Completada ✅]

> ⏱️ **Tempo estimado**: ~45 minutos
>
> 📝 **Objetivo**: Corrigir os casos de borda apontados na 2ª rodada do `/eng.pre-pr` em `markdown_export.py` e `docling.py`.
>
> ⚠️ **Dependência**: FASE 9 completada; decisões "Corrigir" da 2ª rodada do `gate-checklist.md`

### 10.1 Legendas ligadas uma a uma (C8) [Completada ✅]

**O que fazer:** o passo 2 do `alt` percorre `picture.captions` (cada legenda resolvida e normalizada) em vez do `caption_text` concatenado; textos devolvidos sem posição visitada também ficam registrados como usados.
**Onde fazer:** `src/domain/pdf_parser/markdown_export.py` (`_resolve_alt`, `_CaptionIndex.take_text`)

### 10.2 Limpeza e gravação segura (C9) [Completada ✅]

**O que fazer:** regex `\d{3,}`; o `.md` é gravado num arquivo temporário e trocado no fim (`replace`).
**Onde fazer:** `markdown_export.py` (`_ASSET_FILE`, `export_markdown`)

### 10.3 Escape do link e do `alt` (C10) [Completada ✅]

**O que fazer:** destino do link com `urllib.parse.quote(link, safe="/")` (e `<...>` só se ainda houver caractere inseguro); escapar `&` e `<` no `alt`.
**Onde fazer:** `markdown_export.py` (`_link_destination`, `_escape_markdown_inline`)

### 10.4 Busca de legendas com `bisect` (C11) e texto do fallback (D9) [Completada ✅]

**O que fazer:** lista ordenada de posições de legenda e `bisect` em `take_next`/`take_previous`; fallback `_{alt} (image unavailable)_` (decisão do usuário, substitui o texto da Q4).
**Onde fazer:** `markdown_export.py`

### 10.T Testes da Fase 10 (T6–T9) [Completada ✅]

**O que testar:** C8–C11 e D9; escape de `\`, crase e `]` (T6); `pdf_stem` com `<`/`>`, `#`, `%` (T7); limpeza poupa `figure_1.png` e pasta `table_001.csv/` e limpa `figure_1000.png` (T8 e C9); `accelerator_device="gpu"` levanta erro (T9).
**Onde criar:** `tests/unit/domain/pdf_parser/test_markdown_export.py`, `tests/unit/domain/test_pdf_parser.py`
**Como verificar:** unitários, integração (`tests/integration/pdf_parser`), ruff.

### Comentários da Fase 10:

- **Mudança (C8)**: o passo 2 do `alt` resolve cada referência de `picture.captions` separadamente (`_linked_caption_texts`); o `caption_text` concatenado só é usado quando o item não tem `captions` (documentos falsos). Textos ligados que não foram visitados são devolvidos uma vez só (`used_unvisited_texts`).
- **Mudança (C9)**: regex de limpeza com `\d{3,}`. Em vez de gravar o `.md` num temporário, o `.md` antigo é removido junto com os assets antigos no início: com a troca por temporário, uma falha depois da limpeza ainda deixaria o `.md` antigo apontando para assets apagados.
- **Mudança (C10)**: destino do link com `urllib.parse.quote(link, safe="/")` (codifica espaço, parênteses, `#`, `%`, `?`, `<`, `>`), substituindo os `<...>` da FASE 6; nomes do arXiv saem iguais. O `alt` também escapa `<` e `&`.
- **Mudança (C11)**: `_CaptionIndex.positions` (lista ordenada) + `bisect` em `take_next`/`take_previous`.
- **Decisão (D9, usuário)**: fallback `_{alt} (image unavailable)_`, substituindo o texto da Q4.
- **Testes**: 76 unitários passando; `markdown_export.py` 100% das linhas (só os 2 ramos parciais aceitos do `_split_gfm_row`); integração 6 passando.

---

## FASE 11: Documentação (pre-pr 2ª rodada: D7, D8, P10, P11) [Completada ✅]

> ⏱️ **Tempo estimado**: ~20 minutos
>
> ⚠️ **Dependência**: Requer FASE 10 completada

### 11.1 Python 3.12 e dispositivo (D7, D8) [Completada ✅]

**O que fazer:** "3.12+" → 3.12 em `docs/onboarding.md` e no badge do `README.md`; linha de GPU do onboarding e docstring do `DoclingParser` dizem que `accelerator_device` só vale instanciando o `DoclingParser` (pelo serviço/Airflow, `DOCLING_DEVICE=cpu`).

### 11.2 Issue: "Divergências" (P10, P11, D9) [Completada ✅]

**O que fazer:** índice de legendas montado antes da passada; grafia do pipe escapado no CSV; texto `(image unavailable)` no caso extremo 7.

### 11.3 `architecture.md` (D9) [Completada ✅]

**O que fazer:** registrar na Q4 que o texto mudou para `(image unavailable)`.

### 11.T Verificação da Fase 11 [Completada ✅]

**Como verificar:** `make lint`; `make test` com as mesmas falhas anteriores à branch.

### Comentários da Fase 11:

- **Mudança (D7)**: badge do `README.md` e pré-requisitos do `docs/onboarding.md` com Python 3.12 (sem o "+").
- **Mudança (D8)**: linha de GPU do onboarding e docstring do `DoclingParser` dizem que `accelerator_device` só vale ao instanciar o parser; pela API, `PDFParserService` e Airflow, usar `DOCLING_DEVICE=cpu` ou `CUDA_VISIBLE_DEVICES=""`. O serviço não repassa o parâmetro (a sugestão alternativa, repassar, ficou de fora para não mudar a interface do serviço e da factory).
- **Mudança (P10, P11, D9)**: issue com os critérios de link, legendas ligadas e escape atualizados, caso extremo 4 e 7 com o comportamento novo, e quatro itens novos em "Divergências"; `architecture.md` registra na Q4 o texto `(image unavailable)`.
- **Testes**: `make lint` ok; `make test` 290 passando, com as mesmas 3 falhas e 7 erros anteriores à branch.

---

## FASE 12: Ajustes finais (pre-pr 3ª rodada: C12, D10, D11, N2, N3) [Completada ✅]

> ⏱️ **Tempo estimado**: ~20 minutos
>
> ⚠️ **Dependência**: FASE 11 completada; decisões "Corrigir" da 3ª rodada do `gate-checklist.md`

### 12.1 Índice de legendas por texto (C12) [Completada ✅]

**O que fazer:** `_CaptionIndex.by_text` (texto → posições ordenadas) montado no `__post_init__` e usado em `take_text`.
**Onde fazer:** `src/domain/pdf_parser/markdown_export.py`

### 12.2 Docstring do módulo (D11) [Completada ✅]

**O que fazer:** citar o fallback `(image unavailable)` na docstring de `markdown_export.py`.

### 12.3 Onboarding (D10) [Completada ✅]

**O que fazer:** bullet do Docling em `docs/onboarding.md` com a reexportação e o fallback.

### 12.4 Documentos da sessão (N2, N3) [Completada ✅]

**O que fazer:** Q4 do `architecture.md` com o texto `(image unavailable)` na opção e na decisão (nota mantida como histórico); notas nas fases antigas do `plan.md`.

### 12.T Testes da Fase 12 [Completada ✅]

**Como verificar:** unitários do `pdf_parser`, integração, `make lint`.

### Comentários da Fase 12:

- **Mudança (C12)**: `_CaptionIndex.by_text` (texto → posições em ordem) montado no `__post_init__`; `take_text` passa a consultar o dicionário em vez de percorrer todas as legendas. Comportamento igual, coberto pelos testes de `take_text` existentes.
- **Mudança (D10, D11)**: docstring do módulo e bullet do Docling no onboarding citam o fallback `(image unavailable)` e a substituição da saída na reexportação.
- **Mudança (N2, N3)**: Q4 do `architecture.md` mostra o texto atual na opção e na decisão, com o original como histórico; menções antigas no `plan.md` marcadas como substituídas na FASE 10.
- **Testes**: 76 unitários e 6 de integração passando; `make lint` ok; `markdown_export.py` 100% das linhas.

---

## Dependências entre Tarefas

```
FASE 1 (ambiente, modelo, pipeline options)
  ├── 1.1 ──→ 1.2 ──→ 1.3 ──→ 1.T
  └──────────────────────────────→ FASE 2 (texto e tabelas)
                                     ├── 2.1 ┐ (paralelas)
                                     ├── 2.2 ┘──→ 2.T
                                     └──────────→ FASE 3 (figuras e montagem)
                                                    ├── 3.1 ┐ (paralelas)
                                                    ├── 3.2 ┘──→ 3.3 ──→ 3.T
                                                    └──────────────────→ FASE 4 (parser e serviço)
                                                                           ├── 4.1 ──→ 4.2 ──→ 4.T
                                                                           └──────────────────→ FASE 5 (PDF real)
                                                                                 [fixture do usuário] ──→ 5.1 ──→ 5.2 ──→ 5.3 ──→ 5.T
```

**Legenda:**
- `→` = Dependência sequencial (uma após a outra)
- Tarefas marcadas "paralelas" podem ser feitas em qualquer ordem

---

## Notas Importantes

### Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| R1: legenda filha da figura não é visitada | Média | Médio | Fallback `caption_text` (Q1) na 3.1; assert específico na 5.2 |
| R3: formato do GFM do Docling (sem header, `\|` escapado) diferente do esperado | Média | Baixo | Parser tolerante na 2.2; ajustar com o caso real na FASE 5 |
| R4: `generate_picture_images` deixa a ingestão mais lenta | Alta | Médio | Medição na 5.3, registrada no PR |
| R5: fixture sem licença adequada ou com mais de 30 páginas | Baixa | Alto | Conferência na 5.1; FASE 5 bloqueada até o fixture |
| R6: `make test` depende dos modelos do Docling (download, rede) | Média | Médio | Anotar no PR; primeira execução fora do horário crítico |
| R7: figura sem legenda herda a legenda da seguinte na seção | Baixa | Baixo | Mitigado pela Q5 (cada legenda usada uma vez); caso residual documentado em teste (FASE 6) |
| Ambiente: `make setup` falhar (sem `.venv` hoje) | Baixa | Alto | Tarefa 1.1 antes de qualquer código; se falhar, parar e reportar |

### Pontos de Atenção

- `extract_sections_from_document` e o `iterate_items()` dele não mudam (Q1 não altera a passada das seções)
- A ingestão (`src/domain/arxiv_ingestion/service.py:441`) não passa o parâmetro; não alterar
- Nenhuma chamada a `doc.export_to_markdown()` (requisito 1 da TASK-001)
- `PDFParsingException` da exportação precisa passar pelo `except` do `parse_pdf` sem ser reembrulhada (4.1)
- Sem CI: `make lint` e `make test` locais são a validação; `mypy` roda com `ignore_errors = true`

### Decisões Pendentes

- [x] Nenhuma de arquitetura (Q1–Q5 decididas). PDF fixture fornecido pelo usuário.

---

## Histórico de Atualizações

| Data | Fase | Status | Observação |
|------|------|--------|------------|
| 26/10/2026 | - | Criado | Plano inicial criado a partir do `architecture.md` aprovado |
| 26/10/2026 | - | Aprovado | Aprovado pelo usuário ("Aprovar e implementar agora") |
| 26/10/2026 | 1 a 4 | Completadas | Commits 921a049, f2627cd, 9ad0706, 848281b |
| 26/10/2026 | 5 | Bloqueada | Aguardando o PDF fixture do usuário |
| 26/10/2026 | 6–9 | Criado | Fases de correção das 27 lacunas do `gate-checklist.md` (todas "Corrigir") |
| 26/10/2026 | 12 | Completada | Ajustes finais da 3ª rodada do `/eng.pre-pr` |
| 26/10/2026 | 10–11 | Completadas | Commits a56b7b2 e o da FASE 11 |
| 26/10/2026 | 10–11 | Criado | Correções das 13 lacunas ⚪ da 2ª rodada do `/eng.pre-pr` |
| 26/10/2026 | 6–9 | Completadas | Correções das 27 lacunas do `/eng.pre-pr` |
| 26/10/2026 | 5 | Completada | Fixture recebido; Q5 decidida; correções de testes antigos a pedido do usuário (81ef422, 377db54) |

---

## Checklist Final

- [x] Todas as fases completadas ✅
- [x] Todos os testes da tarefa passando (falhas antigas de ambiente e do `graph_builder` documentadas na FASE 5)
- [x] Código revisado (`/eng.pre-pr`; lacunas corrigidas nas fases 6 a 9)
- [x] Documentação atualizada
- [ ] Pronto para PR
