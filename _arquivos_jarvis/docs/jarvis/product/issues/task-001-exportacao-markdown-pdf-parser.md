---
# Modelo de IA recomendado para trabalhar neste documento. Se a sessão estiver em outro,
# o comando avisa e segue (você decide se troca). Opções: opus | sonnet | haiku
recommended_model: opus

# Identificador único, em sequência (ex.: STORY-001, TASK-001, BUG-001). Vai no nome do arquivo e nos campos related_* de outros documentos.
id: TASK-001

# Nome curto do documento. Vai no título e, em kebab-case, no nome do arquivo.
name: Exportação de Markdown no pdf_parser

# Tipo da issue.
# Opções: Story | Task | Bug
# Story = história de usuário (valor para quem usa) | Task = tarefa técnica | Bug = defeito a corrigir
type: Task

# Discovery de origem (o rascunho da ideia, anterior ao PRD). Remova a linha se não houver.
related_discovery: docs/jarvis/discoveries/discovery-001-exportacao-markdown-pdf-parser.md

# Data de criação (YYYY-MM-DD).
created_at: 2026-10-26

# Data da última alteração (YYYY-MM-DD).
updated_at: 2026-10-26

# Em que ponto este documento está no ciclo de vida (lista em rules/product/prod-rules.md).
# Opções: icebox | in_review | backlog | in_progress | in_production | cancelled | deprecated
# icebox = ideia aguardando priorização | in_review = sendo estudado ou revisado | backlog = definido, aguardando implementação
# in_progress = em desenvolvimento | in_production = em produção
# cancelled = cancelado antes de ser entregue | deprecated = esteve em produção e foi descontinuado
status: in_progress

# Quem criou o documento.
created_by: phelipperibeiro@hotmail.com

# Quem editou o documento por último.
last_editor: phelipperibeiro@hotmail.com
---

# TASK-001: Exportação de Markdown no pdf_parser

## Contexto

O `production-rag` converte artigos do arXiv com o Docling no domínio `src/domain/pdf_parser/`. Hoje o resultado (`PdfContent`) guarda só o texto (`raw_text`) e as seções (`sections`); figuras e tabelas são descartadas (`figures=[]`, `tables=[]` em `src/domain/pdf_parser/docling.py`). Para avaliar a qualidade do parsing de um artigo, quem desenvolve precisa inspecionar o banco ou rodar o Docling à mão num notebook, e nenhum desses caminhos mostra figuras e tabelas.

Esta tarefa acrescenta uma exportação **opcional** do artigo para Markdown, com uma pasta de assets ao lado (figuras em PNG, tabelas em CSV), ativada por um parâmetro novo em `parse_pdf`. O uso é **revisão humana e depuração do parsing**: a ingestão (`src/domain/arxiv_ingestion/service.py`) continua chamando `parse_pdf(pdf_path)` sem o parâmetro e não muda de comportamento. A investigação que levou a esta tarefa está no [discovery-001](../../discoveries/discovery-001-exportacao-markdown-pdf-parser.md), com veredito `proceed`.

**Prioridade:** Alta.

**Como** pessoa que desenvolve e depura o `production-rag`
**Eu quero** exportar um artigo parseado para Markdown, com as figuras e tabelas em arquivos ao lado
**Então isso** me deixa conferir, na ordem do PDF, o que o Docling extraiu (texto, tabelas e figuras) sem rodar o Docling à mão

## Critérios de aceitação

#### Parâmetro e compatibilidade
- Quando `DoclingParser.parse_pdf` é chamado sem `markdown_output_dir`, o sistema retorna o `PdfContent` como hoje, com `markdown_path` e `artifacts_dir` iguais a `None`, e não grava nenhum arquivo
- `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf` aceitam o parâmetro opcional `markdown_output_dir: Path | None = None`; o `PDFParserService` repassa o valor ao `DoclingParser`
- Quando `markdown_output_dir` é informado, o sistema roda a exportação **depois** do fluxo existente (`DocumentConverter` + `extract_sections_from_document`), e `sections` e `raw_text` saem iguais aos de uma chamada sem o parâmetro
- Quando `markdown_output_dir` é informado, o sistema grava `{markdown_output_dir}/{pdf_stem}.md` e a pasta `{markdown_output_dir}/{pdf_stem}_assets/`, em que `pdf_stem` é o nome do PDF sem extensão
- Quando a exportação termina, `PdfContent.markdown_path` aponta para o `.md` gerado e `PdfContent.artifacts_dir` aponta para a pasta de assets
- As pipeline options do Docling usam `generate_picture_images=True` em toda conversão, com ou sem `markdown_output_dir`

#### Montagem do Markdown
- O sistema monta o Markdown numa **única** passada guiada por `doc.iterate_items()`, sem chamar `export_to_markdown()` do documento e sem um segundo loop separado só para figuras
- Os elementos aparecem no `.md` na ordem de visita de `doc.iterate_items()`
- `TITLE` e `SECTION_HEADER` viram títulos Markdown (`#`) conforme o nível; `LIST_ITEM` vira item de lista (`-`); `CODE` vira bloco de código cercado; os demais itens de texto (incluindo `CAPTION` e `FORMULA`) viram parágrafo

#### Figuras
- Para cada figura (`DocItemLabel.PICTURE` ou `DocItemLabel.CHART`), o sistema grava `figure_NNN.png` na pasta de assets, com `NNN` de 3 dígitos começando em `001` e seguindo a ordem de `doc.iterate_items()`
- No ponto da figura, o `.md` contém `![alt]({pdf_stem}_assets/figure_NNN.png)`, com caminho relativo ao `.md`, codificado para URL (espaço, parênteses, `#`, `%`, `?`, `<` e `>` viram `%XX`; nomes do arXiv saem iguais)
- Cada legenda `CAPTION` que começa com "Figure" é usada como `alt` por uma figura só. Para cada figura, nesta ordem: (1) a próxima legenda "Figure" ainda não usada, até o fim da seção (próximo `TITLE` ou `SECTION_HEADER`), mesmo passando por outras figuras e tabelas; (2) uma legenda ligada à própria figura pelo Docling (cada uma de `picture.captions`, resolvida separadamente), se começar com "Figure" e não tiver sido usada; (3) a legenda "Figure" não usada mais próxima **antes** da figura, na mesma seção
- Quando nenhuma legenda serve, o `alt` é `Figure NNN`, com o mesmo número do arquivo
- Os caracteres `\`, `` ` ``, `*`, `_`, `[`, `]`, `<` e `&` do `alt` são escapados, para a legenda não quebrar a sintaxe do Markdown nem virar HTML ou entidade

#### Tabelas
- Para cada tabela (`DocItemLabel.TABLE` ou `DocItemLabel.DOCUMENT_INDEX`), o sistema grava `table_NNN.csv` na pasta de assets, com contador próprio de 3 dígitos começando em `001` (independente do contador de figuras)
- No ponto da tabela, o `.md` contém a tabela **inline** como GFM pipe table, sem link para o CSV
- O CSV é gravado em UTF-8, com vírgula como delimitador e `csv.QUOTE_MINIMAL`
- Quando o grid inline tem a linha separadora de cabeçalho, a primeira linha do CSV é esse cabeçalho
- Quando o Docling emite um grid sem cabeçalho, o CSV contém só as linhas de dados, espelhando o grid
- O conteúdo das células do CSV é o mesmo das células da tabela inline no `.md`

#### Testes
- O repositório da tarefa (o fork `phelipperibeiro/production-rag`, não o upstream `apssouza22/production-rag`) commita `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf`
- Existe teste que roda `parse_pdf` com o PDF fixture e `markdown_output_dir` num diretório temporário e verifica: o `.md` e a pasta de assets existem, os nomes começam em `figure_001`/`table_001`, cada figura tem link no `.md` e cada tabela tem CSV
- Existem testes unitários com documentos falsos (no padrão de `tests/unit/domain/pdf_parser/test_docling_sections.py`) que cobrem: contadores separados, regra do `alt` com e sem caption "Figure", CSV com e sem cabeçalho, e chamada sem `markdown_output_dir` não grava nada
- `make lint` e `make test` passam

## Requisitos técnicos

- **Orientação de implementação**: manter a exportação numa função ou módulo próprio dentro de `src/domain/pdf_parser/` (por exemplo, `markdown_export.py`), chamado pelo `DoclingParser.parse_pdf` só quando `markdown_output_dir` vier; isso mantém o `parse_pdf` legível e deixa as regras testáveis com documentos falsos. Para figuras, `PictureItem.get_image(doc)`; para tabelas, montar o grid inline e o CSV a partir da mesma fonte (por exemplo, `TableItem.export_to_dataframe(doc)` ou o grid de células da tabela), para o CSV espelhar o `.md`. Confirmar a API na documentação do Docling 2.52.0 (versão travada no `uv.lock`).
- **Requisitos de desempenho**: `generate_picture_images=True` sempre ligado faz toda conversão renderizar as imagens das figuras, inclusive na ingestão. Medir o tempo de parsing de um lote antes e depois.
- **Considerações de segurança**: gravar só dentro de `markdown_output_dir`; o nome dos arquivos vem do `pdf_stem` e de contadores, sem texto do documento no caminho.
- **Requisitos de dados**: `PdfContent` ganha `markdown_path` e `artifacts_dir`, opcionais e com padrão `None`, para não quebrar quem já instancia `PdfContent` (`src/domain/arxiv_ingestion/service.py` e os testes).

## Casos extremos e tratamento de erros

1. **PDF pulado por limite de páginas ou tamanho**: `parse_pdf` já retorna `None`; nada é gravado.
2. **PDF sem figuras ou sem tabelas**: o `.md` é gravado e a pasta de assets é criada mesmo vazia, para `artifacts_dir` sempre apontar para uma pasta existente.
3. **`markdown_output_dir` inexistente**: o sistema cria o diretório (com os pais) antes de gravar.
4. **Exportação repetida no mesmo diretório**: o `.md` e os `figure_NNN.png`/`table_NNN.csv` da exportação anterior (contadores de 3 ou mais dígitos) são removidos antes de gravar, para uma falha no meio não deixar um `.md` antigo apontando para assets apagados; outros arquivos e pastas da pasta de assets ficam.
5. **Falha ao gravar (permissão, disco)**: o sistema levanta `PDFParsingException` com o caminho e o motivo, já que a exportação foi pedida explicitamente.
6. **Legenda como filha da figura** (resolvido): no pipeline de PDF do Docling a legenda é filha da figura, e o `iterate_items()` a visita logo depois dela mesmo com `traverse_pictures=False`, porque libera os filhos listados em `picture.captions`. No PDF fixture, porém, as legendas podem vir agrupadas depois de várias figuras ou antes da figura, o que levou à regra de "legenda usada uma vez só" dos critérios de Figuras.
7. **Figura sem imagem gerada** (`get_image` sem resultado): o contador avança, nenhum arquivo é gravado, o `.md` mostra `_{alt} (image unavailable)_` e um aviso vai para o log.
8. **PDF fixture acima de 30 páginas** (`PDF_PARSER__MAX_PAGES`): o parser o pularia; conferir o número de páginas e a licença do `2609.17429v1` antes de commitar.

## Fora do escopo

- Passar `markdown_output_dir` na ingestão (`arxiv_ingestion`) ou no DAG do Airflow
- Expor a exportação por CLI ou endpoint da API
- Guardar o Markdown, as figuras ou os CSVs no PostgreSQL ou no OpenSearch, ou usá-los na indexação do RAG
- Configurar a resolução das figuras (`images_scale`) ou ligar/desligar `generate_picture_images` por configuração
- Exportar imagens de páginas inteiras ou imagens das tabelas
- Preencher `PdfContent.figures` e `PdfContent.tables`
- Filtrar os rótulos de texto de diagramas vetoriais (ex.: "live sensor", "dead sensor"), que o Docling emite como itens de texto e saem como parágrafos soltos no `.md`: melhoria futura

### Subtarefas sugeridas

1. Spike: com o PDF fixture, verificar a ordem de `PICTURE` e `CAPTION` em `iterate_items()` e o retorno de `get_image` (casos 6 e 7)
2. Conferir licença e número de páginas do `2609.17429v1` e commitar o fixture
3. Adicionar `markdown_path` e `artifacts_dir` em `PdfContent` e o parâmetro em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`
4. Ligar `generate_picture_images=True` nas pipeline options
5. Implementar a montagem do Markdown (texto, figuras, tabelas) numa passada
6. Implementar a gravação de PNGs e CSVs com os contadores e a regra de cabeçalho
7. Testes unitários com documentos falsos e teste com o PDF fixture
8. Medir o tempo de parsing antes e depois

### Ativos de design

- Não se aplica: a tarefa não tem interface.

## Divergências

Mudanças em relação à primeira versão desta tarefa, decididas durante a implementação (detalhes na seção 8 do `architecture.md` da sessão `.jarvis/sessions/eng/task-001/`):

- **Regra do `alt` (Q5)**: a versão original usava só "o próximo `CAPTION` iniciado com 'Figure'". No PDF fixture isso acertou 5 de 7 figuras: o Docling agrupou as legendas "Figure 1" e "Figure 2" depois das duas figuras e pôs a "Figure 7" antes da sua figura. A regra passou a usar cada legenda uma vez só, com busca adiante, `caption_text` e busca para trás na mesma seção (7 de 7 no fixture).
- **Rótulos de figuras e tabelas**: além de `PICTURE` e `TABLE`, a exportação trata `CHART` e `DOCUMENT_INDEX`, que o Docling usa nos mesmos tipos de item; sem isso, essas figuras e tabelas sumiam do `.md` (achado do `/eng.pre-pr`).
- **Escape no `alt` e no link**: legendas com `[`, `]` ou `_` (comuns em artigos) quebravam a sintaxe da imagem (achado do `/eng.pre-pr`).
- **Reexportação**: em vez de só sobrescrever arquivos com o mesmo nome, a exportação remove os assets de uma exportação anterior, para não deixar arquivos órfãos (achado do `/eng.pre-pr`).
- **Dispositivo do Docling**: `DoclingParser` ganhou o parâmetro opcional `accelerator_device` (padrão: o comportamento atual), usado pelo teste de integração para rodar em CPU.
- **Python 3.12**: `requires-python` passou a `>=3.12,<3.13`, porque o `uv` escolhia o Python 3.15, onde o `torch` 2.8.0 não instala.
- **`make test`**: continua com falhas anteriores a esta tarefa (`tests/api/*` precisam de PostgreSQL e OpenSearch; 2 testes de `tests/unit/platform/graph/test_graph_builder.py` falham também na `main`). Os testes desta tarefa passam.
- **Índice de legendas antes da montagem**: a montagem continua numa única passada por `doc.iterate_items()`, mas a regra do `alt` precisa olhar em volta da figura; por isso a lista de itens é materializada e, antes da passada, um índice das posições das legendas "Figure" e dos inícios de seção é montado sobre ela. Não há segundo loop de figuras nem `export_to_markdown()` do documento.
- **Pipe escapado numa célula**: o `.md` mantém a forma escapada (barra invertida + barra vertical), exigida pelo GFM, e o CSV grava a barra vertical sem escape. O conteúdo da célula é o mesmo; só a grafia muda de formato para formato.
- **Texto do fallback de figura sem imagem**: passou de `(imagem indisponível)` para `(image unavailable)`, em inglês como o restante do `.md` exportado e do código (decisão do usuário na 2ª rodada do `/eng.pre-pr`).
- **Link e legendas ligadas** (2ª rodada do `/eng.pre-pr`): o destino do link passou a ser codificado para URL em vez de ir entre `<...>`, e as legendas ligadas à figura são comparadas uma a uma, porque o `caption_text` do Docling junta várias legendas sem separador.
