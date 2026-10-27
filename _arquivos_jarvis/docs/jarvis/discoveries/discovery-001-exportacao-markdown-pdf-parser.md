---
# Modelo de IA recomendado para trabalhar neste documento. Se a sessão estiver em outro,
# o comando avisa e segue (você decide se troca). Opções: opus | sonnet | haiku
recommended_model: opus

# Identificador único, em sequência (ex.: discovery-001). Vai no nome do arquivo e nos campos related_* de outros documentos.
id: discovery-001

# Nome curto do documento. Vai no título e, em kebab-case, no nome do arquivo.
name: Exportação de Markdown no pdf_parser

# Situação do rascunho. O discovery tem ciclo próprio (rules/product/prod-rules.md).
# Opções: in_review (rascunho em andamento ou concluído) | cancelled (o veredito foi do_not_proceed)
status: in_review

# Recomendação do discovery: vale seguir com essa ideia? Fica vazio durante a entrevista.
# Opções: proceed (vale especificar) | investigate_more (há lacuna crítica) | do_not_proceed (não seguir; status passa a cancelled)
verdict: proceed

# Tamanho da ideia; define quantas perguntas a entrevista faz.
# Opções: small (trabalho já claro, até 2) | bounded (funcionalidade num produto existente, até 5) | broad (várias entregas, até 8)
size: small

# Datas de criação e da última alteração (YYYY-MM-DD).
created_at: 2026-10-26
updated_at: 2026-10-26

# Quem criou o documento e quem o editou por último (USER do ENV.md).
created_by: phelipperibeiro@hotmail.com
last_editor: phelipperibeiro@hotmail.com
---

# discovery-001: Exportação de Markdown no pdf_parser

> **Rascunho.** Registra o que se sabe, o que é suposição e o que ainda é lacuna sobre uma ideia. Não define tarefas, histórias, critérios de aceitação nem estimativas. Um PRD, FRD, épico ou issue criado a partir dele é a fonte da verdade. Nunca registre credenciais, tokens nem dados pessoais de clientes.

## Entendimento

O parser de PDF do `production-rag` (`src/domain/pdf_parser/`) passa a gravar, de forma opcional, uma versão Markdown de cada artigo, com uma pasta de assets ao lado (figuras em PNG, tabelas em CSV). A exportação é ativada por um parâmetro novo, `markdown_output_dir`, e não muda nada quando ele não é passado. É para **revisão humana e depuração**: ver o que o Docling extraiu de cada artigo.

| Dito por você                                                                                                                                                                                  | Suposição (origem)                                                                                                                                                                                                                                                                                                      | Lacuna (quem responde)                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `markdown_output_dir: Path \| None = None` em `DoclingParser.parse_pdf` e `PDFParserService.parse_pdf`; a exportação roda **depois** de `DocumentConverter` + `extract_sections_from_document` | A API do Docling cobre a tarefa: `generate_picture_images`, `PictureItem.get_image(doc)`, `TableItem.export_to_dataframe(doc)` e `export_to_markdown` existem (origem: documentação do Docling via Context7; projeto trava `docling` 2.52.0 no `uv.lock`)                                                               | Risco técnico: `iterate_items()` usa `traverse_pictures=False` por padrão; se a legenda for filha da figura, a regra da "próxima legenda" pode nunca casar e todo `alt` vira `Figure NNN`. Responde: um spike de engenharia com o PDF fixture |
| Saída: `{pdf_stem}.md` + `{pdf_stem}_assets/`                                                                                                                                                  | O "task repo" é o fork `phelipperibeiro/production-rag` (`origin`); o upstream é `apssouza22/production-rag` (origem: `git remote -v`)                                                                                                                                                                                  | Licença do PDF `2609.17429v1`: nem todo artigo do arXiv permite redistribuição. Responde: dono do repositório, conferindo a licença no arXiv                                                                                                  |
| Montagem única guiada por `doc.iterate_items()`, sem misturar `export_to_markdown()` com um segundo loop de figuras                                                                            | Ingestão atual não muda: `parse_pdf(pdf_path)` continua sem o parâmetro, porque o uso é depuração sob demanda (origem: `src/domain/arxiv_ingestion/service.py:441` + resposta da pergunta 1)                                                                                                                            | O fixture precisa ter até 30 páginas (`PDF_PARSER__MAX_PAGES`, padrão em `src/config.py`), senão o parser o pula. Responde: conferir o PDF                                                                                                    |
| `figure_NNN.png` (PICTURE) e `table_NNN.csv` (TABLE), contadores separados a partir de 001, na ordem de `iterate_items()`                                                                      | Sem o parâmetro, `markdown_path` e `artifacts_dir` ficam `None` (origem: suposição, padrão de campo opcional em Pydantic)                                                                                                                                                                                               |                                                                                                                                                                                                                                               |
| Tabelas inline em GFM pipe tables, sem link para o CSV; CSV UTF-8, vírgula, `QUOTE_MINIMAL`; header no CSV só quando o grid inline tem separator row                                           | Resolução das figuras no padrão do Docling (`images_scale` 1.0, cerca de 72 DPI), já que a tarefa não define escala (origem: documentação do Docling)                                                                                                                                                                   |                                                                                                                                                                                                                                               |
| Figuras: `![alt]({pdf_stem}_assets/figure_NNN.png)`; `alt` = próximo `CAPTION` iniciado com "Figure", senão `Figure NNN`                                                                       | **Suposição explícita** (padrão razoável, sem pergunta): os demais itens seguem o Markdown comum: `TITLE` e `SECTION_HEADER` viram `#` conforme o nível, `LIST_ITEM` vira `-`, `CODE` vai em bloco cercado, os demais textos (incluindo `CAPTION` e `FORMULA`) viram parágrafo (origem: suposição; a tarefa não define) |                                                                                                                                                                                                                                               |
| `PdfContent` ganha `markdown_path` e `artifacts_dir`                                                                                                                                           |                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                               |
| O task repo commita `tests/fixtures/arxiv_pdfs/2609.17429v1.pdf` e os testes usam esse PDF                                                                                                     |                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                               |
| Uso: revisão humana e depuração do parsing (pergunta 1)                                                                                                                                        |                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                               |
| `generate_picture_images=True` fica **sempre** ligado, como na tarefa (pergunta 2)                                                                                                             | Custo extra na ingestão (renderizar imagens de figuras em todo PDF) é aceitável; a medir (origem: suposição, sem medição)                                                                                                                                                                                               |                                                                                                                                                                                                                                               |

## Problema

- **Dor:** hoje o parser guarda só texto e seções (`raw_text` e `sections` em `PdfContent`); figuras e tabelas são descartadas (`figures=[]`, `tables=[]` em `src/domain/pdf_parser/docling.py`). Não existe uma versão legível do artigo com figuras e tabelas preservadas.
- **Quem sofre:** quem desenvolve e depura o `production-rag`, ao avaliar a qualidade do parsing de um artigo (pergunta 1).
- **Evidência:** sem evidência além do código atual (nenhum relato de bug de parsing registrado).
- **Como se resolve hoje:** inspecionar `raw_text`/`sections` no banco ou rodar o Docling à mão (por exemplo, num notebook: `jupyter` está nas dependências `dev`); figuras e tabelas não aparecem em nenhum desses caminhos.
- **Por que agora:** sem motivo registrado além da tarefa.
- **Sinal de sucesso:** abrir o `{pdf_stem}.md` de um artigo e ver texto, tabelas e figuras na ordem do PDF, com os assets numerados e os CSVs batendo com as tabelas inline.

## Abordagens

| Abordagem                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Prós                                                                                                             | Contras                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. A menor coisa que funciona: implementar a tarefa como descrita.** Função de exportação no domínio `pdf_parser`, chamada no fim de `parse_pdf` só quando `markdown_output_dir` vier; uma passada por `iterate_items()` gera `.md`, PNGs e CSVs. Testes unitários com documentos falsos (`SimpleNamespace`, como em `tests/unit/domain/pdf_parser/test_docling_sections.py`) para numeração, `alt` e header do CSV, e um teste de integração com o PDF real | Fiel à tarefa; isolado no domínio `pdf_parser`; não afeta a ingestão; regras finas testáveis sem rodar o Docling | Reimplementa a renderização dos textos (sem `export_to_markdown()`); teste com o PDF real é lento (carrega os modelos do Docling); imagens geradas sempre |
| **B. Exportador nativo:** `doc.save_as_markdown(image_mode=REFERENCED)` + loop separado para os CSVs                                                                                                                                                                                                                                                                                                                                                           | Pouco código; renderização de texto pronta                                                                       | Viola o requisito 1 (proíbe misturar `export_to_markdown()` com um segundo loop); sem controle de nomes (`figure_NNN`), `alt` e regra de header           |
| **C. Não fazer / usar o que existe:** depurar no notebook com `export_to_markdown()` quando precisar                                                                                                                                                                                                                                                                                                                                                           | Zero código e zero manutenção                                                                                    | Sem assets nem CSVs; não reproduzível nem testado; não atende a tarefa                                                                                    |

**Recomendação:** A, confirmada pelo usuário: é a única que cumpre os requisitos da tarefa, e os testes com documento falso isolam as regras mais sujeitas a erro. · **Riscos e dependências:** (1) legendas: confirmar num spike com o PDF fixture se o `CAPTION` vem como item seguinte da figura em `iterate_items()` ou como filho dela (nesse caso, `traverse_pictures=True` ou `picture.captions`); confirma: engenharia, no `/eng.start`. (2) licença e número de páginas (≤ 30) do `2609.17429v1`; confirma: dono do repositório. (3) custo de `generate_picture_images=True` na ingestão; confirma: medição antes e depois.

## Scorecard

| Lente                                            | Nota     | Justificativa                                                                                                                                        |
| ------------------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Valor:** alguém vai querer e usar?             | adequado | Uso interno de depuração do parsing (pergunta 1); útil, mas sem evidência medida de dor                                                              |
| **Usabilidade:** as pessoas conseguem usar?      | adequado | Basta passar `markdown_output_dir`; não há CLI nem endpoint, então quem depura chama pelo código                                                     |
| **Técnica:** dá para construir com o que temos?  | adequado | API do Docling confirmada na documentação atual; o comportamento das legendas no `iterate_items()` ainda não foi validado (há fallback `Figure NNN`) |
| **Negócio:** custo, prazo e risco fazem sentido? | adequado | Escopo pequeno e isolado; custo de gerar imagens sempre não medido e licença do fixture pendente                                                     |

## Veredito

**Veredito:** proceed

**Motivo:** a única lacuna crítica (quem usa o Markdown) foi respondida: revisão humana e depuração. Nenhuma lente ficou `desconhecido`. As pendências (legendas, licença e páginas do fixture, custo das imagens) não mudam a decisão e se resolvem com um spike ou uma conferência rápida.

## Próximo passo sugerido

- **Comando** (só uma sugestão, nunca executado sozinho): `/prod.spec.issue discovery-001`
- **Motivo:** `size: small` e requisitos já detalhados; uma Task (tarefa técnica) com critérios de aceitação testáveis basta, sem PRD nem FRD.
- **Alternativa:** ir direto para `/eng.start {TASK_MANAGER_KEY}` usando a descrição da tarefa como especificação, já que ela tem o nível de detalhe de uma issue; ou rodar antes `/eng.light-arch` só para o spike das legendas.

## Histórico

| Data       | Perguntas  | O que ficou pendente                                                                                                       |
| ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-26 | 2 de até 2 | Legendas no `iterate_items()` (spike); licença e páginas do `2609.17429v1`; custo de `generate_picture_images` na ingestão |
