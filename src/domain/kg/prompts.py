"""LLM prompts for knowledge-graph extraction and PathRAG-style dual keywords.

Structured output is enforced via Pydantic schemas in ``schemas.py`` (not tuple-delimited
PathRAG records). Prompt structure and few-shot patterns follow PathRAG's ``prompt.py``.
"""

DEFAULT_ENTITY_TYPES = (
    "METHOD",
    "MODEL",
    "DATASET",
    "METRIC",
    "TASK",
    "CONCEPT",
    "ORGANIZATION",
    "PERSON",
    "PLACE",
    "EVENT",
)

_ENTITY_EXTRACTION_EXAMPLES = """
######################
-Examples-
######################

Example 1

Entity types: METHOD, MODEL, METRIC, TASK, CONCEPT
Section: Experiments
Chunk:
We fine-tune BERT-base on the GLUE benchmark and report F1 on CoNLL-2003 NER. Our approach
outperforms the BiLSTM-CRF baseline by 2.1 points while using half the parameters.

Good extraction (illustrative):
- Entities: BERT-base (MODEL), GLUE (DATASET), F1 (METRIC), CoNLL-2003 NER (TASK/DATASET),
  BiLSTM-CRF baseline (METHOD), parameter count (METRIC/CONCEPT as appropriate).
- Relations: BERT-base → GLUE (fine-tuned on); BERT-base → BiLSTM-CRF (outperforms);
  each relation has a short factual description, comma-separated relation keywords
  (e.g. "fine-tuning, benchmark"), and weight 6–9 when clearly stated in the chunk.

Example 2

Entity types: METHOD, DATASET, ORGANIZATION, CONCEPT
Section: Related work
Chunk:
Graph neural networks propagate information along edges; PathRAG retrieves multi-hop paths
over a knowledge graph. We build on LightRAG-style dual-level retrieval.

Good extraction:
- Do not invent entities or sources not named in the chunk.
- Link METHOD/MODEL entities only when the text clearly states a dependency or comparison.
- Prefer precise names (PathRAG, LightRAG) over vague "the method".

#############################
"""

_ENTITY_TYPES_LINE = ", ".join(DEFAULT_ENTITY_TYPES)

ENTITY_RELATION_EXTRACTION_PROMPT = (
    """-Goal-
Given a text chunk from a source document, identify grounded entities and directed relationships
among them. Output must conform to the tool/response schema (entities and relations arrays).

Use English for names and descriptions unless the chunk is clearly in another language
(use the same language as the chunk for entity names in that case).

-Entity types-
Use one short type label per entity. Prefer: """
    + _ENTITY_TYPES_LINE
    + """. Use UNKNOWN only when no label fits.

-Steps-
1. Identify entities explicitly supported by the chunk. For each entity provide:
   - entity_name: canonical name (capitalize proper nouns in English)
   - entity_type: one of the types above
   - description: 1–2 sentences of factual detail from the chunk only

2. Among those entities, find pairs (source_entity, target_entity) that are *clearly related*
   in this chunk. For each relation provide:
   - source_entity and target_entity: names matching step 1
   - description: why they are related, grounded in the chunk (1–2 sentences)
   - keywords: one short phrase or comma-separated themes summarizing the relation
     (concepts, not full sentences)—similar to PathRAG relationship_keywords
   - weight: 0.0–10.0 strength implied by the text (default 1.0 if weak; up to 10 if central)

3. Do not add entities or edges that require outside knowledge. Limit output to at most
   {max_entities} entities and {max_relations} relations (keep the strongest/most explicit first).

-Rules-
- Concrete nouns, methods, models, datasets, metrics, tasks, organizations, and named concepts.
- Both endpoints of every relation must appear as entity names you extract.
- No speculation beyond the chunk text.

"""
    + _ENTITY_EXTRACTION_EXAMPLES
    + """
#############################
-Real Data-
#############################
Section: {section_title}

Chunk:
{chunk_text}
#############################
Extract entities and relations into the structured response.
"""
)

_KEYWORD_EXTRACTION_EXAMPLES = """
######################
-Examples-
######################

Example 1:

Query: "How does international trade influence global economic stability?"
Good high_level_keywords: International trade, Global economic stability, Economic impact
Good low_level_keywords: Trade agreements, Tariffs, Currency exchange, Imports, Exports

Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
Good high_level_keywords: Environmental consequences, Deforestation, Biodiversity loss
Good low_level_keywords: Species extinction, Habitat destruction, Carbon emissions, Rainforest, Ecosystem

Example 3:

Query: "What is the role of education in reducing poverty?"
Good high_level_keywords: Education, Poverty reduction, Socioeconomic development
Good low_level_keywords: School access, Literacy rates, Job training, Income inequality

Example 4 (CS paper style):

Query: "Which retrieval method does PathRAG use for multi-hop reasoning over a knowledge graph?"
Good high_level_keywords: Multi-hop reasoning, Knowledge graph retrieval, Graph-augmented RAG
Good low_level_keywords: PathRAG, knowledge graph, entity paths, dual-level retrieval

#############################
"""

KEYWORD_EXTRACTION_PROMPT = (
    """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in
the user's query for graph-augmented retrieval (entity/relation search vs chunk search).

---Goal---

Given the query, populate high_level_keywords and low_level_keywords in the structured response:
- High-level keywords: overarching concepts, themes, tasks, or problem settings (global/graph view).
- Low-level keywords: specific entities, methods, model names, datasets, metrics, numbers, or proper nouns (local/entity view).

---Instructions---

- Keep the same language as the query.
- Prefer terms that would match corpus content (methods, benchmarks, architectures, products).
- Include 2–8 keywords per list when the query supports it; omit noise and duplicate ideas.
- Do not answer the question—only extract keywords.

"""
    + _KEYWORD_EXTRACTION_EXAMPLES
    + """
#############################
-Real Data-
#############################
Query: {query}
#############################
Output keyword lists in the structured response.
"""
)
