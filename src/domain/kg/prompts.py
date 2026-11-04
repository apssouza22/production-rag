ENTITY_RELATION_EXTRACTION_PROMPT = """Extract entities and relationships from the text chunk below.

Rules:
- Entities: concrete nouns, methods, models, datasets, metrics, or named concepts.
- Relations: directed links between entities mentioned in this chunk.
- Keep descriptions short (1-2 sentences).
- Do not invent entities not supported by the chunk.
- Limit to at most {max_entities} entities and {max_relations} relations.

Section: {section_title}

Chunk:
{chunk_text}
"""

KEYWORD_EXTRACTION_PROMPT = """Identify high-level and low-level keywords for retrieval.

High-level: themes, topics, abstract concepts.
Low-level: specific entities, methods, names, numbers, datasets.

Query:
{query}
"""
