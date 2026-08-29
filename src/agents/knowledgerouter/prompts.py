CLASSIFY_SYSTEM_PROMPT = """Analyze this query and determine which knowledge sources to consult.
For each relevant source, generate a targeted sub-question optimized for that source.
Example of sub-question: 
- How many transformer papers are in the database and what do they explain? 
    - how many papers about transformers are in the database? -> database
    - what do transformer papers explain? -> documents
- List all papers published in 2023 about reinforcement learning
    - list all papers published in 2023 about reinforcement learning -> database 
- Explain how diffusion models work based on the papers
    - explain how diffusion models work -> documents

Available sources:
- documents: Research paper content, concepts, explanations, methodologies, and findings from arXiv CS papers.
  Use for questions about what papers say, how techniques work, or conceptual understanding.
  Keyword signals (not exhaustive):
  - Information seeking: what is, what are, define, explain, describe
  - Policy/procedure: policy, procedure, guideline, rule, regulation
  - Documentation: guide, manual, handbook, documentation
  - How-to: how to, how do, how can, how should, why
  - References: according to, based on, mentioned in, document says
  - Understanding: summarize, overview, clarify, elaborate
  
- database: Structured metadata stored in PostgreSQL (paper counts, categories, dates, authors, titles).
  Use for questions about quantities, listings, filters, aggregations, or tabular paper metadata.
  Keyword signals (not exhaustive):
  - Aggregation: how many, count, total, sum, average, max, min
  - Listing: list all, show all, find all, display
  - Business data: revenue, sales, orders, customers, products, price
  - Time-based: last, recent, this month, this year, yesterday
  - Comparisons: more than, less than, top, bottom, rank, best
  - Grouping: by segment, by category, group by, per, each

Hybrid signals (route to BOTH documents and database):
  - and explain, and describe, and tell me, also explain, show data and explain, compare and explain, analyze and describe

Return ONLY the sources that are relevant to the query. Each source should have a targeted
sub-question optimized for that specific knowledge domain.

Examples:
- "How many papers about transformers are in the database?" -> database only
- "What is the attention mechanism in transformers?" -> documents only
- "How many BERT papers exist and what do they propose?" -> both database and documents
- "List papers published in 2023 about reinforcement learning" -> database only
- "Explain how diffusion models work based on the papers" -> documents only
- "Show the top 10 cited papers and explain their main contributions" -> both database and documents"""

SYNTHESIZE_SYSTEM_PROMPT = """Synthesize these search results to answer the original question: "{query}"

- Combine information from multiple sources without redundancy
- Highlight the most relevant and actionable information
- Note any discrepancies between sources
- Keep the response concise and well-organized
- If only one source was consulted, present its answer clearly without unnecessary preamble"""
