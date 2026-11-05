# Search in the arXiv Paper Curator: How OpenSearch Powers Retrieval

This article is a deep dive into the search layer of the project. It explains **what** kinds of search are supported, **how** each one is implemented against OpenSearch, and **why** the design choices were made. Hybrid RRF retrieval lives in `src/domain/rrf/`; OpenSearch transport and HTTP DTOs live in `src/domain/opensearch/`.

---

## 1. Overview

The project retrieves relevant excerpts ("chunks") of arXiv papers to feed a RAG pipeline. Retrieval is served entirely by a single OpenSearch index and supports three modes:

| Mode | Signal | When it's used |
|------|--------|----------------|
| **BM25** (keyword) | Lexical / term overlap | No embedding available, or `use_hybrid=False` |
| **Vector** (kNN) | Semantic similarity | Direct semantic lookups |
| **Hybrid** (BM25 + Vector) | Both, fused via RRF | Default for the search endpoint and the agent retriever |

The philosophy is deliberate and mirrors production retrieval systems: **keyword search first, vectors second, fusion on top**. Hybrid is the default because it combines exact term matching (great for names, acronyms, equations) with semantic recall (great for paraphrases and concepts).

The main entry point is `HybridRrfSearch.search_unified()` in `src/domain/rrf/search.py`, which picks BM25-only vs native RRF hybrid based on `query_embedding` and `use_hybrid`:

```132:160:src/domain/rrf/search.py
    def search_unified(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Unified search supporting BM25, vector, and RRF hybrid modes."""
        try:
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(...)
            return self._search_hybrid_native(...)
```

---

## 2. The single hybrid index

Everything is stored in **one** index rather than separate keyword and vector indices. The name is derived from settings as `{index_name}-{chunk_index_suffix}`, which resolves to `arxiv-papers-chunks` by default.

```29:29:src/domain/rrf/indexing.py
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"
```

### Index mapping

The mapping in `chunk_index_schema.py` defines both the text fields (for BM25) and a `knn_vector` field (for semantic search) on the same document, so a single query can score against both.

```66:94:src/domain/rrf/chunk_index_schema.py
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    ...
```

Key points:

- **`index.knn: true`** enables approximate nearest-neighbor search on the index.
- **`text_analyzer`** is a custom analyzer applied to `chunk_text`, `title`, and `abstract`. It lowercases, removes stopwords, and applies **snowball stemming** so that `learning` and `learned` match the same stem. This directly improves BM25 recall.
- **`standard_analyzer`** (with English stopwords) is used for `authors`, where stemming would be inappropriate.
- Text fields also expose a `.keyword` sub-field for exact matching/aggregations, while `categories`, `arxiv_id`, etc. are `keyword` types used for filtering.
- **`dynamic: "strict"`** at the root rejects unknown top-level fields, keeping the core chunk schema explicit.
- Document-type-specific fields (title, authors, categories, custom tags, etc.) live under **`metadata`**, an object with **`dynamic: true`**, so new keys can be indexed without mapping updates.
- Every chunk requires **`document_id`** (for arXiv papers this equals `arxiv_id`). Optional top-level **`arxiv_id`** remains for arXiv denormalization.

The mapping is built by `build_hybrid_chunk_mapping()` in `chunk_index_schema.py` (replacing the static `ARXIV_PAPERS_CHUNKS_MAPPING` body). **After upgrading**, recreate the index (`setup_indices(force=True)`) or re-run ingestion; old documents with paper fields at the top level will not match BM25 on `metadata.*` paths until reindexed.

The main search entry point is `HybridRrfSearch.search_unified()`.

### The vector field

```36:46:src/domain/rrf/chunk_index_schema.py
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
```

- **1024 dimensions** matches the `jina-embeddings-v3` model used throughout the project.
- **HNSW** (Hierarchical Navigable Small World) is the ANN algorithm. It trades a small amount of recall for large speedups versus brute-force kNN.
- **`cosinesimil`** (cosine similarity) is the distance metric — appropriate for normalized text embeddings.
- **`ef_construction: 512`** and **`m: 16`** tune the graph: higher values improve recall at the cost of slower indexing and more memory.

---

## 3. How documents get into the index

Search is only as good as what's indexed. The ingestion pipeline (Airflow DAG) chunks papers, embeds them, and bulk-indexes them:

1. Fetch recently processed papers from PostgreSQL.
2. Chunk each paper into overlapping segments (≈600 words, 100-word overlap).
3. Generate **passage** embeddings with Jina (`task="retrieval.passage"`).
4. Bulk-index chunks + embeddings into OpenSearch.

```39:47:src/domain/arxiv_ingestion/indexing.py
    """Index papers with chunking and vector embeddings for hybrid search.

    This task:
    1. Fetches recently processed papers from PostgreSQL
    2. Chunks them into overlapping segments (600 words, 100 overlap)
    3. Generates embeddings using Jina AI
    4. Indexes chunks with embeddings into OpenSearch
    """
```

On the client side, indexing goes through `index_chunk()` (single) or `bulk_index_chunks()` (batch, using the `opensearchpy.helpers.bulk` API). Both attach the embedding vector to the chunk document before writing, and use `refresh=True` so results are searchable immediately.

> **Important asymmetry:** documents are embedded with `task="retrieval.passage"` while queries use `task="retrieval.query"`. Jina v3 is an asymmetric model — using the matching task on each side is what makes query↔passage similarity meaningful.

---

## 4. BM25 (keyword) search

BM25 is the lexical baseline. It is used directly by `search_papers()` and as the keyword leg of hybrid search. The query itself is assembled by the `QueryBuilder`.

### Field boosting

`QueryBuilder` decides which fields to search and how much to weight them. For chunk search, the chunk text dominates:

```44:51:src/domain/rrf/query_builder.py
        # The ^ sets a boost multiplier for that field’s relevance score.
        if fields is None:
            if search_chunks:
                self.fields = ["chunk_text^3", "title^2", "abstract^1"]
            else:
                self.fields = ["title^3", "abstract^2", "authors^1"]
```

The `^N` suffix is a **boost multiplier** on that field's contribution to `_score`. A hit in `chunk_text` counts 3× more than a hit in `abstract`.

### The text query

```97:111:src/domain/rrf/query_builder.py
    def _build_text_query(self) -> Dict[str, Any]:
        """Build the main text search query.

        :returns: Multi-match query for text search
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }
```

- **`multi_match` + `best_fields`**: scores the document by its single best-matching field (good for "find the most relevant field" rather than summing across all fields).
- **`operator: "or"`**: any query term can match (maximizes recall).
- **`fuzziness: "AUTO"`**: tolerates typos via edit distance, scaled to term length.
- **`prefix_length: 2`**: the first two characters must match exactly, which keeps fuzzy matching fast and avoids nonsense matches.

### Filters, source, highlighting, sorting

The full query body is assembled in `build()`:

```53:71:src/domain/rrf/query_builder.py
    def build(self) -> Dict[str, Any]:
        """Build the complete OpenSearch query.

        :returns: Complete query dictionary ready for OpenSearch
        """
        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }
```

- **Filters** are placed in the `bool.filter` context (not `must`), so category filtering does **not** affect relevance scoring — it only includes/excludes:

```113:123:src/domain/rrf/query_builder.py
    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses for the query.

        :returns: List of filter clauses
        """
        filters = []

        if self.categories:
            filters.append({"terms": {"categories": self.categories}})

        return filters
```

- If the query string is empty, the builder falls back to `match_all` so filters/sorting still work.
- **`_source`** excludes the `embedding` vector from chunk results to keep responses lightweight.
- **Highlighting** wraps matched fragments in `<mark>` tags. For chunks it returns up to 2 fragments of `chunk_text` plus the full `title`.
- **Sorting**: by default results are sorted by relevance (`_score`). When `latest_papers=True`, results are sorted by `published_date` descending with `_score` as a tiebreaker:

```183:194:src/domain/rrf/query_builder.py
    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build sorting configuration.

        :returns: Sort configuration or None for relevance scoring
        """
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]

        if self.query.strip():
            return None

        return [{"published_date": {"order": "desc"}}, "_score"]
```

### Executing BM25

`_search_bm25_only()` builds the body via `QueryBuilder`, runs the search through `OpenSearchClient.search()`, and flattens metadata on each hit:

```166:186:src/domain/rrf/search.py
    def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int,
        categories: list[str] | None,
        latest: bool,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            ingestion_name=self.ingestion_name or None,
            latest_papers=latest,
            profile=self.profile,
        )
        search_body = builder.build()
        results = self._opensearch.search(self.index_name, search_body)
        logger.info("BM25 search for '%s...' returned %s results", query[:50], results["total"])
        return self._flatten_results(results)
```

---

## 5. Vector (kNN) search

Pure semantic search is handled by `HybridRrfSearch.vector_search()`. It runs a `knn` query against the `embedding` field and requests the top `k` nearest neighbors by cosine similarity. Category and ingestion filters wrap the kNN query in a `bool` with a `filter` clause when needed.

```111:130:src/domain/rrf/search.py
    def vector_search(
        self,
        query_embedding: list[float],
        size: int = 10,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Pure vector search on chunks."""
        filter_clause = self._build_search_filters(categories=categories)

        search_body: dict[str, Any] = {
            "size": size,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
            "_source": {"excludes": ["embedding"]},
        }

        if filter_clause:
            search_body["query"] = self._wrap_query_with_filters(search_body["query"], filter_clause)

        results = self._opensearch.search(self.index_name, search_body)
        return self._flatten_results(results)
```

The caller is responsible for producing `query_embedding` (via the Jina client). This mode is strong for paraphrased or conceptual queries where the exact keywords don't appear in the text.

---

## 6. Hybrid search (the default)

Hybrid search combines BM25 and vector results and fuses them into a single ranking. It is implemented natively in OpenSearch using the **`hybrid` query** plus an **RRF search pipeline**, rather than fusing scores in Python.

### Building the hybrid query

`_search_hybrid_native()` reuses the `QueryBuilder` to produce the BM25 leg, builds a kNN leg, and wraps both in a `hybrid` query. It over-fetches (`size * 2` on each leg) so fusion has a richer candidate pool to work with:

```188:231:src/domain/rrf/search.py
    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        categories: list[str] | None,
        min_score: float,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            ingestion_name=self.ingestion_name or None,
            latest_papers=False,
            profile=self.profile,
        )
        bm25_search_body = builder.build()
        ...
        results = self._opensearch.search(
            self.index_name,
            search_body,
            params={"search_pipeline": self._rrf_pipeline_id},
            min_score=min_score,
            total_from_filtered_count=True,
        )
        return self._flatten_results(results)
```

The two legs run independently, each producing its own ranked list. The magic is in the `search_pipeline` parameter, which tells OpenSearch to post-process and fuse those two lists.

### Reciprocal Rank Fusion (RRF)

The fusion strategy is defined as a search pipeline:

```6:20:src/domain/rrf/pipeline.py
HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}
```

RRF fuses ranked lists using only **rank position**, not raw scores. Each document's fused score is:

\[
\text{RRF}(d) = \sum_{\text{list } l} \frac{1}{k + \text{rank}_l(d)}
\]

with `k = rank_constant = 60`. A document ranked #1 in a list contributes `1/61`; ranked #2 contributes `1/62`; and so on. Documents that appear near the top of **both** the BM25 and vector lists accumulate the highest fused scores.

**Why RRF instead of weighted score averaging?**

- BM25 scores and cosine similarities live on completely different, unbounded scales. Naively combining them requires normalization and hand-tuned weights.
- RRF sidesteps that entirely by using rank, so it's robust with no tuning.
- The file keeps a commented-out weighted-average alternative (`HYBRID_SEARCH_PIPELINE`) for reference, but RRF is the default precisely because it "generally provides better results without manual weight tuning."

### Post-filtering

After fusion, `OpenSearchClient.search()` filters by `min_score` in `normalize_search_hits()` (called from `HybridRrfSearch._search_hybrid_native()` with `total_from_filtered_count=True`):

```137:151:src/domain/opensearch/client.py
        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue
            ...
        if total_from_filtered_count:
            results["total"] = len(results["hits"])
```

---

## 7. Index & pipeline lifecycle

Before any search can run, the index and the RRF pipeline must exist. `setup_indices()` creates both idempotently.

```39:45:src/domain/rrf/indexing.py
    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        mapping = build_hybrid_chunk_mapping(self.settings.opensearch)
        hybrid_index = self._opensearch.create_index(
            self.index_name,
            mapping,
            force=force,
```

Two production-minded details worth calling out:

- **Race-condition handling:** when multiple workers boot simultaneously they may all see "index doesn't exist" and race to create it. The code catches `resource_already_exists_exception` and treats it as success rather than crashing.
- **Pipeline creation via raw transport:** the RRF search pipeline is registered with a direct `PUT /_search/pipeline/{id}` call, because search pipelines aren't exposed through the high-level ingest-pipeline helpers.

The `OpenSearchClient` is created through a cached factory so the whole app shares one instance:

```11:23:src/domain/opensearch/factory.py
@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Uses lru_cache to maintain a singleton instance for efficiency.

    :param settings: Optional settings instance
    :returns: Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings)
```

(`make_opensearch_client_fresh()` exists for tests and Airflow tasks that need a non-cached instance.)

---

## 8. How search is consumed

### The `/hybrid-search` API endpoint

The FastAPI router ties everything together: health-check the cluster, generate a query embedding, call `search_unified()`, and map results into `SearchResponse`. Note the **graceful fallback** — if embedding generation fails, the request still succeeds as a BM25 search.

```23:43:src/api/hybrid_search.py
        query_embedding = None
        if request.use_hybrid:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                query_embedding = None

        logger.info(f"Hybrid search: '{request.query}' (hybrid: {request.use_hybrid and query_embedding is not None})")

        results = paper_rrf_search.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=request.use_hybrid,
            min_score=request.min_score,
        )
```

The request/response contracts are defined in `schemas.py` (`HybridSearchRequest`, `SearchResponse`, `SearchHit`), including validation like `size` bounds and the `from` alias for pagination.

### The agent retriever tool

The LangGraph agent does **not** call `search_unified()` directly. `create_retriever_tool()` in `src/agents/fusionsearch/tools.py` calls `SearchService.search_rrf()`, which embeds the query, runs `HybridRrfSearch.search_unified()` for hybrid recall, optionally reranks with Jina, then returns formatted context and confidence metadata:

```42:52:src/agents/fusionsearch/tools.py
        search_result = await rerank_search_service.search_rrf(query=query, top_k=top_k)
        rerank_applied = search_result.rerank_applied

        confident_documents = filter_confident_documents(
            search_result.after_rerank,
            min_document_score=graph_config.min_document_score,
            rerank_applied=rerank_applied,
        )
        retrieval_confidence = compute_retrieval_confidence(confident_documents, rerank_applied=rerank_applied)
        sources = build_source_items(confident_documents, rerank_applied=rerank_applied)
        formatted_context = format_documents_for_context(confident_documents)
```

The `/hybrid-search` HTTP route still calls `HybridRrfSearch.search_unified()` without the `SearchService` rerank layer. For parity with the agent, use `/api/v1/rerank-search` or call `SearchService.search_rrf()` in process.

---

## 9. Request flow at a glance

```
User query
   │
   ▼
embed_query()  ──(Jina v3, task=retrieval.query, 1024-d)
   │
   ▼
search_unified(query, query_embedding, use_hybrid=True)
   │
   ├─ no embedding / use_hybrid=False ─▶ _search_bm25_only ─▶ multi_match (BM25)
   │
   └─ hybrid ─▶ _search_hybrid_native
                    │
                    ├─ leg A: multi_match (BM25),  size*2
                    ├─ leg B: knn on embedding,    size*2
                    │
                    └─ hybrid query + RRF search pipeline (rank_constant=60)
                             │
                             ▼
                       fused ranking → min_score filter → top `size`
                             │
                             ▼
                    hits (chunk_text, metadata, score, highlights)
```

---

## 10. Design decisions summary

| Decision | Rationale |
|----------|-----------|
| Single hybrid index | One document carries both text and vector, so one query can score against both signals. |
| Snowball `text_analyzer` | Stemming boosts BM25 recall; stopwords cut noise. Authors use a plain analyzer to avoid bad stems. |
| HNSW + cosine, 1024-d | Fast ANN matched to Jina v3 embeddings; `ef_construction`/`m` tuned for recall vs. cost. |
| Passage vs. query embedding tasks | Jina v3 is asymmetric; matching tasks make similarity meaningful. |
| Hybrid as default | Combines exact-term precision with semantic recall. |
| RRF fusion (`k=60`) | Rank-based fusion avoids score-scale mismatch and manual weight tuning. |
| Over-fetch `size*2` per leg | Gives RRF a richer candidate pool before truncating to `size`. |
| Filters in `filter` context | Category filtering doesn't distort relevance scores. |
| Graceful BM25 fallback | Search stays available even if the embedding service is down. |
| Cached client factory | One shared, efficient client across the app. |

---

## 11. Where to look in the code

| Concern | File |
|---------|------|
| Hybrid recall orchestration | `src/domain/rrf/search.py` |
| Hybrid + rerank + optional `dual_search` | `src/domain/search/service.py` |
| BM25 query construction (boosts, fuzziness, filters, highlight, sort) | `src/domain/rrf/query_builder.py` |
| Index mapping, HNSW config | `src/domain/rrf/chunk_index_schema.py` |
| RRF search pipeline body | `src/domain/rrf/pipeline.py` |
| OpenSearch transport (search, bulk index, pipelines) | `src/domain/opensearch/client.py` |
| Request/response models | `src/domain/opensearch/schemas.py` |
| OpenSearch client factory (cached + fresh) | `src/domain/opensearch/factory.py` |
| RRF search factory (cached + fresh) | `src/domain/rrf/factory.py` |
| HTTP search endpoint | `src/api/hybrid_search.py` |
| Agent retriever tool | `src/agents/fusionsearch/tools.py` (via `SearchService.search_rrf`) |
| Ingestion / indexing DAG | `src/domain/arxiv_ingestion/indexing.py` |
| Query/passage embeddings | `src/domain/jinaai/jina_client.py` |
