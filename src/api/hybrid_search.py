import logging

from fastapi import APIRouter, HTTPException

from src.dependencies import CacheDep, EmbeddingsDep, OpenSearchDep
from src.domain.opensearch.schemas import HybridSearchRequest, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-search", tags=["hybrid-search"])


def _build_search_response(request: HybridSearchRequest, results: dict, query_embedding) -> SearchResponse:
    hits = []
    for hit in results.get("hits", []):
        hits.append(
            SearchHit(
                arxiv_id=hit.get("arxiv_id", ""),
                title=hit.get("title", ""),
                authors=hit.get("authors"),
                abstract=hit.get("abstract"),
                published_date=hit.get("published_date"),
                pdf_url=hit.get("pdf_url"),
                score=hit.get("score", 0.0),
                highlights=hit.get("highlights"),
                chunk_text=hit.get("chunk_text"),
                chunk_id=hit.get("chunk_id"),
                section_name=hit.get("section_name"),
            )
        )

    return SearchResponse(
        query=request.query,
        total=results.get("total", 0),
        hits=hits,
        size=request.size,
        **{"from": request.from_},
        search_mode="hybrid" if (request.use_hybrid and query_embedding) else "bm25",
    )


@router.post("/", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    cache_client: CacheDep,
) -> SearchResponse:
    """Hybrid search endpoint supporting multiple search modes."""
    try:
        if not opensearch_client.health_check():
            raise HTTPException(status_code=503, detail="Search service is currently unavailable")

        query_embedding = None
        if cache_client:
            try:
                cache_result = await cache_client.lookup(request)
                if cache_result.response:
                    hit_label = "exact" if cache_result.hit_type == "exact" else "confidence-based"
                    logger.info("Returning %s cached hybrid search response", hit_label)
                    return cache_result.response
                query_embedding = cache_result.query_embedding
            except Exception as e:
                logger.warning("Cache check failed, proceeding with normal search: %s", e)

        if request.use_hybrid and query_embedding is None:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning("Failed to generate embeddings, falling back to BM25: %s", e)
                query_embedding = None

        logger.info("Hybrid search: '%s' (hybrid: %s)", request.query, request.use_hybrid and query_embedding is not None)

        results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.size,
            from_=request.from_,
            categories=request.categories,
            latest=request.latest_papers,
            use_hybrid=request.use_hybrid,
            min_score=request.min_score,
        )

        search_response = _build_search_response(request, results, query_embedding)
        logger.info("Search completed: %s results returned", search_response.total)

        if cache_client:
            try:
                await cache_client.store(request, search_response, query_embedding=query_embedding)
            except Exception as e:
                logger.warning("Failed to store hybrid search response in cache: %s", e)

        return search_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Hybrid search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
