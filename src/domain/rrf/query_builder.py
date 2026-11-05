import logging
from typing import Any, Dict, List, Optional

from .chunk_index_schema import DEFAULT_CHUNK_PROFILE, ChunkSearchProfile

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Build OpenSearch queries for hybrid chunk search (BM25, filters, highlighting)."""

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        ingestion_name: Optional[str] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        profile: ChunkSearchProfile | None = None,
    ):
        """Initialize query builder.

        :param query: Search query text
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param fields: BM25 fields (defaults to profile.bm25_fields)
        :param categories: Filter by categories
        :param ingestion_name: Filter by ingestion partition (exact match)
        :param track_total_hits: Whether to track total hits accurately
        :param latest_papers: Sort by profile.sort_field (newest first) instead of relevance
        :param profile: BM25/filter/highlight profile for chunk search
        """
        self.query = query
        self.size = size
        self.from_ = from_
        self.categories = categories
        self.ingestion_name = ingestion_name
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.profile = profile or DEFAULT_CHUNK_PROFILE
        self.fields = list(fields) if fields is not None else list(self.profile.bm25_fields)

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

        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:
        """Build the main query with filters.

        :returns: Query dictionary with bool structure
        """
        must_clauses = []

        if self.query.strip():
            must_clauses.append(self._build_text_query())

        filter_clauses = self._build_filters()

        bool_query = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

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

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses for the query.

        :returns: List of filter clauses
        """
        filters = []

        if self.categories:
            field = self.profile.term_filters.get("categories", "categories")
            filters.append({"terms": {field: self.categories}})

        if self.ingestion_name:
            filters.append({"term": {"ingestion_name": self.ingestion_name}})

        return filters

    def _build_source_fields(self) -> dict[str, list[str]]:
        """Define which fields to return in results."""
        return {"excludes": ["embedding"]}

    def _build_highlight(self) -> Dict[str, Any]:
        """Build highlighting configuration.

        :returns: Highlight configuration dictionary
        """
        return {
            "fields": dict(self.profile.highlight_fields),
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build sorting configuration.

        :returns: Sort configuration or None for relevance scoring
        """
        if self.latest_papers:
            return [{self.profile.sort_field: {"order": "desc"}}, "_score"]

        if self.query.strip():
            return None

        return [{self.profile.sort_field: {"order": "desc"}}, "_score"]
