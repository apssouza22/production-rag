"""OpenSearch index mappings for KG entity and relation vector stores.

These indices hold only KG embeddings; chunk vectors remain in the hybrid chunk index.
"""


def build_kg_vector_mapping(vector_dimension: int) -> dict:
    knn_field = {
        "type": "knn_vector",
        "dimension": vector_dimension,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "nmslib",
            "parameters": {"ef_construction": 512, "m": 16},
        },
    }
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.knn": True,
            "index.knn.space_type": "cosinesimil",
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "ingestion_name": {"type": "keyword"},
                "embedding": knn_field,
            },
        },
    }


KG_ENTITY_MAPPING = {
    **build_kg_vector_mapping(1024),
}
KG_ENTITY_MAPPING["mappings"]["properties"].update(
    {
        "entity_id": {"type": "keyword"},
        "entity_name": {"type": "text"},
        "entity_type": {"type": "keyword"},
        "description": {"type": "text"},
    }
)

KG_RELATION_MAPPING = {
    **build_kg_vector_mapping(1024),
}
KG_RELATION_MAPPING["mappings"]["properties"].update(
    {
        "relation_id": {"type": "keyword"},
        "source_entity_id": {"type": "keyword"},
        "target_entity_id": {"type": "keyword"},
        "description": {"type": "text"},
        "keywords": {"type": "text"},
        "weight": {"type": "float"},
    }
)
