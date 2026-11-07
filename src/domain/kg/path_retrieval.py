"""PathRAG-style multi-hop path discovery and narration for the knowledge graph.

This module implements query-time path retrieval inspired by
`BUPT-GAMMA/PathRAG` (see ``operate._find_most_related_edges_from_entities3``).
It does not index paths separately: it walks a bounded subgraph already loaded
from ``kg_nodes`` / ``kg_edges`` and turns selected routes into text the LLM can
use alongside dual-level entity and relation hits.

Role in the stack
-----------------
``DualLevelKGRetrievalService`` (``dual_level_retrieval.py``) obtains seed
``entity_id`` values from low-level keyword / entity kNN search, loads incident
edges via ``KgGraphRepository.get_incident_edges_for_entity_ids``, then calls
``PathDiscover.discover()``. Results are ``KgPathHit`` rows (entity chain, hop
count, narrative) passed into ``format_kg_context()`` as the
``--- Relations (KG paths) ---`` block when ``KG__PATH_RETRIEVAL_ENABLED`` is on.

``make_path_discover()`` in ``factory.py`` wires knobs from ``KGSettings``.
When ``enabled=False`` (default), ``discover()`` returns ``[]`` so dual-level
behavior stays backward compatible.

End-to-end pipeline (``PathDiscover.discover``)
-----------------------------------------------
1. **Topology** — ``_build_topology`` builds an undirected adjacency list
   (``_UndirectedGraph``) and a bidirectional ``(source, target) -> KgEdge`` map.
2. **Enumeration** — ``_enumerate_paths`` runs DFS from each ordered pair of
   distinct seeds, collecting simple paths up to ``max_hops`` (default 3). Paths
   are grouped by seed pair and by hop bucket (1-, 2-, 3-hop lists).
3. **Selection** — ``_select_paths`` prefers paths ranked by PathRAG edge-flow
   weighting (``_score_paths_for_pair``: threshold + alpha propagation along
   follow sets). If that yields nothing, a hop-bucket fallback takes slices of
   enumerated paths (reference behavior). Output is capped at ``max_paths``.
4. **Narration** — ``_to_path_hits`` / ``_render_narrative`` expand each path into
   prose: entity name/type/description at endpoints and ``through edge (keywords)``
   between hops, using merged node text from the caller's ``node_map``.

Early exits (empty list, no error): disabled discoverer, fewer than two seeds,
no edges, or seeds not present in the built graph.

Configurable parameters (constructor / ``KGSettings``)
------------------------------------------------------
- ``enabled`` — master switch for path retrieval.
- ``max_hops`` — maximum edges on a path (DFS depth cap).
- ``max_paths`` — maximum paths returned after ranking / fallback.
- ``weight_threshold`` — drop weak edge weights during flow scoring.
- ``weight_alpha`` — decay factor when propagating weight to the next hop.

Public API
----------
- ``PathDiscover`` — stateless discoverer; inject in tests or via factory.
- ``discover(seed_entity_ids, node_map, edges) -> list[KgPathHit]`` — main entry.

Private helpers (``_UndirectedGraph``, ``_PathEnumeration``) keep graph search and
DFS bookkeeping local to this file.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.domain.kg.models import KgEdge, KgNode
from src.domain.kg.schemas import KgPathHit


@dataclass
class _UndirectedGraph:
    """In-memory undirected adjacency for bounded subgraph path search."""

    _adj: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, source: str, target: str) -> None:
        self._adj[source].add(target)
        self._adj[target].add(source)

    def neighbors(self, node: str) -> set[str]:
        return self._adj.get(node, set())

    @property
    def nodes(self) -> set[str]:
        return set(self._adj.keys())


@dataclass(frozen=True)
class _PathEnumeration:
    """DFS output grouped by seed pair and hop bucket."""

    pair_paths: dict[tuple[str, str], dict[str, list]]
    one_hop_paths: list[list[str]]
    two_hop_paths: list[list[str]]
    three_hop_paths: list[list[str]]


def cap_subgraph_edges_by_nodes(
    edges: list[KgEdge],
    seed_entity_ids: list[str],
    max_nodes: int,
) -> list[KgEdge]:
    """Keep highest-weight incident edges without growing the node set beyond ``max_nodes``.

    ``edges`` should already be ordered by weight (desc). Seeds are always retained in the
    allowed node set even when there are more seeds than ``max_nodes`` (no edges are dropped
    for that case).
    """
    if max_nodes <= 0 or not edges:
        return []
    allowed = set(seed_entity_ids)
    kept: list[KgEdge] = []
    for edge in edges:
        endpoints = {edge.source_entity_id, edge.target_entity_id}
        new_nodes = endpoints - allowed
        if new_nodes and len(allowed) + len(new_nodes) > max_nodes:
            continue
        allowed |= endpoints
        kept.append(edge)
    return kept


class PathDiscover:
    """Finds and narrates ranked multi-hop paths between entity seeds on a subgraph."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_hops: int = 3,
        max_paths: int = 15,
        weight_threshold: float = 0.3,
        weight_alpha: float = 0.8,
    ) -> None:
        self._enabled = enabled
        self._max_hops = max_hops
        self._max_paths = max_paths
        self._weight_threshold = weight_threshold
        self._weight_alpha = weight_alpha

    def discover(
        self,
        seed_entity_ids: list[str],
        node_map: dict[str, KgNode],
        edges: list[KgEdge],
    ) -> list[KgPathHit]:
        """Return ranked path narratives for PathRAG-style relation context."""
        if not self._enabled or len(seed_entity_ids) < 2 or not edges:
            return []

        graph, edge_map = self._build_topology(edges)
        seeds = [entity_id for entity_id in seed_entity_ids if entity_id in graph.nodes]
        if len(seeds) < 2:
            return []

        enumeration = self._enumerate_paths(graph, seeds)
        selected = self._select_paths(seeds, enumeration)
        return self._to_path_hits(selected, node_map, edge_map)

    def _build_topology(self, edges: list[KgEdge]) -> tuple[_UndirectedGraph, dict[tuple[str, str], KgEdge]]:
        graph = _UndirectedGraph()
        edge_map: dict[tuple[str, str], KgEdge] = {}
        for edge in edges:
            graph.add_edge(edge.source_entity_id, edge.target_entity_id)
            edge_map[(edge.source_entity_id, edge.target_entity_id)] = edge
            edge_map[(edge.target_entity_id, edge.source_entity_id)] = edge
        return graph, edge_map

    def _enumerate_paths(self, graph: _UndirectedGraph, seeds: list[str]) -> _PathEnumeration:
        pair_paths: dict[tuple[str, str], dict] = defaultdict(lambda: {"paths": [], "edges": set()})
        one_hop: list[list[str]] = []
        two_hop: list[list[str]] = []
        three_hop: list[list[str]] = []

        def dfs(current: str, target: str, path: list[str], depth: int) -> None:
            if depth > self._max_hops:
                return
            if current == target:
                pair_paths[(path[0], target)]["paths"].append(list(path))
                for u, v in zip(path[:-1], path[1:], strict=False):
                    pair_paths[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
                if depth == 1:
                    one_hop.append(list(path))
                elif depth == 2:
                    two_hop.append(list(path))
                elif depth == 3:
                    three_hop.append(list(path))
                return
            for neighbor in graph.neighbors(current):
                if neighbor not in path:
                    dfs(neighbor, target, path + [neighbor], depth + 1)

        for node1 in seeds:
            for node2 in seeds:
                if node1 != node2:
                    dfs(node1, node2, [node1], 0)

        for key in pair_paths:
            pair_paths[key]["edges"] = list(pair_paths[key]["edges"])

        return _PathEnumeration(
            pair_paths=dict(pair_paths),
            one_hop_paths=one_hop,
            two_hop_paths=two_hop,
            three_hop_paths=three_hop,
        )

    def _select_paths(self, seeds: list[str], enumeration: _PathEnumeration) -> list[list[str]]:
        weighted = self._rank_weighted_paths(seeds, enumeration.pair_paths)
        hop_fallback = self._hop_bucket_paths(enumeration)
        total = min(self._max_paths, max(len(hop_fallback), len(weighted)))
        if not total:
            return []
        if weighted:
            return weighted[:total] if len(weighted) > total else weighted
        return hop_fallback[:total]

    def _rank_weighted_paths(
        self,
        seeds: list[str],
        pair_paths: dict[tuple[str, str], dict],
    ) -> list[list[str]]:
        all_weighted: list[tuple[list[str], float]] = []
        for node1 in seeds:
            for node2 in seeds:
                if node1 == node2:
                    continue
                key = (node1, node2)
                paths = pair_paths.get(key, {}).get("paths") or []
                if not paths:
                    continue
                all_weighted.extend(
                    self._score_paths_for_pair(paths, node1, node2),
                )

        all_weighted.sort(key=lambda item: item[1], reverse=True)
        seen: set[tuple[str, ...]] = set()
        ranked: list[list[str]] = []
        for path, _weight in all_weighted:
            if len(path) < 2:
                continue
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(path)
        return ranked

    def _hop_bucket_paths(self, enumeration: _PathEnumeration) -> list[list[str]]:
        bucket: list[list[str]] = []
        if enumeration.one_hop_paths:
            bucket = enumeration.one_hop_paths[: len(enumeration.one_hop_paths) // 2]
        if enumeration.two_hop_paths:
            bucket = bucket + enumeration.two_hop_paths[: len(enumeration.two_hop_paths) // 2]
        if enumeration.three_hop_paths:
            bucket = bucket + enumeration.three_hop_paths[: len(enumeration.three_hop_paths) // 2]
        return bucket

    def _score_paths_for_pair(
        self,
        paths: list[list[str]],
        source: str,
        target: str,
    ) -> list[tuple[list[str], float]]:
        """PathRAG edge-flow weighting over candidate paths between one source/target pair."""
        follow_dict: dict[str, set[str]] = {}
        for path in paths:
            for i in range(len(path) - 1):
                current, next_node = path[i], path[i + 1]
                follow_dict.setdefault(current, set()).add(next_node)

        edge_weights: dict[tuple[str, str], float] = defaultdict(float)
        for neighbor in follow_dict.get(source, ()):
            edge_weights[(source, neighbor)] += 1 / len(follow_dict[source])
            if neighbor == target:
                continue
            if edge_weights[(source, neighbor)] <= self._weight_threshold:
                continue
            for second in follow_dict.get(neighbor, ()):
                edge_weights[(neighbor, second)] += (
                    edge_weights[(source, neighbor)] * self._weight_alpha / len(follow_dict[neighbor])
                )
                if second == target:
                    continue
                if edge_weights[(neighbor, second)] <= self._weight_threshold:
                    continue
                for third in follow_dict.get(second, ()):
                    edge_weights[(second, third)] += (
                        edge_weights[(neighbor, second)] * self._weight_alpha / len(follow_dict[second])
                    )

        scored: list[tuple[list[str], float]] = []
        for path in paths:
            path_weight = sum(
                edge_weights.get((path[i], path[i + 1]), 0.0) for i in range(len(path) - 1)
            )
            scored.append((path, path_weight / max(len(path) - 1, 1)))
        return scored

    def _to_path_hits(
        self,
        paths: list[list[str]],
        node_map: dict[str, KgNode],
        edge_map: dict[tuple[str, str], KgEdge],
    ) -> list[KgPathHit]:
        hits: list[KgPathHit] = []
        for path in paths:
            if len(path) < 2:
                continue
            narrative = self._render_narrative(path, node_map, edge_map)
            if not narrative.strip():
                continue
            hits.append(KgPathHit(entity_ids=path, hop_count=len(path) - 1, narrative=narrative))
        return hits

    @staticmethod
    def _lookup_edge(edge_map: dict[tuple[str, str], KgEdge], a: str, b: str) -> KgEdge | None:
        return edge_map.get((a, b)) or edge_map.get((b, a))

    def _render_narrative(
        self,
        path: list[str],
        node_map: dict[str, KgNode],
        edge_map: dict[tuple[str, str], KgEdge],
    ) -> str:
        parts: list[str] = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edge = self._lookup_edge(edge_map, a, b)
            kw = edge.keywords if edge and edge.keywords else "related"
            na = node_map.get(a)
            nb = node_map.get(b)
            if i == 0 and na:
                parts.append(
                    f"The entity {na.entity_name} is a {na.entity_type} with the description ({na.description})"
                )
            if edge and nb:
                parts.append(f" through edge ({kw}) to connect to {na.entity_name if na else a} and {nb.entity_name}")
                if i + 1 == len(path) - 1:
                    parts.append(
                        f"The entity {nb.entity_name} is a {nb.entity_type} with the description ({nb.description})"
                    )
        return " ".join(parts)
