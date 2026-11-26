"""Load and format schema ontology YAML for text-to-SQL prompts."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SchemaNode(BaseModel):
    id: str
    description: str
    example: str | None = None
    query: str | None = None


class SchemaEdge(BaseModel):
    relation: str
    source_id: str
    target_id: str
    description: str
    query: str | None = None


class SchemaMetric(BaseModel):
    id: str
    node_id: str
    kind: str = "sql"
    description: str
    query: str
    embeddable: bool = False


class SchemaOntology(BaseModel):
    nodes: list[SchemaNode] = Field(default_factory=list)
    edges: list[SchemaEdge] = Field(default_factory=list)
    metrics: list[SchemaMetric] = Field(default_factory=list)

    def table_names(self) -> list[str]:
        return [node.id for node in self.nodes]

    def format_for_prompt(self) -> str:
        sections: list[str] = []

        sections.append("## Tables (nodes)")
        for node in self.nodes:
            block = [
                f"### table: `{node.id}`",
                node.description,
            ]
            if node.example:
                block.append(f"Example row: {node.example}")
            if node.query:
                block.append(f"Sample query:\n```sql\n{node.query}\n```")
            sections.append("\n".join(block))

        if self.edges:
            sections.append("## Relationships (edges)")
            for edge in self.edges:
                block = [
                    f"### {edge.relation} (`{edge.source_id}` -> `{edge.target_id}`)",
                    edge.description,
                ]
                if edge.query:
                    block.append(f"Join pattern:\n```sql\n{edge.query}\n```")
                sections.append("\n".join(block))

        if self.metrics:
            sections.append("## Metrics")
            for metric in self.metrics:
                block = [
                    f"### metric: `{metric.id}` (on `{metric.node_id}`)",
                    metric.description,
                    f"SQL:\n```sql\n{metric.query}\n```",
                ]
                sections.append("\n".join(block))

        return "\n\n".join(sections)


def default_ontology_path() -> Path:
    return Path(__file__).parent / "pagila-ontology.yaml"


def load_schema_ontology(path: Path | None = None) -> SchemaOntology:
    ontology_path = path or default_ontology_path()
    with ontology_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid ontology file: {ontology_path}")
    return SchemaOntology.model_validate(raw)