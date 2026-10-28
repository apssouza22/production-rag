# Central docs

Registro do que **foi implementado** neste workspace. Os documentos são gerados pelo `/eng.spec-consolidate` depois de cada PR. Rascunhos de trabalho (PRD, FRD, ARD, RFC) ficam em `docs/jarvis/` e não vêm para cá.

## Estrutura

```
{SQUAD}/
├── index.md                              # gerado pelo `jarvis docs publish` (não editar à mão)
├── product/{produto}/feature-*.md        # um documento completo por funcionalidade (estado atual)
└── engineering/{produto}/delivery-*.md   # um registro por entrega
```

## Regras

- Leitura e publicação sempre por `jarvis docs fetch | sync | publish`.
- Esta pasta fica em `docs/jarvis/central-docs/`, fora do versionamento dos repositórios de código (a raiz do workspace não é um repositório).
