# Contexto Detectado

## CONTEXT_PROFILE

```yaml
tipo: feature
urgencia: normal
rigor: padrão
comunicacao: didático
autonomia: alta

projeto:
  testes: existentes
  typescript: ausente
  cicd: ausente
  linter: configurado
  cobertura: baixa
```

## Resumo Visual

| Aspecto | Valor | Impacto |
|---------|-------|---------|
| Tipo | feature | Fluxo completo (branch `main`, chave `TASK-001` sem prefixo de bug/hotfix) |
| Urgência | normal | Sem sinais de urgência no pedido; a TASK-001 tem prioridade Alta de produto, o que não muda o rigor |
| Rigor | padrão | Todas as validações |
| Comunicação | didático | `POSITION=GENERALIST` não está na tabela do skill: usado o padrão (`pleno`) |
| Autonomia | alta | `MAX_AI_EXECUTION_PERCENTAGE=100` |

## Características do Projeto (`production-rag`)

| Aspecto | Valor | Sinal |
|---------|-------|-------|
| Testes | existentes | 40 arquivos `test_*.py` em `tests/` |
| Tipagem | ausente (Python) | Sem `tsconfig.json`; mypy roda com `ignore_errors = true` (`pyproject.toml`) |
| CI/CD | ausente | Sem `.github/workflows/` nem `.gitlab-ci.yml` |
| Linter | configurado | `[tool.ruff]` no `pyproject.toml` |
| Cobertura | baixa | Sem relatório de cobertura gerado |

## Ajustes Aplicados

- Testes: exigir testes para código novo (pytest, padrão espelhado em `tests/unit/domain/`)
- Tipagem: anotar assinaturas (regra do repositório), sem contar com o mypy para barrar erros
- CI/CD: ausente, então validar localmente com `make lint` e `make test` antes do PR
- Linter: garantir que o código passa no `ruff`

---

> Este perfil será usado pelos workflows `eng.start`, `eng.plan` e `eng.work`.
> Para recalibrar, execute `/jarvis-context-detect` novamente.
