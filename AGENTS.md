# AGENTS.md — Regras do Projeto

## Definições gerais
- Sempre responda em **português (pt-BR)**.
- Antes de editar qualquer arquivo de código, **verifique a existência da pasta `BCK`**:
  - Se não existir, **crie-a automaticamente** na raiz do projeto.
- **Antes de qualquer alteração em código-fonte**, execute o seguinte procedimento de backup:
  1. Crie uma cópia do arquivo original dentro da pasta `BCK`.
  2. Nomeie a cópia com **numeração sequencial** (exemplo: `nome_001.py`, `nome_002.py`, etc.), preservando a extensão original.
  3. Somente após o backup, prossiga com a edição solicitada.
- Este processo garante **rastreabilidade e recuperação de versões** em caso de erro.

## Observações para o agente
- Caso a tarefa envolva múltiplos arquivos, repita o procedimento para cada um antes da primeira modificação.
- O agente deve **confirmar no log** (mensagem no VS Code) que o backup foi criado com sucesso antes de iniciar a edição.
