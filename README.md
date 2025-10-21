# Projeto LangChain – QA sobre PDFs

Este projeto cria uma base vetorial a partir de PDFs e responde perguntas usando o conteúdo indexado.

## Pré-requisitos
- Python 3.10+
- Chave da API da OpenAI (`OPENAI_API_KEY`)

## Configuração
1. Crie e ative um ambiente virtual (opcional, recomendado):
   - Windows:
     ```bash
     python -m venv env
     .\env\Scripts\activate
     ```

2. Instale as dependências (pip ou uv):
   - Com pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Com uv (recomendado):
     - Instale o uv: https://docs.astral.sh/uv/getting-started/ (Windows: PowerShell `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`)
     - Crie o venv e sincronize dependências do `pyproject.toml`:
       ```bash       
       uv sync
       ```
     - Rodar scripts com o venv correto:
       ```bash
       uv run python criar_db.py
       uv run python main.py
       ```

3. Configure as variáveis de ambiente em um arquivo `.env` na raiz do projeto:
   ```env
   OPENAI_API_KEY=seu_token_aqui
   ```

4. Adicione seus PDFs na pasta `base/`.

## Criar a base vetorial
Gera os embeddings e persiste o banco de dados em `db/`.
```bash
python criar_db.py
```
- Se a base já existir, a criação é pulada.
- Logs mostram quantos chunks foram gerados.

## Executar o assistente
Inicia o loop interativo de perguntas e respostas:
```bash
python main.py
```
- Digite sua pergunta e pressione Enter.
- Use `sair` ou deixe vazio para encerrar.
- As respostas incluem referências às fontes (arquivos PDF utilizados).

## Estrutura
- `base/`: PDFs de origem
- `db/`: base vetorial persistente (ChromaDB)
- `criar_db.py`: pipeline de carga, chunking e vetorização
- `main.py`: consulta à base e geração de resposta
- `requirements.txt`: dependências (pip tradicional)
- `pyproject.toml`: dependências gerenciadas pelo uv

## Observações
- Se nenhum contexto relevante for encontrado, o assistente informa que não há dados suficientes.
- Ajuste os parâmetros `chunk_size`, `chunk_overlap`, `K_RESULTADOS` e `RELEVANCIA_MINIMA` conforme suas necessidades.

## Gerenciar dependências com uv
O `uv` é um gerenciador super rápido e determinístico:
- Crie/ative o ambiente: `uv venv` (cria `.venv/` no projeto).
- Sincronize dependências do `pyproject.toml`: `uv sync`.
- Adicione/remova pacotes: `uv add nome-do-pacote`, `uv remove nome-do-pacote`.
- Atualize versões e lockfile: `uv lock --upgrade` e depois `uv sync`.
- Instale uma versão de Python: `uv python install 3.11` (opcional, por projeto).
- Execute comandos no ambiente correto: `uv run python main.py`.