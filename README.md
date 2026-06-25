# Edge AI Local RAG Chat Demo

Local-only Python RAG demo for the HP edge AI take-home assignment. The app ingests Markdown and text files, chunks them, embeds chunks with a local Ollama embedding model, stores retrieval metadata locally, retrieves source chunks, rewrites follow-up questions, enforces a 6,000-token working-context budget, streams grounded answers through Streamlit, and reports local inference metrics such as TTFT, prompt/output tokens per second, and estimated effective throughput.

## Architecture

- UI: Streamlit
- LLM runtime: local Ollama at `http://localhost:11434`
- LLM model: configurable, default `llama3.2:3b`
- Embedding model: configurable, default `embeddinggemma`
- Vector store: LanceDB under `data/index`
- Manifest and conversation summaries: SQLite at `data/app.db`
- Orchestration: explicit Python services in `rag_demo/`
- Instrumentation: pipeline timing plus Ollama stream metrics for local inference

No cloud LLM, cloud embedding, or hosted vector service is used.

## Prerequisites

- Python 3.11 or newer. Python 3.11 or 3.12 64-bit is recommended on Windows.
- `uv` for dependency management, or `pip` as a fallback.
- Ollama installed and running locally on `http://localhost:11434`. Verify with:

```bash
ollama --version
ollama list
```

If `ollama --version` fails, install Ollama first from:

- Windows: https://ollama.com/download/windows
- macOS/Linux: https://ollama.com/download

On Windows, install with the official `OllamaSetup.exe`, reopen PowerShell, and
run:

```powershell
ollama --version
ollama list
Invoke-RestMethod http://localhost:11434/api/version
```

If the API check fails, start Ollama from the Start menu or run `ollama serve`
in another terminal.

On macOS with Homebrew, you can also install and start the CLI/server with:

```bash
brew install ollama
brew services start ollama
```

If `uv` is not installed, use the official installer docs:
https://docs.astral.sh/uv/getting-started/installation/

## Setup

### macOS/Linux

```bash
uv sync --extra dev
```

### Windows PowerShell

```powershell
uv sync --extra dev
```

`uv` creates and manages `.venv` for this project, so you do not need to
activate the virtual environment manually before using `uv run`.
If `uv` cannot find a compatible Python on Windows, run:

```powershell
uv python install 3.12
uv sync --extra dev
```

If you are not using `uv`, install the project dependencies into your Python environment.
On macOS/Linux:

```bash
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -3 -m pip install -e ".[dev]"
```

With Ollama running, pull the local models:

```bash
ollama pull llama3.2:3b
ollama pull embeddinggemma
```

The model names are configurable in `config.toml`. If `embeddinggemma` is not available in your local Ollama setup, switch `embedding_model` to another local embedding model such as `all-minilm` or `nomic-embed-text`.

The default chat model is `llama3.2:3b` to avoid Qwen3-style thinking output during the demo. The generation config still sets `think = false`; if you switch back to a thinking-capable model such as `qwen3:4b`, the Ollama adapter also prepends `/no_think` to system/user prompts and filters raw `<think>...</think>` text if the model still emits it as normal content. Query rewrite uses Ollama's prompt completion path instead of chat, with `think = false`, a short token cap, and a newline stop sequence. The config caps answers with `num_predict = 512` and retrieves `top_k = 4` chunks by default to keep the local prompt smaller. Raise those values for deeper answers, or set `think` to `true`, `"low"`, `"medium"`, `"high"`, or `"max"` if you want Ollama to expose that model's thinking mode. When Ollama returns a separate `message.thinking` stream, the app renders it under "Model thinking" and keeps it out of the saved answer.

## Run

### macOS/Linux

```bash
uv run streamlit run app.py
```

### Windows PowerShell

```powershell
uv run streamlit run app.py
```

Open the Streamlit URL, usually `http://localhost:8501`, go to Documents,
upload files from `sample_docs/` or `demo_docs/`, index them, then return to
Chat and ask a question.

## Windows Notes

- Clone the repo into a normal user-writable folder such as
  `C:\Users\<you>\Documents\local-RAG`.
- Use PowerShell, Windows Terminal, or VS Code's PowerShell terminal.
- Forward-slash paths in `config.toml`, such as `data/uploads`, are intentional
  and work on Windows through Python's path handling.
- If `uv` or `ollama` is not found after installation, close and reopen the
  terminal so your `PATH` is refreshed.
- If Ollama is reachable but generation is slow on a laptop, keep the default
  `llama3.2:3b` model and lower `num_predict` or `top_k` in `config.toml`.
- To reset all local indexed data, stop Streamlit and delete the local `data`
  folder. It is ignored by Git and will be recreated on the next run.

## Demo Questions

After indexing the sample documents, try:

- `Why was Ollama selected?`
- `How does the app stay within the 6k context limit?`
- `What happens if query rewrite fails?`
- `How are documents deleted?`

Then ask a follow-up:

- `What about the embedding model?`

For a larger demo set, index the files in `demo_docs/` and use
`demo_docs/sample_questions.md` as the question bank.

The live pipeline panel shows memory loading, query rewriting, retrieval, prompt building, answer generation, memory updates, and summary refresh timings. The performance metrics section reports TTFT, prompt eval throughput, output throughput, output token count, generation time, and estimated effective GFLOP/s. The GFLOP/s value is derived from output tokens/sec and estimated model parameters; it is not a hardware-counter measurement. The debug expander shows the rewritten retrieval query, estimated prompt tokens, dropped sources, and retrieved source chunks.

## Verification

Run these before pushing or demoing:

```bash
uv run pytest -q
uv run python -m compileall app.py rag_demo tests
```

Current automated coverage includes loader validation, chunking, document manifest lifecycle, token budgeting, prompt/source alignment, query rewrite behavior, conversation memory, Ollama adapter behavior, default config, and chat orchestration.

## Design Notes

- `rag_demo/ingestion/loader.py` validates `.md` and `.txt`, decodes UTF-8, normalizes line endings, and computes content hashes.
- `rag_demo/ingestion/chunker.py` uses a transparent token-window splitter with overlap and source offsets.
- `rag_demo/retrieval/store.py` wraps LanceDB, attempts hybrid vector/full-text retrieval, and falls back to dense retrieval.
- `rag_demo/chat/token_budget.py` enforces the 6,000-token working-context budget before generation.
- `rag_demo/chat/service.py` is the explicit RAG orchestration path: memory, rewrite, retrieve, prompt, stream, summarize.
- `rag_demo/chat/service.py` also aggregates TTFT, Ollama token counts, token throughput, and estimated effective throughput for edge-runtime discussion.
- `rag_demo/ui/` keeps Streamlit rendering separate from the RAG logic.
