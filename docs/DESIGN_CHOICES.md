# Edge AI RAG Demo: Design Choices

Status: Updated for implemented v1 demo  
Date: 2026-06-25  
Related docs:

- [Take-home assignment](../TAKE_HOME_ASSIGNMENT.md)
- [Full design spec](./EDGE_AI_RAG_DESIGN_SPEC.md)

## 1. Executive Summary

For this implemented take-home demo, the selected stack is:

- **Model runtime:** Ollama
- **Default chat model:** `llama3.2:3b`
- **Embedding runtime:** Ollama-served local embedding model
- **Vector store:** LanceDB
- **Retrieval:** hybrid retrieval with dense vector search plus full-text search
- **Orchestration:** explicit Python service code instead of LangChain for the first version
- **UI:** Streamlit

The core reason is that this stack fits the assignment and the edge AI context: it runs locally, avoids cloud services, has a small operational footprint, supports source-grounded RAG, and is explainable end to end.

The design is intentionally not the most abstract or enterprise-heavy version of RAG. It is a prototype architecture that proves the full on-device pipeline while leaving room to swap model runtimes, vector stores, or retrieval strategies later.

## 2. Edge AI Constraints Driving The Design

An edge AI team is usually optimizing for a different set of constraints than a cloud-first AI application:

- **Privacy:** user documents and prompts should stay on-device.
- **Offline capability:** the application should continue working without internet access after models are installed.
- **Latency:** retrieval and generation should avoid network round trips.
- **Resource awareness:** model size, context length, chunk size, and retrieval volume must fit local memory and compute.
- **Reproducibility:** setup should be scriptable and demoable on a developer machine.
- **Operational simplicity:** a prototype should avoid unnecessary servers, containers, and cloud dependencies.
- **Explainability:** the team should be able to inspect every step from document ingestion to answer generation.

These constraints are why the design favors local serving, embedded storage, explicit orchestration, and source-visible outputs.

## 3. Decision Matrix

| Layer | Selected Option | Main Reason | Main Tradeoff |
| --- | --- | --- | --- |
| Model runtime | Ollama | Simple local runtime with Python support, streaming, embeddings, and context configuration | Requires installing and pulling models locally |
| Embeddings | Ollama embedding model | Keeps generation and embeddings under one local runtime | Less direct control than using `sentence-transformers` manually |
| Vector store | LanceDB | Embedded local database with vector, full-text, and hybrid search | Less familiar than Chroma for many RAG tutorials |
| Retrieval | Hybrid search | Handles semantic similarity and exact keyword matches | Slightly more setup than dense-only retrieval |
| Orchestration | Raw Python | Most explainable and easiest to debug | More code than using LangChain/LlamaIndex |
| UI | Streamlit | Fast Python-native demo UI with chat, upload, sidebar, and streaming support | Not a production frontend |

## 4. Model Runtime Choice

### Recommendation

Use **Ollama** as the default local model runtime.

### Options Compared

| Option | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| Ollama | Easy local setup, official Python package, streaming chat, embedding API, configurable context behavior | Requires local model pulls; less low-level than llama.cpp | Best default for a reproducible local prototype |
| LM Studio | Excellent GUI, local server, OpenAI-compatible API, easy model discovery | GUI state can be less reproducible; more manual for take-home setup | Great fallback if already installed |
| llama.cpp server | Maximum control over GGUF models, quantization, low-level serving, OpenAI-compatible routes | More setup and tuning; easier to lose time | Strong later optimization or deep-dive option |
| MLX / MLX-LM | Apple Silicon optimized | More specialized and less universally familiar | Good future path for Mac/Apple edge work |
| vLLM / vLLM-Metal | High-throughput serving architecture | More infrastructure than this one-user demo needs | Better for multi-user serving than MVP |

### Why Ollama Fits The Assignment

Ollama is the right starting point because the demo needs to be local, Python-based, and explainable quickly. Its Python library supports chat, generation, model management, embeddings, and streamed responses. Its embedding docs explicitly describe embeddings for semantic search, retrieval, and RAG, and recommend local embedding models such as `embeddinggemma`, `qwen3-embedding`, and `all-minilm`.

Ollama also exposes context length configuration. Since the assignment asks for a 6k context window, the app can enforce a 6,000-token working context budget and set the runtime context near that budget. This gives us two levels of control:

- **Application-level control:** prompt builder keeps the prompt and reserved answer space within the planned token budget.
- **Runtime-level control:** model runtime is configured with enough context for the packed prompt.

The implementation explicitly uses the local Ollama endpoint and local model names. Ollama cloud models or any other cloud-hosted model path are out of scope for this assignment.

The current demo uses `llama3.2:3b` for chat generation because it gives cleaner
non-thinking output for a live demo than `qwen3:4b`. Qwen3 is still a useful
local model, but in testing it sometimes emitted thinking-style text even with
`think = false`; the Ollama adapter contains defensive handling for that case,
but the default demo path should be boring and reliable.

### Short Explanation

> I chose Ollama because it gives me a reproducible local model runtime with Python integration, streaming output, local embeddings, and context configuration. LM Studio is a good fallback, but Ollama is easier to script and explain in a take-home demo.

## 5. Embedding Model Choice

### Recommendation

Use an **Ollama-served embedding model** for the first version, starting with `embeddinggemma` or falling back to `all-minilm`.

### Options Compared

| Option | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| Ollama embeddings | One local runtime for chat and embeddings; simple Python API; easy local story | Less custom batching/control than direct model loading | Best MVP choice |
| Sentence Transformers | Large model ecosystem, direct Python control, easy reranker path | Adds separate model runtime/caching path | Good fallback or upgrade |
| llama.cpp embeddings | Strong low-level local option | More setup complexity | Useful if optimizing runtime footprint |
| Cloud embeddings | High quality and easy APIs | Violates assignment and edge AI privacy goals | Not allowed |

### Why This Matters For Edge AI

Embedding generation is part of the privacy boundary. If document chunks are sent to a cloud embedding service, the system is no longer local-first. Running embeddings locally keeps the entire indexing pipeline on-device.

The embedding model must also be consistent: the same embedding model should be used for both indexing and querying. The design records the embedding model in the document manifest so the app can warn or reindex when the embedding model changes.

### Short Explanation

> I treat embeddings as part of the model layer, not just a database detail. For an edge system, embeddings must run locally because every document chunk passes through that model.

## 6. Vector Store Choice

### Recommendation

Use **LanceDB** as the default local vector store.

### Options Compared

| Option | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| LanceDB | Embedded local storage, vector search, full-text search, hybrid search, metadata support | Less common than Chroma in beginner examples | Best fit for local hybrid RAG |
| ChromaDB | Very common RAG stack, simple persistent client, good dense vector search | Hybrid search usually needs extra BM25/TF-IDF sidecar logic | Good fallback if LanceDB setup is slower |
| Qdrant | Strong vector DB, filtering, hybrid query support, production-friendly | Usually feels like extra service infrastructure for this prototype | Better for production-like deployment |
| FAISS | Very fast vector similarity library | Not a full document management store; metadata/delete/update need extra code | Too low-level for this demo |
| SQLite + vectors manually | Maximum control, minimal dependency | Easy to build something fragile | Good learning exercise, poor take-home use of time |

### Why LanceDB Fits The Assignment

The demo needs source management, metadata, retrieval, and ideally hybrid search. LanceDB gives an embedded local database model, which means no cloud service and no separate server is required for the MVP. Its docs describe hybrid search as combining vector and full-text search with reranking. That is a strong fit for local RAG because dense embeddings and keyword search solve different failure modes.

Dense retrieval is good at semantic similarity:

- "How do I configure context length?"
- "What are the local-only constraints?"

Full-text retrieval is good at exact terms:

- `num_ctx`
- `OpenRouter`
- `embeddinggemma`
- filenames
- error codes
- API names

Hybrid retrieval is valuable because technical documents often contain exact terms that embeddings may blur.

### Short Explanation

> I chose LanceDB because it keeps the vector store embedded and local, while giving me a path to hybrid retrieval without adding a second database or a custom BM25 sidecar.

## 7. Retrieval Strategy Choice

### Recommendation

Use **hybrid retrieval** with query rewrite enabled for conversational cases.

### Options Compared

| Strategy | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| Direct dense retrieval | Simple, fast, easy to debug | Misses exact keyword matches sometimes | Good baseline |
| Direct hybrid retrieval | Captures semantic and exact-term matches | Slightly more setup | Best default |
| LLM query rewrite | Helps follow-up questions and vague queries | Adds latency and can rewrite incorrectly | Enabled with fallback and debug visibility |
| Agentic retrieval | Flexible multi-step reasoning | More moving parts and harder to verify | Not needed for MVP |

### Why Query Rewrite Is Enabled

The initial baseline can retrieve directly from the user query, but the current
demo enables query rewrite because it makes conversational follow-ups much
easier to show. A question like "what about chat?" is too vague for retrieval
by itself, so the app resolves it into a standalone retrieval query before
searching.

The implementation keeps the risk controlled:

1. Simple "what about X?" follow-ups are handled deterministically when possible.
2. Other rewrites use Ollama's prompt completion path, not chat, with
   `think = false`, a short token cap, and a newline stop sequence.
3. If rewrite fails or returns empty output, retrieval falls back to the original
   user question.
4. The rewritten query is visible in the debug panel, so this step is auditable.

### Short Explanation

> I enabled query rewrite because follow-up questions are part of a realistic chat demo, but I kept it bounded: deterministic handling for common follow-ups, a short no-thinking local completion call for harder cases, and fallback to the original query.

## 8. RAG Framework Choice

### Recommendation

Use **raw Python orchestration** for the first version.

### Options Compared

| Option | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| Raw Python | Transparent flow, easy to explain, easy to test each step | More code than a framework | Best fit for an explainable prototype |
| LangChain | Rich ecosystem, mature integrations, fast to scaffold | Abstractions can hide important details | Good if building a broader app |
| LlamaIndex | Strong document/RAG abstractions | Can hide ingestion and retrieval decisions | Good for data-heavy RAG projects |
| Haystack | Pipeline-oriented and production-minded | More framework structure than needed | Better for larger search systems |

### Why Raw Python Fits This Prototype

The assignment explicitly says the developer should understand and explain each line of code. A small raw Python orchestration layer makes the data flow obvious:

```text
load documents -> chunk text -> embed chunks -> store vectors
user query -> embed query -> retrieve chunks -> pack prompt -> stream LLM answer
```

Frameworks are useful, but the first version should make the architectural decisions visible.

### Short Explanation

> I avoided LangChain in the first version because the assignment is about proving and explaining the RAG pipeline. I can add a framework later, but I do not want framework abstractions to hide chunking, retrieval, prompt packing, or source handling.

## 9. Context Window And Token Budget Choice

### Recommendation

Enforce a **6,000-token application working-context budget**, even if the selected local model supports more context.

### Why This Matters

The assignment requires a 6k context window. Treating this as an application-level invariant is safer than relying only on the model runtime. The prompt builder should count or estimate tokens, reserve space for the answer, and pack only the chunks that fit.

Suggested budget:

| Prompt Component | Budget |
| --- | ---: |
| System prompt | 300 tokens |
| User question | 300 tokens |
| Short chat history | 600 tokens |
| Retrieved chunks | 3,800 tokens |
| Formatting and safety margin | 200 tokens |
| Reserved answer space | 800 tokens |
| Total working context | 6,000 tokens |

The runtime can be configured near 6144 tokens, while the prompt builder keeps the combined prompt plus reserved answer space at or below 6000 tokens.

### Short Explanation

> I enforce the context limit in my own prompt packing logic, not just in the model runtime. That makes the 6k requirement testable.

## 10. UI Choice

### Recommendation

Use **Streamlit** for the demo UI.

### Options Compared

| Option | Strengths | Weaknesses | Position |
| --- | --- | --- | --- |
| Streamlit | Fast Python UI, chat input, file upload, sidebar, streaming support | Not a production web frontend | Best take-home demo option |
| Gradio | Very fast ML demos, simple upload/chat patterns | Less natural for multi-page document management | Good fallback |
| FastAPI + React | Production-style separation | Too much UI work for the timeline | Overkill for MVP |
| CLI only | Simple and testable | Less impressive for demo and source inspection | Useful fallback, not ideal final demo |

### Why Streamlit Fits

The assignment needs a working demo, not a production web app. Streamlit lets us build a chat page and a document management page quickly while staying in Python. It also supports chat-style input, file upload, sidebar layout, and generator-based streaming.

### Short Explanation

> I chose Streamlit because the UI is not the core research problem. It gives me enough interface to demonstrate ingestion, chat, streaming, and sources without spending the take-home on frontend plumbing.

## 11. Source Display Choice

### Recommendation

Always show retrieved source chunks with the answer.

### Why This Matters

For RAG, source visibility is part of correctness. It helps a reviewer verify that:

- The answer is grounded in retrieved text.
- Retrieval is actually happening.
- The model is not relying only on parametric memory.
- Chunking and metadata are working.
- The system can be debugged when the answer is wrong.

Each answer should show:

- Filename
- Chunk index
- Retrieval score
- Short excerpt
- Expandable full chunk text

### Short Explanation

> I show sources because RAG without source inspection is hard to debug. In a local AI setting, observability is not just logs; it is making the retrieval context visible.

## 12. Document Management Choice

### Recommendation

Use a local document manifest plus vector store metadata.

The manifest tracks:

- Document ID
- Filename
- Content hash
- File type
- Chunk count
- Embedding model
- Indexing status
- Error message

### Why This Matters

Document management is easy to underestimate in RAG. Without a manifest, it is hard to answer:

- Is this document already indexed?
- Which embedding model was used?
- How many chunks did it create?
- Can I delete all chunks for this document?
- Do I need to reindex after changing models?

For the prototype, SQLite is a good manifest store because it is local, familiar, and easy to inspect.

### Short Explanation

> I separated the document manifest from the vector index because document lifecycle management is different from similarity search.

## 13. Risks And Fallbacks

| Risk | Mitigation |
| --- | --- |
| Ollama is not installed or model pulls are slow | Keep LM Studio/OpenAI-compatible adapter as a fallback |
| Selected LLM is too slow on 16 GB Mac | Use a smaller 4B model or more aggressive quantization |
| Embedding model is unavailable in Ollama | Fall back to `all-minilm` or direct `sentence-transformers` |
| LanceDB setup takes longer than expected | Fall back to ChromaDB dense retrieval for MVP |
| Hybrid search behavior is hard to tune | Start dense-only, then add full-text/hybrid after baseline works |
| Prompt plus reserved answer space exceeds 6k tokens | Token budgeter drops lower-ranked chunks before generation |
| Answer is unsupported by sources | Prompt instructs model to say the documents lack enough information |

## 14. How I Would Explain The Final Architecture

> This is a local-first RAG system designed for an edge AI context. Ollama runs the local chat and embedding models. The ingestion pipeline loads Markdown and text files, chunks them, embeds each chunk locally, and stores them in LanceDB with metadata. The chat flow loads conversation memory, rewrites follow-up questions into standalone retrieval queries, runs hybrid retrieval over vector and full-text indexes, packs the highest-value chunks into a 6k working-context budget, and streams the answer back through Streamlit. The UI shows pipeline timings and retrieved source chunks so the answer can be inspected and debugged. I used raw Python orchestration so the full pipeline is easy to explain and test.

## 15. Sources Checked

- Ollama Python library, including streaming and embedding APIs: https://github.com/ollama/ollama-python
- Ollama embeddings documentation: https://docs.ollama.com/capabilities/embeddings
- Ollama context length documentation: https://docs.ollama.com/context-length
- LanceDB hybrid search documentation: https://docs.lancedb.com/search/hybrid-search
- Chroma query and get documentation: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Qdrant local quickstart: https://qdrant.tech/documentation/quickstart/
- Qdrant hybrid query documentation: https://qdrant.tech/documentation/search/hybrid-queries/
- LM Studio OpenAI-compatible endpoint documentation: https://lmstudio.ai/docs/developer/openai-compat
- llama.cpp server documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- LangChain retrieval documentation: https://docs.langchain.com/oss/python/langchain/retrieval
- Streamlit chat input documentation: https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
