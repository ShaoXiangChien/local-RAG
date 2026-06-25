# Demo Sample Questions

Use these questions to test the local RAG demo after indexing the documents in
this folder. The "Expected focus" column is not an answer key; it is a quick
sanity check for whether retrieval is landing on the right parts of each
document.

## ollama_api.md

| Question | Expected focus |
| --- | --- |
| What Ollama endpoint should this app use for embeddings? | `POST /api/embed`, with `/api/embeddings` noted as superseded |
| What parameters does `/api/embed` accept for generating embeddings? | `model`, `input`, and advanced parameters like `truncate`, `options`, `keep_alive`, and `dimensions` |
| How do I send multiple texts to Ollama for embeddings in one request? | The `input` array example for `/api/embed` |
| How can streaming be disabled for Ollama generation APIs? | `stream: false` in the request payload |
| What timing fields does Ollama return at the end of a streamed generation response? | `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` |
| What endpoint lists models that are currently loaded into memory? | `GET /api/ps` |

## ollama_embeddings.md

| Question | Expected focus |
| --- | --- |
| What are embeddings used for in semantic search and RAG? | Numeric vectors for vector databases, cosine similarity, and retrieval |
| Which Ollama embedding models are recommended in this document? | `embeddinggemma`, `qwen3-embedding`, and `all-minilm` |
| How can I generate an embedding from the command line with Ollama? | `ollama run embeddinggemma "Hello world"` and piping text |
| What does the `/api/embed` endpoint return about the vectors? | L2-normalized unit-length vectors |
| What is the key rule for choosing embedding models when indexing and querying? | Use the same embedding model for both indexing and querying |
| How do I generate a batch of embeddings through the Ollama API? | Pass an array of strings to `input` |

## microsoft_edge_ai_overview.md

| Question | Expected focus |
| --- | --- |
| What is the AI on Edge Flagship Accelerator intended to provide? | Production-ready infrastructure as code, reusable components, blueprints, and AI-assisted practices |
| What are the main user paths described in the repository overview? | Quick deploy with blueprints, custom solutions with components, and docs exploration |
| What are the nine blueprint types listed in the overview? | Full single node, full multi node, full Arc multi node, minimal single node, partial single node, Edge IoT only, cloud only, CNCF cluster script, and Fabric |
| Which component areas combine to build edge solutions? | Cloud Foundation, Edge Infrastructure, IoT Platform, Observability, Analytics Platform, Integration Services, and Automation Tools |
| What does the Observability component provide for edge solutions? | Cloud monitoring, edge monitoring, health checks, dashboards, and metrics |
| How can Terraform telemetry to Microsoft be disabled? | Set `ARM_DISABLE_TERRAFORM_PARTNER_ID=true` before Terraform commands |

## microsoft_edge_ai_general_user.md

| Question | Expected focus |
| --- | --- |
| What prerequisites are required before deploying the AI on Edge accelerator? | Azure subscription, registered resource providers, GitHub access, Docker Desktop, VS Code, Dev Containers, and HVE Core |
| What Azure role is recommended for deployment and why? | Owner role because blueprints create resources and assign RBAC roles |
| What is the recommended development environment for a general user? | The default Dev Container |
| What are the core Terraform variables for a minimal blueprint deployment? | `environment`, `location`, and `resource_prefix` in `terraform.auto.tfvars` |
| What should I check before running `terraform apply`? | Correct variables, correct subscription via `az account show`, and expected `terraform plan` resources |
| How do I verify the deployed Kubernetes cluster after deployment? | Use `az connectedk8s proxy`, then `kubectl get nodes` and `kubectl get pods --all-namespaces` |

## microsoft_edge_ai_hybrid_workloads.md

| Question | Expected focus |
| --- | --- |
| What is hybrid edge-cloud data processing in this document? | Shared processing responsibilities between edge systems such as Azure IoT Operations and cloud systems such as Microsoft Fabric |
| Why can data gravity create challenges for edge applications? | Centralization, latency, security, compliance, infrastructure complexity, and cost |
| When should data be processed locally but archived in the cloud? | When data must stay at the edge because of cost, size, bandwidth, or timing, but older data should move to cloud storage |
| What edge storage options are listed for hybrid workloads? | InfluxDB, distributed cache, Azure IoT Operations State Store, Azure Container Storage enabled by Azure Arc, and Azure Arc Data Services |
| Why is Fabric Eventhouse preferred for real-time edge data over relational database approaches? | KQL/Eventhouse is built for high-volume, low-latency, scalable real-time analytics |
| What are the Bronze, Silver, and Gold layers in the Eventhouse medallion pattern? | Raw landing data, cleaned and normalized data, and curated business-ready data |
| How does lambda architecture split processing between edge and Fabric? | Batch/cold paths and low-latency streaming/hot paths, with Azure IoT Operations DataFlows filtering and routing data |

## opentelemetry_overview.md

| Question | Expected focus |
| --- | --- |
| What is OpenTelemetry, and what is it not? | An observability framework and toolkit, not an observability backend |
| What telemetry data types does OpenTelemetry help generate, collect, and export? | Traces, metrics, and logs |
| Why does OpenTelemetry help avoid vendor lock-in? | You own generated data and use a single set of APIs and conventions across backends |
| What are the main components of OpenTelemetry? | Specification, protocol, semantic conventions, APIs, SDKs, instrumentation ecosystem, collector, and platform tools |
| What does it mean for a system to be observable? | Understanding internal state by examining outputs such as traces, metrics, and logs |
| How can OpenTelemetry be extended? | Custom collector receivers, SDK instrumentation libraries, distributions, exporters, or propagators |

## opentelemetry_collector_architecture.md

| Question | Expected focus |
| --- | --- |
| What are the three main parts of an OpenTelemetry Collector pipeline? | Receivers, processors, and exporters |
| What telemetry data types can Collector pipelines operate on? | Traces, metrics, and logs |
| What happens if a receiver is shared by multiple pipelines? | One receiver instance fans out synchronously to multiple pipelines; blocking in one can block the others |
| How are processors shared across multiple pipelines? | The same processor config can be referenced, but each pipeline gets its own processor instance |
| What does an exporter do in the Collector architecture? | Sends processed telemetry outside the Collector, such as to backends or debug logs |
| When should the Collector run as an agent? | When applications send telemetry to a local VM/container/pod/node agent that can export to backends independently |
| How is running the Collector as a gateway different from running it as an agent? | A gateway receives telemetry from multiple agents, libraries, or tasks and forwards it to configured exporters |

## Cross-document demo questions

| Question | Expected focus |
| --- | --- |
| How would I explain the relationship between OpenTelemetry and the Edge AI accelerator's observability goals? | OpenTelemetry concepts plus Edge AI observability components |
| If this RAG app is indexing local docs with Ollama, which embedding endpoint and model guidance should it follow? | `ollama_api.md` and `ollama_embeddings.md` together |
| What would a good edge-to-cloud observability architecture need to collect, process, and export? | Collector pipelines plus Edge AI monitoring needs |
| Which docs should I read first to deploy an Edge AI blueprint and then monitor it? | General user guide, Edge AI overview, OpenTelemetry overview, and Collector architecture |
| What are the trade-offs between local edge processing and cloud analytics for high-volume IoT data? | Hybrid workloads data gravity, latency, cost, archival, Eventhouse, and Fabric |
