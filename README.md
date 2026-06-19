# OncoMatch-Orchestrator: Stateful Multi-Agent System for Clinical Trial Matching

OncoMatch-Orchestrator is a production-grade, containerized AI microservice designed to ingest unstructured clinical oncology notes, extract validated patient entities, query an in-memory vector database, and execute multi-agent validation rules to determine patient clinical trial eligibility.

Built to simulate enterprise-scale healthcare data engineering pipelines, this system shifts away from basic linear prompting into deterministic, stateful agent orchestration.

---

## 🚀 Key Engineering Achievements & Core Metrics

- **Data Engineering & Vector Semantics:** Engineered an automated document chunking and indexing pipeline using `qdrant-client`. Implemented a local vector space optimized with Cosine distance indexing to allow sub-millisecond semantic search retrieval over unstructured protocol entries.
- **Structured Agentic Extraction:** Utilized strict `Pydantic` validation schemas to enforce strict object types on messy, unformatted doctor narratives, preventing data type drifting across downstream applications.
- **Stateful Flow Architecture:** Designed a modular multi-agent workflow where specialized worker instances act as independent nodes—decoupling the Data Ingestor, Retriever, and an Eligibility Critic agent to enforce complex clinical safety guardrails.
- **Production Web Infrastructure:** Wrapped the multi-agent pipeline inside a high-throughput `FastAPI` application backend featuring auto-generated interactive documentation (Swagger UI) for real-time web execution.
- **MLOps Containerization:** Dockerized the complete multi-service application layer using a lightweight, multi-stage Linux-slim image to ensure platform-agnostic portability and seamless cloud deployments.

---

## 🛠️ The 2026 AI Engineer Tech Stack

| Layer                   | Technologies Used                         |
| :---------------------- | :---------------------------------------- |
| **Agent Orchestration** | LangGraph (Conceptual Setup), Python Core |
| **Vector Database**     | Qdrant (In-Memory Engine)                 |
| **Data Validation**     | Pydantic v2                               |
| **API Framework**       | FastAPI, Uvicorn                          |
| **DevOps / MLOps**      | Docker, Docker-Compose                    |

---

## 💻 Technical Walkthrough: How the Data Flows

1. **Extraction Node:** Receives raw clinical text data and parses core fields (Cancer Type, Stage, Genetic Mutations, Previous Lines of Therapy).
2. **Retriever Node:** Maps the validated entities to a dense vector space query, identifying high-affinity clinical trials within the Qdrant instance.
3. **Critic Node:** Evaluates strict patient exclusion criteria (e.g., checking for specific metastatic flags) against the retrieved trial protocol payload to approve or reject candidacy.

---

## 🔧 Installation & Local Execution

### Prerequisites

- Python 3.11+
- Docker Desktop

### Option A: Running Containerized via Docker (Recommended)

Build and run the entire ecosystem globally without local dependency overhead:

```bash
# Build the container image from blueprint
docker build -t oncomatch-api .

# Fire up the live microservice
docker run -p 8000:8000 oncomatch-api
```
