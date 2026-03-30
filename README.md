# CityPark Parking Assistant — Stage 1

An intelligent parking assistant chatbot built with **LangChain**, **LangGraph**, **Pinecone**, and **OpenAI**, using a Retrieval-Augmented Generation (RAG) architecture.

---

## Architecture

```
User (Streamlit UI)
       │
       ▼
  LangGraph Workflow
  ┌─────────────────────────────────────────┐
  │  input_guard  ──(unsafe)──► END         │
  │       │                                 │
  │  classify_intent                        │
  │       ├── "info"  ──► retrieve          │
  │       │                  └──► generate  │
  │       └── "reservation" ──► manage_reservation
  │                                  │      │
  │                           output_guard  │
  │                                  └──► END
  └─────────────────────────────────────────┘
       │
       ▼
  Pinecone (static parking docs)
  OpenAI  (embeddings + chat)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| `ParkingState` has Optional Stage 2-4 fields | Zero-change state migration when adding future stages |
| Node functions return partial dicts | LangGraph merges; nodes stay independently testable |
| Guardrails are pure functions outside the graph | Can be unit-tested without LangGraph |
| Evaluation module has no app dependency | Can be run offline as a standalone script |

---

## Project structure

```
parking_assistant/
├── app.py                    # Streamlit UI
├── config.py                 # Pydantic settings (reads .env)
├── requirements.txt
├── .env.example
├── data/
│   └── parking_documents.py  # Static documents seeded into Pinecone
├── rag/
│   ├── retriever.py          # Pinecone vector store + retrieve()
│   └── prompts.py            # All prompt templates
├── graph/
│   ├── state.py              # ParkingState TypedDict (forward-compatible)
│   ├── nodes.py              # Node functions (input_guard, retrieve, generate, …)
│   └── builder.py            # Graph assembly + compile_graph()
├── guardrails/
│   └── filter.py             # check_input() and check_output()
├── evaluation/
│   └── metrics.py            # precision_at_k, recall_at_k, mrr, measure_latency
├── scripts/
│   ├── seed_pinecone.py      # One-time Pinecone seeding script
│   └── run_eval.py           # Offline RAG evaluation script
└── tests/
    ├── test_rag.py
    ├── test_guardrails.py
    └── test_evaluation.py
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and PINECONE_API_KEY
```

### 3. Seed Pinecone

Run once to embed and upload the parking documents:

```bash
python scripts/seed_pinecone.py
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

### 5. (Optional) Run the RAG evaluation

```bash
python scripts/run_eval.py
```

### 6. Run the tests

```bash
pytest tests/ -v
```

---

## Guardrails

Two-layer filtering is applied to every conversation turn:

**Input guardrail** (before RAG):
1. Fast regex check for prompt injection keywords
2. LLM-based topic relevance check (must be parking-related)

**Output guardrail** (after generation):
1. Regex check for sensitive patterns (API keys, admin credentials)
2. LLM-based check for subtle data leakage (other users' personal data, etc.)

---

## RAG Evaluation metrics

| Metric | Description |
|---|---|
| `Precision@K` | Of the top-K retrieved docs, what fraction are relevant? |
| `Recall@K` | Of all relevant docs, what fraction appear in top-K? |
| `MRR` | Mean Reciprocal Rank — average 1/rank of first relevant doc |
| `Latency` | Wall-clock time for retrieval + generation |

---

## Roadmap

| Stage | Description |
|---|---|
| **1 (current)** | RAG chatbot, Pinecone, guardrails, evaluation |
| **2** | Human-in-the-loop admin approval via email/REST |
| **3** | MCP server writes confirmed reservations to file |
| **4** | Full LangGraph orchestration of all components |
