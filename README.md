# DSN × BCT LLM Agent Challenge

**Team:** Vancus 
**Hackathon:** DSN × BCT Data & AI Summit · Hackathon 3.0  
**Submission deadline:** 24 May 2026

---

## Overview

Two LLM-powered agentic systems built on the Yelp Open Dataset:

| Task | Description | Endpoint |
|------|-------------|----------|
| **A — Review Simulator** | Given a user persona + business, generate a realistic star rating and written review | `POST /simulate-review` |
| **B — Recommendation Agent** | Given a user persona (+ optional conversation context), return personalised ranked recommendations | `POST /recommend` |

Both tasks share a **Persona Engine** that extracts behavioural fingerprints from user history (rating bias, vocabulary profile, tone signals) and a **Nigerian style adapter** applied at the LLM layer.

---

## Architecture

```
Yelp dataset (5 JSON files)
        ↓
  Vector store (ChromaDB / FAISS)
        ↓
  Persona engine  ← shared core
    ↙         ↘
Task A        Task B
Review        Recommendation
Simulator     Agent
    ↘         ↙
  LLM backbone + Nigerian style adapter
  (Claude / GPT-4o)
    ↙         ↘
POST            POST
/simulate-review  /recommend
```

---

## Quickstart

### 1. Prerequisites
- Docker + docker-compose
- Python 3.14.2+ (for running preprocessing scripts locally)
- Yelp Open Dataset JSON files

### 2. Setup

```bash
git clone <repo>
cd bcthack

# Copy and fill in your API key
cp .env.example .env
# Edit .env — set GEMINI_API_KEY

# Install script dependencies (local only, not in containers)
pip install pandas pyarrow chromadb sentence-transformers python-dotenv
```

### 3. Data Pipeline

Place all 5 Yelp JSON files in `data/raw/`, then:

```bash
# Step 1: Parse + join Yelp data → parquet files
python -m scripts.extract_data

# Step 2: Build ChromaDB vector index
python -m scripts.build_index

# (Optional) Run EDA first
python -m scripts.yelp_eda --data_dir ./data/raw
```

### 4. Run with Docker

```bash
docker-compose up --build
```

- Task A: http://localhost:8001
- Task B: http://localhost:8002
- Task A docs: http://localhost:8001/docs
- Task B docs: http://localhost:8002/docs

### 5. Test the endpoints

```bash
# Task A
curl -X POST http://localhost:8001/simulate-review \
  -H "Content-Type: application/json" \
  -d '{"user_id": "abc123", "business_id": "xyz456"}'

# Task B
curl -X POST http://localhost:8002/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "abc123", "context": "Looking for a good suya spot"}'
```

---

## Project Structure

```
bcthack/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── raw/                  # Yelp JSON files (gitignored)
│   └── processed/            # Parquet + ChromaDB index (gitignored)
├── shared/                   # Shared library (mounted into both containers)
│   ├── persona/              # PersonaBuilder — extracts user fingerprints
│   ├── vectorstore/          # ChromaDB client wrapper
│   ├── llm/                  # LLM API client (Claude / GPT-4o)
│   └── nigerian/             # Nigerian style adapter
├── scripts/
│   ├── yelp_eda.py           # Exploratory data analysis
│   ├── extract_data.py       # Yelp JSON → parquet
│   └── build_index.py        # Parquet → ChromaDB
├── task_a/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py               # FastAPI app
│   └── simulator.py          # Review simulator logic
└── task_b/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py               # FastAPI app
    └── recommender.py        # Recommendation agent logic
```

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| ROUGE-L (Task A review text) | > 0.25 |
| BERTScore F1 (Task A) | > 0.85 |
| Rating RMSE (Task A) | < 1.0 |
| NDCG@10 (Task B) | > 0.35 |
| Hit Rate@10 (Task B) | > 0.40 |
| Nigerian style fidelity | ✅ explicit adapter |
| Cold-start handling | ✅ content-based fallback |

---

## Solution Paper

See `solution_paper.pdf` (submitted separately).
