# PersonaRAG

**An Agentic LLM Framework for Behavioural User Modelling and Contextual Recommendation**

Built for the DSN × BCT LLM Agent Challenge 2026. PersonaRAG is a two-task agentic system over the Yelp Open Dataset:

- **Task A — Review Simulator.** Given a user ID and a target business, predicts the star rating and generates a Yelp review faithful to that user's tone, rating bias, and writing style.
- **Task B — Recommendation Agent.** Three-stage agentic pipeline (intent reasoning → semantic retrieval over 150K businesses → LLM reranking) that handles warm users, cold-start users, cross-domain transfer, and multi-turn conversations.

Both tasks share a Nigerian Cultural Adapter that injects authentic Nigerian English and Naija Pidgin into outputs when Nigerian signals are detected.

---

## Architecture

```
                      ┌─────────────────┐
   User ID + Context ─┤  PersonaBuilder ├── ChromaDB (150K businesses)
                      └────────┬────────┘           │
                               │                    │
                ┌──────────────┴──────────────┐    │
                │                             │    │
            Task A pipeline             Task B pipeline
            ├ Rating Predictor          ├ Intent Reasoner (LLM)
            ├ RAG Context selector      ├ Dense Retrieval ◄──┘
            └ Review Generation (LLM)   └ LLM Reranking
                               │
                               ▼
              Nigerian Cultural Adapter (prompt-level)
                               │
                               ▼
                Review + Rating  /  Ranked Recommendations
```

## Final evaluation results

Leave-one-out protocol on 200 held-out Yelp users:

### Task A (n=199)

| Metric          | Value     |
|-----------------|-----------|
| RMSE ↓          | **1.162** |
| ROUGE-L F1 ↑    | **0.133** |
| BERTScore F1 ↑  | **0.850** |
| Mean predicted  | 3.972     |
| Mean true       | 3.769     |

### Task B (n=200)

| Protocol            | NDCG@10 ↑ | HR@10 ↑ |
|---------------------|-----------|---------|
| Open retrieval      | 0.042     | 0.050   |
| Candidate-100       | 0.043     | 0.050   |

Open retrieval ranks the ground-truth business against all 150,346 candidates; candidate-100 reranks within a pre-filtered set of 100. A random baseline over the full corpus yields NDCG@10 ≈ 0.001, so the system performs ~40× better than random.

---

## Repository layout

```
.
├── data/                          # Not committed — Yelp parquet + ChromaDB index
│   ├── raw/                       # Original Yelp JSON files
│   └── processed/                 # users.parquet, reviews.parquet, chroma/, ui_metadata.json
│
├── scripts/
│   ├── extract_data.py            # Raw Yelp JSON → parquet
│   ├── yelp_eda.py                # EDA pipeline
│   ├── build_index.py             # Build 150K-business ChromaDB index
│   ├── build_ui_metadata.py       # Pre-compute dropdown data for UI
│   ├── smoke_test.py              # End-to-end sanity check
│   └── evaluate.py                # Run dual-protocol evaluation
│
├── shared/
│   ├── persona/builder.py         # In-memory user fingerprint extractor
│   ├── vectorstore/store.py       # ChromaDB wrapper
│   ├── llm/client.py              # Gemini / Anthropic / OpenAI factory
│   └── nigerian/adapter.py        # Cultural-prompt injection layer
│
├── task_a/                        # FastAPI service on port 8001
│   ├── main.py
│   ├── simulator.py               # Rating predictor + RAG review generation
│   └── Dockerfile
│
├── task_b/                        # FastAPI service on port 8002
│   ├── main.py
│   ├── recommender.py             # Async 3-stage agentic pipeline
│   └── Dockerfile
│
├── ui/                            # Single-page web client (vanilla JS, no build step)
│   ├── index.html
│   ├── css/                       # Modular CSS: base, dropdown, components, skeleton, animations
│   └── js/                        # Modular ES6: config, utils, dropdown, api, renderers, skeletons, main
│
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.14+
- Docker + Docker Compose
- A Gemini API key (free tier works) or Anthropic/OpenAI key
- ~10 GB free disk for the Yelp dataset and ChromaDB index
- ~8 GB free RAM (4 GB per container at steady state)

### One-time setup

```bash
# 1. Clone and create env file
git clone https://github.com/Mopheshi/bcthack.git
cd bcthack
cp .env.example .env
# Edit .env: set LLM_PROVIDER, GOOGLE_API_KEY (or other provider key)

# 2. Download Yelp Open Dataset to data/raw/
#    https://www.yelp.com/dataset

# 3. Build the data pipeline
python -m scripts.extract_data           # ~3 min
python -m scripts.build_index            # ~2 hours (150K embeddings)
python -m scripts.build_ui_metadata      # ~30s
```

### Running the services

```bash
docker compose up -d
# Wait ~30 seconds for both containers to load persona data into memory
# Then visit:
#   Task A:  http://localhost:8001/docs
#   Task B:  http://localhost:8002/docs
#   UI:      open ui/index.html in your browser
```

### Smoke test

```bash
python -m scripts.smoke_test
```

### Run the evaluation

```bash
python -m scripts.evaluate --task both --n 200 --protocol both
```

---

## Performance characteristics

Measured on a 16 GB consumer laptop (Intel i5, no GPU):

| Stage                              | Task A   | Task B    |
|------------------------------------|----------|-----------|
| Container startup (incl. data load)| 25-35s   | 25-35s    |
| PersonaBuilder lookup (warm)       | <1ms     | <1ms      |
| PersonaBuilder lookup (cold)       | <1ms     | <1ms      |
| Vector retrieval                   | —        | ~50ms     |
| LLM call(s)                        | 4-10s    | 8-20s     |
| **End-to-end (warm user)**         | **5-12s**| **10-25s**|
| **End-to-end (cold start)**        | **4-8s** | **8-18s** |

Task B is slower because it issues two sequential Gemini API calls (intent reasoning, then JSON reranking). Latency is dominated by network round-trip to the LLM provider, not local computation.

### Memory footprint after warm-up

- `users.parquet` (eager): ~150 MB
- `reviews.parquet` + user-to-index map (eager): ~3.5 GB
- ChromaDB index + SentenceTransformer model: ~600 MB

**Total per container at steady state: ~4 GB.**

---

## Key design decisions

**Decoupled rating prediction.** A calibrated statistical predictor anchored to per-user mean and bias produces star ratings, *not* the LLM. This eliminates temperature-induced variance on a regression task and lets the LLM focus on text fidelity.

**Business-level vector index.** Indexing 5.7M reviews caused ChromaDB compaction failures at scale. Indexing 150K businesses with aggregated review snippets is architecturally correct for Task B (recommendations surface businesses) and completed reliably in ~2 hours.

**In-memory persona cache.** The first iteration loaded review history from parquet on every request, with 30-60s latency. The production build loads `reviews.parquet` once at startup and builds a `user_id → row_indices` map using numpy argsort for ~10× faster index construction. Trades 3.5 GB of RAM for sub-millisecond lookups.

**Async-native Task B pipeline.** Intent reasoning and initial retrieval run in parallel via `asyncio.gather`, saving 5-10s on every warm-user request.

**Graceful fallbacks.** If intent reasoning fails → use raw context. If retrieval returns too few candidates → category fallback. If LLM reranking fails → vector-distance ranking with locally synthesised reasons. The system never returns empty recommendations.

**Nigerian cultural layer.** A lexical scanner detects Nigerian Pidgin function words (na, abi, sha, wahala), cultural vocabulary (suya, jollof, mama put, egusi), and city names (Lagos, Abuja, 18 others). When triggered, it appends a cultural instruction block to the LLM system prompt, producing authentic outputs like *"Chai!"*, *"I no go lie"*, *"abeg"*, *"wahala"*.

---

## Configuration

All knobs live in `.env`. Sensible defaults are provided; override as needed.

| Variable                       | Default              | Purpose                                  |
|--------------------------------|----------------------|------------------------------------------|
| `LLM_PROVIDER`                 | `gemini`             | `gemini`, `anthropic`, or `openai`       |
| `LLM_MODEL`                    | `gemini-2.5-flash`   | Model name for the chosen provider       |
| `GOOGLE_API_KEY`               |                      | Required for Gemini                      |
| `ANTHROPIC_API_KEY`            |                      | Required for Anthropic                   |
| `OPENAI_API_KEY`               |                      | Required for OpenAI                      |
| `EAGER_LOAD_REVIEWS`           | `true`               | Load reviews.parquet at startup          |
| `MIN_REVIEWS_FOR_WARM_USER`    | `5`                  | Threshold for warm vs cold persona       |
| `TOP_K_REVIEWS`                | `10`                 | Per-user history depth                   |
| `MAX_RAG_REVIEWS`              | `3`                  | Sample reviews in Task A prompt          |
| `RAG_SNIPPET_LEN`              | `150`                | Max chars per sample review snippet      |
| `TOP_K_RETRIEVE`               | `15`                 | Candidates from vector store             |
| `TOP_K_RETURN`                 | `10`                 | Final recommendations returned           |
| `LLM_REVIEW_TOKENS`            | `600`                | Max tokens for Task A generation         |
| `LLM_RERANK_TOKENS`            | `1500`               | Max tokens for Task B rerank JSON        |

---

## API reference

### Task A — `POST /simulate-review`

```json
{
  "user_id": "yelp_user_abc123",
  "business_id": "yelp_biz_xyz456",
  "product_details": {
    "name": "Mama Put Kitchen",
    "categories": "Nigerian, African, Restaurants",
    "city": "Lagos",
    "state": "LA",
    "stars": 4.2,
    "review_count": 88
  }
}
```

Response:
```json
{
  "predicted_stars": 4.0,
  "review_text": "Chai! Mama Put Kitchen is the real deal...",
  "persona_summary": {
    "is_cold": false,
    "review_count": 28,
    "rating_bias": "generous",
    "style": "medium",
    "nigerian": true,
    "top_cats": ["Restaurants", "Food", "Seafood"]
  },
  "rating_confidence": 0.56
}
```

### Task B — `POST /recommend`

```json
{
  "user_id": "yelp_user_abc123",
  "context": "I want good Nigerian food tonight",
  "conversation_history": [
    {"role": "user", "content": "something spicy"},
    {"role": "assistant", "content": "Indian or Nigerian?"}
  ],
  "top_k": 5
}
```

Response:
```json
{
  "search_intent": "Nigerian restaurants spicy authentic",
  "cold_start": false,
  "persona_summary": { "...": "..." },
  "recommendations": [
    {
      "business_id": "abc...",
      "name": "Ify's Nigerian Cuisine",
      "categories": "African, Restaurants",
      "city": "Antioch",
      "stars": "4.5",
      "score": 0.98,
      "reason": "Explicitly Nigerian cuisine with a 4.5-star rating..."
    }
  ]
}
```

### Auxiliary endpoints

- `GET /health` — readiness probe
- `GET /metadata` — dropdown data for the UI (top users, businesses, cities, states)
- `GET /docs` — interactive Swagger UI

---

## License

MIT. Yelp dataset is subject to its own [license terms](https://www.yelp.com/dataset/download).

## Citation

If you use this work, please cite:

```bibtex
@misc{personarag2026,
  title  = {PersonaRAG: An Agentic LLM Framework for Behavioural User Modelling and Contextual Recommendation},
  author = {Ndachimya Magaji Edward},
  year   = {2026},
  url    = {https://github.com/Mopheshi/bcthack}
}
```

---

## Acknowledgements

Built for the DSN × BCT LLM Agent Challenge 2026 (Hackathon 3.0). Uses the Yelp Open Dataset, Google Gemini, ChromaDB, sentence-transformers, and FastAPI.