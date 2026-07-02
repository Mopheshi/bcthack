# PersonaRAG

**An Agentic LLM Framework for Behavioural User Modelling and Contextual Recommendation**

Built for the DSN × BCT LLM Agent Challenge 2026. PersonaRAG is a two-task agentic system over the Yelp Open Dataset, deployed as a live web service.

- **Task A — Review Simulator.** Given a user ID and a target business, predicts the star rating and generates a Yelp review faithful to that user's tone, rating bias, and writing style.
- **Task B — Recommendation Agent.** Three-stage agentic pipeline (intent reasoning → semantic retrieval over 150K businesses → LLM reranking) that handles warm users, cold-start users, cross-domain transfer, and multi-turn conversations.

Both tasks share a Nigerian Cultural Adapter that injects authentic Nigerian English and Naija Pidgin into outputs when Nigerian signals are detected.

**Live application:** https://bcthack.vancus.app
**API endpoint:** https://bcthack.vancus.app/api
**Source code:** https://github.com/Mopheshi/bcthack

---

## Architecture

PersonaRAG runs as a **single consolidated FastAPI service**. Task A and Task B share one PersonaBuilder and one vector index, so each is loaded into memory exactly once.

```
                       ┌─────────────────┐
   User ID + Context ──┤  PersonaBuilder ├── Persona Store (precomputed)
                       └────────┬────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
             Task A pipeline             Task B pipeline
             ├ Rating Predictor          ├ Intent Reasoner (LLM)
             ├ RAG Context selector      ├ Dense Retrieval ──► ChromaDB
             └ Review Generation (LLM)   └ LLM Reranking        (150K biz)
                                │
                                ▼
               Nigerian Cultural Adapter (prompt-level)
                                │
                                ▼
                 Review + Rating  /  Ranked Recommendations
```

A key architectural principle is the **separation of build time from runtime**. The raw Yelp corpus and intermediate tables are processed once, offline, into two compact artefacts — a precomputed persona store and a ChromaDB business index. The running service loads only those. It never touches the multi-gigabyte review corpus.

---

## Final evaluation results

Leave-one-out protocol on 200 held-out Yelp users.

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

Open retrieval ranks the ground-truth business against all 150,346 candidates — the strictest possible protocol, corresponding directly to deployment behaviour. A random baseline over the full corpus yields NDCG@10 ≈ 0.001, so the system performs ~40× better than random.

The evaluation harness also implements a **candidate-100** protocol (`--protocol candidate`), in which the agent reranks within a fixed 100-business pool with the ground truth injected. This isolates rerank quality from large-scale retrieval difficulty; because the pool is far smaller than the full corpus, candidate-100 scores would be expected to exceed the open-retrieval figures. A full candidate-100 run over the 200-user test set is left as a reproducible exercise — the protocol is in the harness but not run here, owing to the LLM API budget it requires.

---

## Repository layout

```
.
├── data/                          # Not committed — Yelp parquet + ChromaDB index
│   ├── raw/                       # Original Yelp JSON files (build-time only)
│   └── processed/                 # persona_store.parquet, chroma/, ui_metadata.json
│
├── scripts/
│   ├── extract_data.py            # Raw Yelp JSON → parquet
│   ├── yelp_eda.py                # EDA pipeline
│   ├── build_index.py             # Build 150K-business ChromaDB index
│   ├── build_persona_store.py     # Precompute the persona store (build-time)
│   ├── build_ui_metadata.py       # Pre-compute dropdown data for the UI
│   ├── smoke_test.py              # End-to-end sanity check
│   └── evaluate.py                # Run dual-protocol evaluation
│
├── shared/
│   ├── persona/builder.py         # Loads the precomputed persona store
│   ├── vectorstore/store.py       # ChromaDB wrapper
│   ├── llm/client.py              # Gemini / Anthropic / OpenAI factory
│   └── nigerian/adapter.py        # Cultural-prompt injection layer
│
├── task_a/
│   └── simulator.py               # Rating predictor + RAG review generation
│
├── task_b/
│   └── recommender.py             # Async 3-stage agentic pipeline
│
├── app/
│   └── main.py                    # Consolidated FastAPI service (both tasks)
│
├── ui/                            # Single-page web client (vanilla JS, no build step)
│   ├── index.html
│   ├── css/                       # Modular CSS
│   └── js/                        # Modular ES6: config, utils, dropdown, api, renderers, main
│
├── Dockerfile                     # Single consolidated image
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.14+
- Docker + Docker Compose
- A Gemini API key
- ~10 GB free disk for the Yelp dataset and ChromaDB index

### One-time setup

```bash
# 1. Clone and create env file
git clone https://github.com/Mopheshi/bcthack.git
cd bcthack
cp .env.example .env
# Edit .env: set LLM_PROVIDER and GOOGLE_API_KEY (or another provider key)

# 2. Install dependencies
#    Runtime only (what the deployed service needs — lean, no PyTorch):
pip install -r requirements.txt
#    Build/eval extras (adds sentence-transformers + bert-score, for the
#    offline pipeline and the evaluation harness only):
pip install -r requirements-dev.txt

# 3. Download the Yelp Open Dataset into data/raw/
#    https://www.yelp.com/dataset

# 4. Build the data pipeline (build-time, runs once)
python -m scripts.extract_data           # ~3 min   — raw JSON → parquet
python -m scripts.build_index            # ~2 hours — 150K business embeddings
python -m scripts.build_persona_store    # ~2-4 min — precompute the persona store
python -m scripts.build_ui_metadata      # ~30s     — UI dropdown data
```

Step 4 is the build-time stage. After it completes, the running service
needs only `data/processed/persona_store.parquet`, `data/processed/chroma/`,
and `data/processed/ui_metadata.json`.

### Running locally

The system is one service. Run it natively:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Or with Docker:

```bash
docker compose up
```

Then visit:
- API docs: http://localhost:8080/docs
- Health:   http://localhost:8080/health
- UI:       open `ui/index.html`, or http://localhost:8081 under Docker

Startup takes roughly 5–10 seconds — the service loads the persona store
and ChromaDB index, nothing more.

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

Measured against the live Cloud Run deployment, warm instance:

| Stage                       | Task A           | Task B           |
|-----------------------------|------------------|------------------|
| Persona lookup              | O(1), <1 ms      | O(1), <1 ms      |
| Rating prediction           | <1 ms            | —                |
| Dense retrieval (ChromaDB)  | —                | tens of ms       |
| LLM call(s)                 | 1 call           | 2 calls          |
| **End-to-end (typical)**    | **1.4–2.1 s**    | **2.3–2.7 s**    |

Task B is slower because it issues two sequential LLM calls (intent
reasoning, then JSON reranking) where Task A issues one. In both tasks the
LLM round-trip dominates; local computation contributes a few milliseconds.

### Runtime footprint

The deployed container holds only the derived artefacts:

- `persona_store.parquet` — ~84 MB, loaded into an in-memory dict
- ChromaDB index + ONNX MiniLM-L6-v2 encoder — ~1.5 GB on disk

The raw review corpus is never loaded at runtime.

The runtime image is deliberately lean: it installs only `requirements.txt`,
which carries **no PyTorch**. The dense retriever runs the ONNX build of
MiniLM-L6-v2 bundled with ChromaDB, so `sentence-transformers` and
`bert-score` (both of which pull in torch) are confined to
`requirements-dev.txt` and used only by the offline build and evaluation
scripts. This keeps the deployed image small — faster cold starts and less
Artifact Registry storage.

Repeated identical requests are served from a small in-process LRU
(`RESPONSE_CACHE_SIZE`, default 256), so a duplicate call costs zero LLM
round-trips.

---

## Key design decisions

**Decoupled rating prediction.** A calibrated statistical predictor anchored to per-user mean and bias produces star ratings, *not* the LLM. This eliminates temperature-induced variance on a regression task and lets the LLM focus on text fidelity.

**Business-level vector index.** Indexing 5.7M reviews caused ChromaDB compaction failures at scale. Indexing 150K businesses with aggregated review snippets is architecturally correct for Task B (recommendations surface businesses) and completed reliably in ~2 hours.

**Build-time persona precomputation.** An earlier design built personas from the review corpus inside the request path, which forced the runtime to hold several gigabytes of review data in memory and made startup slow and fragile. The production build precomputes one fingerprint per warm user — including representative review snippets — into a compact persona store (~84 MB). Runtime persona lookup is an O(1) dict read; the raw corpus is never touched. This is the change that made serverless deployment practical.

**Consolidated single service.** Task A and Task B share the PersonaBuilder and vector index, so they run as one FastAPI application rather than two containers. One copy of the data in memory, one cold start, one origin.

**ONNX embedding runtime.** The dense retriever uses all-MiniLM-L6-v2 via an ONNX-runtime implementation rather than the PyTorch-backed one. Identical 384-dimensional embeddings, fully compatible with the prebuilt index, without the PyTorch memory overhead.

**Async-native Task B pipeline.** Intent reasoning and initial retrieval run in parallel via `asyncio.gather`, saving wall-clock time on every request.

**Graceful fallbacks.** If intent reasoning fails → use raw context. If retrieval returns too few candidates → category fallback. If LLM reranking fails → vector-distance ranking with locally synthesised reasons. The LLM client also retries transient API errors with bounded backoff. The system never returns empty recommendations.

**Nigerian cultural layer.** A lexical scanner detects Nigerian Pidgin function words (na, abi, sha, wahala), cultural vocabulary (suya, jollof, mama put, egusi), and city names (Lagos, Abuja, and 18 others). When triggered, it appends a cultural instruction block to the LLM system prompt, producing authentic outputs like *"Chai!"*, *"I no go lie"*, *"abeg"*, *"wahala"*.

---

## Configuration

All knobs live in `.env`. Sensible defaults are provided; override as needed.

| Variable                    | Default                  | Purpose                                 |
|-----------------------------|--------------------------|-----------------------------------------|
| `LLM_PROVIDER`              | `gemini`                 | `gemini`, or `openai` or `anthropic`    |
| `LLM_MODEL`                 | `gemini-3.1-flash-lite`  | Model name for the chosen provider      |
| `LLM_THINKING_LEVEL`        | `minimal`                | Reasoning level for Gemini 3.x models   |
| `GOOGLE_API_KEY`            |                          | Required for Gemini                     |
| `OPENAI_API_KEY`            |                          | Required for OpenAI                     |
| `ANTHROPIC_API_KEY`         |                          | Required for Anthropic                  |
| `MIN_REVIEWS_FOR_WARM_USER` | `5`                      | Threshold for warm vs cold persona      |
| `PERSONA_SAMPLES_PER_USER`  | `3`                      | Review snippets stored per persona      |
| `PERSONA_SNIPPET_LEN`       | `220`                    | Max chars per stored review snippet     |
| `MAX_RAG_REVIEWS`           | `3`                      | Sample reviews in the Task A prompt     |
| `RAG_SNIPPET_LEN`           | `150`                    | Max chars per RAG snippet in the prompt |
| `TOP_K_RETRIEVE`            | `15`                     | Candidates from the vector store        |
| `TOP_K_RETURN`              | `10`                     | Final recommendations returned          |
| `LLM_REVIEW_TOKENS`         | `600`                    | Max tokens for Task A generation        |
| `LLM_RERANK_TOKENS`         | `1500`                   | Max tokens for Task B rerank JSON       |
| `LLM_MAX_RETRIES`           | `2`                      | Transient-error retry budget            |
| `RESPONSE_CACHE_SIZE`       | `256`                    | Bounded LRU of API responses; `0` disables. Collapses identical requests to zero LLM calls |

---

## API reference

The consolidated service exposes both tasks under one origin.

### Task A — `POST /api/simulate`

```json
{
  "user_id": "yelp_user_abc123",
  "business_id": "yelp_biz_xyz456",
  "product_details": {
    "name": "Mama Put Kitchen",
    "categories": "Nigerian, African, Restaurants",
    "city": "Abuja",
    "state": "FC",
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

### Task B — `POST /api/recommend`

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

## Deployment

The production system runs on Google Cloud Run, with the web client served
by Firebase Hosting at `bcthack.vancus.app`. Firebase rewrites proxy
`/api/*` to the Cloud Run service, so the UI and API share a single origin.
The Gemini API key is held in Google Secret Manager and injected at deploy
time, never baked into the image.

Deploy with the codified, cost-optimal target:

```bash
make deploy                    # SERVICE/REGION/SECRET overridable, e.g. make deploy REGION=europe-west1
make cost-check                # show the live cost-relevant settings
```

The service is tuned to **bill only while a request is being handled**:

| Setting            | Value  | Why                                                        |
|--------------------|--------|------------------------------------------------------------|
| `--min-instances`  | `0`    | No always-on instance → **zero cost when idle**            |
| `--cpu-throttling`  | on     | CPU allocated only during request processing               |
| `--max-instances`  | `3`    | Caps cost blast radius of a traffic/retry spike            |
| `--memory / --cpu` | `8Gi / 2` | Measured floor — startup OOMs at both 2Gi and 4Gi (the in-memory persona store dominates) |
| `--concurrency`    | `80`   | I/O-bound (LLM-dominated) work → many requests per instance |

The decisive saving is **`min-instances 0`**: for low-traffic use the old
always-on instance *was* essentially the entire bill, and it is now gone —
you pay only while a request is being served. Per-instance size is a minor
factor by comparison, so memory stays at the proven `8Gi`; dropping it to
`4Gi` first requires shrinking the persona store's in-memory footprint.

Because the app initialises heavy state in a background thread and exposes a
`/healthz/startup` probe, scale-to-zero cold starts stay in the ~5–10 s range.
Two `make` targets are provided: `make tune` applies just these settings to the
running revision (no rebuild, no data needed), while `make deploy` rebuilds the
image from source *and* applies them.

---

## License

[MIT](LICENSE). The Yelp dataset is subject to its own [license terms](https://www.yelp.com/dataset/download).

## Citation

```bibtex
@misc{personarag2026,
  title  = {PersonaRAG: An Agentic LLM Framework for Behavioural User Modelling and Contextual Recommendation},
  author = {Ndachimya Magaji Edward},
  year   = {2026},
  url    = {https://github.com/Mopheshi/bcthack}
}
```

## Acknowledgements

Built for the DSN × BCT LLM Agent Challenge 2026 (Hackathon 3.0). Uses the Yelp Open Dataset, Google Gemini, ChromaDB, and FastAPI.