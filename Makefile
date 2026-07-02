# ── DSN x BCT Hackathon — Build Commands ─────────────────────────────────────

.PHONY: build up down logs test eval smoke deploy tune cost-check registry-cleanup

# ── Cloud Run deploy — cost-optimal, scale-to-zero ───────────────────────────
# Region/service/secret can be overridden on the command line, e.g.
#   make deploy REGION=europe-west1
SERVICE ?= bcthack-api
REGION  ?= us-central1
# The Gemini key: env var GOOGLE_API_KEY is fed from the Secret Manager
# secret `gemini-api-key` (this is how the live service is already wired).
SECRET_ENV  ?= GOOGLE_API_KEY
SECRET_NAME ?= gemini-api-key
# `gcloud run deploy --source .` pushes images to this Artifact Registry repo.
REPO    ?= cloud-run-source-deploy

# Full source deploy — rebuilds the image (ships code changes) AND applies the
# cost-optimal, scale-to-zero settings below. REQUIRES the build artefacts to be
# present locally, because the Dockerfile bakes them in:
#   data/processed/persona_store.parquet, ui_metadata.json, chroma/
# Run this from your build machine/CI where the data pipeline output exists —
# a checkout without that data will fail at the COPY step.
#
# Settings that bill ONLY while a request is being handled:
#   --min-instances 0   no always-on instance → zero cost when idle
#   --cpu-throttling    CPU allocated only during request processing (default,
#                       stated explicitly so a future change can't silently
#                       flip it to always-allocated billing)
#   --max-instances 3   caps blast radius of a traffic/retry spike
#   --memory 8Gi --cpu 2  measured floor: startup OOMs at 2Gi AND 4Gi (the
#                       in-memory persona store dominates), so this stays at the
#                       proven size. The real saving here is min-instances 0, not
#                       per-instance size. To shrink RAM you must first reduce the
#                       persona-store in-memory footprint, then drop this to 4Gi/1.
deploy:
	gcloud run deploy $(SERVICE) \
	  --source . \
	  --region $(REGION) \
	  --memory 8Gi \
	  --cpu 2 \
	  --min-instances 0 \
	  --max-instances 3 \
	  --concurrency 80 \
	  --cpu-throttling \
	  --timeout 300 \
	  --allow-unauthenticated \
	  --set-secrets $(SECRET_ENV)=$(SECRET_NAME):latest

# Apply ONLY the cost-optimal settings to the existing revision — no rebuild,
# no data needed. Safe to run anywhere; this is the change that stops idle billing.
tune:
	gcloud run services update $(SERVICE) \
	  --region $(REGION) \
	  --memory 8Gi \
	  --cpu 2 \
	  --min-instances 0 \
	  --max-instances 3 \
	  --concurrency 80 \
	  --cpu-throttling

# Show the cost-relevant knobs on the live service.
cost-check:
	gcloud run services describe $(SERVICE) --region $(REGION) \
	  --format="yaml(spec.template.spec.containerConcurrency, spec.template.spec.containers[0].resources, spec.template.metadata.annotations)"

# Apply the Artifact Registry cleanup policy: keep the 3 most recent images,
# delete anything older than 30 days. Stops old ~1.5 GB revisions accumulating.
registry-cleanup:
	gcloud artifacts repositories set-cleanup-policies $(REPO) \
	  --location $(REGION) \
	  --policy artifact-cleanup-policy.json

# Build both containers
build:
	docker-compose build --no-cache

# Start both services
up:
	docker-compose up -d
	@echo "Task A: http://localhost:8001/docs"
	@echo "Task B: http://localhost:8002/docs"

# Stop both services
down:
	docker-compose down

# Stream logs from both containers
logs:
	docker-compose logs -f

# Smoke test (local, no Docker)
smoke:
	python -m scripts.smoke_test

# Full evaluation (local, no Docker)
eval:
	python -m scripts.evaluate --task both --n 200

# Test endpoints after docker-compose up
test:
	@echo "--- Task A health ---"
	curl -s http://localhost:8001/health | python -m json.tool
	@echo "\n--- Task B health ---"
	curl -s http://localhost:8002/health | python -m json.tool
	@echo "\n--- Task A simulate ---"
	curl -s -X POST http://localhost:8001/simulate-review \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"cold_user_test","business_id":"biz_001","product_details":{"name":"Mama Put Kitchen","categories":"Nigerian, African, Restaurants","city":"Lagos","state":"LA","stars":4.2,"review_count":88}}' \
	  | python -m json.tool
	@echo "\n--- Task B recommend ---"
	curl -s -X POST http://localhost:8002/recommend \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"cold_user_test","context":"Looking for good Nigerian food tonight","top_k":3}' \
	  | python -m json.tool
