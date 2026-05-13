# ── DSN x BCT Hackathon — Build Commands ─────────────────────────────────────

.PHONY: build up down logs test eval smoke

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
