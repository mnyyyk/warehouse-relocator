SHELL := /bin/bash

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

bash-pg:
	docker exec -it wh_optim_pg bash

bash-redis:
	docker exec -it wh_optim_redis sh

dev:            ## backend & frontend 開発同時起動
	cd backend && poetry run uvicorn app.main:app --reload --port 8000 &
	cd frontend && npm run dev &
	wait