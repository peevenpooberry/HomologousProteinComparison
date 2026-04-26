PORT ?= 8050
NAME ?= "peevenpooberry/conserved-docking-dashboard:1.0.0"

filter:
	docker ps --filter "expose=${PORT}" --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"

build:
	docker build -t ${NAME} .

run: build
	docker run -d -p 8050:8050 ${NAME}

stop:
	@CONTAINERS=$(shell docker ps -q --filter "ancestor=${NAME}"); \
	if [ -n "$$CONTAINERS" ]; then docker stop $$CONTAINERS; else echo "No running containers for ${NAME}"; fi

all: stop run

compose-down:
	docker compose down

compose-up:
	docker compose up --build -d

compose: compose-down compose-up

compose-down-staging:
	docker compose -f docker-compose-staging.yml down

compose-up-staging:
	docker compose -f docker-compose-staging.yml up --build -d

compose-staging: compose-down-staging compose-up-staging