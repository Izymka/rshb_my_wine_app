.DEFAULT_GOAL := help

.PHONY: help web-env web web-install web-build web-start web-dev tunnel tunnel-stop scanner-init release-prepare release-publish

PYTHON ?= python3

scanner-init: ## Скачать релиз, проверить SHA256, распаковать и запустить scanner (повторяемо)
	$(PYTHON) scripts/release_scanner.py deploy

release-prepare: ## Пересобрать два релизных архива, SHA256SUMS и инструкцию
	$(PYTHON) scripts/release_scanner.py prepare

release-publish: release-prepare ## С подтверждением загрузить релиз в Google Drive (DRIVE_REMOTE=gdrive:)
	$(PYTHON) scripts/release_scanner.py publish

help: ## Показать команды
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

web-env: ## Создать web/.env из примера, если файла ещё нет
	@test -f web/.env || cp web/.env.example web/.env

web: web-env ## Установить зависимости, собрать и запустить веб-приложение
	$(MAKE) web-install
	$(MAKE) web-build
	$(MAKE) web-start

web-install: ## Установить зависимости из package-lock.json
	cd web && npm ci

web-build: web-env ## Собрать production-версию
	cd web && npm run build

web-start: web-env ## Запустить готовую сборку с настройками web/.env (Node >= 20.12)
	cd web && node --env-file=.env .output/server/index.mjs

web-dev: web-env ## Запустить Nuxt с горячей перезагрузкой
	cd web && npm run dev

tunnel: web-env ## Поднять HTTPS-туннель ngrok к работающему web (Ctrl+C — остановить)
	cd web && node --env-file=.env --input-type=module -e 'import { spawnSync } from "node:child_process"; const r = spawnSync("bash", ["../scripts/tunnel.sh", "start"], { stdio: "inherit" }); process.exit(r.status ?? 1);'

tunnel-stop: ## Остановить только туннель этого проекта
	bash scripts/tunnel.sh stop
