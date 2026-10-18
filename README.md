# Сканер российских вин для платформы «Своё Вино»

## Быстрый запуск теста

### Подготовка сервера

Выберите один из трёх вариантов. Для собственного сервера нужны Linux-машина с NVIDIA Turing
или новее (от 6 ГБ VRAM), драйвер r570+, Docker Compose и NVIDIA Container Toolkit.
На время установки нужно около 40 ГБ свободного места и интернет.

#### 1. Ручная подготовка

Релизные файлы лежат в [папке Google Drive](https://drive.google.com/drive/u/0/folders/1xZ1ye2XNLb3Zd6witTz5ZjVWpEfFdTmQ).
Для загрузки скриптом можно использовать команду `gdown` (версии 6.1 или новее (`pipx install gdown`)), 
либо скачайте файлы вручную в папку data/release 

```bash
git clone https://github.com/Izymka/rshb_my_wine_app.git
cd rshb_my_wine_app
mkdir -p data/release
gdown 'https://drive.google.com/drive/folders/1xZ1ye2XNLb3Zd6witTz5ZjVWpEfFdTmQ' -O data/release
(cd data/release && sha256sum -c SHA256SUMS)
tar -xf data/release/wine_scanner_artifacts.tar
tar -xf data/release/siglip2.tar
cp .env.example .env
# При наличии ключа Yandex впишите его в .env; без ключа задайте WINE_VLM=0.
docker compose up -d --build scanner
curl -s http://localhost:8080/health | jq .
```

#### 2. Команда `make scanner-init`

Установите `make`, Python 3.11+ и `gdown` 6.1+; для закрытой папки вместо `gdown` нужен
настроенный `rclone`.

```bash
git clone https://github.com/Izymka/rshb_my_wine_app.git
cd rshb_my_wine_app
make scanner-init
```

Команда скачивает и проверяет архивы, распаковывает их, создаёт `.env` при необходимости,
запускает scanner и проверяет `/health`. Повторный запуск использует установленные артефакты.
Для обоих вариантов ожидаются `status: ok`, версии `994899aea9e7` / `8ba1722e86f7`,
порог `0.533` и `devices.ocr: gpu:0`. Первая сборка занимает около 20–40 минут.

#### 3. Готовый сервер

Можно сразу использовать [сервер разработчика](http://212.46.3.67:58080/v1/eval/predict).
Для запуска скрипта нужен только код проекта; модели и Docker не требуются:

```bash
git clone https://github.com/Izymka/rshb_my_wine_app.git
cd rshb_my_wine_app
```

Доступность стенда зависит от машины разработчика.

> При тестах с удалённым сервером время ответа включает сетевые издержки. Учитывайте их при
> сравнении скорости с локальным запуском.

### Запуск скрипта валидации

Скрипту нужны `bash`, `curl`, `jq`, `awk`, `mktemp` и `sha256sum` (или `shasum`). Команды ниже
выполняются из корня клонированного репозитория.

#### С локальным сервером

```bash
./data/eval/participant_test.sh --images-dir ./data/eval/queries \
  --manifest ./data/eval/queries.tsv \
  --endpoint http://127.0.0.1:8080/v1/eval/predict \
  --output ./data/eval/output/predictions_local_$(date +%Y%m%d_%H%M%S).jsonl
```

#### С удалённым сервером

```bash
./data/eval/participant_test.sh --images-dir ./data/eval/queries \
  --manifest ./data/eval/queries.tsv \
  --endpoint http://212.46.3.67:58080/v1/eval/predict \
  --output ./data/eval/output/predictions_remote_$(date +%Y%m%d_%H%M%S).jsonl
```

> Результаты каждого прогона находятся в `data/eval/output/predictions_*.jsonl` на машине,
> где запущен скрипт. В каждой строке — `query_id`, `image_path`, `image_sha256`,
> `predicted_slug` и время ответа `latency_ms`.

Разбор кропов и разметки: [ноутбук](notebooks/02_rtdetr_crop_audit.ipynb).
Переобучение, padding и OCR/XFeat: [эксперименты](notebooks/03_pipeline_rebuild_experiments.ipynb).
Обучение решающего слоя: [ноутбук 06](notebooks/06_decider_eda_and_training.ipynb).
Инференс v7 на тесте, ошибки и важность признаков: [ноутбук 08](notebooks/08_inference_test_eval_errors_importance.ipynb).
Облачные шаги (hybrid OCR, VLM-судья) на тесте: [ноутбук 07](notebooks/07_cloud_steps_hybrid_ocr_and_vlm_judge.ipynb).

Фотография этикетки → карточка вина из каталога платформы, калиброванная уверенность, честный
отказ с похожими и аналогами, если вина в каталоге нет. Хакатон РСХБ, сентябрь 2026.

Как это устроено в двух словах: задача решается как поиск по картинке, а не как классификация.
Визуальная ветка находит **семью** вина (линейку, винодельню), текстовая — **конкретную
карточку внутри семьи**: близнецы одной серии различаются словами «красное / белое / брют /
полусладкое», а не рисунком этикетки.

## Быстрый старт

### Docker (машина с NVIDIA)

```bash
cp .env.example .env          # ключи облачного OCR и LLM, если используются; без них тоже работает
docker compose up --build     # сервис на :8080, интерфейс на :3000, PostgreSQL для аккаунтов
curl -s localhost:8080/health | jq .devices   # визуальные модели — cuda, PaddleOCR — gpu:0
```

В `./models` должны лежать артефакты (см. «Сборка артефактов»), в `./data/catalog` — каталог.

### Аккаунты и отзывы

Сканер и полка работают без входа: полка хранится в браузере и доступна без сети. Аккаунт
нужен, чтобы публиковать отзывы и держать полку в облаке. Сервер Nuxt хранит в PostgreSQL
пользователей, сессии, облачную полку и отзывы (`web/server/utils/db.ts`); схема создаётся сама
при первом запросе к базе. Без `NUXT_DATABASE_URL` вход и отзывы просто не показываются.

- Вход — по номеру телефона и четырёхзначному коду. СМС-провайдер в этой версии не подключён:
  сервер возвращает код в ответе, и интерфейс показывает его на экране
  (`NUXT_AUTH_DEMO_CODES=0` — не отдавать код). Кнопки Яндекс ID и VK ID пока неактивны.
- Профиль: телефон, аватар, имя, email; при первом входе — имя и подтверждение 18+.
- После входа полка устройства сливается с аккаунтом: по каждому вину побеждает более поздняя
  правка, удаления тоже синхронизируются.
- Отзыв (оценка, блюда, заметка) публикуется из шторки отзыва от имени или анонимно; карточка
  вина показывает, сколько человек держат его на полке, долю «понравилось», с чем его пьют и
  последние отзывы.

### Отдельный деплой фронтенда

GitHub Actions → GHCR → Node.js-контейнер Nuxt/Nitro (`.github/workflows/deploy-web.yml`);
scanner может оставаться на локальном сервере разработки с GPU.

### Локально (uv, Python 3.12)

```bash
uv sync --extra api --extra rerank --extra ocr --extra decide
WINE_INDEX=models/index_platform_sq_v3_site_1810 WINE_DECIDER=models/decider_site1810_abd \
  uv run uvicorn api.main:app --port 8080 --env-file .env     # .env необязателен
cd web && npm install && NUXT_SCANNER_URL=http://127.0.0.1:8080 npm run dev   # интерфейс на :3000
# аккаунты и отзывы: PostgreSQL в Docker на 127.0.0.1:5432 (make db, остановить — make db-stop);
# адрес базы берётся из web/.env (NUXT_DATABASE_URL, см. web/.env.example)
make db && make web-dev
```

Первый запрос после старта сервис делает сам (прогрев), поэтому поднимается он около минуты;
`/health` показывает `load_seconds` и `warmup_seconds`.

Переключатель `WINE_EVAL_REFUSE=0` заставляет и ниже порога отдавать лучшего кандидата.

## Переменные окружения

| Переменная | Значение по умолчанию | Смысл |
|---|---|---|
| `WINE_INDEX` | `models/index` | Индекс каталога (в Docker — `models/index_platform_sq_v3_site_1810`: 2272 карточки, каталог сайта на 18.10.2026) |
| `WINE_DECIDER` | `models/decider` | Решающий слой (в Docker — `models/decider_site1810_abd`, обучен под OCR из `WINE_OCR_DET` и `WINE_OCR_REC`) |
| `WINE_OCR` | `paddle` | `yandex` — Vision с откатом на PaddleOCR; `hybrid` — Vision только при слабом PaddleOCR; нужны `YANDEX_OCR_API_KEY`, `YANDEX_FOLDER_ID` |
| `WINE_OCR_DET` | `PP-OCRv5_mobile_det` (в Docker и `.env.example` — `PP-OCRv6_medium_det`) | Детектор строк PaddleOCR. PP-OCRv6 medium на тесте 18.10: текст R@5 0.82 → 0.89, верных при пороге 0.4 87 → 89 из 100, общее время не растёт. Имя детектора входит в ключ кэша OCR |
| `WINE_OCR_REC` | `eslav_PP-OCRv5_mobile_rec` (в Docker и `.env.example` — `models/ocr_rec/PP-OCRv6_medium_rec_ru`) | Распознаватель строк: имя модели PaddleOCR или папка модели; папка с `inference.onnx` читается через onnxruntime (`onnxruntime-gpu` в окружении OCR). Русский PP-OCRv6 medium с решающим слоем v8 на тесте 18.10: верных при пороге 0.4 89 → 94 из 100, ложных приёмов те же 14 из 102, общее время p50 0.98 → 1.03 с. Решающий слой обучен под распознаватель: v8 — под русский, v7 — под eslav. Входит в ключ кэша OCR |
| `WINE_OCR_PYTHON`, `WINE_OCR_DEVICE` | —, `gpu:0` | Интерпретатор отдельного окружения с `paddlepaddle-gpu` и устройство PaddleOCR (`src/wine_scanner/ocr/worker.py`); в Docker заданы в образе (`/opt/ocr`). Пусто — OCR на CPU в отдельном процессе (`WINE_OCR_PROCESS=0` — в потоке сервиса) |
| `WINE_VLM` | `0` (в Docker `1`) | `1` — VLM-судья между близнецами одной винодельни; работает только в `/v1/eval/predict` |
| `WINE_EVAL_LOG`, `WINE_EVAL_LOG_DIR` | `0`, `data/eval_requests` | `WINE_EVAL_LOG=1` включает журнал всех POST-запросов `/v1/eval/predict`. В Docker каталог `./data/eval_requests` смонтирован в контейнер и сохраняет JSONL и оригинальные фото между перезапусками. Просмотр: `http://localhost:8080/v1/eval/logs` (или `/v1/eval/logs/data` для JSON). |
| `WINE_EVAL_LOG_USER`, `WINE_EVAL_LOG_PASSWORD` | пусто | Если оба заданы в `.env`, страница журнала, данные и фотографии требуют HTTP Basic Auth. Для доступа извне используйте HTTPS. |
| `WINE_VLM_SCAN` | `0` | `1` — звать судью и в продуктовом `/scan` (ложные приёмы незнакомых 0.07 → 0.34) |
| `WINE_VLM_OPTIONS`, `WINE_VLM_TRIGGER`, `WINE_VLM_VERIFY`, `WINE_VLM_TIMEOUT` | `family`, `family`, `1`, `4` | Режим судьи family-verify: показывать карточки своей винодельни, звать при соседках в выдаче, смену лидера принимать только при подтверждении текстом этикетки |
| `WINE_VLM_DEADLINE`, `WINE_VLM_COOLDOWN`, `WINE_VLM_MAX_FAILURES` | `7`, `60`, `2` | Фолбэк на решающий слой: судья отвечает только до 7 с от начала запроса (осталось меньше секунды — не зовём); после 2 сбоев облака подряд — пауза 60 с. Счётчики пропусков — `llm.judge_skipped` в `/health` |
| `WINE_VLM_CHECK` | `1` | Сверять выбор судьи с текстом, который он сам прочитал: противоречие по цвету, сорту или сладости — выбор переходит к показанной соседке, которую текст подтверждает |
| `WINE_VLM_OCR` | `off` | Строки PaddleOCR судье: `hint` — в промпт подсказкой, `hint-verify` — ещё и для подтверждения смены лидера (на стенде не помогло) |
| `WINE_GRAPE` | `0` | `1` — сорт этикетки против поля `grapes` в правиле выбора внутри семьи без судьи |
| `WINE_VLM_PROVIDER`, `WINE_VLM_FALLBACK` | `yandex`, — | Провайдер судьи и откат: `yandex` (Yandex AI Studio, без VPN) / `openai` (любой OpenAI-совместимый чат: `WINE_VLM_BASE_URL`, `WINE_VLM_MODEL`, `WINE_VLM_API_KEY`) |
| `YANDEX_LLM_API_KEY`, `YANDEX_FOLDER_ID` | — (ключ OCR), — | Ключ AI Studio: роль `ai.languageModels.user`, область ключа `yc.ai.languageModels.execute`; модели `YANDEX_VLM_MODEL` (`qwen3.6-35b-a3b`) и `YANDEX_LLM_MODEL` (`yandexgpt-5-lite`) |
| `WINE_PORTAL` | `1` | `GET /catalog/{slug}` добавляет к карточке народный рейтинг, крепость, температуру подачи и гастрономию с портала vino-svoe.ru (кэш 6 ч); `0` — без портала |
| `WINE_SOMMELIER` | `0` | `1` — цифровой сомелье (`/sommelier`); провайдер `WINE_LLM_PROVIDER` (`yandex` / `openai` — `WINE_LLM_*` или те же `WINE_VLM_*`), откат `WINE_LLM_FALLBACK` |
| `WINE_THRESHOLD` | порог из `meta.json` решающего слоя (0.533); в `docker-compose.yml` — `0.4` | Порог отказа `/scan`. 0.40 вместо 0.533: больше ответов на ужатых кадрах ценой ложных приёмов (OOF train 5.8 → 7.1 %, тест 9.8 → 12.7 %) |
| `WINE_SIBLING` | `0` | `1` — правило выбора внутри семьи по тексту поверх решающего слоя; выключено 29.09: на стенде train 176 → 162 верных из 185 |
| `WINE_GUARD` | `warn` | Защита от близнеца: `warn` возвращает карточку с предупреждением; `twin` / `strict` отказывают, `off` выключает правило |
| `WINE_RERANK_CANDIDATES` | `25` | Окно ре-ранкинга локальными признаками; признаки решающего слоя собираются с тем же окном 25 |
| `WINE_VISUAL_CANDIDATES`, `WINE_TEXT_CANDIDATES` | `100`, `50` | Ширина визуальной и текстовой веток |
| `WINE_FAMILY` | `1` | Расширение кандидатов роднёй по винодельне |
| `WINE_EVAL_REFUSE` | `1` | Политика eval-ручки на незнакомом вине |
| `NUXT_SCANNER_URL` | `http://127.0.0.1:8080` | Адрес scanner для серверного прокси Nuxt; задаётся при запуске контейнера |
| `NUXT_DATABASE_URL` | пусто (в Docker — сервис `postgres`) | PostgreSQL для аккаунтов, облачной полки и отзывов; пусто — без входа |
| `NUXT_AUTH_DEMO_CODES` | `1` | Код входа показывается на экране, пока СМС-провайдер не подключён; `0` — не отдавать код |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `wine` | Учётные данные сервиса `postgres` в docker-compose |
| `POSTGRES_PORT` | `5432` | Порт базы на хосте (только 127.0.0.1) для web, запущенного вне compose |

Без единого внешнего ключа сервис полностью работоспособен: PaddleOCR локально, судья и
сомелье выключены.

## Сборка артефактов

Каталог платформы (дамп CSV + медиа) → индекс → whitening → признаки → решающий слой:

```bash
uv run python scripts/build_catalog.py                      # data/catalog/: catalog.csv + originals/<slug>.png
uv run python scripts/check_catalog_crops.py --write-overrides  # вырезка этикетки на эталонах -> label_overrides.csv
uv run python scripts/build_index.py --catalog platform --detect cascade --fit pad \
  --model models/siglip2 --weights models/rtdetr_label_recrop_bf16 --local-preprocess clahe2 \
  --out models/index_platform_sq_v3                          # ~1 ч на RTX 4060 Ti
uv run python scripts/fit_whitening.py --index models/index_platform_sq_v3 --dim 256
uv run python scripts/synthesize_queries.py --n 300         # псевдофото из вырезок каталога
uv run python scripts/import_live_photos.py                  # новые съёмки из data/incoming -> data/test | data/train
uv run python scripts/import_organizer_photos.py            # organizer_100 -> data/train по таблице проверки
uv run python scripts/build_platform_features.py --index models/index_platform_sq_v3 \
  --sources synthetic,train --out eval/results/features_platform_sq_v3_v7.jsonl
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  notebooks/06_decider_eda_and_training.ipynb               # EDA, CV, подбор -> decider_platform_sq_v3_v6
uv run python scripts/train_decider.py --features eval/results/features_platform_sq_v3_v7.jsonl \
  --params-from models/decider_platform_sq_v3_v6 --calibration live-split --scenario real \
  --out models/decider_platform_sq_v3_v7                    # параметры v6, признаки v7
uv run python eval/threshold_selection.py --decider models/decider_platform_sq_v3_v7 \
  --apply utility                                           # порог по пользе, бутстреп по винодельням
# v8 (сервис 18.10.2026): те же параметры, индекс сайта 18.10 и OCR из .env (WINE_OCR_DET,
# WINE_OCR_REC — признаки зависят от того, как читается текст)
uv run python scripts/build_platform_features.py --index models/index_platform_sq_v3_site_1810 \
  --sources synthetic,train --out eval/results/features_platform_site_v6det_ru.jsonl
uv run python scripts/train_decider.py --features eval/results/features_platform_site_v6det_ru.jsonl \
  --params-from models/decider_platform_sq_v3_v6 --calibration live-split --scenario real \
  --out models/decider_platform_sq_v3_v8
uv run python eval/threshold_selection.py --decider models/decider_platform_sq_v3_v8 \
  --features eval/results/features_platform_site_v6det_ru.jsonl --apply utility
# А+Б+Г (сервис с 19.10.2026): признаки пересобраны кодом с дозаполненными атрибутами карточек,
# монотонные признаки текста, визуальный отрыв только от соседей, не опровергнутых текстом
export WINE_OCR_PYTHON=.venv-ocr/Scripts/python.exe   # иначе распознаватель поднимется без GPU
uv run python scripts/build_platform_features.py --index models/index_platform_sq_v3_site_1810 \
  --sources synthetic,train --out eval/results/features_platform_site_v6det_ru_A.jsonl
uv run python scripts/train_decider.py --features eval/results/features_platform_site_v6det_ru_A.jsonl \
  --params-from models/decider_platform_sq_v3_v8 --calibration live-split --scenario real \
  --monotone text --gap-rivals text --out models/decider_site1810_abd
uv run python eval/threshold_selection.py --decider models/decider_site1810_abd \
  --features eval/results/features_platform_site_v6det_ru_A.jsonl --apply utility
```

Решающий слой обучен на признаках, которые зависят от того, как OCR прочитал этикетку. Поэтому
после смены детектора строк или распознавателя его нужно переобучить: русский распознаватель со
старым решающим слоем v7 на тесте даёт 88 верных при пороге 0.4 вместо 94 с переобученным v8.
Русский распознаватель скачивается один раз с Hugging Face в `models/ocr_rec/`:

```bash
mkdir -p models/ocr_rec/PP-OCRv6_medium_rec_ru && cd models/ocr_rec/PP-OCRv6_medium_rec_ru
R=https://huggingface.co/vladlinv/PP-OCRv6_medium_rec_ru_onnx/resolve/d44f62c323b78182d4e2b4aa153e638dde0d99bc
curl -L -o inference.onnx $R/model.onnx && curl -L -o inference.yml $R/inference.yml
sha256sum inference.onnx   # a8cdd7ad0747b29b79de6e187be18c32225b01447d1ae5837c901ab8a296031d
```

Вырезка этикетки на эталоне каталога проверяется отдельно: ошибка детектора на эталоне не видна
нигде, кроме ответов, а повторяется на каждом запросе этого вина. `check_catalog_crops.py`
сравнивает рамку бутылки с маской переднего плана вырезки (эталон — бутылка на белом), ищет
этикетку, обрезанную краем рамки, найденную в горлышке или не найденную вовсе, и при обрезанной
бутылке повторяет каскад с бутылкой по маске. Починенные квадраты пишутся в
`data/catalog/label_overrides.csv` (`source=auto`); туда же человек вписывает свой квадрат
(`manual`) или решение «оставить как есть» (`accepted`). `build_index.py` применяет поправки
сам. Чтобы не пересобирать весь каталог, исправленные карточки можно пересчитать в копии индекса:

```bash
uv run python scripts/check_catalog_crops.py --slugs <новые slug> --write-overrides  # при заведении вин
uv run python scripts/patch_index_crops.py --index models/index_platform_sq_v3 \
  --out models/index_platform_sq_v3_fix                      # вектор, XFeat, бокс года — только у поправок
```

Главный режим ошибки на эталонах — первая ступень каскада: на высокой вырезке детектор бутылки
уверенней находит её верх, чем бутылку целиком, хотя полная рамка у него тоже есть.
`BoxCropper.partial_min_score` берёт полную рамку, если она содержит выбранную, стоит на той же
вертикали и уверенность не ниже порога (0.15). Порог хранится в `config.json` индекса
(`bottle_partial_min_score`) и тем же значением режет запрос; у индекса без поля правило выключено.
Перевести готовый индекс на правило без пересборки (пересчитываются только карточки, у которых
сменилась вырезка):

```bash
uv run python scripts/patch_index_crops.py --index models/index_platform_sq_v3_fix \
  --out models/index_platform_sq_v3_partial --recrop 0.15
```

Вина, появившиеся на сайте после дампа, заводит `add_site_wines.py`: свежий sitemap против
`catalog.csv` → карточка со страницы вина → эталон → проверка вырезки (как
`check_catalog_crops.py`, плюс проверка содержимого вырезки SigLIP 2 — этикетка, а не горлышко) →
ближайший сосед после whitening (почти совпадение — дубль) → самопоиск по трём псевдофото через
весь пайплайн. В новый индекс и `catalog.csv` попадают только прошедшие; остальное — в
`data/derived/site_add/report.jsonl` и `review.jpg`, код выхода 1. Обновлять раз в неделю и
перед каждой проверкой: пробный проход → разбор `review.jpg` (свой квадрат — `source=manual` в
`label_overrides.csv`, вырезка годится — `source=accepted`) → копия `catalog.csv` и
`label_overrides.csv` → боевой проход в новую папку → `platform_benchmark.py` на новом индексе →
`WINE_INDEX` в `.env`, `docker-compose.yml` и `scripts/release_scanner.py`. Фотографии,
сменившиеся на сайте (`--update-images`), заменять только после просмотра: 18.10 у
`muskat-premium` на сайте стояла фотография красного «Merlot».

```bash
uv run python scripts/add_site_wines.py --index models/index_platform_sq_v3_site_1810 \
  --decider models/decider_platform_sq_v3_v7 --dry-run          # только проверки, ничего не пишет
uv run python scripts/add_site_wines.py --index models/index_platform_sq_v3_site_1810 \
  --decider models/decider_platform_sq_v3_v7 --out models/index_platform_sq_v3_site_<дата>
```

Веса детектора этикетки (`models/rtdetr_label`) обучаются `scripts/train_rtdetr.py`.

## Как измеряется

`eval/platform_benchmark.py` гоняет тот же `WineScanner`, что стоит за API, по живым кадрам
(`data/test/manifest.csv` — изолированный тест: все снятые вина из каталога и 50 незнакомых) и раскладывает путь ответа по ступеням: визуальный ранг верной
карточки, текстовый ранг, попала ли в окно ре-ранкинга, итоговый top-1, исход (верно /
близнец / чужое / отказ), ложные приёмы на незнакомых винах. Каждый прогон дописывается
локально в `eval/results/platform_runs.jsonl` — это и есть таблица абляций.

```bash
uv run python eval/platform_benchmark.py --tag <метка>
uv run python eval/platform_benchmark.py --decider none --text-scorer bm25 --tag ablation-bm25
uv run python eval/latency.py --n 20          # разбивка задержки по блокам без кэшей
uv run python -m pytest tests/                # 130+ тестов без единой модели
```

## Структура

```
api/            FastAPI: /scan, /v1/eval/predict, /health, /catalog/{slug}, /catalog/image/{slug}, /sommelier
web/            Nuxt 3, mobile-first интерфейс в стилистике портала; server/ — вход, полка, отзывы (PostgreSQL)
src/wine_scanner/
  detect/       детекция бутылки и этикетки (каскад RT-DETR)
  embed/        SigLIP 2 / DINOv2 (фабрика по config.json) + whitening
  index/        FAISS
  ocr/          PaddleOCR / Yandex Vision, сворачивание алфавитов, n-граммный текстовый индекс, атрибуты
  rerank/       XFeat + LighterGlue + RANSAC
  vintage/      год урожая
  decide/       CatBoost, калибровка, защита от близнеца, VLM-судья
  analogues.py  аналоги из других виноделен
  sommelier.py  цифровой сомелье
  pipeline.py   сквозной WineScanner — единственная точка входа
scripts/        сборка каталога, индекса, признаков, обучение
eval/           бенчмарки
data/           каталог, живые кадры, публичные кадры организаторов (в git не входит)
```

## Ограничения

- Эталоны платформы — студийные вырезки, часто 140–300 px шириной; визуальная ветка находит
  семью, а внутри семьи ответ держится на тексте. Кадр без читаемой этикетки (сильный смаз,
  этикетка сбоку) с большой вероятностью уйдёт в отказ.
- 55 карточек каталога делят одну и ту же картинку с другой карточкой; их различает только
  текст, а если он совпадает — различить нельзя в принципе.
- Латентность на ноутбуке без видеокарты 6–9 с; на RTX 4060 Ti ожидаемо до 2 с.
