# Сканер российских вин для платформы «Своё Вино»

Состояние после переноса с Mac и перехода на RT-DETR: [аудит 20.09.2026](docs/RTDETR_STATUS.md).
Разбор кропов и разметки: [ноутбук](notebooks/02_rtdetr_crop_audit.ipynb).
Переобучение, padding и OCR/XFeat: [эксперименты](notebooks/03_pipeline_rebuild_experiments.ipynb).
Разбор ошибок действующего решающего слоя: [ноутбук 04](notebooks/04_catboost_importance_and_errors.ipynb).
Облачные шаги (hybrid OCR, VLM-судья) на тесте: [ноутбук 07](notebooks/07_cloud_steps_hybrid_ocr_and_vlm_judge.ipynb).

Фотография этикетки → карточка вина из каталога платформы, калиброванная уверенность, честный
отказ с похожими и аналогами, если вина в каталоге нет. Хакатон РСХБ, сентябрь 2026.

Как это устроено в двух словах: задача решается как поиск по картинке, а не как классификация.
Визуальная ветка находит **семью** вина (линейку, винодельню), текстовая — **конкретную
карточку внутри семьи**: близнецы одной серии различаются словами «красное / белое / брют /
полусладкое», а не рисунком этикетки. Подробно — в [ARCHITECTURE.md](ARCHITECTURE.md),
контракт API — в [API.md](API.md).

## Быстрый старт

### Docker (машина с NVIDIA)

```bash
cp .env.example .env          # ключи облачного OCR и LLM, если используются; без них тоже работает
docker compose up --build     # сервис на :8080, интерфейс на :3000
curl -s localhost:8080/health | jq .devices   # визуальные модели — cuda, PaddleOCR — cpu
```

В `./models` должны лежать артефакты (см. «Сборка артефактов»), в `./data/catalog` — каталог.

### Локально (uv, Python 3.12)

```bash
uv sync --extra api --extra rerank --extra ocr --extra decide
WINE_INDEX=models/index_platform_sq_v3 WINE_DECIDER=models/decider_platform_sq_v3_slug \
  uv run uvicorn api.main:app --port 8080 --env-file .env     # .env необязателен
cd web && npm install && NUXT_SCANNER_URL=http://127.0.0.1:8080 npm run dev   # интерфейс на :3000
```

Первый запрос после старта сервис делает сам (прогрев), поэтому поднимается он около минуты;
`/health` показывает `load_seconds` и `warmup_seconds`.

### Скрипт оценки

```bash
cd data/eval && ./participant_test.sh --images-dir ./queries --manifest ./queries.tsv \
  --endpoint http://127.0.0.1:8080/v1/eval/predict --output /tmp/predictions.jsonl
```

Ручка отвечает `{"slug": ...}` либо `{"slug": null, "similar": [...]}` на незнакомое вино.
Переключатель `WINE_EVAL_REFUSE=0` заставляет и ниже порога отдавать лучшего кандидата.

## Переменные окружения

| Переменная | Значение по умолчанию | Смысл |
|---|---|---|
| `WINE_INDEX` | `models/index` | Индекс каталога (в Docker — `models/index_platform_sq_v3`) |
| `WINE_DECIDER` | `models/decider` | Решающий слой (в Docker — `models/decider_platform_sq_v3_slug`) |
| `WINE_OCR` | `paddle` | `yandex` — Vision с откатом на PaddleOCR; `hybrid` — Vision только при слабом PaddleOCR; нужны `YANDEX_OCR_API_KEY`, `YANDEX_FOLDER_ID` |
| `WINE_VLM` | `0` (в Docker `1`) | `1` — VLM-судья между близнецами одной винодельни; работает только в `/v1/eval/predict` |
| `WINE_VLM_SCAN` | `0` | `1` — звать судью и в продуктовом `/scan` (ложные приёмы незнакомых 0.07 → 0.34) |
| `WINE_VLM_OPTIONS`, `WINE_VLM_TRIGGER`, `WINE_VLM_VERIFY`, `WINE_VLM_TIMEOUT` | `family`, `family`, `1`, `4` | Режим судьи family-verify: показывать карточки своей винодельни, звать при соседках в выдаче, смену лидера принимать только при подтверждении текстом этикетки |
| `WINE_VLM_PROVIDER`, `WINE_VLM_FALLBACK` | `yandex`, — | Провайдер судьи и откат: `yandex` (Yandex AI Studio, без VPN) / `openai` (любой OpenAI-совместимый чат: `WINE_VLM_BASE_URL`, `WINE_VLM_MODEL`, `WINE_VLM_API_KEY`) |
| `YANDEX_LLM_API_KEY`, `YANDEX_FOLDER_ID` | — (ключ OCR), — | Ключ AI Studio: роль `ai.languageModels.user`, область ключа `yc.ai.languageModels.execute`; модели `YANDEX_VLM_MODEL` (`qwen3.6-35b-a3b`) и `YANDEX_LLM_MODEL` (`yandexgpt-5-lite`) |
| `WINE_SOMMELIER` | `0` | `1` — цифровой сомелье (`/sommelier`); провайдер `WINE_LLM_PROVIDER` (`yandex` / `openai` — `WINE_LLM_*` или те же `WINE_VLM_*`), откат `WINE_LLM_FALLBACK` |
| `WINE_GUARD` | `warn` | Защита от близнеца: `warn` возвращает карточку с предупреждением; `twin` / `strict` отказывают, `off` выключает правило |
| `WINE_RERANK_CANDIDATES` | `25` | Окно ре-ранкинга локальными признаками; на видеокарте 50 |
| `WINE_VISUAL_CANDIDATES`, `WINE_TEXT_CANDIDATES` | `100`, `50` | Ширина визуальной и текстовой веток |
| `WINE_FAMILY` | `1` | Расширение кандидатов роднёй по винодельне |
| `WINE_EVAL_REFUSE` | `1` | Политика eval-ручки на незнакомом вине |
| `NUXT_SCANNER_URL` | `http://127.0.0.1:8080` | Адрес сервиса для интерфейса |

Без единого внешнего ключа сервис полностью работоспособен: PaddleOCR локально, судья и
сомелье выключены.

## Сборка артефактов

Каталог платформы (дамп CSV + медиа) → индекс → whitening → признаки → решающий слой:

```bash
uv run python scripts/build_catalog.py                      # data/catalog/: catalog.csv + originals/<slug>.png
uv run python scripts/build_index.py --catalog platform --detect cascade --fit pad \
  --model models/siglip2 --weights models/rtdetr_label_recrop_bf16 --local-preprocess clahe2 \
  --out models/index_platform_sq_v3                          # ~1 ч на RTX 4060 Ti
uv run python scripts/fit_whitening.py --index models/index_platform_sq_v3 --dim 256
uv run python scripts/synthesize_queries.py --n 300         # псевдофото из вырезок каталога
uv run python scripts/import_live_photos.py                  # новые съёмки из data/incoming -> data/test | data/train
uv run python scripts/import_organizer_photos.py            # organizer_100 -> data/train по таблице проверки
uv run python scripts/build_platform_features.py --index models/index_platform_sq_v3 \
  --sources synthetic,train --out eval/results/features_platform_sq_v3.jsonl
uv run python scripts/build_decider_training_notebook.py    # ноутбук 06: EDA, CV, подбор, K/P
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  notebooks/06_decider_eda_and_training.ipynb               # -> models/decider_platform_sq_v3_slug
```

Веса детектора этикетки (`models/rtdetr_label`) обучаются `scripts/train_rtdetr.py`.

## Как измеряется

`eval/platform_benchmark.py` гоняет тот же `WineScanner`, что стоит за API, по живым кадрам
(`data/test/manifest.csv` — изолированный тест: все снятые вина из каталога и 50 незнакомых) и раскладывает путь ответа по ступеням: визуальный ранг верной
карточки, текстовый ранг, попала ли в окно ре-ранкинга, итоговый top-1, исход (верно /
близнец / чужое / отказ), ложные приёмы на незнакомых винах. Каждый прогон дописывается в
`eval/results/platform_runs.jsonl` — это и есть таблица абляций.

```bash
uv run python eval/platform_benchmark.py --tag <метка>
uv run python eval/platform_benchmark.py --decider none --text-scorer bm25 --tag ablation-bm25
uv run python eval/latency.py --n 20          # разбивка задержки по блокам без кэшей
uv run python -m pytest tests/                # 130+ тестов без единой модели
```

Цифры на живых кадрах каталога платформы (16–17.09.2026, 24 известных кадра, 152 незнакомых)
— в [ARCHITECTURE.md](ARCHITECTURE.md), раздел «Что измерено».

## Структура

```
api/            FastAPI: /scan, /v1/eval/predict, /health, /catalog/image/{slug}, /sommelier
web/            Nuxt 3, mobile-first интерфейс в стилистике портала
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
eval/           бенчмарки и результаты
data/           каталог, живые кадры, публичные кадры организаторов (в git не входит)
```

## Ограничения

- Эталоны платформы — студийные вырезки, часто 140–300 px шириной; визуальная ветка находит
  семью, а внутри семьи ответ держится на тексте. Кадр без читаемой этикетки (сильный смаз,
  этикетка сбоку) с большой вероятностью уйдёт в отказ.
- 55 карточек каталога делят одну и ту же картинку с другой карточкой; их различает только
  текст, а если он совпадает — различить нельзя в принципе.
- Латентность на ноутбуке без видеокарты 6–9 с; на RTX 4060 Ti ожидаемо до 2 с (замер — в
  ARCHITECTURE.md после переезда).
