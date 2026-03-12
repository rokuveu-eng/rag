# Hybrid Search API + AI Orchestrator

Упрощённый проект для:
- загрузки Excel-прайса в Qdrant;
- обновления остатков отдельным Excel;
- загрузки PDF-паспортов (OCR + чанкинг);
- гибридного поиска (`hybrid`/`dense`/`sparse`) через веб-интерфейс и API;
- AI-оркестратора для подбора позиций и генерации Excel-спецификации.

Интеграции, runtime-конфиги, HF-cache API и инструменты создания/теста бота удалены.

## Запуск

```bash
docker compose up -d --build
```

Сервисы:
- Qdrant: http://localhost:6333
- Ollama (локально, GPU): http://localhost:11434
- API + UI: http://localhost:8424
- AI Orchestrator: http://localhost:8430
- Redis (память): http://localhost:6379
- Bitrix Adapter: http://localhost:8440

### GPU для Ollama

В `docker-compose.yml` для `ollama` используется простая настройка:

- `gpus: all`

Это ускоряет получение эмбеддингов в `api/search` и загрузочных пайплайнах.
После изменения перезапустите сервисы:

```bash
docker compose up -d --build
```

## Основные эндпоинты

- `POST /upload_processed_xlsx_async` + `GET /upload_status/{job_id}`
- `POST /upload_stock_async` + `GET /stock_status/{job_id}`
- `POST /upload_passports_async` + `GET /passports_status/{job_id}`
- `GET /search`
- `GET /search_passports`
- `DELETE /collection?collection_name=...`

### Эндпоинты AI Orchestrator

- `POST /agent/chat`
- `POST /agent/spec`
- `POST /memory/reset`
- `GET /agent/files/{file_id}`
- `GET /health`

### Эндпоинты Bitrix Adapter

- `POST /bitrix/webhook`
- `POST /bitrix/send-test`
- `GET /health`

Параметры `GET /search`:
- `query` — текст запроса;
- `collection_name` — имя коллекции;
- `mode` — `hybrid|dense|sparse`;
- `only_in_stock` — только позиции с остатком > 0;
- `limit` — сколько результатов вернуть (по умолчанию 15);
- `candidate_limit` — сколько кандидатов брать из dense/sparse в hybrid перед fusion (по умолчанию 10);
- `dense_weight` / `sparse_weight` — веса для hybrid-ранжирования (по умолчанию `0.8/0.2`, то есть приоритет dense);
- `include_breakdown` — добавить в debug отдельные top-списки по `dense/sparse/hybrid` (по умолчанию `false`).

В `mode=hybrid` используется **взвешенное RRF-ранжирование**:
- берутся top-K из dense и top-K из sparse,
- для каждого кандидата считается `final_score = dense_weight * dense_rrf + sparse_weight * sparse_rrf`,
- возвращается top `limit` по `final_score`.

Debug в ответе `/search` (`debug.score_trace`) для `hybrid` включает трейс взвешенного RRF:
- `final_rank_source` (`weighted_rrf`),
- `trace_top[]` с полями `dense_rank`, `sparse_rank`, `dense_rrf`, `sparse_rrf`, `final_score`.

Если включён `include_breakdown=true`, дополнительно возвращается:
- `debug.score_trace.breakdown.dense_top[]`
- `debug.score_trace.breakdown.sparse_top[]`
- `debug.score_trace.breakdown.hybrid_top[]`

Каждый элемент breakdown содержит:
- `id`, `rank`, `score`, `title`

Пример запроса:

```bash
curl -s "http://localhost:8424/search?query=автомат%2016а&collection_name=my_collection&mode=hybrid&limit=10&candidate_limit=30&include_breakdown=true" | jq '.debug.score_trace'
```

Пример тела запроса для `/agent/chat` и `/agent/spec`:

```json
{
  "message": "Автомат 16А 4 шт; Контактор 25А 2 шт; Кабель 3х2.5 100 м",
  "collection_name": "my_collection",
  "ai_base_url": "https://polza.ai/api/v1",
  "ai_api_key": "<POLZA_AI_API_KEY>",
  "ai_model": "openai/gpt-4o",
  "probable_limit": 3
}
```

Дополнительные поля для памяти в `/agent/chat`:
- `tenant_id` — идентификатор tenant/портала (по умолчанию `default`)
- `dialog_id` — идентификатор диалога (если передан, включается краткая память)
- `reset_memory` — если `true`, очистить память диалога перед обработкой

Пример:

```json
{
  "message": "подбери автомат 16А",
  "collection_name": "CHINT",
  "tenant_id": "my-company",
  "dialog_id": "chat123",
  "reset_memory": false
}
```

Сброс краткой памяти:

```bash
curl -s -X POST "http://localhost:8430/memory/reset" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"my-company","dialog_id":"chat123"}' | jq
```

Если позиций больше 2 (или вызван `/agent/spec`) будет сформирован xlsx-файл
с колонками: `Запрос`, `Артикул замены`, `Наименование замены`, `Score`, `ScoreNorm(0..1)`, `Цена`, `Кол-во`, `Сумма`, `Вероятные замены`, `Dense top`, `Sparse top`, `Hybrid top`.

Также для каждой строки подбора в JSON (`rows[]`) и в `reply_text` доступны:
- `score`
- `score_norm` (нормализация в диапазон `0..1` внутри кандидатов текущего запроса)
- отдельные списки: `dense_top`, `sparse_top`, `hybrid_top`.

## UI

Откройте `http://localhost:8424`:
- вкладка **Основная загрузка**;
- вкладка **Остатки**;
- вкладка **Паспорта**;
- вкладка **AI чат** (настройки endpoint/API key/model + тестовый чат);
- вкладка **Удаление**.

## Bitrix24 интеграция и память

В `docker-compose.yml` добавлены сервисы:
- `redis` — хранение краткой LLM-памяти;
- `bitrix-adapter` — webhook-адаптер Bitrix24.

Ключевые переменные окружения Bitrix Adapter:
- `BITRIX_WEBHOOK_URL` — базовый webhook URL Bitrix24 (без `/imbot.message.add` в конце),
- `BITRIX_BOT_ID` — ID чат-бота (опционально),
- `BITRIX_CLIENT_ID` — CLIENT_ID для webhook-вызовов (если требуется),
- `BITRIX_WEBHOOK_VERIFY_TOKEN` — токен верификации входящего webhook.

Поведение команды `/reset`:
- если в входящем сообщении из Bitrix24 текст `/reset`, адаптер вызывает `POST /memory/reset` в orchestrator;
- после успешного сброса отправляет подтверждение в диалог Bitrix24.
