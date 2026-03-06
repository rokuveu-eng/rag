# Гибридный поиск по Excel (Qdrant + Ollama + FastEmbed)

Этот проект — готовый API для индексации данных из `.xlsx` и гибридного поиска по ним. 
Используется **Qdrant Query API** (v1.10+) с **Fusion RRF**, что позволяет объединять результаты **семантического поиска** (dense) и **лексического поиска** (sparse).

---

## 🚀 Компоненты

### ✅ Qdrant (векторная база)
Хранит:
- `text-dense` — плотные эмбеддинги (Ollama / bge-m3)
- `text-sparse` — разреженные эмбеддинги (FastEmbed / SPLADE)

### ✅ Ollama
Используется для получения dense-векторов.
Модель: **bge-m3**.

### ✅ FastEmbed
Используется для sparse-векторов (SPLADE).
Модель: **prithivida/Splade_PP_en_v1**.

### ✅ FastAPI API
Индексирует данные, сохраняет их в Qdrant и выполняет гибридный поиск.

---

## ⚙️ Архитектура

```
[Excel (.xlsx)]
      ↓
 FastAPI API
      ↓
  ┌────────────┐    ┌────────────┐
  │ Ollama     │    │ FastEmbed  │
  │ (bge-m3)   │    │ (SPLADE)   │
  └────────────┘    └────────────┘
      ↓                  ↓
      └──────┬───────────┘
             ↓
         Qdrant
      (dense + sparse)
             ↓
      Fusion RRF Query
             ↓
         Результаты
```

---

## ✅ Требования

- Docker / Docker Compose
- Qdrant Client **>= 1.10.0**
- NVIDIA GPU (опционально, только для ускорения Ollama)
- Установленный **nvidia-container-toolkit** на хосте (если используете GPU)

> Qdrant работает на CPU и **не требует** GPU. Он полностью совместим с GPU‑сервисами рядом.

---

## 🧩 Запуск

```bash
docker compose up -d --build
```

После запуска:
- Qdrant: http://localhost:6333
- API: http://localhost:8424
- Веб-интерфейс: http://localhost:8424

> По умолчанию API обращается к Ollama по адресу `http://localhost:11434`.
> Если запускаете API внутри Docker и Ollama в контейнере, укажите `OLLAMA_BASE_URL=http://ollama:11434`.
> В контейнере `localhost` указывает на сам API, поэтому для Docker нужен hostname `ollama`.
> Ollama может отдавать эмбеддинги через `/api/embed`, `/api/embeddings` или `/v1/embeddings` — API автоматически пробует все варианты и проверяет размер батча.

---

## 📥 Индексация Excel

Эндпоинт:
```
POST /upload_processed_xlsx
```

### Пример curl:
```bash
curl -X POST \
  -F "file=@/path/to/data.xlsx" \
  -F "skip_rows=0" \
  -F "mappings={\"Артикул\":0,\"Наименование\":1,\"Тариф с НДС, руб\":2}" \
  -F "collection_name=my_collection" \
  -F "batch_size=16" \
  -F "points_batch_size=200" \
  http://localhost:8424/upload_processed_xlsx
```

### Асинхронная загрузка (с прогрессом)
```bash
curl -X POST \
  -F "file=@/path/to/data.xlsx" \
  -F "skip_rows=0" \
  -F "mappings={\"Артикул\":0,\"Наименование\":1,\"Тариф с НДС, руб\":2}" \
  -F "collection_name=my_collection" \
  -F "batch_size=16" \
  -F "points_batch_size=200" \
  http://localhost:8424/upload_processed_xlsx_async
```
Ответ:
```
{"status":"started","job_id":"..."}
```
Статус:
```
GET http://localhost:8424/upload_status/{job_id}
```

Пример ответа статуса:
```json
{
  "status": "running",
  "progress": 37.5,
  "indexed_rows": 1500,
  "total_rows": 4000,
  "rate": 120.5,
  "eta": 20.3
}
```

### Пояснение:
- `skip_rows` — сколько строк пропустить перед заголовком.
- `mappings` — соответствие колонок (индексы идут с 0).
- `batch_size` — размер пачки для батч‑генерации dense эмбеддингов через Ollama.
- `points_batch_size` — размер чанка при записи в Qdrant (уменьшайте при ошибке лимита payload). Интерфейс использует асинхронную загрузку и показывает прогресс в форме.

---

## 📦 Обновление остатков

Эндпоинт:
```
POST /upload_stock_async
```

Требуются столбцы: **Артикул** и **Остаток** (выбираются вручную в интерфейсе). Перед загрузкой все остатки обнуляются (`Остаток = 0`), затем обновляются значения из XLSX. Обновляется только payload поле `Остаток`, векторы не трогаются. Если артикула нет в коллекции — строка пропускается. Артикул нормализуется (значения вида `="00050"` приводятся к `00050`).

> При основной загрузке (индексации) `Остаток` также устанавливается в `0`.

### Пример curl:
```bash
curl -X POST \
  -F "file=@/path/to/stock.xlsx" \
  -F "skip_rows=0" \
  -F "collection_name=my_collection" \
  -F "article_col=0" \
  -F "stock_col=3" \
  -F "batch_size=200" \
  http://localhost:8424/upload_stock_async
```

Статус:
```
GET http://localhost:8424/stock_status/{job_id}
```

Пример ответа статуса:
```json
{
  "status": "running",
  "progress": 42.5,
  "processed_rows": 2000,
  "updated_rows": 1500,
  "skipped_rows": 500,
  "total_rows": 4000,
  "rate": 120.5,
  "eta": 20.3
}
```

> Батчинг ускоряет индексирование. Сначала используется `/api/embed` (если поддерживается Ollama),
> иначе выполняются параллельные запросы к `/api/embeddings`. Запись в Qdrant идёт чанками.

---

## 🔍 Поиск

Эндпоинт:
```
GET /search
```

Параметры:
- `only_in_stock=true|false` — если `true`, возвращаются только позиции с остатком > 0 (до 15 шт). Если `false`, выдача комбинируется: до 5 позиций с остатком > 0 + до 10 обычных, всего не более 15.

### Пример curl:
```bash
curl -G "http://localhost:8424/search" \
  --data-urlencode "collection_name=my_collection" \
  --data-urlencode "query=поисковый запрос"
```

### Что происходит:
1. Генерируется dense-вектор через Ollama (`bge-m3`)
2. Генерируется sparse-вектор через FastEmbed (SPLADE)
3. Qdrant объединяет результаты через **Fusion RRF** (`query_points + prefetch`)

> Ollama запущена с GPU (если он доступен), что ускоряет генерацию dense‑векторов.

---

## 🖥️ Веб-версия (UI)

В проекте есть встроенный веб-интерфейс для загрузки Excel/PDF и поиска.
Он доступен по адресу:

```
http://localhost:8424
```

### Что умеет UI
- Основная загрузка Excel с подбором колонок и отображением прогресса.
- Асинхронная загрузка остатков.
- Поиск по основной коллекции (режимы: hybrid/dense/sparse) с выводом `payload` и `score`.
- Поиск по паспортам (PDF) с отображением score, payload и диагностикой по векторам.
- Вкладка **«Интеграции»** для настройки Polza.ai и Bitrix24 прямо из веб-интерфейса.

### Как пользоваться
1. Откройте `http://localhost:8424` в браузере.
2. Вкладка **«Основная загрузка»** — загрузите Excel и дождитесь окончания индексации.
3. В этой же вкладке используйте блок **«Поиск (основная коллекция)»**: введите запрос, выберите режим и нажмите **«Искать»**.
4. Вкладка **«Остатки»** — загрузите файл остатков, если нужно фильтровать по наличию.
5. Вкладка **«Паспорта»** — загрузите PDF и выполняйте поиск по паспорту изделия.
6. Вкладка **«Интеграции»** — укажите параметры Polza.ai и Bitrix24 для чат-бота.

---

## 🤖 Интеграции: Polza.ai + Bitrix24 (OAuth + webhook)

Чат-бот сценарий поддерживает два режима вызова Bitrix REST:

1. **Входящий webhook URL** (быстрый старт).
2. **OAuth-приложение Bitrix24** (рекомендуется для production).

Сценарий работы:

1. Бот получает команду `/price <запрос>` из Bitrix24.
2. Backend выполняет поиск по коллекции (`hybrid` / `dense` / `sparse`).
3. Из найденных результатов формируется контекст.
4. Контекст отправляется в Polza Chat Completions.
5. Готовый ответ + найденные позиции отправляются обратно в Bitrix24.

Поддерживаемые команды:
- `/help` — список команд;
- `/search <запрос>` — поиск по коллекции документации + LLM;
- `/newchat` и `/clear` — очистка контекста диалога.
- `/stock <запрос>` — поиск только позиций в наличии.

Контекст сообщений хранится в памяти backend (по `dialog_id`), без записи в БД.

---

## ⚙️ Runtime-конфигурация интеграций

Настройки Polza и Bitrix сохраняются **в памяти процесса FastAPI** (runtime-only):
- не пишутся в файл;
- сбрасываются после рестарта контейнера/API;
- секреты в ответах отдаются в маскированном виде.

### `GET /runtime_config`

Возвращает текущую конфигурацию интеграций в безопасном виде:
- `polza.configured`, `polza.api_key_masked`, `model`, `base_url`, `temperature`, `max_tokens`;
- `bitrix.client_id`, `client_secret_masked`, `redirect_uri`, `webhook_url`, `bot_id`, `collection_name`, `search_mode`.

Дополнительно для OAuth:
- `bitrix.portal_base_url`
- `bitrix.oauth_auth_url`
- `bitrix.oauth_token_url`
- `bitrix.oauth_connected`
- `bitrix.access_token_masked`
- `bitrix.refresh_token_masked`
- `bitrix.token_expires_at`
- `bitrix.docs_collection_name` (для команды `/search`)

Пример:

```bash
curl http://localhost:8424/runtime_config
```

### `POST /runtime_config`

Обновляет runtime-параметры.

Пример:

```bash
curl -X POST http://localhost:8424/runtime_config \
  -H "Content-Type: application/json" \
  -d '{
    "polza": {
      "api_key": "pk_xxx",
      "model": "openai/gpt-4o",
      "base_url": "https://polza.ai/api/v1/chat/completions",
      "temperature": 0.2,
      "max_tokens": 500
    },
    "bitrix": {
      "client_id": "local.xxxxx",
      "client_secret": "secret",
      "redirect_uri": "https://example.com/oauth/callback",
      "webhook_url": "https://your-domain.bitrix24.ru/rest/1/xxxxxxxx",
      "bot_id": "62",
      "collection_name": "my_collection",
      "search_mode": "hybrid"
    }
  }'
```

> `bitrix.search_mode` должен быть одним из: `hybrid`, `dense`, `sparse`.

### OAuth-эндпоинты Bitrix24

- `GET /bitrix/oauth/connect_url` — получить ссылку авторизации (client_id + redirect_uri + state).
- `GET /bitrix/oauth/callback` — callback для обмена `code` на `access/refresh token`.
- `POST /bitrix/oauth/refresh` — принудительное обновление access token.
- `GET /bitrix/oauth/status` — проверить состояние OAuth-подключения.

В UI (вкладка **Интеграции**) добавлены кнопки:
- «Подключить Bitrix24 (OAuth)»
- «Обновить OAuth токен»
- «Проверить OAuth статус»

---

## 🧠 Bitrix24 webhook API

Эндпоинт входящих сообщений от Bitrix:

```
POST /bitrix/webhook
```

Поддерживаемые команды:
- `/help`
- `/price <запрос>`
- `/search <запрос>`
- `/stock <запрос>`
- `/newchat`
- `/clear`

Исходящие ответы отправляются в Bitrix методом:

```
imbot.message.add
```

на основе `bitrix.webhook_url` (добавляется `/imbot.message.add.json`).

Минимальный локальный smoke-пример:

```bash
curl -X POST http://localhost:8424/bitrix/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "MESSAGE": "/price автоматический выключатель 16А",
      "DIALOG_ID": "chat123",
      "FROM_USER_ID": "1"
    }
  }'
```

---

## 🔐 Примечания по безопасности

- API-ключ Polza и Bitrix secret не сохраняются на диск (только runtime память).
- После сохранения через UI поля секретов очищаются в браузере.
- В API-ответах используется маскирование секретов.
- Для production рекомендуется вынести хранение секретов в защищённое хранилище (Vault/KMS/secret manager).

---

## ✅ Пример запроса в Qdrant (новый API)

```python
client.query_points(
    collection_name="my_collection",
    prefetch=[
        models.Prefetch(
            query=sparse_vector,
            using="text-sparse",
            limit=20,
        ),
        models.Prefetch(
            query=dense_vector,
            using="text-dense",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

---

## 💡 Важно
- При первом запуске FastEmbed скачивает модель SPLADE с Hugging Face (около 500MB).
- Ollama скачивает модель `bge-m3` при первом обращении.
- При отсутствии GPU Ollama будет работать на CPU.

---

## 📌 Развёртывание через Portainer

1. Создайте новый Stack
2. Вставьте содержимое `docker-compose.yml`
3. Нажмите Deploy

---

Если нужно — добавлю раздел с API ошибками, возвратом payload или схемой данных.
