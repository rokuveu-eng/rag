const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

const DEFAULT_TIMEOUT_MS = 45000;
const LOG_MAX_LINES = 200;

const switchTab = (tabName) => {
    tabButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.tab === tabName);
    });
    tabContents.forEach((content) => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
};

tabButtons.forEach((button) => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
});

const setStatus = (el, text) => {
    if (el) el.textContent = `Статус: ${text}`;
};

const setProgress = (el, value) => {
    if (!el) return;
    const safe = Math.max(0, Math.min(100, value));
    el.style.width = `${safe}%`;
};

const addLog = (el, text) => {
    if (!el) return;
    const t = new Date().toLocaleTimeString();
    const next = `[${t}] ${text}`;
    const prev = (el.textContent || '').split('\n').filter(Boolean);
    const merged = [next, ...prev].slice(0, LOG_MAX_LINES);
    el.textContent = `${merged.join('\n')}${merged.length ? '\n' : ''}`;
};

const fetchJson = async (url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) => {
    const timeoutController = new AbortController();
    const externalSignal = options.signal;
    if (externalSignal) {
        externalSignal.addEventListener('abort', () => timeoutController.abort(), { once: true });
    }

    const timer = setTimeout(() => timeoutController.abort(), timeoutMs);

    try {
        const response = await fetch(url, { ...options, signal: timeoutController.signal });
        let payload = null;
        try {
            payload = await response.json();
        } catch {
            payload = null;
        }

        if (!response.ok) {
            throw new Error(payload?.detail || payload?.message || `HTTP ${response.status}`);
        }

        return payload;
    } catch (e) {
        if (e?.name === 'AbortError') {
            throw new Error('Превышено время ожидания запроса');
        }
        throw e;
    } finally {
        clearTimeout(timer);
    }
};

const mainRequiredColumns = ['Артикул', 'Наименование', 'Тариф с НДС, руб'];

const processButton = document.getElementById('process-file');
const uploadButton = document.getElementById('upload-data');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const logOutput = document.getElementById('log-output');
const sheetSelect = document.getElementById('sheet-name');

const stockProcessButton = document.getElementById('stock-process-file');
const stockUploadButton = document.getElementById('stock-upload-data');
const stockStatusText = document.getElementById('stock-status-text');
const stockProgressFill = document.getElementById('stock-progress-fill');
const stockLogOutput = document.getElementById('stock-log-output');

const passportsUploadButton = document.getElementById('passports-upload-button');
const passportsStatusText = document.getElementById('passports-status-text');
const passportsLogOutput = document.getElementById('passports-log-output');
const passportsSearchButton = document.getElementById('passports-search-button');
const passportsSearchStatus = document.getElementById('passports-search-status');
const passportsSearchOutput = document.getElementById('passports-search-output');

const deleteCollectionButton = document.getElementById('delete-collection-button');
const deleteStatusText = document.getElementById('delete-status-text');
const deleteLogOutput = document.getElementById('delete-log-output');

const mainSearchButton = document.getElementById('main-search-button');
const mainSearchStatus = document.getElementById('main-search-status');
const mainSearchOutput = document.getElementById('main-search-output');
const mainSearchDebug = document.getElementById('main-search-debug');

const aiEndpointInput = document.getElementById('ai-endpoint');
const aiModelInput = document.getElementById('ai-model');
const aiApiKeyInput = document.getElementById('ai-api-key');
const orchestratorUrlInput = document.getElementById('orchestrator-url');
const aiCollectionNameInput = document.getElementById('ai-collection-name');
const aiProbableLimitInput = document.getElementById('ai-probable-limit');
const aiSaveSettingsButton = document.getElementById('ai-save-settings');
const aiSettingsStatus = document.getElementById('ai-settings-status');
const aiMessageInput = document.getElementById('ai-message');
const aiSendChatButton = document.getElementById('ai-send-chat');
const aiSendSpecButton = document.getElementById('ai-send-spec');
const aiChatStatus = document.getElementById('ai-chat-status');
const aiChatOutput = document.getElementById('ai-chat-output');

const xlsxFileInput = document.getElementById('xlsx-file');
const skipRowsInput = document.getElementById('skip-rows');
const collectionNameInput = document.getElementById('collection-name');
const batchSizeInput = document.getElementById('batch-size');
const pointsBatchSizeInput = document.getElementById('points-batch-size');
const articleModeInput = document.getElementById('article-mode');

const stockXlsxFileInput = document.getElementById('stock-xlsx-file');
const stockSkipRowsInput = document.getElementById('stock-skip-rows');
const stockCollectionNameInput = document.getElementById('stock-collection-name');
const stockBatchSizeInput = document.getElementById('stock-batch-size');

const passportsFilesInput = document.getElementById('passports-files');
const passportsCollectionInput = document.getElementById('passports-collection-name');
const passportsBatchSizeInput = document.getElementById('passports-batch-size');
const passportsPointsBatchSizeInput = document.getElementById('passports-points-batch-size');
const passportsSearchQueryInput = document.getElementById('passports-search-query');
const passportsSearchLimitInput = document.getElementById('passports-search-limit');

const mainSearchQueryInput = document.getElementById('main-search-query');
const mainSearchModeInput = document.getElementById('main-search-mode');
const mainSearchInStockInput = document.getElementById('main-search-in-stock');
const mainSearchLimitInput = document.getElementById('main-search-limit');
const mainSearchCandidateLimitInput = document.getElementById('main-search-candidate-limit');
const deleteCollectionNameInput = document.getElementById('delete-collection-name');

let mainSearchController = null;
let aiRequestController = null;

let cachedWorkbook = null;

const aiSettingsKey = 'orchestrator_ai_settings_v1';

const defaultAiSettings = {
    ai_endpoint: 'http://ollama:11434',
    ai_model: 'qwen3.5:9b',
    ai_api_key: '',
    orchestrator_url: 'http://localhost:8430',
    collection_name: 'CHINT',
    probable_limit: 3,
};

const syncCollectionInputs = (name) => {
    if (!name) return;
    const ids = ['collection-name', 'stock-collection-name', 'delete-collection-name', 'ai-collection-name'];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el && !el.value) el.value = name;
    });
};

const preloadCollectionName = async () => {
    try {
        const payload = await fetchJson('/collections');
        const names = payload.collections || [];
        if (!names.length) return;
        const preferred = names.includes('CHINT') ? 'CHINT' : names[0];
        syncCollectionInputs(preferred);
    } catch {
        // ignore preload errors
    }
};

const getAiSettingsFromForm = () => ({
    ai_endpoint: aiEndpointInput?.value?.trim() || defaultAiSettings.ai_endpoint,
    ai_model: aiModelInput?.value?.trim() || defaultAiSettings.ai_model,
    ai_api_key: aiApiKeyInput?.value?.trim() || '',
    orchestrator_url: orchestratorUrlInput?.value?.trim() || defaultAiSettings.orchestrator_url,
    collection_name: aiCollectionNameInput?.value?.trim() || defaultAiSettings.collection_name,
    probable_limit: Math.max(1, Math.min(10, parseInt(aiProbableLimitInput?.value || '3', 10) || 3)),
});

const applyAiSettingsToForm = (settings) => {
    if (aiEndpointInput) aiEndpointInput.value = settings.ai_endpoint || defaultAiSettings.ai_endpoint;
    if (aiModelInput) aiModelInput.value = settings.ai_model || defaultAiSettings.ai_model;
    if (aiApiKeyInput) aiApiKeyInput.value = settings.ai_api_key || '';
    if (orchestratorUrlInput) orchestratorUrlInput.value = settings.orchestrator_url || defaultAiSettings.orchestrator_url;
    if (aiCollectionNameInput) aiCollectionNameInput.value = settings.collection_name || defaultAiSettings.collection_name;
    if (aiProbableLimitInput) aiProbableLimitInput.value = String(settings.probable_limit || defaultAiSettings.probable_limit);
};

const loadAiSettings = () => {
    try {
        const raw = localStorage.getItem(aiSettingsKey);
        if (!raw) {
            applyAiSettingsToForm(defaultAiSettings);
            return;
        }
        const parsed = JSON.parse(raw);
        applyAiSettingsToForm({ ...defaultAiSettings, ...parsed });
    } catch {
        applyAiSettingsToForm(defaultAiSettings);
    }
};

const saveAiSettings = () => {
    const settings = getAiSettingsFromForm();
    localStorage.setItem(aiSettingsKey, JSON.stringify(settings));
    return settings;
};

const renderAiResponse = (payload, settings) => {
    const lines = [];
    lines.push(payload.reply_text || 'Пустой ответ');

    if (Array.isArray(payload.rows) && payload.rows.length) {
        lines.push('');
        lines.push('Rows breakdown:');
        payload.rows.forEach((row, idx) => {
            lines.push(`${idx + 1}) ${row.query_text || 'n/a'}`);
            lines.push(`   score: ${row.score ?? 'n/a'} | score_norm: ${row.score_norm ?? 'n/a'}`);
            lines.push(`   dense: ${(row.dense_top || []).join(' ; ') || 'нет'}`);
            lines.push(`   sparse: ${(row.sparse_top || []).join(' ; ') || 'нет'}`);
            lines.push(`   hybrid: ${(row.hybrid_top || []).join(' ; ') || 'нет'}`);
        });
    }

    if (payload.spec_download_url) {
        const url = `${(settings.orchestrator_url || '').replace(/\/$/, '')}${payload.spec_download_url}`;
        lines.push('');
        lines.push(`Скачать спецификацию: ${url}`);
    }
    if (payload.debug) {
        lines.push('');
        lines.push('Debug:');
        lines.push(JSON.stringify(payload.debug, null, 2));
    }
    return lines.join('\n');
};

const sendAiRequest = async (path) => {
    const message = aiMessageInput?.value?.trim();
    if (!message) return alert('Введите сообщение для AI');

    const settings = saveAiSettings();
    const orchestratorBase = (settings.orchestrator_url || '').replace(/\/$/, '');
    if (!orchestratorBase) return alert('Укажите Orchestrator URL');

    try {
        if (aiRequestController) aiRequestController.abort();
        aiRequestController = new AbortController();
        setStatus(aiChatStatus, 'отправка запроса...');
        const payload = await fetchJson(`${orchestratorBase}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: aiRequestController.signal,
            body: JSON.stringify({
                message,
                collection_name: settings.collection_name,
                ai_base_url: settings.ai_endpoint,
                ai_api_key: settings.ai_api_key,
                ai_model: settings.ai_model,
                probable_limit: settings.probable_limit,
            }),
        }, 90000);

        aiChatOutput.textContent = renderAiResponse(payload, settings);
        setStatus(aiChatStatus, 'готово');
    } catch (e) {
        setStatus(aiChatStatus, 'ошибка');
        aiChatOutput.textContent = `Ошибка: ${e.message}`;
        alert(e.message);
    } finally {
        aiRequestController = null;
    }
};

const renderMappings = (containerId, requiredColumns, headers, prefix = '') => {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    requiredColumns.forEach((requiredCol) => {
        const row = document.createElement('div');
        row.className = 'mapping-row';
        const label = document.createElement('label');
        label.textContent = requiredCol;
        const select = document.createElement('select');
        select.id = `${prefix}select-${requiredCol}`;

        const def = document.createElement('option');
        def.value = '';
        def.textContent = `Выберите столбец для "${requiredCol}"`;
        select.appendChild(def);

        headers.forEach((header, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = header;
            select.appendChild(option);
        });

        row.appendChild(label);
        row.appendChild(select);
        container.appendChild(row);
    });
};

const pollJob = async (url, onProgress) => {
    let done = false;
    while (!done) {
        await new Promise((r) => setTimeout(r, 1500));
        const status = await fetchJson(url);
        onProgress(status);
        if (status.status === 'failed') throw new Error(status.error || 'Ошибка обработки');
        if (status.status === 'completed') done = true;
    }
};

const renderMainSearchOutput = (results = [], debug = null) => {
    const lines = [];
    if (!results.length) {
        return 'Ничего не найдено';
    }

    lines.push('Результаты:');
    lines.push(
        results
            .map((r, i) => `#${i + 1}\nscore: ${r.score ?? 'n/a'}\npayload: ${JSON.stringify(r.payload || {}, null, 2)}\n`)
            .join('\n')
    );

    const breakdown = debug?.score_trace?.breakdown;
    if (breakdown) {
        const block = (title, arr = []) => {
            lines.push('');
            lines.push(`${title}:`);
            if (!arr.length) {
                lines.push('  нет');
                return;
            }
            arr.forEach((item, idx) => {
                lines.push(`  ${idx + 1}. ${item.title || 'n/a'} | rank=${item.rank ?? 'n/a'} | score=${item.score ?? 'n/a'}`);
            });
        };

        lines.push('');
        lines.push('Breakdown (dense / sparse / hybrid):');
        block('Dense top', breakdown.dense_top || []);
        block('Sparse top', breakdown.sparse_top || []);
        block('Hybrid top', breakdown.hybrid_top || []);
    }

    return lines.join('\n');
};

if (processButton) {
    processButton.addEventListener('click', () => {
        const skipRows = parseInt(skipRowsInput?.value || '0', 10) || 0;
        if (!xlsxFileInput?.files?.length) return alert('Выберите XLSX файл');

        const reader = new FileReader();
        setStatus(statusText, 'чтение файла...');
        setProgress(progressFill, 20);

        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            cachedWorkbook = XLSX.read(data, { type: 'array' });
            const sheets = cachedWorkbook.SheetNames || [];
            sheetSelect.innerHTML = '';
            sheets.forEach((name, index) => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                if (index === 0) option.selected = true;
                sheetSelect.appendChild(option);
            });
            sheetSelect.disabled = sheets.length === 0;

            if (!sheets.length) {
                setStatus(statusText, 'листы не найдены');
                setProgress(progressFill, 0);
                return;
            }

            const ws = cachedWorkbook.Sheets[sheetSelect.value || sheets[0]];
            const rows = XLSX.utils.sheet_to_json(ws, { header: 1 });
            const headers = rows[skipRows] || [];
            renderMappings('column-mappings', mainRequiredColumns, headers);
            uploadButton.style.display = 'inline-flex';
            setStatus(statusText, 'готово к загрузке');
            setProgress(progressFill, 70);
        };

        reader.readAsArrayBuffer(xlsxFileInput.files[0]);
    });
}

if (sheetSelect) {
    sheetSelect.addEventListener('change', () => {
        if (!cachedWorkbook) return;
        const skipRows = parseInt(skipRowsInput?.value || '0', 10) || 0;
        const ws = cachedWorkbook.Sheets[sheetSelect.value];
        const rows = XLSX.utils.sheet_to_json(ws, { header: 1 });
        renderMappings('column-mappings', mainRequiredColumns, rows[skipRows] || []);
    });
}

if (uploadButton) {
    uploadButton.addEventListener('click', async () => {
        const file = xlsxFileInput?.files?.[0];
        if (!file) return alert('Выберите файл');

        const mappings = {};
        for (const col of mainRequiredColumns) {
            const v = document.getElementById(`select-${col}`).value;
            if (v !== '') mappings[col] = parseInt(v, 10);
        }
        if (Object.keys(mappings).length !== mainRequiredColumns.length) {
            return alert('Заполните маппинг обязательных столбцов');
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('skip_rows', skipRowsInput?.value || '0');
        formData.append('mappings', JSON.stringify(mappings));
        formData.append('collection_name', collectionNameInput?.value || 'my_collection');
        formData.append('article_mode', articleModeInput?.value || 'price');
        formData.append('batch_size', batchSizeInput?.value || '16');
        formData.append('points_batch_size', pointsBatchSizeInput?.value || '200');
        if (sheetSelect.value) formData.append('sheet_name', sheetSelect.value);

        try {
            setStatus(statusText, 'запуск задачи...');
            const { job_id } = await fetchJson('/upload_processed_xlsx_async', { method: 'POST', body: formData });
            addLog(logOutput, `Задача: ${job_id}`);

            await pollJob(`/upload_status/${job_id}`, (s) => {
                setProgress(progressFill, s.progress || 0);
                setStatus(statusText, `в процессе... ${s.progress || 0}%`);
            });

            setStatus(statusText, 'успех');
            setProgress(progressFill, 100);
        } catch (e) {
            setStatus(statusText, 'ошибка');
            setProgress(progressFill, 0);
            addLog(logOutput, `Ошибка: ${e.message}`);
            alert(e.message);
        }
    });
}

if (stockProcessButton) {
    stockProcessButton.addEventListener('click', () => {
        const skipRows = parseInt(stockSkipRowsInput?.value || '0', 10) || 0;
        if (!stockXlsxFileInput?.files?.length) return alert('Выберите XLSX файл');

        const reader = new FileReader();
        setStatus(stockStatusText, 'чтение файла...');
        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            const workbook = XLSX.read(data, { type: 'array' });
            const ws = workbook.Sheets[workbook.SheetNames[0]];
            const rows = XLSX.utils.sheet_to_json(ws, { header: 1 });
            renderMappings('stock-column-mappings', ['Артикул', 'Остаток'], rows[skipRows] || [], 'stock-');
            stockUploadButton.style.display = 'inline-flex';
            setStatus(stockStatusText, 'готово к обновлению');
            setProgress(stockProgressFill, 60);
        };
        reader.readAsArrayBuffer(stockXlsxFileInput.files[0]);
    });
}

if (stockUploadButton) {
    stockUploadButton.addEventListener('click', async () => {
        const file = stockXlsxFileInput?.files?.[0];
        if (!file) return alert('Выберите файл');

        const articleCol = document.getElementById('stock-select-Артикул').value;
        const stockCol = document.getElementById('stock-select-Остаток').value;
        if (articleCol === '' || stockCol === '') return alert('Заполните маппинг');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('skip_rows', stockSkipRowsInput?.value || '0');
        formData.append('collection_name', stockCollectionNameInput?.value || 'my_collection');
        formData.append('article_col', articleCol);
        formData.append('stock_col', stockCol);
        formData.append('batch_size', stockBatchSizeInput?.value || '200');

        try {
            const { job_id } = await fetchJson('/upload_stock_async', { method: 'POST', body: formData });
            addLog(stockLogOutput, `Задача: ${job_id}`);

            await pollJob(`/stock_status/${job_id}`, (s) => {
                setProgress(stockProgressFill, s.progress || 0);
                setStatus(stockStatusText, `в процессе... ${s.progress || 0}%`);
            });

            setStatus(stockStatusText, 'успех');
            setProgress(stockProgressFill, 100);
        } catch (e) {
            setStatus(stockStatusText, 'ошибка');
            setProgress(stockProgressFill, 0);
            addLog(stockLogOutput, `Ошибка: ${e.message}`);
            alert(e.message);
        }
    });
}

if (passportsUploadButton) {
    passportsUploadButton.addEventListener('click', async () => {
        const files = passportsFilesInput?.files;
        if (!files.length) return alert('Выберите PDF файлы');

        const formData = new FormData();
        Array.from(files).forEach((f) => formData.append('files', f));
        formData.append('collection_name', passportsCollectionInput?.value || 'passports_collection');
        formData.append('batch_size', passportsBatchSizeInput?.value || '8');
        formData.append('points_batch_size', passportsPointsBatchSizeInput?.value || '200');

        try {
            const { job_id } = await fetchJson('/upload_passports_async', { method: 'POST', body: formData });
            addLog(passportsLogOutput, `Задача: ${job_id}`);

            await pollJob(`/passports_status/${job_id}`, (s) => {
                setStatus(passportsStatusText, `в процессе... ${s.progress || 0}%`);
            });

            setStatus(passportsStatusText, 'успех');
        } catch (e) {
            setStatus(passportsStatusText, 'ошибка');
            addLog(passportsLogOutput, `Ошибка: ${e.message}`);
            alert(e.message);
        }
    });
}

if (passportsSearchButton) {
    passportsSearchButton.addEventListener('click', async () => {
        const q = passportsSearchQueryInput?.value?.trim();
        if (!q) return alert('Введите запрос');
        const collection = passportsCollectionInput?.value || 'passports_collection';
        const limit = passportsSearchLimitInput?.value || '5';

        try {
            setStatus(passportsSearchStatus, 'поиск...');
            const payload = await fetchJson(`/search_passports?collection_name=${encodeURIComponent(collection)}&query=${encodeURIComponent(q)}&limit=${encodeURIComponent(limit)}`);
            const results = payload.results || [];
            if (!results.length) {
                passportsSearchOutput.textContent = 'Ничего не найдено';
            } else {
                passportsSearchOutput.textContent = results.map((r, i) => {
                    return `#${i + 1}\nscore: ${r.score ?? 'n/a'}\npayload: ${JSON.stringify(r.payload || {}, null, 2)}\n`;
                }).join('\n');
            }
            setStatus(passportsSearchStatus, 'готово');
        } catch (e) {
            setStatus(passportsSearchStatus, 'ошибка');
            passportsSearchOutput.textContent = `Ошибка: ${e.message}`;
            alert(e.message);
        }
    });
}

if (mainSearchButton) {
    mainSearchButton.addEventListener('click', async () => {
        const q = mainSearchQueryInput?.value?.trim();
        if (!q) return alert('Введите запрос');

        const collection = collectionNameInput?.value || 'my_collection';
        const mode = mainSearchModeInput?.value || 'hybrid';
        const onlyInStock = mainSearchInStockInput?.value === 'true';
        const limit = parseInt(mainSearchLimitInput?.value || '15', 10) || 15;
        const candidateLimit = parseInt(mainSearchCandidateLimitInput?.value || '20', 10) || 20;

        const params = new URLSearchParams({
            collection_name: collection,
            query: q,
            mode,
            only_in_stock: String(onlyInStock),
            limit: String(limit),
            candidate_limit: String(candidateLimit),
            include_breakdown: String(mode === 'hybrid'),
        });

        try {
            if (mainSearchController) mainSearchController.abort();
            mainSearchController = new AbortController();
            setStatus(mainSearchStatus, 'поиск...');
            const payload = await fetchJson(`/search?${params.toString()}`, { signal: mainSearchController.signal }, 60000);
            const results = payload.results || [];
            mainSearchOutput.textContent = renderMainSearchOutput(results, payload.debug || null);
            mainSearchDebug.textContent = payload.debug
                ? JSON.stringify(payload.debug, null, 2)
                : 'Debug не вернулся';
            setStatus(mainSearchStatus, 'готово');
        } catch (e) {
            if (e.message === 'Превышено время ожидания запроса') {
                addLog(mainSearchOutput, 'Поиск прерван по таймауту или отменён новым запросом');
            }
            setStatus(mainSearchStatus, 'ошибка');
            mainSearchOutput.textContent = `Ошибка: ${e.message}`;
            if (mainSearchDebug) mainSearchDebug.textContent = 'Debug недоступен';
            alert(e.message);
        } finally {
            mainSearchController = null;
        }
    });
}

if (deleteCollectionButton) {
    deleteCollectionButton.addEventListener('click', async () => {
        const collection = deleteCollectionNameInput?.value?.trim();
        if (!collection) return alert('Введите имя коллекции');
        if (!confirm(`Удалить коллекцию ${collection}?`)) return;

        try {
            setStatus(deleteStatusText, 'удаление...');
            await fetchJson(`/collection?collection_name=${encodeURIComponent(collection)}`, { method: 'DELETE' });
            setStatus(deleteStatusText, 'удалено');
            addLog(deleteLogOutput, `Удалена коллекция ${collection}`);
        } catch (e) {
            setStatus(deleteStatusText, 'ошибка');
            addLog(deleteLogOutput, `Ошибка: ${e.message}`);
            alert(e.message);
        }
    });
}

if (aiSaveSettingsButton) {
    loadAiSettings();
    setStatus(aiSettingsStatus, 'настройки загружены');
    aiSaveSettingsButton.addEventListener('click', () => {
        saveAiSettings();
        setStatus(aiSettingsStatus, 'настройки сохранены локально');
    });
}

preloadCollectionName();

if (aiSendChatButton) {
    aiSendChatButton.addEventListener('click', async () => {
        await sendAiRequest('/agent/chat');
    });
}

if (aiSendSpecButton) {
    aiSendSpecButton.addEventListener('click', async () => {
        await sendAiRequest('/agent/spec');
    });
}
