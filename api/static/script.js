const processButton = document.getElementById('process-file');
const uploadButton = document.getElementById('upload-data');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const logOutput = document.getElementById('log-output');
const sheetSelect = document.getElementById('sheet-name');
let cachedWorkbook = null;
let cachedFileBuffer = null;
const mainRequiredColumns = ['Артикул', 'Наименование', 'Тариф с НДС, руб'];

const stockProcessButton = document.getElementById('stock-process-file');
const stockUploadButton = document.getElementById('stock-upload-data');
const stockStatusText = document.getElementById('stock-status-text');
const stockProgressFill = document.getElementById('stock-progress-fill');
const stockLogOutput = document.getElementById('stock-log-output');

const deleteCollectionButton = document.getElementById('delete-collection-button');
const deleteStatusText = document.getElementById('delete-status-text');
const deleteLogOutput = document.getElementById('delete-log-output');

const passportsUploadButton = document.getElementById('passports-upload-button');
const passportsStatusText = document.getElementById('passports-status-text');
const passportsLogOutput = document.getElementById('passports-log-output');
const passportsSearchButton = document.getElementById('passports-search-button');
const passportsSearchStatus = document.getElementById('passports-search-status');
const passportsSearchOutput = document.getElementById('passports-search-output');

const hfCacheRefreshButton = document.getElementById('hf-cache-refresh');
const hfCacheClearButton = document.getElementById('hf-cache-clear');
const hfCacheStatus = document.getElementById('hf-cache-status');
const hfCacheLog = document.getElementById('hf-cache-log');
const hfHomePath = document.getElementById('hf-home-path');
const hfHubCachePath = document.getElementById('hf-hub-cache-path');
const hfCacheSize = document.getElementById('hf-cache-size');
const hfCacheFiles = document.getElementById('hf-cache-files');

const mainSearchButton = document.getElementById('main-search-button');
const mainSearchStatus = document.getElementById('main-search-status');
const mainSearchOutput = document.getElementById('main-search-output');

const integrationsRefreshButton = document.getElementById('integrations-refresh');
const integrationsSaveButton = document.getElementById('integrations-save');
const bitrixOauthConnectButton = document.getElementById('bitrix-oauth-connect');
const bitrixOauthRefreshButton = document.getElementById('bitrix-oauth-refresh');
const bitrixOauthStatusButton = document.getElementById('bitrix-oauth-status');
const integrationsStatus = document.getElementById('integrations-status');
const integrationsLog = document.getElementById('integrations-log');

const setStatus = (message) => {
    statusText.textContent = `Статус: ${message}`;
};

const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    logOutput.textContent = `[${timestamp}] ${message}\n` + logOutput.textContent;
};

const setProgress = (value) => {
    const safeValue = Math.max(0, Math.min(100, value));
    progressFill.style.width = `${safeValue}%`;
};

const setStockStatus = (message) => {
    stockStatusText.textContent = `Статус: ${message}`;
};

const addStockLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    stockLogOutput.textContent = `[${timestamp}] ${message}\n` + stockLogOutput.textContent;
};

const setStockProgress = (value) => {
    const safeValue = Math.max(0, Math.min(100, value));
    stockProgressFill.style.width = `${safeValue}%`;
};

const setDeleteStatus = (message) => {
    if (!deleteStatusText) {
        return;
    }
    deleteStatusText.textContent = `Статус: ${message}`;
};

const addDeleteLog = (message) => {
    if (!deleteLogOutput) {
        return;
    }
    const timestamp = new Date().toLocaleTimeString();
    deleteLogOutput.textContent = `[${timestamp}] ${message}\n` + deleteLogOutput.textContent;
};

const setPassportsStatus = (message) => {
    if (!passportsStatusText) {
        return;
    }
    passportsStatusText.textContent = `Статус: ${message}`;
};

const addPassportsLog = (message) => {
    if (!passportsLogOutput) {
        return;
    }
    const timestamp = new Date().toLocaleTimeString();
    passportsLogOutput.textContent = `[${timestamp}] ${message}\n` + passportsLogOutput.textContent;
};

const setPassportsSearchStatus = (message) => {
    if (!passportsSearchStatus) {
        return;
    }
    passportsSearchStatus.textContent = `Статус: ${message}`;
};

const setPassportsSearchOutput = (message) => {
    if (!passportsSearchOutput) {
        return;
    }
    passportsSearchOutput.textContent = message;
};

const setHfCacheStatus = (message) => {
    if (!hfCacheStatus) {
        return;
    }
    hfCacheStatus.textContent = `Статус: ${message}`;
};

const setMainSearchStatus = (message) => {
    if (!mainSearchStatus) {
        return;
    }
    mainSearchStatus.textContent = `Статус: ${message}`;
};

const setMainSearchOutput = (message) => {
    if (!mainSearchOutput) {
        return;
    }
    mainSearchOutput.textContent = message;
};

const setIntegrationsStatus = (message) => {
    if (!integrationsStatus) {
        return;
    }
    integrationsStatus.textContent = `Статус: ${message}`;
};

const addIntegrationsLog = (message) => {
    if (!integrationsLog) {
        return;
    }
    const timestamp = new Date().toLocaleTimeString();
    integrationsLog.textContent = `[${timestamp}] ${message}\n` + integrationsLog.textContent;
};

const renderRuntimeConfig = (payload) => {
    const polza = payload?.polza || {};
    const bitrix = payload?.bitrix || {};

    const polzaApiKeyMasked = document.getElementById('polza-api-key-masked');
    const polzaModel = document.getElementById('polza-model');
    const polzaTemperature = document.getElementById('polza-temperature');
    const polzaMaxTokens = document.getElementById('polza-max-tokens');
    const polzaBaseUrl = document.getElementById('polza-base-url');

    const bitrixClientId = document.getElementById('bitrix-client-id');
    const bitrixClientSecretMasked = document.getElementById('bitrix-client-secret-masked');
    const bitrixRedirectUri = document.getElementById('bitrix-redirect-uri');
    const bitrixWebhookUrl = document.getElementById('bitrix-webhook-url');
    const bitrixPortalBaseUrl = document.getElementById('bitrix-portal-base-url');
    const bitrixOauthAuthUrl = document.getElementById('bitrix-oauth-auth-url');
    const bitrixOauthTokenUrl = document.getElementById('bitrix-oauth-token-url');
    const bitrixBotId = document.getElementById('bitrix-bot-id');
    const bitrixCollectionName = document.getElementById('bitrix-collection-name');
    const bitrixDocsCollectionName = document.getElementById('bitrix-docs-collection-name');
    const bitrixSearchMode = document.getElementById('bitrix-search-mode');

    if (polzaApiKeyMasked) polzaApiKeyMasked.value = polza.api_key_masked || '';
    if (polzaModel) polzaModel.value = polza.model || 'openai/gpt-4o';
    if (polzaTemperature) polzaTemperature.value = polza.temperature ?? 0.2;
    if (polzaMaxTokens) polzaMaxTokens.value = polza.max_tokens ?? 500;
    if (polzaBaseUrl) polzaBaseUrl.value = polza.base_url || 'https://polza.ai/api/v1/chat/completions';

    if (bitrixClientId) bitrixClientId.value = bitrix.client_id || '';
    if (bitrixClientSecretMasked) bitrixClientSecretMasked.value = bitrix.client_secret_masked || '';
    if (bitrixRedirectUri) bitrixRedirectUri.value = bitrix.redirect_uri || '';
    if (bitrixWebhookUrl) bitrixWebhookUrl.value = bitrix.webhook_url || '';
    if (bitrixPortalBaseUrl) bitrixPortalBaseUrl.value = bitrix.portal_base_url || '';
    if (bitrixOauthAuthUrl) bitrixOauthAuthUrl.value = bitrix.oauth_auth_url || 'https://oauth.bitrix.info/oauth/authorize/';
    if (bitrixOauthTokenUrl) bitrixOauthTokenUrl.value = bitrix.oauth_token_url || 'https://oauth.bitrix.info/oauth/token/';
    if (bitrixBotId) bitrixBotId.value = bitrix.bot_id || '';
    if (bitrixCollectionName) bitrixCollectionName.value = bitrix.collection_name || 'my_collection';
    if (bitrixDocsCollectionName) bitrixDocsCollectionName.value = bitrix.docs_collection_name || 'passports_collection';
    if (bitrixSearchMode) bitrixSearchMode.value = bitrix.search_mode || 'hybrid';

    if (bitrix.oauth_connected) {
        addIntegrationsLog(
            `OAuth подключен. portal=${bitrix.portal_base_url || 'n/a'}, token=${bitrix.access_token_masked || '***'}`
        );
    }
};

const loadRuntimeConfig = async () => {
    const response = await fetch('/runtime_config');
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Не удалось получить runtime_config');
    }
    return response.json();
};

const addHfCacheLog = (message) => {
    if (!hfCacheLog) {
        return;
    }
    const timestamp = new Date().toLocaleTimeString();
    hfCacheLog.textContent = `[${timestamp}] ${message}\n` + hfCacheLog.textContent;
};

const renderHfCacheInfo = (payload) => {
    if (hfHomePath) {
        hfHomePath.value = payload.hf_home || '';
    }
    if (hfHubCachePath) {
        hfHubCachePath.value = payload.hf_hub_cache || '';
    }
    if (hfCacheSize) {
        hfCacheSize.value = (payload.size_mb ?? 0).toString();
    }
    if (hfCacheFiles) {
        hfCacheFiles.value = (payload.file_count ?? 0).toString();
    }
};

const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

const switchTab = (tabName) => {
    tabButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.tab === tabName);
    });
    tabContents.forEach((content) => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
};

if (tabButtons.length) {
    tabButtons.forEach((button) => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });
}

const renderMainMappings = (headers) => {
    const columnMappingsDiv = document.getElementById('column-mappings');
    columnMappingsDiv.innerHTML = '';

    mainRequiredColumns.forEach(requiredCol => {
        const row = document.createElement('div');
        row.className = 'mapping-row';

        const label = document.createElement('label');
        label.textContent = requiredCol;

        const select = document.createElement('select');
        select.id = `select-${requiredCol}`;

        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = `Выберите столбец для "${requiredCol}"`;
        select.appendChild(defaultOption);

        headers.forEach((header, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = header;
            select.appendChild(option);
        });

        row.appendChild(label);
        row.appendChild(select);
        columnMappingsDiv.appendChild(row);
    });
};

const updateMainSheetMappings = () => {
    if (!cachedWorkbook) {
        return;
    }
    const skipRows = parseInt(document.getElementById('skip-rows').value, 10) || 0;
    const selectedSheet = sheetSelect.value || cachedWorkbook.SheetNames[0];
    const worksheet = cachedWorkbook.Sheets[selectedSheet];
    if (!worksheet) {
        return;
    }
    const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    const headers = json[skipRows] || [];
    renderMainMappings(headers);
};

sheetSelect.addEventListener('change', () => {
    updateMainSheetMappings();
});

processButton.addEventListener('click', () => {
    const fileInput = document.getElementById('xlsx-file');
    const skipRows = parseInt(document.getElementById('skip-rows').value, 10);

    if (fileInput.files.length === 0) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    setStatus('чтение файла...');
    setProgress(5);

    reader.onprogress = function(event) {
        if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 40);
            setProgress(percent);
        }
    };

    reader.onload = function(e) {
        setStatus('обработка заголовков...');
        setProgress(55);

        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, {type: 'array'});
        cachedWorkbook = workbook;
        cachedFileBuffer = data;
        const sheetNames = workbook.SheetNames || [];

        sheetSelect.innerHTML = '';
        if (!sheetNames.length) {
            sheetSelect.disabled = true;
            sheetSelect.innerHTML = '<option value="">Листы не найдены</option>';
            setStatus('листов не найдено');
            setProgress(0);
            return;
        }

        sheetNames.forEach((name, index) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            if (index === 0) {
                option.selected = true;
            }
            sheetSelect.appendChild(option);
        });
        sheetSelect.disabled = false;

        updateMainSheetMappings();

        uploadButton.style.display = 'inline-flex';
        setStatus('готово к загрузке. Настройте маппинг.');
        setProgress(70);
    };

    reader.onerror = function() {
        setStatus('ошибка чтения файла');
        setProgress(0);
    };

    reader.readAsArrayBuffer(file);
});

uploadButton.addEventListener('click', async () => {
    const fileInput = document.getElementById('xlsx-file');
    const skipRows = parseInt(document.getElementById('skip-rows').value, 10);
    const batchSize = parseInt(document.getElementById('batch-size').value, 10);
    const pointsBatchSize = parseInt(document.getElementById('points-batch-size').value, 10);
    const file = fileInput.files[0];
    const collectionName = document.getElementById('collection-name').value;
    const articleMode = document.getElementById('article-mode').value;
    const sheetName = sheetSelect ? sheetSelect.value : '';

    if (!file) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const mappings = {};
    const requiredColumns = mainRequiredColumns;
    requiredColumns.forEach(requiredCol => {
        const selectedIndex = document.getElementById(`select-${requiredCol}`).value;
        if (selectedIndex !== '') {
            mappings[requiredCol] = parseInt(selectedIndex, 10);
        }
    });

    if (Object.keys(mappings).length !== requiredColumns.length) {
        alert('Заполните сопоставление всех обязательных столбцов.');
        return;
    }

    setStatus('отправка данных на сервер...');
    setProgress(80);
    addLog('Старт загрузки.');
    processButton.disabled = true;
    uploadButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('skip_rows', skipRows);
    formData.append('mappings', JSON.stringify(mappings));
    formData.append('collection_name', collectionName);
    formData.append('article_mode', articleMode);
    formData.append('batch_size', isNaN(batchSize) ? 16 : batchSize);
    formData.append('points_batch_size', isNaN(pointsBatchSize) ? 200 : pointsBatchSize);
    if (sheetName) {
        formData.append('sheet_name', sheetName);
    }

    const pollStatus = async (jobId) => {
        const response = await fetch(`/upload_status/${jobId}`);
        if (!response.ok) {
            throw new Error('Ошибка получения статуса загрузки.');
        }
        return response.json();
    };

    try {
        const response = await fetch('/upload_processed_xlsx_async', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Произошла ошибка при старте загрузки.');
        }

        const { job_id: jobId } = await response.json();
        addLog(`Задача запущена: ${jobId}`);

        let isRunning = true;
        while (isRunning) {
            await new Promise((resolve) => setTimeout(resolve, 1500));
            const status = await pollStatus(jobId);

            if (status.status === 'failed') {
                throw new Error(status.error || 'Ошибка обработки файла.');
            }

            const progressValue = status.progress ?? 0;
            setProgress(progressValue);
            setStatus(`в процессе... ${progressValue}%`);

            if (status.total_rows) {
                addLog(
                    `Прогресс: ${status.indexed_rows}/${status.total_rows} | ` +
                    `${status.rate || 0} строк/сек | ETA ${status.eta || 0}с`
                );
            }

            if (status.status === 'completed') {
                isRunning = false;
                setStatus(`успех! Проиндексировано строк: ${status.indexed_rows}`);
                setProgress(100);
                addLog(`Готово за ${status.duration_sec || 0} сек.`);
            }
        }
    } catch (error) {
        setStatus('ошибка загрузки данных.');
        setProgress(0);
        addLog(`Ошибка: ${error.message}`);
        alert(error.message);
    } finally {
        processButton.disabled = false;
        uploadButton.disabled = false;
    }
});

stockProcessButton.addEventListener('click', () => {
    const fileInput = document.getElementById('stock-xlsx-file');
    const skipRows = parseInt(document.getElementById('stock-skip-rows').value, 10);

    if (fileInput.files.length === 0) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    setStockStatus('чтение файла...');
    setStockProgress(5);

    reader.onprogress = function(event) {
        if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 40);
            setStockProgress(percent);
        }
    };

    reader.onload = function(e) {
        setStockStatus('обработка заголовков...');
        setStockProgress(55);

        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, {type: 'array'});
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        const headers = json[skipRows];
        const columnMappingsDiv = document.getElementById('stock-column-mappings');
        columnMappingsDiv.innerHTML = '';

        const requiredColumns = ['Артикул', 'Остаток'];

        requiredColumns.forEach(requiredCol => {
            const row = document.createElement('div');
            row.className = 'mapping-row';

            const label = document.createElement('label');
            label.textContent = requiredCol;

            const select = document.createElement('select');
            select.id = `stock-select-${requiredCol}`;

            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = `Выберите столбец для "${requiredCol}"`;
            select.appendChild(defaultOption);

            headers.forEach((header, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = header;
                select.appendChild(option);
            });

            row.appendChild(label);
            row.appendChild(select);
            columnMappingsDiv.appendChild(row);
        });

        stockUploadButton.style.display = 'inline-flex';
        setStockStatus('готово к обновлению. Настройте маппинг.');
        setStockProgress(70);
    };

    reader.onerror = function() {
        setStockStatus('ошибка чтения файла');
        setStockProgress(0);
    };

    reader.readAsArrayBuffer(file);
});

stockUploadButton.addEventListener('click', async () => {
    const fileInput = document.getElementById('stock-xlsx-file');
    const skipRows = parseInt(document.getElementById('stock-skip-rows').value, 10);
    const batchSize = parseInt(document.getElementById('stock-batch-size').value, 10);
    const file = fileInput.files[0];
    const collectionName = document.getElementById('stock-collection-name').value;

    if (!file) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const mappings = {};
    const requiredColumns = ['Артикул', 'Остаток'];
    requiredColumns.forEach(requiredCol => {
        const selectedIndex = document.getElementById(`stock-select-${requiredCol}`).value;
        if (selectedIndex !== '') {
            mappings[requiredCol] = parseInt(selectedIndex, 10);
        }
    });

    if (Object.keys(mappings).length !== requiredColumns.length) {
        alert('Заполните сопоставление всех обязательных столбцов.');
        return;
    }

    setStockStatus('отправка данных на сервер...');
    setStockProgress(80);
    addStockLog('Старт обновления.');
    stockProcessButton.disabled = true;
    stockUploadButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('skip_rows', skipRows);
    formData.append('collection_name', collectionName);
    formData.append('article_col', mappings['Артикул']);
    formData.append('stock_col', mappings['Остаток']);
    formData.append('batch_size', isNaN(batchSize) ? 200 : batchSize);

    const pollStatus = async (jobId) => {
        const response = await fetch(`/stock_status/${jobId}`);
        if (!response.ok) {
            throw new Error('Ошибка получения статуса обновления.');
        }
        return response.json();
    };

    try {
        const response = await fetch('/upload_stock_async', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Произошла ошибка при старте обновления.');
        }

        const { job_id: jobId } = await response.json();
        addStockLog(`Задача запущена: ${jobId}`);

        let isRunning = true;
        while (isRunning) {
            await new Promise((resolve) => setTimeout(resolve, 1500));
            const status = await pollStatus(jobId);

            if (status.status === 'failed') {
                throw new Error(status.error || 'Ошибка обработки файла.');
            }

            const progressValue = status.progress ?? 0;
            setStockProgress(progressValue);
            setStockStatus(`в процессе... ${progressValue}%`);

            if (status.total_rows) {
                addStockLog(
                    `Прогресс: ${status.processed_rows}/${status.total_rows} | ` +
                    `обновлено ${status.updated_rows || 0}, пропущено ${status.skipped_rows || 0} | ` +
                    `${status.rate || 0} строк/сек | ETA ${status.eta || 0}с`
                );
            }

            if (status.status === 'completed') {
                isRunning = false;
                setStockStatus(`успех! Обновлено: ${status.updated_rows}`);
                setStockProgress(100);
                addStockLog(`Готово за ${status.duration_sec || 0} сек.`);
            }

            if (status.status === 'resetting') {
                setStockStatus('обнуление остатков...');
                setStockProgress(5);
                addStockLog('Обнуление остатков по всей коллекции.');
            }
        }
    } catch (error) {
        setStockStatus('ошибка обновления.');
        setStockProgress(0);
        addStockLog(`Ошибка: ${error.message}`);
        alert(error.message);
    } finally {
        stockProcessButton.disabled = false;
        stockUploadButton.disabled = false;
    }
});

if (deleteCollectionButton) {
    deleteCollectionButton.addEventListener('click', async () => {
        const collectionName = document.getElementById('delete-collection-name').value;
        if (!collectionName) {
            alert('Введите имя коллекции.');
            return;
        }

        const confirmed = confirm(
            `Удалить коллекцию "${collectionName}"? Все данные, dense/sparse векторы и индексы будут удалены.`
        );
        if (!confirmed) {
            return;
        }

        setDeleteStatus('удаление коллекции...');
        addDeleteLog(`Запрос удаления: ${collectionName}`);
        deleteCollectionButton.disabled = true;

        try {
            const response = await fetch(`/collection?collection_name=${encodeURIComponent(collectionName)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось удалить коллекцию.');
            }

            const payload = await response.json();
            setDeleteStatus('коллекция удалена');
            addDeleteLog(`Удалено: ${payload.collection_name}`);
        } catch (error) {
            setDeleteStatus('ошибка удаления');
            addDeleteLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            deleteCollectionButton.disabled = false;
        }
    });
}

if (passportsUploadButton) {
    passportsUploadButton.addEventListener('click', async () => {
        const fileInput = document.getElementById('passports-files');
        const collectionName = document.getElementById('passports-collection-name').value;
        const batchSize = parseInt(document.getElementById('passports-batch-size').value, 10);
        const pointsBatchSize = parseInt(document.getElementById('passports-points-batch-size').value, 10);
        const files = fileInput.files;

        if (!files || files.length === 0) {
            alert('Пожалуйста, выберите PDF файлы.');
            return;
        }

        if (files.length > 30) {
            alert('Можно выбрать не более 30 файлов за раз.');
            return;
        }

        setPassportsStatus('загрузка файлов...');
        addPassportsLog(`Отправка ${files.length} файлов.`);
        passportsUploadButton.disabled = true;

        const formData = new FormData();
        Array.from(files).forEach((file) => formData.append('files', file));
        formData.append('collection_name', collectionName);
        formData.append('batch_size', isNaN(batchSize) ? 8 : batchSize);
        formData.append('points_batch_size', isNaN(pointsBatchSize) ? 200 : pointsBatchSize);

        const pollStatus = async (jobId) => {
            const response = await fetch(`/passports_status/${jobId}`);
            if (!response.ok) {
                throw new Error('Ошибка получения статуса загрузки паспортов.');
            }
            return response.json();
        };

        try {
            const response = await fetch('/upload_passports_async', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Ошибка старта загрузки паспортов.');
            }

            const { job_id: jobId } = await response.json();
            addPassportsLog(`Задача запущена: ${jobId}`);

            let isRunning = true;
            while (isRunning) {
                await new Promise((resolve) => setTimeout(resolve, 1500));
                const status = await pollStatus(jobId);

                if (status.status === 'failed') {
                    throw new Error(status.error || 'Ошибка обработки паспортов.');
                }

                const progressValue = status.progress ?? 0;
                setPassportsStatus(`в процессе... ${progressValue}%`);

                if (status.total_chunks) {
                    addPassportsLog(
                        `Прогресс: ${status.indexed_chunks || 0}/${status.total_chunks}`
                    );
                }

                if (status.status === 'completed') {
                    isRunning = false;
                    setPassportsStatus(
                        `успех! Чанков: ${status.indexed_chunks || 0}, пропущено файлов: ${status.skipped_files || 0}`
                    );
                    addPassportsLog(`Готово за ${status.duration_sec || 0} сек.`);
                }
            }
        } catch (error) {
            setPassportsStatus('ошибка загрузки');
            addPassportsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            passportsUploadButton.disabled = false;
        }
    });
}

if (passportsSearchButton) {
    passportsSearchButton.addEventListener('click', async () => {
        const collectionName = document.getElementById('passports-collection-name').value;
        const query = document.getElementById('passports-search-query').value.trim();
        const limit = parseInt(document.getElementById('passports-search-limit').value, 10) || 5;

        if (!query) {
            alert('Введите поисковый запрос.');
            return;
        }

        setPassportsSearchStatus('поиск...');
        setPassportsSearchOutput('');
        passportsSearchButton.disabled = true;

        try {
            const response = await fetch(
                `/search_passports?collection_name=${encodeURIComponent(collectionName)}` +
                `&query=${encodeURIComponent(query)}&limit=${limit}`
            );

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Ошибка поиска.');
            }

            const payload = await response.json();
            const results = payload.results || [];
            const stage1Results = payload.stage1_results || [];
            const selectedPdf = payload.selected_pdf || 'не выбран';
            const debug = payload.debug || {};
            const stage1Hits = debug.stage1_hits ?? 'n/a';
            const sparseNonzero = debug.sparse_nonzero ?? 'n/a';
            const denseDim = debug.dense_dim ?? 'n/a';
            const pdfCounts = debug.stage1_pdf_counts || {};

            addPassportsLog(
                `Выбранный PDF: ${selectedPdf} | stage1 hits: ${stage1Hits} | ` +
                `dense_dim: ${denseDim} | sparse_nonzero: ${sparseNonzero}`
            );

            const formattedStage1 = stage1Results.map((item, index) => {
                const payloadData = item.payload || {};
                const text = payloadData.text || '';
                const preview = text.length > 200 ? `${text.slice(0, 200)}...` : text;
                const score = item.score ?? 'n/a';
                return [
                    `S1 #${index + 1} score: ${score} | PDF: ${payloadData.pdf_name || 'unknown'} | страницы ${payloadData.page_range || ''}`,
                    preview,
                    ''
                ].join('\n');
            }).join('\n');

            if (!results.length) {
                const formattedSummary = [
                    `Выбранный PDF: ${selectedPdf}`,
                    `Stage1 hits: ${stage1Hits}`,
                    `Dense dim: ${denseDim}`,
                    `Sparse nonzero: ${sparseNonzero}`,
                    `PDF статистика: ${JSON.stringify(pdfCounts, null, 2)}`,
                    '',
                    'Stage1 результаты (score):',
                    formattedStage1 || 'n/a',
                ].join('\n');

                setPassportsSearchStatus('ничего не найдено');
                setPassportsSearchOutput(formattedSummary);
                return;
            }

            const formattedResults = results.map((item, index) => {
                const payloadData = item.payload || {};
                const text = payloadData.text || '';
                const preview = text.length > 500 ? `${text.slice(0, 500)}...` : text;
                const vectors = item.vector || {};
                const denseVector = vectors['text-dense'] || [];
                const sparseVector = vectors['text-sparse'] || {};
                const densePreview = denseVector.length
                    ? JSON.stringify(denseVector.slice(0, 10)) + (denseVector.length > 10 ? ' ...' : '')
                    : 'n/a';
                const sparseIndices = sparseVector.indices || [];
                const sparseValues = sparseVector.values || [];
                const sparsePreview = sparseIndices.length
                    ? JSON.stringify(
                        sparseIndices.slice(0, 10).map((idx, i) => [idx, sparseValues[i]])
                    ) + (sparseIndices.length > 10 ? ' ...' : '')
                    : 'n/a';
                return [
                    `#${index + 1} PDF: ${payloadData.pdf_name || 'unknown'} | страницы ${payloadData.page_range || ''}`,
                    preview,
                    `Dense vector (first 10): ${densePreview}`,
                    `Sparse vector (idx,val first 10): ${sparsePreview}`,
                    ''
                ].join('\n');
            }).join('\n');

            const formattedSummary = [
                `Выбранный PDF: ${selectedPdf}`,
                `Stage1 hits: ${stage1Hits}`,
                `Dense dim: ${denseDim}`,
                `Sparse nonzero: ${sparseNonzero}`,
                `PDF статистика: ${JSON.stringify(pdfCounts, null, 2)}`,
                '',
                'Stage1 результаты (score):',
                formattedStage1 || 'n/a',
                '',
                formattedResults,
            ].join('\n');

            setPassportsSearchStatus('готово');
            setPassportsSearchOutput(formattedSummary);
        } catch (error) {
            setPassportsSearchStatus('ошибка поиска');
            setPassportsSearchOutput(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            passportsSearchButton.disabled = false;
        }
    });
}

if (mainSearchButton) {
    mainSearchButton.addEventListener('click', async () => {
        const collectionName = document.getElementById('collection-name').value;
        const query = document.getElementById('main-search-query').value.trim();
        const mode = document.getElementById('main-search-mode').value;
        const onlyInStock = document.getElementById('main-search-in-stock').value === 'true';

        if (!query) {
            alert('Введите поисковый запрос.');
            return;
        }

        setMainSearchStatus('поиск...');
        setMainSearchOutput('');
        mainSearchButton.disabled = true;

        try {
            const response = await fetch(
                `/search?collection_name=${encodeURIComponent(collectionName)}` +
                `&query=${encodeURIComponent(query)}` +
                `&mode=${encodeURIComponent(mode)}` +
                `&only_in_stock=${onlyInStock}`
            );

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Ошибка поиска.');
            }

            const payload = await response.json();
            const results = payload.results || [];
            const debug = payload.debug || {};

            if (!results.length) {
                setMainSearchStatus('ничего не найдено');
                setMainSearchOutput(
                    `Результаты не найдены.\n` +
                    `mode: ${debug.mode || mode}\n` +
                    `dense_dim: ${debug.dense_dim ?? 'n/a'}\n` +
                    `sparse_nonzero: ${debug.sparse_nonzero ?? 'n/a'}`
                );
                return;
            }

            const formattedResults = results.map((item, index) => {
                const payloadData = item.payload || {};
                const score = item.score ?? 'n/a';
                return [
                    `#${index + 1} score: ${score}`,
                    `payload: ${JSON.stringify(payloadData, null, 2)}`,
                    ''
                ].join('\n');
            }).join('\n');

            const formattedSummary = [
                `mode: ${debug.mode || mode}`,
                `dense_dim: ${debug.dense_dim ?? 'n/a'}`,
                `sparse_nonzero: ${debug.sparse_nonzero ?? 'n/a'}`,
                '',
                formattedResults,
            ].join('\n');

            setMainSearchStatus('готово');
            setMainSearchOutput(formattedSummary);
        } catch (error) {
            setMainSearchStatus('ошибка поиска');
            setMainSearchOutput(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            mainSearchButton.disabled = false;
        }
    });
}

const fetchHfCacheInfo = async () => {
    const response = await fetch('/hf_cache/info');
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Не удалось получить информацию о кэше.');
    }
    return response.json();
};

if (hfCacheRefreshButton) {
    hfCacheRefreshButton.addEventListener('click', async () => {
        setHfCacheStatus('обновление информации...');
        hfCacheRefreshButton.disabled = true;
        try {
            const payload = await fetchHfCacheInfo();
            renderHfCacheInfo(payload);
            addHfCacheLog('Информация о кэше обновлена.');
            setHfCacheStatus('готово');
        } catch (error) {
            setHfCacheStatus('ошибка');
            addHfCacheLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            hfCacheRefreshButton.disabled = false;
        }
    });
}

if (hfCacheClearButton) {
    hfCacheClearButton.addEventListener('click', async () => {
        const confirmed = confirm('Очистить HF‑кэш? Модели будут скачиваться заново.');
        if (!confirmed) {
            return;
        }

        setHfCacheStatus('очистка кэша...');
        hfCacheClearButton.disabled = true;
        try {
            const response = await fetch('/hf_cache/clear', { method: 'POST' });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось очистить кэш.');
            }
            const payload = await response.json();
            addHfCacheLog(`Кэш очищен: ${payload.hf_hub_cache || ''}`);
            setHfCacheStatus('кэш очищен');
            const info = await fetchHfCacheInfo();
            renderHfCacheInfo(info);
        } catch (error) {
            setHfCacheStatus('ошибка');
            addHfCacheLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            hfCacheClearButton.disabled = false;
        }
    });
}

if (integrationsRefreshButton) {
    integrationsRefreshButton.addEventListener('click', async () => {
        integrationsRefreshButton.disabled = true;
        setIntegrationsStatus('загрузка настроек...');
        try {
            const payload = await loadRuntimeConfig();
            renderRuntimeConfig(payload);
            addIntegrationsLog('Настройки интеграций обновлены.');
            setIntegrationsStatus('готово');
        } catch (error) {
            setIntegrationsStatus('ошибка');
            addIntegrationsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            integrationsRefreshButton.disabled = false;
        }
    });
}

if (integrationsSaveButton) {
    integrationsSaveButton.addEventListener('click', async () => {
        const polzaApiKey = document.getElementById('polza-api-key')?.value || '';
        const polzaModel = document.getElementById('polza-model')?.value || 'openai/gpt-4o';
        const polzaTemperature = parseFloat(document.getElementById('polza-temperature')?.value || '0.2');
        const polzaMaxTokens = parseInt(document.getElementById('polza-max-tokens')?.value || '500', 10);
        const polzaBaseUrl = document.getElementById('polza-base-url')?.value || 'https://polza.ai/api/v1/chat/completions';

        const bitrixClientId = document.getElementById('bitrix-client-id')?.value || '';
        const bitrixClientSecret = document.getElementById('bitrix-client-secret')?.value || '';
        const bitrixRedirectUri = document.getElementById('bitrix-redirect-uri')?.value || '';
        const bitrixWebhookUrl = document.getElementById('bitrix-webhook-url')?.value || '';
        const bitrixPortalBaseUrl = document.getElementById('bitrix-portal-base-url')?.value || '';
        const bitrixOauthAuthUrl = document.getElementById('bitrix-oauth-auth-url')?.value || 'https://oauth.bitrix.info/oauth/authorize/';
        const bitrixOauthTokenUrl = document.getElementById('bitrix-oauth-token-url')?.value || 'https://oauth.bitrix.info/oauth/token/';
        const bitrixBotId = document.getElementById('bitrix-bot-id')?.value || '';
        const bitrixCollectionName = document.getElementById('bitrix-collection-name')?.value || 'my_collection';
        const bitrixDocsCollectionName = document.getElementById('bitrix-docs-collection-name')?.value || 'passports_collection';
        const bitrixSearchMode = document.getElementById('bitrix-search-mode')?.value || 'hybrid';

        const polzaPayload = {
            model: polzaModel,
            temperature: Number.isFinite(polzaTemperature) ? polzaTemperature : 0.2,
            max_tokens: Number.isFinite(polzaMaxTokens) ? polzaMaxTokens : 500,
            base_url: polzaBaseUrl,
        };
        if (polzaApiKey.trim()) {
            polzaPayload.api_key = polzaApiKey.trim();
        }

        const bitrixPayload = {
            client_id: bitrixClientId,
            redirect_uri: bitrixRedirectUri,
            webhook_url: bitrixWebhookUrl,
            portal_base_url: bitrixPortalBaseUrl,
            oauth_auth_url: bitrixOauthAuthUrl,
            oauth_token_url: bitrixOauthTokenUrl,
            bot_id: bitrixBotId,
            collection_name: bitrixCollectionName,
            docs_collection_name: bitrixDocsCollectionName,
            search_mode: bitrixSearchMode,
        };
        if (bitrixClientSecret.trim()) {
            bitrixPayload.client_secret = bitrixClientSecret.trim();
        }

        integrationsSaveButton.disabled = true;
        setIntegrationsStatus('сохранение...');
        try {
            const response = await fetch('/runtime_config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    polza: polzaPayload,
                    bitrix: bitrixPayload,
                })
            });

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось сохранить настройки интеграций');
            }

            const payload = await response.json();
            renderRuntimeConfig(payload.config || {});

            const polzaApiKeyInput = document.getElementById('polza-api-key');
            const bitrixClientSecretInput = document.getElementById('bitrix-client-secret');
            if (polzaApiKeyInput) polzaApiKeyInput.value = '';
            if (bitrixClientSecretInput) bitrixClientSecretInput.value = '';

            addIntegrationsLog('Настройки интеграций сохранены.');
            setIntegrationsStatus('сохранено');
        } catch (error) {
            setIntegrationsStatus('ошибка');
            addIntegrationsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            integrationsSaveButton.disabled = false;
        }
    });
}

if (bitrixOauthConnectButton) {
    bitrixOauthConnectButton.addEventListener('click', async () => {
        bitrixOauthConnectButton.disabled = true;
        setIntegrationsStatus('получение OAuth URL...');
        try {
            const response = await fetch('/bitrix/oauth/connect_url');
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось получить OAuth URL');
            }
            const payload = await response.json();
            const connectUrl = payload.connect_url;
            addIntegrationsLog(`OAuth URL получен. state=${payload.state}`);
            setIntegrationsStatus('откройте окно OAuth и завершите авторизацию');
            window.open(connectUrl, '_blank', 'noopener,noreferrer');
        } catch (error) {
            setIntegrationsStatus('ошибка OAuth');
            addIntegrationsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            bitrixOauthConnectButton.disabled = false;
        }
    });
}

if (bitrixOauthRefreshButton) {
    bitrixOauthRefreshButton.addEventListener('click', async () => {
        bitrixOauthRefreshButton.disabled = true;
        setIntegrationsStatus('обновление OAuth токена...');
        try {
            const response = await fetch('/bitrix/oauth/refresh', { method: 'POST' });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось обновить токен');
            }
            const payload = await response.json();
            addIntegrationsLog(`OAuth токен обновлен. expires_at=${payload.expires_at}`);
            setIntegrationsStatus('OAuth токен обновлён');
        } catch (error) {
            setIntegrationsStatus('ошибка OAuth refresh');
            addIntegrationsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            bitrixOauthRefreshButton.disabled = false;
        }
    });
}

if (bitrixOauthStatusButton) {
    bitrixOauthStatusButton.addEventListener('click', async () => {
        bitrixOauthStatusButton.disabled = true;
        setIntegrationsStatus('проверка OAuth статуса...');
        try {
            const response = await fetch('/bitrix/oauth/status');
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Не удалось получить OAuth статус');
            }
            const payload = await response.json();
            addIntegrationsLog(
                `OAuth status: connected=${payload.connected}, portal=${payload.portal_base_url || 'n/a'}, expires_in=${payload.expires_in || 0}s`
            );
            setIntegrationsStatus(payload.connected ? 'OAuth подключен' : 'OAuth не подключен');
        } catch (error) {
            setIntegrationsStatus('ошибка OAuth status');
            addIntegrationsLog(`Ошибка: ${error.message}`);
            alert(error.message);
        } finally {
            bitrixOauthStatusButton.disabled = false;
        }
    });
}

(async () => {
    if (!integrationsRefreshButton) {
        return;
    }
    try {
        const payload = await loadRuntimeConfig();
        renderRuntimeConfig(payload);
    } catch {
        // ignore bootstrap error
    }
})();
