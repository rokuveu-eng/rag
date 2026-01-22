const processButton = document.getElementById('process-file');
const uploadButton = document.getElementById('upload-data');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const logOutput = document.getElementById('log-output');

const stockProcessButton = document.getElementById('stock-process-file');
const stockUploadButton = document.getElementById('stock-upload-data');
const stockStatusText = document.getElementById('stock-status-text');
const stockProgressFill = document.getElementById('stock-progress-fill');
const stockLogOutput = document.getElementById('stock-log-output');

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
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

        const headers = json[skipRows];
        const columnMappingsDiv = document.getElementById('column-mappings');
        columnMappingsDiv.innerHTML = '';

        const requiredColumns = ['Артикул', 'Наименование', 'Тариф с НДС, руб'];

        requiredColumns.forEach(requiredCol => {
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

    if (!file) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const mappings = {};
    const requiredColumns = ['Артикул', 'Наименование', 'Тариф с НДС, руб'];
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
    formData.append('batch_size', isNaN(batchSize) ? 16 : batchSize);
    formData.append('points_batch_size', isNaN(pointsBatchSize) ? 200 : pointsBatchSize);

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
