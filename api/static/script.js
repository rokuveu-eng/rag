const processButton = document.getElementById('process-file');
const uploadButton = document.getElementById('upload-data');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');

const setStatus = (message) => {
    statusText.textContent = `Статус: ${message}`;
};

const setProgress = (value) => {
    const safeValue = Math.max(0, Math.min(100, value));
    progressFill.style.width = `${safeValue}%`;
};

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
    processButton.disabled = true;
    uploadButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('skip_rows', skipRows);
    formData.append('mappings', JSON.stringify(mappings));
    formData.append('collection_name', collectionName);
    formData.append('batch_size', isNaN(batchSize) ? 16 : batchSize);

    try {
        const response = await fetch('/upload_processed_xlsx', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Произошла ошибка при загрузке данных.');
        }

        const data = await response.json();
        setStatus(`успех! Проиндексировано строк: ${data.indexed_rows}`);
        setProgress(100);
    } catch (error) {
        setStatus('ошибка загрузки данных.');
        setProgress(0);
        alert(error.message);
    } finally {
        processButton.disabled = false;
        uploadButton.disabled = false;
    }
});
