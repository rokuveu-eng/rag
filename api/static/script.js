document.getElementById('process-file').addEventListener('click', () => {
    const fileInput = document.getElementById('xlsx-file');
    const skipRows = parseInt(document.getElementById('skip-rows').value, 10);

    if (fileInput.files.length === 0) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
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

            columnMappingsDiv.innerHTML += `<label>${requiredCol}: </label>`;
            columnMappingsDiv.appendChild(select);
            columnMappingsDiv.innerHTML += '<br/>';
        });

        document.getElementById('upload-data').style.display = 'block';
    };

    reader.readAsArrayBuffer(file);
});

document.getElementById('upload-data').addEventListener('click', async () => {
    const fileInput = document.getElementById('xlsx-file');
    const skipRows = parseInt(document.getElementById('skip-rows').value, 10);
    const file = fileInput.files[0];
    const collectionName = document.getElementById('collection-name').value;

    const mappings = {};
    const requiredColumns = ['Артикул', 'Наименование', 'Тариф с НДС, руб'];
    requiredColumns.forEach(requiredCol => {
        const selectedIndex = document.getElementById(`select-${requiredCol}`).value;
        if (selectedIndex) {
            mappings[requiredCol] = parseInt(selectedIndex, 10);
        }
    });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('skip_rows', skipRows);
    formData.append('mappings', JSON.stringify(mappings));
    formData.append('collection_name', collectionName);

    const response = await fetch('/upload_processed_xlsx', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        alert('Данные успешно загружены!');
    } else {
        alert('Произошла ошибка при загрузке данных.');
    }
});
