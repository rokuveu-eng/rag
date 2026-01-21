from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
import httpx
from rank_bm25 import BM25Okapi
import json
import openpyxl
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("api/static/index.html") as f:
        return f.read()

qdrant_client = QdrantClient(host="qdrant", port=6333)
ollama_api_url = "http://ollama:11434/api/embeddings"

def create_collection(collection_name: str):
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
        )

def load_corpus(collection_name: str):
    corpus_file = f"/app/{collection_name}_corpus.json"
    try:
        with open(corpus_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

async def get_embedding(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(ollama_api_url, json={"model": "bge-m3", "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

@app.post("/upload_xlsx")
async def upload_xlsx(file: UploadFile = File(...)):
    global corpus
    try:
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        headers = [cell.value for cell in sheet[1]]
        
        points = []
        new_chunks = []

        for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True)):
            row_data = dict(zip(headers, row))

            # Create text for embedding and BM25
            text_to_embed = f"Артикул - {row_data.get('Артикул')}, Наименование - {row_data.get('Наименование')}, Тариф с НДС, руб - {row_data.get('Тариф с НДС, руб')}, Имя файла - {file.filename}"
            new_chunks.append(text_to_embed)

            # Create payload
            payload = row_data
            payload["Остаток"] = ""

            embedding = await get_embedding(text_to_embed)
            points.append(models.PointStruct(id=len(corpus) + row_idx, vector=embedding, payload=payload))

        corpus.extend(new_chunks)
        with open(corpus_file, 'w') as f:
            json.dump(corpus, f)

        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        return {"status": "success", "indexed_rows": len(new_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_stock")
async def update_stock(file: UploadFile = File(...), stock_column: str = Form(...)):
    try:
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        headers = [cell.value for cell in sheet[1]]
        if stock_column not in headers:
            raise HTTPException(status_code=400, detail=f"Column '{stock_column}' not found in the file.")

        for row in sheet.iter_rows(min_row=2, values_only=True):
            row_data = dict(zip(headers, row))
            article = row_data.get("Артикул")
            stock_value = row_data.get(stock_column)

            if article:
                # Find points with the matching article
                search_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="Артикул",
                                match=models.MatchValue(value=article),
                            )
                        ]
                    ),
                    limit=1
                )
                if search_result[0]:
                    point_id = search_result[0][0].id
                    qdrant_client.set_payload(
                        collection_name=collection_name,
                        payload={"Остаток": stock_value},
                        points=[point_id],
                        wait=True
                    )

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_processed_xlsx")
async def upload_processed_xlsx(file: UploadFile = File(...), skip_rows: int = Form(...), mappings: str = Form(...), collection_name: str = Form(...)):
    create_collection(collection_name)
    corpus = load_corpus(collection_name)
    corpus_file = f"/app/{collection_name}_corpus.json"

    try:
        mappings = json.loads(mappings)
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        _ = [cell.value for cell in sheet[1]]
        
        points = []
        new_chunks = []

        for row_idx, row in enumerate(sheet.iter_rows(min_row=skip_rows + 2, values_only=True)):
            
            # Create text for embedding and BM25
            text_to_embed = f"Артикул - {row[mappings['Артикул']]}, Наименование - {row[mappings['Наименование']]}, ариф с НДС, руб - {row[mappings['Тариф с НДС, руб']]}, Имя файла - {file.filename}"
            new_chunks.append(text_to_embed)

            # Create payload
            payload = {}
            for header, col_idx in mappings.items():
                payload[header] = row[col_idx]
            payload["Остаток"] = ""

            embedding = await get_embedding(text_to_embed)
            points.append(models.PointStruct(id=len(corpus) + row_idx, vector=embedding, payload=payload))

        corpus.extend(new_chunks)
        with open(corpus_file, 'w') as f:
            json.dump(corpus, f)

        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        return {"status": "success", "indexed_rows": len(new_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str, collection_name: str):
    corpus = load_corpus(collection_name)
    # Vector search
    embedding = await get_embedding(query)
    vector_search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=5, 
    )

    # BM25 search
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Combine results (simple approach: return top 5 from each)
    bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
    bm25_docs = [corpus[i] for i in bm25_results]
    

    return {"vector_search_results": vector_search_result, "bm25_search_results": bm25_docs}
