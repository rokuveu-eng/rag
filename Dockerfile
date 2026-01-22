FROM python:3.9

WORKDIR /app

COPY ./api/requirements.txt /app/
RUN apt-get update && apt-get install -y build-essential g++ make cmake
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8424"]
