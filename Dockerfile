FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install \
      --default-timeout=600 \
      --retries=10 \
      --no-cache-dir \
      -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]