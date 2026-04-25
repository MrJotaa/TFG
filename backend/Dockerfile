FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["fastapi", "dev", "app/main.py", "--host", "0.0.0.0"]
