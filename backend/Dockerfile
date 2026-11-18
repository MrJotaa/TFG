FROM python:3.10-slim@sha256:38758a82a44d1acb9bae3dd5f7d2a55452fb44a5ceca7c4589f360f2c4aa3d0c

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock.txt ./

RUN pip install --no-cache-dir -r requirements.lock.txt \
    && python -m pip check

COPY app ./app

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
