FROM python:3.10-slim

# Установка необходимых утилит
RUN apt-get update && apt-get install -y ffmpeg git wget unzip && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app/ ./app/

# Создание необходимых папок для конфигов/моделей (volume монтируются на host)
RUN mkdir -p /models /config /tmp

# Экспонируемый порт
EXPOSE 55667

# Старт приложения через uvicorn (FastAPI)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "55667"]
