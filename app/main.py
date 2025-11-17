import os
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Папка для хранения временных файлов
RESULTS_TMP_DIR = "/tmp"
os.makedirs(RESULTS_TMP_DIR, exist_ok=True)

# Инициализация FastAPI и шаблонов
app = FastAPI(title="InsightAudio")
templates = Jinja2Templates(directory="app/templates")

# Демонстрационные опции моделей (для прототипа). Реально список моделей кешируется и читается из /models, /config!
MODEL_OPTIONS = {
    "transcribe": [
        {"name": "whisper-tiny", "display_name": "Whisper Tiny (RU/EN)", "downloaded": True},
        {"name": "vosk-ru", "display_name": "Vosk Russian", "downloaded": False}
    ],
    "summarize": [
        {"name": "llama3-8b", "display_name": "Llama3 8B", "downloaded": True},
        {"name": "gemma-7b", "display_name": "Gemma 7B", "downloaded": False}
    ]
}
CONFIG = {"MODEL_DIR": "../models", "CONFIG_DIR": "../config"}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_options": MODEL_OPTIONS,
        "config": CONFIG
    })

@app.post("/process")
async def process(request: Request, file: UploadFile, transcribe_model: str = Form(), summarize_model: str = Form()):
    # Сохраняем загруженный файл
    temp_path = os.path.join(RESULTS_TMP_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    # Здесь должен быть вызов конвертации, транскрипции и summary через ваши модули
    # transcriber.transcribe(temp_path, model=transcribe_model)
    # summarizer.summarize(текст, model=summarize_model)
    dummy_transcript = "Транскрипция: пример текста из аудио/видео."
    dummy_summary = "Пересказ: ключевые мысли, структурированное summary."
    transcript_file = os.path.join(RESULTS_TMP_DIR, "result_transcript.txt")
    summary_file = os.path.join(RESULTS_TMP_DIR, "result_summary.txt")
    with open(transcript_file, "w") as f:
        f.write(dummy_transcript)
    with open(summary_file, "w") as f:
        f.write(dummy_summary)
    result_files = ["result_transcript.txt", "result_summary.txt"]
    return templates.TemplateResponse("results.html", {
        "request": request,
        "files": result_files
    })

@app.get("/download/{filename}")
def download(filename: str):
    filepath = os.path.join(RESULTS_TMP_DIR, filename)
    return FileResponse(filepath, media_type='application/octet-stream', filename=filename)

# Ниже - заготовки API endpoints для работы со списком моделей, конфигом, скачиванием моделей
@app.get("/list_models")
def list_models():
    return MODEL_OPTIONS

@app.post("/download_model")
def download_model(model_name: str = Form(), model_type: str = Form()):
    # Здесь должна быть интеграция с загрузкой и распаковкой моделей
    return {"status": "OK", "message": f"Модель {model_name} запрошена для скачивания"}

@app.get("/config")
def get_config():
    return CONFIG

@app.post("/config")
def update_config(cfg: dict):
    # Сохраняйте/изменяйте конфиг здесь
    return {"status": "OK", "cfg": cfg}

# Healthcheck для докера
@app.get("/health")
def health():
    return {"status": "InsightAudio is alive"}
