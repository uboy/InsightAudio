import subprocess
import tempfile
import os

DEFAULT_SUMMARY_PROMPT = (
    "Подготовь структурированный пересказ по критериям:\n"
    "1. Тема/Название встречи/лекции\n"
    "2. Участники, роли\n"
    "3. Хронология вопросов и решений\n"
    "4. Ключевые задачи, обсуждения, решения\n"
    "5. Даты, документы, ссылки\n"
    "6. Итоги и открытые вопросы\n"
    "Оформи результат структурированно, разделы, список, в формате Markdown."
)

def summarize(text, model="llama3-8b", prompt=DEFAULT_SUMMARY_PROMPT):
    """
    Генерирует пересказ (summary) текста через локальную LLM (например Ollama).
    Args:
      text (str): исходная транскрипция
      model (str): имя модели в Ollama или локальной LLM
      prompt (str): промпт для summary
    Returns:
      summary (str): готовый структурированный пересказ
    """
    llm_prompt = f"{prompt}\n---\n{text}\n---\n"
    # Используется ollama (ollama должны быть запущен на хосте!),
    # Если надо — весь текст пишется во временный файл (чтобы не перегнать длинную команду)
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        tmp.write(llm_prompt)
        tmp.flush()
        file_path = tmp.name

    # Пример вызова Ollama CLI:
    # ollama run llama3-8b < file_path
    run_cmd = ["ollama", "run", model]
    summary = ""
    try:
        with open(file_path, "r", encoding="utf-8") as inp:
            res = subprocess.run(run_cmd, input=inp.read(), text=True, capture_output=True)
            summary = res.stdout.strip()
    except Exception as e:
        summary = f"Ошибка генерации пересказа: {str(e)}"
    finally:
        os.remove(file_path)
    return summary

# Для локального теста
if __name__ == "__main__":
    test_text = "Это пример транскрипта, на основе которого будет создан структурированный пересказ."
    result = summarize(test_text, model="llama3-8b")
    print(result)
