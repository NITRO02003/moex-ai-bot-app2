import requests
import os
import re

# Настройки сервера LM Studio
API_URL = "http://127.0.0.1:1234/v1/chat/completions"


def execute_command(command_text):
    """
    Выполняет команды ИИ.
    Поддерживает:
    - [LIST_DIR: path]
    - [READ_FILE: path]
    - [WRITE_FILE: path] content [/WRITE_FILE]
    - [SCAN_PROJECT: path] <--- НОВАЯ КОМАНДА ДЛЯ ГЛУБОКОГО ПОГРУЖЕНИЯ
    """
    try:
        # 1. [LIST_DIR: path]
        list_match = re.search(r'\[LIST_DIR:\s*(.*?)\s*\]', command_text)
        if list_match:
            path = list_match.group(1).strip() or "."
            if not os.path.exists(path): return f"Ошибка: Путь {path} не найден."
            return "Список файлов:\n" + "\n".join(os.listdir(path))

        # 2. [READ_FILE: path]
        read_match = re.search(r'\[READ_FILE:\s*(.*?)\s*\]', command_text)
        if read_match:
            path = read_match.group(1).strip()
            if not os.path.exists(path): return f"Ошибка: Файл {path} не найден."
            with open(path, 'r', encoding='utf-8') as f:
                return f"--- Содержимое {path} ---\n{f.read()}\n--- Конец ---"

        # 3. [WRITE_FILE: path] content [/WRITE_FILE]
        write_pattern = r'\[WRITE_FILE:\s*(.*?)\s*\](.*?)\[/WRITE_FILE\]'
        write_match = re.search(write_pattern, command_text, re.DOTALL)
        if write_match:
            path = write_match.group(1).strip()
            content = write_match.group(2)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Файл {path} успешно записан."

        # 4. [SCAN_PROJECT: path] <--- ГЛУБОКОЕ СКАНИРОВАНИЕ
        scan_match = re.search(r'\[SCAN_PROJECT:\s*(.*?)\s*\]', command_text)
        if scan_match:
            root_path = scan_match.group(1).strip() or "."
            if not os.path.exists(root_path): return f"Ошибка: Путь {root_path} не найден."

            full_report = []
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            full_report.append(f"=== FILE: {file_path} ===\n{content}\n")
                    except Exception:
                        full_report.append(f"=== FILE: {file_path} ===\n[Ошибка чтения или бинарный файл]\n")

            return "ГЛУБОКОЕ СКАНИРОВАНИЕ ЗАВЕРШЕНО:\n" + "\n".join(full_report)

        return None
    except Exception as e:
        return f"Критическая ошибка выполнения: {str(e)}"


def main():
    print("=== Агент 'Deep Understanding' запущен ===")
    print(f"Связь с LM Studio: {API_URL}")

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — автономный ИИ-агент. Твоя цель — глубокое понимание проекта. "
                "Твои инструменты: "
                "1. [LIST_DIR: path] - просмотр структуры. "
                "2. [READ_FILE: path] - чтение конкретного файла. "
                "3. [WRITE_FILE: path] content [/WRITE_FILE] - изменение кода. "
                "4. [SCAN_PROJECT: path] - САМЫЙ ВАЖНЫЙ ИНСТРУМЕНТ. Используй его, чтобы прочитать ВСЕ файлы в папке (например, docs) за один раз. "
                "После SCAN_PROJECT ты получишь всё содержимое файлов и сможешь анализировать проект целиком."
            )
        }
    ]

    while True:
        user_input = input("\n[ВЫ]: ")
        if user_input.lower() in ['exit', 'quit']: break

        messages.append({"role": "user", "content": user_input})

        # АГЕНТНЫЙ ЦИКЛ (Автономное выполнение без участия человека)
        while True:
            try:
                response = requests.post(API_URL, json={"messages": messages}, timeout=300)
                response.raise_for_status()
                ai_text = response.json()['choices'][0]['message']['content']

                print(f"\n[ИИ]:\n{ai_text}")
                messages.append({"role": "assistant", "content": ai_text})

                # Проверяем, есть ли команды в ответе
                result = execute_command(ai_text)

                if result:
                    print(f"\n[СИСТЕМА]: {result}")
                    messages.append({"role": "system", "content": f"Результат выполнения: {result}"})
                    # ВАЖНО: Мы НЕ выходим из цикла, а сразу шлем результат обратно ИИ
                    continue
                else:
                    # Если команд больше нет, возвращаемся к ожиданию ввода пользователя
                    break

            except Exception as e:
                print(f"\n[ОШИБКА]: {str(e)}")
                break


if __name__ == "__main__":
    main()
