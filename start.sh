#!/bin/bash

## Переход в директорию с приложением
#cd /path/to/your/app

# Запуск FastAPI приложения с помощью Uvicorn в фоновом режиме
# uvicorn file_keeper:app --reload --host 0.0.0.0 --port 8000 &

# Задержка 1 секунда
sleep 1

# Запуск Streamlit приложения
streamlit run timeline_app.py
