@echo off
cd /d "C:\Users\avglinsky\PycharmProjects\nlp-course-timeline"

:: Запуск FastAPI приложения с помощью Uvicorn
:: start "" cmd /c "uvicorn file_keeper:app --reload --host 0.0.0.0 --port 8000"

:: Задержка 1 секунда
ping 127.0.0.1 -n 1 -w 1000 > nul

:: Запуск Streamlit приложения
start "" cmd /c "streamlit run .\timeline_app.py"
