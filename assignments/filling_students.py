import subprocess
import pandas as pd
import os

# Ваша bash-команда
bash_command = "nbgrader db student import assignments\students.csv"

# Запуск bash-команды
# subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)



# Чтение students.csv с использованием pandas
df = pd.read_csv('students.csv', encoding="windows-1251")

# Создание списка ids из колонки 'id'
ids = df['id'].tolist()

# Путь к папке 'submitted'
submitted_dir = 'submitted'

# Создание папок и подпапок
for student_id in ids:
    student_dir = os.path.join(submitted_dir, str(student_id))
    os.makedirs(student_dir, exist_ok=True)  # Создаем папку id

    # Создаем подпапки lecture_N_assignment
    for index in range(2, 13):
        assignment_dir = os.path.join(student_dir, f"lecture_{str.zfill(str(index), 2)}_assignment")
        os.makedirs(assignment_dir, exist_ok=True)

    # Создаем папку 'final_test'
    final_test_dir = os.path.join(student_dir, 'final_test')
    os.makedirs(final_test_dir, exist_ok=True)