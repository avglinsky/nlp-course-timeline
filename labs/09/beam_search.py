import math


def beam_search(initial_state, transition_func, beam_width, max_length):
    # Создаем начальное состояние
    initial_beam = [(initial_state, [], 0)]

    for _ in range(max_length):
        new_beam = []
        for state, sequence, score in initial_beam:
            # Получаем возможные следующие состояния и их оценки
            next_states_and_scores = transition_func(state)

            # Сортируем их по убыванию оценок
            next_states_and_scores.sort(key=lambda x: x[1], reverse=True)

            # Выбираем лучшие `beam_width` состояний
            next_states_and_scores = next_states_and_scores[:beam_width]

            for next_state, next_score in next_states_and_scores:
                new_sequence = sequence + [next_state]
                new_score = score + next_score
                new_beam.append((next_state, new_sequence, new_score))

        # Сортируем новый пучок по убыванию оценок
        new_beam.sort(key=lambda x: x[2], reverse=True)

        # Выбираем лучшие `beam_width` состояний для следующей итерации
        initial_beam = new_beam[:beam_width]

    # Возвращаем лучший найденный путь
    best_sequence = initial_beam[0][1]
    best_score = initial_beam[0][2]
    return best_sequence, best_score


# Пример использования
if __name__ == "__main__":
    # Пример функции перехода для задачи поиска наибольшей суммы в матрице
    def transition_function(state):
        row, col = state
        if row < len(matrix) - 1:
            next_states_and_scores = [((row + 1, col), matrix[row + 1][col])]
        if col < len(matrix[0]) - 1:
            next_states_and_scores.append(((row, col + 1), matrix[row][col + 1]))
        return next_states_and_scores


    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    initial_state = (0, 0)
    beam_width = 2
    max_length = 3

    best_path, best_score = beam_search(initial_state, transition_function, beam_width, max_length)
    print("Best Path:", best_path)
    print("Best Score:", best_score)
