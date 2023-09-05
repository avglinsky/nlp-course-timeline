import numpy as np

# Генерируем случайные данные для обучения
np.random.seed(42)
observed_data = np.concatenate([np.random.randn(100), np.random.randn(100) + 4])

# Инициализируем параметры HMM случайным образом
num_hidden_states = 2
num_observation_symbols = 1
num_samples = len(observed_data)

initial_probabilities = np.random.rand(num_hidden_states)
initial_probabilities /= np.sum(initial_probabilities)

transition_matrix = np.random.rand(num_hidden_states, num_hidden_states)
transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

emission_means = np.random.rand(num_hidden_states, num_observation_symbols)
emission_covariances = np.random.rand(num_hidden_states, num_observation_symbols, num_observation_symbols)

# Обучение HMM методом Expectation-Maximization (EM)
num_iterations = 100

for iteration in range(num_iterations):
    # E-шаг (Expectation): Вычисляем апостериорные вероятности скрытых состояний с помощью алгоритма прямого-обратного прохода (forward-backward)

    # Прямой проход (forward)
    forward_probs = np.zeros((num_samples, num_hidden_states))
    forward_probs[0] = initial_probabilities * emission_means[:, int(observed_data[0])]
    for t in range(1, num_samples):
        for j in range(num_hidden_states):
            forward_probs[t, j] = np.sum(forward_probs[t - 1] * transition_matrix[:, j]) * emission_means[
                j, int(observed_data[t])]

    # Обратный проход (backward)
    backward_probs = np.zeros((num_samples, num_hidden_states))
    backward_probs[-1] = 1.0
    for t in range(num_samples - 2, -1, -1):
        for i in range(num_hidden_states):
            backward_probs[t, i] = np.sum(
                transition_matrix[i, :] * emission_means[:, int(observed_data[t + 1])] * backward_probs[t + 1, :])

    # Обновление параметров с использованием апостериорных вероятностей
    new_initial_probabilities = forward_probs[0] * backward_probs[0]
    new_transition_matrix = np.zeros((num_hidden_states, num_hidden_states))
    new_emission_means = np.zeros((num_hidden_states, num_observation_symbols))

    for i in range(num_hidden_states):
        for j in range(num_hidden_states):
            new_transition_matrix[i, j] = np.sum((forward_probs[:-1, i] * transition_matrix[i, j] * emission_means[
                j, int(observed_data[1:])] * backward_probs[1:, j]) / np.sum(
                forward_probs[:-1, :] * backward_probs[1:, :], axis=1))

    for i in range(num_hidden_states):
        new_emission_means[i] = np.sum(forward_probs[:, i] * backward_probs[:, i] * (observed_data == i)) / np.sum(
            forward_probs[:, i] * backward_probs[:, i])

    # Нормализация параметров
    new_initial_probabilities /= np.sum(new_initial_probabilities)
    new_transition_matrix /= np.sum(new_transition_matrix, axis=1, keepdims=True)

    # Обновление параметров
    initial_probabilities = new_initial_probabilities
    transition_matrix = new_transition_matrix
    emission_means = new_emission_means

# Вывод обученных параметров
print("Оцененные начальные вероятности:")
print(initial_probabilities)

print("Оцененная матрица переходов:")
print(transition_matrix)

print("Оцененные средние значения эмиссий:")
print(emission_means)
