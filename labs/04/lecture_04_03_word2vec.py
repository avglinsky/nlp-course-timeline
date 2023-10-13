#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict
from tqdm import tqdm

class Word2VecSkipGram:
    def __init__(self, window_size=2, n=100, epochs=200, learning_rate=0.01):
        self.window_size = window_size
        self.n = n
        self.epochs = epochs
        self.learning_rate = learning_rate

    def generate_training_data(self, corpus):
        # Инициализация словаря для подсчета частоты слов
        word_count = defaultdict(int)
    
        # Подсчитываем частоту каждого слова в корпусе
        for sentence in corpus:
            for word in sentence:
                word_count[word] += 1
    
        # Создаем отображение слов в индексы и обратно
        self.word2index = {word: i for i, word in enumerate(word_count.keys())}
        self.index2word = {i: word for word, i in self.word2index.items()}
        
        # Определяем размер словаря (количество уникальных слов)
        self.vocab_size = len(self.word2index)
    
        # Создаем список обучающих пар (target_word, context_word)
        training_data = []
        for sentence in corpus:
            for i, target_word in enumerate(sentence):
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and 0 <= j < len(sentence):
                        context_word = sentence[j]
                        training_data.append((target_word, context_word))
    
        return training_data


    def initialize_weights(self):
        self.W1 = np.random.uniform(-1, 1, (self.vocab_size, self.n))
        self.W2 = np.random.uniform(-1, 1, (self.n, self.vocab_size))

    def forward(self, target_word):
        # Получаем индекс целевого слова в словаре
        target_idx = self.word2index[target_word]
        
        # Инициализируем входной слой (input layer) нулевым вектором
        self.input_layer = np.zeros(self.vocab_size)
        
        # Устанавливаем значение индекса целевого слова в 1, чтобы представить его как one-hot вектор
        self.input_layer[target_idx] = 1
        
        # Производим вычисление скрытого слоя (hidden layer) путем умножения входного слоя на матрицу W1
        self.hidden_layer = np.dot(self.input_layer, self.W1)
        
        # Производим вычисление выходного слоя (output layer) путем умножения скрытого слоя на матрицу W2
        self.output_layer = np.dot(self.hidden_layer, self.W2)
        
        # Применяем softmax к выходному слою для получения вероятностей контекстных слов
        self.output_probs = self.softmax(self.output_layer)
        
        # Возвращаем выходные вероятности контекстных слов
        return self.output_probs


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, target_word, context_word):
        # Получаем индексы целевого и контекстного слов в словаре
        target_idx = self.word2index[target_word]
        context_idx = self.word2index[context_word]
    
        # Вычисляем разницу между предсказанными вероятностями и фактическими вероятностями
        delta_output = self.output_probs
        delta_output[context_idx] -= 1
    
        # Вычисляем ошибку на скрытом слое, умножая ошибку на выходном слое на матрицу W2
        delta_hidden = np.dot(self.W2, delta_output)
    
        # Обновляем матрицу W1 с учетом ошибки на скрытом слое и входного слоя
        self.W1 -= self.learning_rate * np.outer(self.input_layer, delta_hidden)
    
        # Обновляем матрицу W2 с учетом ошибки на выходном слое и скрытого слоя
        self.W2 -= self.learning_rate * np.outer(self.hidden_layer, delta_output)


    def train(self, corpus):
        # Генерируем обучающие данные из предоставленного корпуса
        training_data = self.generate_training_data(corpus)
        
        # Инициализируем веса модели
        self.initialize_weights()
    
        # Начинаем обучение на заданном количестве эпох (self.epochs)
        for epoch in tqdm(range(self.epochs)):
            # Проходим по всем обучающим парам (target_word, context_word) в обучающих данных
            for target_word, context_word in training_data:
                # Выполняем прямой проход (forward pass) для текущего целевого слова
                self.forward(target_word)
                
                # Выполняем обратное распространение (backpropagation) и обновляем веса модели
                self.backprop(target_word, context_word)


    def get_word_vector(self, word):
        if word in self.word2index:
            target_idx = self.word2index[word]
            return self.W1[target_idx]

    def most_similar(self, word, top_n=10):
        if word in self.word2index:
            target_vector = self.get_word_vector(word)
            word_sim = {}

            for i in range(self.vocab_size):
                if i != self.word2index[word]:
                    context_word = self.index2word[i]
                    context_vector = self.get_word_vector(context_word)
                    cosine_similarity = np.dot(target_vector, context_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(context_vector))
                    word_sim[context_word] = cosine_similarity

            return sorted(word_sim.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Пример использования
corpus = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
model = Word2VecSkipGram(window_size=2, n=100, epochs=200, learning_rate=0.01)
model.train(corpus)
similar_words = model.most_similar('fox', top_n=5)
print(similar_words)


# In[13]:


class Word2VecCBOW:
    def __init__(self, window_size=2, n=100, epochs=200, learning_rate=0.01):
        self.window_size = window_size
        self.n = n
        self.epochs = epochs
        self.learning_rate = learning_rate

    def generate_training_data(self, corpus):
        # Инициализация словаря для подсчета частоты слов
        word_count = defaultdict(int)
    
        # Подсчитываем частоту каждого слова в корпусе
        for sentence in corpus:
            for word in sentence:
                word_count[word] += 1
    
        # Создаем отображение слов в индексы и обратно
        self.word2index = {word: i for i, word in enumerate(word_count.keys())}
        self.index2word = {i: word for word, i in self.word2index.items()}
        
        # Определяем размер словаря (количество уникальных слов)
        self.vocab_size = len(self.word2index)
    
        # Создаем список обучающих пар (context_words, target_word)
        training_data = []
        for sentence in corpus:
            for i, target_word in enumerate(sentence):
                context_words = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and 0 <= j < len(sentence):
                        context_words.append(sentence[j])
                if context_words:
                    training_data.append((context_words, target_word))
    
        return training_data

    def initialize_weights(self):
        self.W1 = np.random.uniform(-1, 1, (self.vocab_size, self.n))
        self.W2 = np.random.uniform(-1, 1, (self.n, self.vocab_size))


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def forward(self, context_words):
        # Инициализируем входной слой (input layer) нулевым вектором
        self.input_layer = np.zeros(self.vocab_size)
        
        # Вычисляем средний вектор контекстных слов путем суммирования их векторов
        for word in context_words:
            if word in self.word2index:
                word_idx = self.word2index[word]
                self.input_layer[word_idx] += 1
        
        # Производим вычисление скрытого слоя (hidden layer) путем умножения входного слоя на матрицу W1
        self.hidden_layer = np.dot(self.input_layer, self.W1)
        
        # Производим вычисление выходного слоя (output layer) путем умножения скрытого слоя на матрицу W2
        self.output_layer = np.dot(self.hidden_layer, self.W2)
        
        # Применяем softmax к выходному слою для получения вероятностей целевого слова
        self.output_probs = self.softmax(self.output_layer)
        
        # Возвращаем выходные вероятности целевого слова
        return self.output_probs

    def backprop(self, context_words, target_word):
        # Инициализируем delta_output нулевым вектором
        delta_output = np.zeros(self.vocab_size)
        
        # Вычисляем ошибку на выходном слое
        if target_word in self.word2index:
            target_idx = self.word2index[target_word]
            delta_output[target_idx] = 1
        
        # Вычисляем ошибку на скрытом слое, умножая ошибку на выходном слое на матрицу W2
        delta_hidden = np.dot(self.W2, delta_output)
        
        # Обновляем матрицу W1 с учетом ошибки на скрытом слое и входного слоя
        self.W1 -= self.learning_rate * np.outer(self.input_layer, delta_hidden)
        
        # Обновляем матрицу W2 с учетом ошибки на выходном слое и скрытого слоя
        self.W2 -= self.learning_rate * np.outer(self.hidden_layer, delta_output)


    def train(self, corpus):
        # Генерируем обучающие данные из предоставленного корпуса
        training_data = self.generate_training_data(corpus)
        
        # Инициализируем веса модели
        self.initialize_weights()
    
        # Начинаем обучение на заданном количестве эпох (self.epochs)
        for epoch in tqdm(range(self.epochs)):
            # Проходим по всем обучающим парам (context_words, target_word) в обучающих данных
            for context_words, target_word in training_data:
                # Выполняем прямой проход (forward pass) для текущего контекста и целевого слова
                self.forward(context_words)
                
                # Выполняем обратное распространение (backpropagation) и обновляем веса модели
                self.backprop(context_words, target_word)

    def most_similar(self, word, top_n=10):
        if word in self.word2index:
            target_vector = self.get_word_vector(word)
            word_sim = {}

            for i in range(self.vocab_size):
                if i != self.word2index[word]:
                    context_word = self.index2word[i]
                    context_vector = self.get_word_vector(context_word)
                    cosine_similarity = np.dot(target_vector, context_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(context_vector))
                    word_sim[context_word] = cosine_similarity

            return sorted(word_sim.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_word_vector(self, word):
        if word in self.word2index:
            target_idx = self.word2index[word]
            return self.W1[target_idx]


# Пример использования
corpus = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
model = Word2VecCBOW(window_size=2, n=100, epochs=200, learning_rate=0.01)
model.train(corpus)
similar_words = model.most_similar('fox', top_n=5)
print(similar_words)


# In[ ]:




