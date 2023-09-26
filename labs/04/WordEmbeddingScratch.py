#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict
from tqdm import tqdm


# In[2]:


class word2vec():
    def __init__(self):
        self.lr = settings['learning_rate']
        self.epoch = settings['epochs']
        self.window_size = settings['window_size']
        self.dimension = settings['n']
        
    def generate_training_data(self,setting,corpus):
        word_count = defaultdict(int)
        for row in corpus:
            for word in row:
                word_count[word] += 1
          
        self.word_len = len(word_count.keys())
        self.word_list = list(word_count.keys())
        self.word_index = dict((word,i) for i,word in enumerate(self.word_list))
        self.index_word = dict((i,word) for i,word in enumerate(self.word_list))
  
        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            
            for i,word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                
                for j in range(i - self.window_size,i + self.window_size):
                    if j!= i and j>=0 and j<= sent_len-1:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target,w_context])
       
        
        return np.array(training_data)
       

   
    def word2onehot(self,word):
        word_vec = np.zeros(self.word_len)
        word_index1 = self.word_index[word]
        word_vec[word_index1] = 1
        return word_vec
    

    def skipgramtrain(self,train_data):
        # using skipgram model in which we predict the context word from the target word to get the word embedding 
        self.w1 = np.random.uniform(-1, 1, (self.word_len, self.dimension))
        self.w2 = np.random.uniform(-1, 1, (self.dimension, self.word_len)) 
        for i in tqdm(range(self.epoch)):
  
            for w_t ,w_c in train_data:
                y_u,h,u = self.ford_prop(w_t)

                EI = np.sum([np.subtract(y_u,word) for word in w_c],axis=0)
                self.back_prop(EI,h,w_t)
                
    def cbowtrain(self,train_data):
        # using cbow model in which we predict the target word from the context word
        self.w1 = np.random.uniform(-1, 1, (self.word_len, self.dimension))
        self.w2 = np.random.uniform(-1, 1, (self.dimension, self.word_len))
        for i in tqdm(range(self.epoch)):
            for w_t ,w_c in train_data:
                for vec in w_c:
                    y_u,h,u = self.ford_prop(vec)
                    EI = np.subtract(y_u,w_t)
                    self.back_prop(EI,h,w_t)  
            
    def ford_prop(self,x):
        h = np.dot(x,self.w1)
        u = np.dot(h,self.w2)
        
        y_u = self.softmax(u)
        
        return y_u,h,u
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def back_prop(self,e,h,x):
        dl_dw2 = np.outer(h,e)
        dl_dw1 =np.outer(x,np.dot(self.w2,e))
        
        self.w1 = self.w1 - (self.lr*dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)
        
    def word_vec(self,x):
        w_index = self.word_index[x]
        return self.w1[w_index]
    
    def cosinevec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.word_len):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
            word = self.index_word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
            
    def eculvec_sim(self,word,top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}
        
        for i in range(self.word_len):
            v_w2 = self.w1[i]
            eculedian = np.linalg.norm(v_w1-v_w2)
            word_sim[self.index_word[i]] = eculedian
    
        word_sorted = sorted(word_sim.items(),key=lambda k:k[1],reverse=False)
        
        for word , sim in word_sorted[:top_n]:
            print(word,sim)


# In[3]:


settings = {
	'window_size': 2,
	'n': 3,
	'epochs': 200,
	'learning_rate': 0.01
}


# In[4]:


text = "the day is friday and the day is sunday and the day is thursday and the day is wednesday and the day is Monday"


# In[5]:


corpus = [[word.lower() for word in text.split()]]


# In[6]:


w2v = word2vec()
training_data = w2v.generate_training_data(settings, corpus)
w2v.skipgramtrain(training_data)


# In[ ]:


x = 'monday'
w2v.word_vec(x)
w2v.cosinevec_sim(x, 7)


# In[ ]:


w2v.eculvec_sim(x, 7)


# In[ ]:


w2v.cbowtrain(training_data)


# In[ ]:


x = 'monday'
w2v.word_vec(x)
w2v.cosinevec_sim(x, 7)


# In[ ]:


w2v.eculvec_sim(x, 7)


# In[ ]:




