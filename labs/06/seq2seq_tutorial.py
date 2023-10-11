# %%
# code by Tae Hwan Jung @graykode
# Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# S: Symbol that shows the starting of decoding input
# E: Symbol that shows the starting of decoding output
# P: Symbol that will fill in blank sequence if the current batch data size is shorter than time steps

# Define a function to create a batch of data
def make_batch():
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    
    # Convert data to PyTorch tensors
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

# Define the Attention class which is a subclass of nn.Module
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        # Linear layer for attention
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        enc_inputs = enc_inputs.transpose(0, 1)  # Transpose input for RNN
        dec_inputs = dec_inputs.transpose(0, 1)  # Transpose input for RNN

        # Pass encoder inputs through RNN
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        model = torch.empty([n_step, 1, n_class])

        for i in range(n_step):  # Loop over each time step
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(dec_output, enc_outputs)  # Calculate attention weights
            trained_attn.append(attn_weights.squeeze().data.numpy())

            context = attn_weights.bmm(enc_outputs.transpose(0, 1))  # Calculate context vector
            dec_output = dec_output.squeeze(0)
            context = context.squeeze(1)
            model[i] = self.out(torch.cat((dec_output, context), 1))  # Concatenate and pass through linear layer

        # Reshape the model
        return model.transpose(0, 1).squeeze(0), trained_attn

    def get_att_weight(self, dec_output, enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)  # Initialize attention scores
        
        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])  # Calculate attention scores
        
        # Normalize scores to weights in the range 0 to 1 using softmax
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):
        score = self.attn(enc_output)  # Apply linear layer for attention
        return torch.dot(dec_output.view(-1), score.view(-1))  # Calculate dot product

if __name__ == '__main__':
    n_step = 5  # Number of cells (steps)
    n_hidden = 128  # Number of hidden units in one cell

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))  # Create a list of unique words
    word_dict = {w: i for i, w in enumerate(word_list)}  # Create a dictionary mapping words to indices
    number_dict = {i: w for i, w in enumerate(word_list)}  # Create a reverse dictionary mapping indices to words
    n_class = len(word_dict)  # Vocabulary size

    # Initialize hidden state for RNN
    hidden = torch.zeros(1, 1, n_hidden)

    model = Attention()
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer

    input_batch, output_batch, target_batch = make_batch()  # Create a batch of input, output, and target data

    # Training loop
    for epoch in range(2000):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)  # Forward pass

        loss = criterion(output, target_batch.squeeze(0))  # Calculate the loss
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()  # Backpropagation
        optimizer.step()

    # Testing
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]  # Test input
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)  # Get predictions
    predict = predict.data.max(1, keepdim=True)[1]  # Select the word with the highest probability
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])  # Print the result

    # Show attention visualization
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()
