#Julian, you have to change the inputs and the number of epochs/iterations 
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

torch.manual_seed(58)

# Example: An LSTM for Part-of-Speech Tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The model is as follows: let our input sentence be
# :math:`w_1, \dots, w_M`, where :math:`w_i \in V`, our vocab. Also, let
# :math:`T` be our tag set, and :math:`y_i` the tag of word :math:`w_i`.
# Denote our prediction of the tag of word :math:`w_i` by
# :math:`\hat{y}_i`.
#
# To do the prediction, pass an LSTM over the sentence. Denote the hidden
# state at timestep :math:`i` as :math:`h_i`. Also, assign each tag a
# unique index (like how we had word\_to\_ix in the word embeddings
# section). Then our prediction rule for :math:`\hat{y}_i` is
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# That is, take the log softmax of the affine map of the hidden state,
# and the predicted tag is the tag that has the maximum value in this
# vector. Note this implies immediately that the dimensionality of the
# target space of :math:`A` is :math:`|T|`.






EMBEDDING_DIM   = 1019
HIDDEN_DIM      = 128
EPOCHS          = 50

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_years, word2yearvectordict):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        print('loading yearbook')
        self.w2yv = pickle.load(open('../../data_sets/w2yv.pickle'))
        print('Loaded yearbook')
       
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, num_years)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def sent2years(self, sentence):
        list_o_embeddings = []
        for word in sentence:
            list_o_embeddings.append( Variable(torch.tensor(self.w2yv[word])) )
        return torch.stack(list_o_embeddings)

    def forward(self, sentence):
        embeds = self.sent2years(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model           = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DIM)
loss_function   = nn.NLLLoss()
optimizer       = optim.SGD(model.parameters(), lr=0.003, nesterov=True, momentum=0.9)
training_data   = 
questions       = open(



for epoch in range(50):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence)
        targets = prepare_sequence(tags)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


with torch.no_grad():
    inputs      = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores  = model(inputs)
    print(tag_scores)

