#Julian, you have to change the inputs and the number of epochs/iterations
#

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class YearLSTM(nn.Module):

    def __init__(self, word2yearvectordict):
        super(YearLSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.w2yv = word2yearvectordict

        self.lstm           = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        self.hidden2tag     = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)
        self.hidden         = self.init_hidden()


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def sent2years(self, sentence):
        list_o_embeddings = []
        for word in sentence:
            if word in self.w2yv:
                list_o_embeddings.append( Variable(torch.tensor(self.w2yv[word])) )
        return torch.stack(list_o_embeddings)

    def forward(self, sentence):
        embeds = self.sent2years(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
