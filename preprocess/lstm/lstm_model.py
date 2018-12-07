#Julian, you have to change the inputs and the number of epochs/iterations
#
from torch.autograd import Variable
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


HIDDEN_DIM      = 128
EPOCHS          = 10
class YearLSTM(nn.Module):

    def __init__(self, embedding_dim, batch_size):
        super(YearLSTM, self).__init__()
        self.EMBEDDING_DIM  = embedding_dim
        self.hidden_dim     = HIDDEN_DIM
        self.BATCH_SIZE     = batch_size
        self.lstm           = nn.LSTM(self.EMBEDDING_DIM, HIDDEN_DIM)
        self.hidden2tag     = nn.Linear(HIDDEN_DIM, self.EMBEDDING_DIM)


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.BATCH_SIZE, self.hidden_dim),
                torch.zeros(1, self.BATCH_SIZE, self.hidden_dim))

    def forward(self, batch):
        self.hidden = self.init_hidden()
        #for wLi in range(len(sentence)):
        #    sentence[wLi] = torch.Tensor(sentence[])
        print('BS',batch.shape)
        #[[print(word) for word in sentence if word.shape[0]>1] for sentence in batch]
        #embeds = sentence.view(l, -1, EMBEDDING_DIM)
        lstm_out, self.hidden = self.lstm( batch, self.hidden)

        pred_year = self.hidden2tag(lstm_out.view(l, -1)[-1])
        tag_scores = F.log_softmax(pred_year, dim=0).view(-1, self.EMBEDDING_DIM)
        return tag_scores
