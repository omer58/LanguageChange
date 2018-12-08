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


HIDDEN_DIM      = 256

class YearLSTM(nn.Module):

    def __init__(self, embedding_dim, batch_size, sent_len, device):
        super(YearLSTM, self).__init__()
        self.EMBEDDING_DIM  = embedding_dim #1019
        self.BATCH_SIZE     = batch_size
        self.SENT_LEN       = sent_len
        self.hidden_dim     = HIDDEN_DIM
        self.lstm           = nn.LSTM(self.EMBEDDING_DIM, HIDDEN_DIM)
        self.hidden2tag     = nn.Linear(HIDDEN_DIM, 1)
        self.device         = device
        self.features       = nn.Sequential(
                                nn.Conv1d(self.SENT_LEN, 16, 5, stride=2),
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True), #=(1019 - 3 )/2 = 508

                                nn.Conv1d(16, 8, 5), # 504
                                nn.BatchNorm1d(8),
                                nn.ReLU(inplace=True),

                                nn.Conv1d(8, 2, 5), #500
                                nn.BatchNorm1d(2),
                                nn.ReLU(inplace=True)
                                )
        self.feat2hid       = nn.Linear(1000, HIDDEN_DIM)




    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        return (torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(self.device),
                torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(self.device))

    def forward(self, batch):
        self.hidden = self.init_hidden()
        #embeds = batch.view(self.SENT_LEN, -1, self.EMBEDDING_DIM)
        #BATCH SEQLEN FEATURES ([64, 128, 1019])
        embeds = batch.permute(0,1,2)
        #print(embeds.shape)
        #print(batch.shape)
        conv_out  = self.features(embeds)
        #print('A',conv_out.shape)
        conv_out = conv_out.view(self.BATCH_SIZE, -1)
        conv_out  = nn.ReLU(inplace=True)(self.feat2hid(conv_out))
        #print('B',conv_out.shape)

        #lstm_out, self.hidden = self.lstm( embeds, self.hidden)
        pred_year = self.hidden2tag(conv_out)
        tag_scores = pred_year #.view(-1, self.EMBEDDING_DIM)

        return tag_scores
