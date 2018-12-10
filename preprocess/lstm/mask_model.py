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


HIDDEN_DIM      = 16
CONV_OUT_NUM = 32
class YearLSTM(nn.Module):

    def __init__(self, embedding_dim, batch_size, sent_len, device):
        super(YearLSTM, self).__init__()
        self.EMBEDDING_DIM  = embedding_dim #1019
        self.BATCH_SIZE     = batch_size
        self.SENT_LEN       = sent_len
        self.hidden_dim     = HIDDEN_DIM
        self.lay1           = nn.Linear(1119, HIDDEN_DIM)
        self.lay2           = nn.Linear(HIDDEN_DIM, 1)
        self.device         = device
        self.mask           = nn.Sequential(
                                nn.Linear( CONV_OUT_NUM , HIDDEN_DIM),
                                nn.ReLU(inplace=True),
                                nn.Linear(HIDDEN_DIM, 1),
                                )
        self.maskfeats      = nn.Sequential(
                                #BATCH SEQLEN FEATURES ([64, 128, 1019])

                                nn.Conv1d(1, 16, 31, stride=10, padding=6),
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True), #=(1019 - 3 )/2 = 1019

                                nn.Conv1d(16, 16, 7,stride=2), # 101
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True),

                                nn.MaxPool1d(2), #48

                                nn.Conv1d(16, 24, 6,stride=2), # 24
                                nn.BatchNorm1d(24),
                                nn.ReLU(inplace=True),

                                nn.Conv1d(24, CONV_OUT_NUM, 4, stride=2), # 10
                                nn.BatchNorm1d(CONV_OUT_NUM),
                                nn.ReLU(inplace=True),

                                nn.MaxPool1d(3), # 3 OUT 1
                                )
        self.sigmoid        = nn.Sigmoid()

    def forward(self, batch, word_emb, show=False):
        #BATCH SEQLEN FEATURES ([64, 128, 1019]) [BS, WL, Y]

        m = batch.view(self.BATCH_SIZE*self.SENT_LEN, 1, -1)
        m  = self.maskfeats(m)
        m  = m.permute(0,2,1)
        m  = self.mask( m)    # Batch x SentLen x Year   [32 x 64 x 1019]
        m  = self.sigmoid(m) # Batch x SentLen          [32 x 64 x    1]
        m = m.view(self.BATCH_SIZE,self.SENT_LEN, -1)

        batch = m*(torch.cat((batch, word_emb), dim=2))      # Batch x SentLen x Year   [32 x 64 x 1019]
        batch = batch.sum(1) # Batch x Year   [32 x 1019]

        a1 = nn.ReLU(inplace=True)(self.lay1(batch))
        a2 = self.lay2(a1)
        if show:
            return a2, m
        return a2
