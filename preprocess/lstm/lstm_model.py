#Julian, you have to change the inputs and the number of epochs/iterations
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from matplotlib import pyplot as plt
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

class LSTM(nn.Module):

    def __init__(self, word2yearvectordict, file_name_to_save):
        super(LSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.name = str(file_name_to_save)
        print('loading yearbook')
        self.w2yv = word2yearvectordict
        print('Loaded yearbook')

        self.lstm           = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        self.hidden2tag     = nn.Linear(HIDDEN_DIM, NUM_YEARS)
        self.hidden         = self.init_hidden()
        self.model          = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_DIM)
        self.loss_function  = nn.NLLLoss()
        self.optimizer      = optim.SGD(model.parameters(), lr=0.003, nesterov=True, momentum=0.9)

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

    def save_data_model(self, trnA, trnL, tstA, tstL, model):
        fig, (ax1,ax2) = plt.subplots( nrows=1, ncols=2 )
        ax1.plot(trnA, label='train acc')                   #Plot 1
        ax1.plot(tstA, label='test  acc')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(trnL, label='train loss')                  #Plot 2
        ax2.plot(tstL, label='test  loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        fig.savefig('../../data_sets/results/'+self.name)   # save the figure to file
        plt.close(fig)    # close the figure

        torch.save(model, '../../models/'+self.name)


    def train(self, training_data, validation=[], num_epochs=50):
        train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
        for epoch in range(num_epochs):
            epoch_correct, epoch_loss, valid_correct, valid_loss = 0.0, 0.0, 0.0, 0.0
            for sentence, tag in training_data:
                model.zero_grad()
                model.hidden = model.init_hidden()
                target = Variable(tag)
                pred_year = model(sentence)
                loss = loss_function(pred_year, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                if pred_year == target:
                    epoch_correct +=1
            train_accuracy.append(epoch_correct)
            train_loss.append(epoch_loss)

            if validation:
                with torch.no_grad():
                    for example_X, example_Y in validation:
                        pred_year  = model(inputs)
                        target = Variable(example_Y)
                        v_loss = loss_function(pred_year, target)
                        if pred_year == target:
                            valid_correct +=1
                        valid_loss += loss
                    test_accuracy.append(valid_correct)
                    test_loss.append(valid_loss)
        self.save_data_model(train_accuracy, train_loss, test_accuracy, test_loss, self.lstm)
        return self.lstm
