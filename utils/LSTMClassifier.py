import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2fc1 = nn.Linear(hidden_dim, 250)
        self.fc1 = nn.Linear(250, 125)
        self.fc2 = nn.Linear(125, 62)
        self.fc3 = nn.Linear(62, 31)
        self.fc4 = nn.Linear(31, 10)
        self.fc5 = nn.Linear(10, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2fc1(lstm_out[-1])
        y1 = self.fc1(y)
        y1_relu = nn.ReLU(y1)
        y2 = self.fc2(y1_relu)
        y2_relu = nn.ReLU(y2)
        y3 = self.fc3(y2_relu)
        y3_relu = nn.ReLU(y3)
        y4 = self.fc4(y3_relu)
        y4_relu = nn.ReLU(y4)
        y5 = self.fc4(y4_relu)
        y5_relu = nn.ReLU(y5)
        predictions = self.fc5(y5_relu)
        return predictions
