import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from utils.DataLoading import GetURLcharset, URLCharDataset
use_plot = False
use_save = True
if use_save:
    import pickle
    from datetime import datetime

TRAIN_URLS = 'urls.txt'
TRAIN_LABELS = 'labels.txt'
TEST_URLS = 'urls.txt'
TEST_LABELS = 'labels.txt'

## parameter setting
epochs = 50
batch_size = 64
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting
    embedding_dim = 100
    hidden_dim = 50
    url_len = 32
    nlabel = 2
    regularset = set("}} {{ '""~`[]|+-_*^=()1234567890qwertyuiop[]\\asdfghjkl;/.mnbvcxz!?><&*$%QWERTYUIOPASDFGHJKLZXCVBNM#@")  
    chars = tuple(regularset)
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    ### data processing
    dtrain_set = URLCharDataset(int2char, char2int, url_len, TRAIN_URLS, TRAIN_LABELS)

    train_loader = DataLoader(dtrain_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
    dtest_set = URLCharDataset(int2char, char2int, url_len, TEST_URLS, TEST_LABELS)

    test_loader = DataLoader(dtest_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4
                         )
    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                           vocab_size=dtrain_set.vocab_size,label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:
        print("CUDA-compatible GPU was detected, accelerating with GPU-compute")
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_labels = traindata
            train_labels = torch.squeeze(train_labels)

            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: train_inputs = Variable(train_inputs)

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs.t())

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.data[0]

        train_loss_.append(total_loss / total)
        train_acc_.append(float(total_acc) / float(total))
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else: test_inputs = Variable(test_inputs)

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.data[0]
        test_loss_.append(total_loss / total)
        test_acc_.append(float(total_acc) / float(total))

        print('[Epoch: %d/%d] Training Loss: %.6f, Testing Loss: %.6f, Train Accuracy: %.3f, Test Accuracy: %.3f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = url_len

    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param

    if use_plot:
        import PlotFigure as PF
        PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
