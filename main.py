import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.DataLoading import GetURLcharset, URLCharDataset
import argparse
import time
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
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting    
    parser = argparse.ArgumentParser(description="LSTM URL Classification Training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size")
    args = parser.parse_args()
    batch_size = args.batch_size
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
                          shuffle=True,
                          num_workers=4
                         )
    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                           vocab_size=dtrain_set.vocab_size,label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:
        print("CUDA-compatible GPU was detected, accelerating with GPU-compute")
        model = model.cuda()
    else:
        print("No GPU detected, using slow-as-balls CPU training")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    model.batch_size = batch_size
    for epoch in range(epochs):

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for (iter, traindata) in enumerate(train_loader):
            train_inputs, train_labels = traindata
          #  print("Train Inputs", train_inputs)
            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: train_inputs = Variable(train_inputs)

            model.hidden = model.init_hidden()
            output = model(train_inputs.t())
           # print("Raw Outputs", output)
          #  print("Labels", train_labels)

            loss = loss_function(output, Variable(train_labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = F.softmax(output,dim=1)
           # print("Softmax Outputs: ", predictions)

            # calc training acc
            _, predicted = torch.max(predictions, 1)
          #  print("Max of the Softmaxes: ", predicted)
           # print("Train Labels: ", train_labels)
            num_right = (predicted == train_labels).sum().item()
           # print("Got ", num_right, " correct")
            total_acc += num_right
            total += len(train_labels)
            total_loss += loss.data.item()
            percent_correct = float(total_acc)/float(total)
            print("Percent Correct: ", percent_correct)
            print("Average Loss: ", total_loss/total)
            time.sleep(1)
            
        train_loss_.append(float(total_loss) / float(total))
        train_acc_.append(float(total_acc) / float(total))
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for (iter, testdata) in enumerate(test_loader):
            test_inputs, test_labels = testdata

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else: test_inputs = Variable(test_inputs)

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())
           # print("Raw Outputs", output)
          #  print("Labels", train_labels)
            loss = loss_function(output, Variable(train_labels))
            predictions = F.softmax(output,dim=1)
           # print("Softmax Outputs: ", predictions)

            # calc training acc
            _, predicted = torch.max(predictions, 1)
          #  print("Max of the Softmaxes: ", predicted)
           # print("Train Labels: ", train_labels)
            num_right = (predicted == train_labels).sum().item()
           # print("Got ", num_right, " correct")
            total_acc += num_right
            total += len(train_labels)
            total_loss += loss.data.item()
            percent_correct = float(total_acc)/float(total)
            print("Validation Percent Correct: ", percent_correct)
            print("Validation Average Loss: ", total_loss/total)

        test_loss_.append(float(total_loss) / float(total))
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
