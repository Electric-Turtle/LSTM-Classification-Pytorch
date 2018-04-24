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

TRAIN_URLS = 'train_urls.txt'
TRAIN_LABELS = 'train_labels.txt'
TEST_URLS = 'val_urls.txt'
TEST_LABELS = 'val_labels.txt'

## parameter setting
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting    
    parser = argparse.ArgumentParser(description="LSTM URL Classification Training")
    parser.add_argument("--use_gpu", type=bool, default=False, help="Accelerate with GPU")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size")
    parser.add_argument("--hidden_dim", type=int, default=30, help="Hidden Dimension of the LSTM")
    parser.add_argument("--embedding_dim", type=int, default=80, help="Embedding Dimension of the URL Tokens")
    parser.add_argument("--url_len", type=int, default=60, help="Clips all URLs to this length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs to run")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Number of training epochs to run")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    use_gpu = args.use_gpu
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    url_len = args.url_len
    epochs = args.epochs
    learning_rate=args.learning_rate
    nlabel = 2
    regularset = set("}} {{ '""~`[]|+-_*^=()1234567890qwertyuiop[]\\asdfghjkl;/.mnbvcxz!?><&*$%QWERTYUIOPASDFGHJKLZXCVBNM#@")  
    chars = tuple(regularset)
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    ### data processing
    dtrain_set = URLCharDataset(int2char, char2int, url_len, TRAIN_URLS, TRAIN_LABELS)
    dtest_set = URLCharDataset(int2char, char2int, url_len, TEST_URLS, TEST_LABELS)
    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                           vocab_size=dtrain_set.vocab_size,label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    ### training procedure
    train_loader = DataLoader(dtrain_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4
                    )

    test_loader = DataLoader(dtest_set,
                        batch_size=10,
                        shuffle=True,
                        num_workers=4
                        )
    for epoch in range(epochs):

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        print("Starting Training Epoch: ", epoch+1)
        time.sleep(3)
        model.batch_size = batch_size
        for (i, traindata) in enumerate(train_loader):
            train_inputs, train_labels = traindata
            if(train_inputs.size()[0]!=batch_size):
                continue
          #  print("Train Inputs", train_inputs)
           
            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else: train_inputs = Variable(train_inputs)

            model.hidden = model.init_hidden()
            try:
                output = model(train_inputs.t())
            except:
                print("Output failed to compute for some reason.")
                continue
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
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        model.batch_size = dtest_set.__len__()/10
        for (i, testdata) in enumerate(test_loader):
            test_inputs, test_labels = testdata

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else: test_inputs = Variable(test_inputs)

            model.hidden = model.init_hidden()
            try:
                output = model(test_inputs.t())
            except:
                print("Output failed to compute for some reason... Skipping that input")
                continue
            #print("Raw Outputs", output)
            loss = loss_function(output, Variable(test_labels))
            predictions = F.softmax(output,dim=1)
            #print("Softmax Outputs: ", predictions)

            # calc training acc
            _, predicted = torch.max(predictions, 1)
            #print("Max of the Softmaxes: ", predicted)
            #print("Labels", test_labels)
            num_right = (predicted == test_labels).sum().item()
            #print("Got ", num_right, " correct")
            total_acc += num_right
            total += len(test_labels)
            total_loss += loss.data.item()
            percent_correct = float(total_acc)/float(total)
            print("Validation Percent Correct: ", percent_correct)
            print("Validation Average Loss: ", total_loss/total)
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
        dt = datetime.now().strftime("%d-%h-%m-%s")

        filename = 'log/LSTM_classifier_model_' + dt + '.pkl'
        fp = open(filename, 'wb')
        pickle.dump(model, fp)
        fp.close()
        print('File %s is saved.' % filename)

        filename = 'log/LSTM_classifier_trainset_' + dt + '.pkl'
        fp = open(filename, 'wb')
        pickle.dump(dtrain_set, fp)
        fp.close()
        print('File %s is saved.' % filename)

        filename = 'log/LSTM_classifier_valset_' + dt + '.pkl'
        fp = open(filename, 'wb')
        pickle.dump(dtest_set, fp)
        fp.close()
        print('File %s is saved.' % filename)
