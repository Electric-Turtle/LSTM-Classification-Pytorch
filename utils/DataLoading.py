import torch.utils.data as data
import torch
import torch.nn.functional as F
import pickle
class URLCharDataset(data.Dataset):
    # Load annotations in the initialization of the object.
    def __init__(self, int2char, char2int, standard_length, url_file, label_file):
        self.int2char = int2char
        self.char2int = char2int
        self.standard_length=standard_length
        self.unknown_char_id = len(self.char2int)
        self.padding_char_id = len(self.char2int)+1
        self.vocab_size = len(self.char2int)+2
        self.read_files(url_file,label_file)
    def read_files(self, url_file, label_file):
        urls_in = open(url_file,'r')
        labels_in = open(label_file,'r')
        self.urls = urls_in.readlines()
        self.labels = labels_in.readlines()
    # Transform a caption into a list of word ids.
    def caption2ids(self, caption):
        caption_ids = [self.char2int.get(c, self.unknown_char_id) for c in tuple(caption)]
        temp = torch.LongTensor(caption_ids)
        l = len(temp)
        if l  > self.standard_length:
            return temp[:self.standard_length]
        elif l  < self.standard_length:
            return torch.cat((temp,torch.LongTensor(self.standard_length-l).fill_(self.padding_char_id)),0)
        else:
            return temp
    
    # Transform a list of word ids into a caption.
    def ids2caption(self, caption_ids):
        return ''.join([self.int2char.get(C,'UNK') for C in caption_ids])
    
    # Return imgId, and a random caption for that image.
    def __getitem__(self, index):
        return self.caption2ids(self.urls[index].replace('\n','')), int(self.labels[index])
    
    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.urls)
class HTMLCharDataset(data.Dataset):
    # Load annotations in the initialization of the object.
    def __init__(self, int2char, char2int, standard_length, pkl_file):
        self.int2char = int2char
        self.char2int = char2int
        self.standard_length=standard_length
        self.unknown_char_id = len(self.char2int)
        self.padding_char_id = len(self.char2int)+1
        self.vocab_size = len(self.char2int)+2
        self.read_file(pkl_file)
    def read_file(self, pkl_file):
        f = open(pkl_file,'rb')
        self.examples = pickle.load(f)
        f.close()
    # Transform a caption into a list of word ids.
    def caption2ids(self, caption):
        caption_ids = [self.char2int.get(c, self.unknown_char_id) for c in tuple(caption)]
        temp = torch.LongTensor(caption_ids)
        l = len(temp)
        if l  > self.standard_length:
            return temp[:self.standard_length]
        elif l  < self.standard_length:
            return torch.cat((temp,torch.LongTensor(self.standard_length-l).fill_(self.padding_char_id)),0)
        else:
            return temp
    
    # Transform a list of word ids into a caption.
    def ids2caption(self, caption_ids):
        return ''.join([self.int2char.get(C,'UNK') for C in caption_ids])
    
    # Return imgId, and a random caption for that image.
    def __getitem__(self, index):
        example = self.examples[index]
        return self.caption2ids(example[0]), example[1]
    
    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.examples)
