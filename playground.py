from utils.DataLoading import HTMLCharDataset, URLCharDataset
import torch.utils.data as data
from torch.utils.data import DataLoader
regularset = set("}} {{ '""~`[]|+-_*^=()1234567890qwertyuiop[]\\asdfghjkl;/.mnbvcxz!?><&*$%QWERTYUIOPASDFGHJKLZXCVBNM#@")  
chars = tuple(regularset)
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
### data processing
dtrain_set = HTMLCharDataset(int2char, char2int, 50, 'html_trainset.pkl')
dtest_set = HTMLCharDataset(int2char, char2int, 50, 'html_valset.pkl')
train_loader = DataLoader(dtrain_set,
                batch_size=64,
                shuffle=True,
                num_workers=4
                )
train_load = enumerate(train_loader)
test_loader = DataLoader(dtest_set,
                    batch_size= int(dtest_set.__len__()/10),
                    shuffle=True,
                    num_workers=4
                    )
test_load = enumerate(test_loader)