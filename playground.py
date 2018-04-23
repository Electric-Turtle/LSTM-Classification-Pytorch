from utils.DataLoading import GetURLcharset, URLCharDataset
import torch.utils.data as data

charset = GetURLcharset('urls.txt')
regularset = set("}} {{ '""~`[]|+-_*^=()1234567890qwertyuiop[]\\asdfghjkl;/.mnbvcxz!?><&*$%QWERTYUIOPASDFGHJKLZXCVBNM#@")
print(charset)
print(charset - regularset)    
chars = tuple(regularset)
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
ds = URLCharDataset(int2char, char2int, 25, 'urls.txt', 'labels.txt')
test=[1,2,3,4]
test_string="asdfjkl;1234567"
print(int2char)
print(ds.ids2caption(test))
print(ds.caption2ids(test_string))
dl = data.DataLoader(ds,batch_size=4,shuffle=True, num_workers=2)
for iter, sample_batched in enumerate(dl):
    urls, labels = sample_batched
    print(urls.size())
    print(labels.size())