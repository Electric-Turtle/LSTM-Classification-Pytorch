from random import shuffle
from os import listdir
from os.path import isfile, join
import pickle
htmlfiles = [f for f in listdir('raw_html') if isfile(join('raw_html', f))]
malicious_examples = []
benign_examples = []
labelsfile = open('labels.txt')
labels = labelsfile.readlines()
labelsfile.close()
for htmlfile in htmlfiles:
  f = join('raw_html', htmlfile)
  ff = open(f)
  try:
    ff_string = ff.read()
  except:
    ff.close()
    continue
  ff.close()
  ff_string = ff_string.replace('\n','')
  if("<" not in ff_string or '404 Not Found' in ff_string):
    continue
  index_str, _ = htmlfile.split('.')
  index = int(index_str)
  if labels[index].replace('\n','')=='1':
    benign_examples.append(ff_string)
  else:
    malicious_examples.append(ff_string)
shuffle(benign_examples)
shuffle(malicious_examples)
print("Num benign examples: ", len(benign_examples))
print("Num malicious examples: ", len(malicious_examples))
val_ratio = 0.2
malicious_val_count = int(val_ratio * len(malicious_examples))
valset=[]
trainset=[]
for i in range(len(malicious_examples)):
  if i <= malicious_val_count:
    valset.append((malicious_examples[i],0))
  else:
    trainset.append((malicious_examples[i],0))
benign_val_count = int(val_ratio * len(benign_examples))
for i in range(len(benign_examples)):
  if i <= benign_val_count:
    valset.append((benign_examples[i],1))
  else:
    trainset.append((benign_examples[i],1))
#print("Valset: ", valset)
#print("Trainset: ", trainset)
print(valset[0])
print(trainset[0])
print("Valset Size: ", len(valset))
print("Trainset Size: ", len(trainset))



filename = 'html_valset.pkl'
fp = open(filename, 'wb')
pickle.dump(valset, fp)
fp.close()


filename = 'html_trainset.pkl'
fp = open(filename, 'wb')
pickle.dump(trainset, fp)
fp.close()










