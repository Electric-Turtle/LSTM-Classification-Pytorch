from random import shuffle

urls_in = open('urls.txt','r')
labels_in = open('labels.txt','r')
urls = urls_in.readlines()
labels = labels_in.readlines()

if(len(urls)!=len(labels)):
    print("Error! The labels file has %d columns, but the urls file has %d colums" % len(labels),len(urls))
else:
    print("Splitting %d values into a train/val split." % len(urls))
urls_in.close()
labels_in.close()

benign_examples=[]
malicious_examples=[]
for i in range(len(urls)):
    url = urls[i].replace('\n','')
    label = labels[i].replace('\n','')
    if(label == '1'):
        benign_examples.append(url)
    else:
        malicious_examples.append(url)
shuffle(benign_examples)
shuffle(malicious_examples)
shuffle(benign_examples)
shuffle(malicious_examples)
shuffle(benign_examples)
shuffle(malicious_examples)


num_benign = len(benign_examples)
num_malicious = len(malicious_examples)
print("Total Benign Examples:", num_benign)
print("Total Malicious Examples:", num_malicious)
train_urls_out = open('train_urls.txt','w')
train_labels_out = open('train_labels.txt','w')
val_urls_out = open('val_urls.txt','w')
val_labels_out = open('val_labels.txt','w')

val_percentage = 0.1

num_malicious_val = int(val_percentage*num_malicious)
malicious_val = malicious_examples[:num_malicious_val]
malicious_train = malicious_examples[num_malicious_val:]
for url in malicious_val:
  val_urls_out.write("%s\n" % url)
  val_labels_out.write("%d\n" % 0)
for url in malicious_train:
  train_urls_out.write("%s\n" % url)
  train_labels_out.write("%d\n" % 0)


num_benign_val = int(val_percentage*num_benign)
benign_val = benign_examples[:num_benign_val]
benign_train = benign_examples[num_benign_val:]
for url in benign_val:
  val_urls_out.write("%s\n" % url)
  val_labels_out.write("%d\n" % 1)
for url in benign_train:
  train_urls_out.write("%s\n" % url)
  train_labels_out.write("%d\n" % 1)











