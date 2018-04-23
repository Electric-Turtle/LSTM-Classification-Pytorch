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
train_urls_out = open('train_urls.txt','w')
train_labels_out = open('train_labels.txt','w')


val_urls_out = open('val_urls.txt','w')
val_labels_out = open('val_labels.txt','w')


val_percentage = 0.2

num_benign_val = int(val_percentage * num_benign)
num_benign_train = num_benign - num_benign_val


print(num_benign_val, num_benign_train)

for i in range(len(benign_examples)):
    if i <= num_benign_val:
        val_urls_out.write(benign_examples[i] + "\n")
        val_labels_out.write("1\n")
    else:
        train_urls_out.write(benign_examples[i] + "\n")
        train_labels_out.write("1\n")        


num_malicious_val = int(val_percentage * num_malicious)
num_malicious_train = num_malicious - num_malicious_val
print(num_malicious_val, num_malicious_train)
for i in range(len(malicious_examples)):
    if i <= num_benign_val:
        val_urls_out.write(malicious_examples[i] + "\n")
        val_labels_out.write("0\n")
    else:
        train_urls_out.write(malicious_examples[i] + "\n")
        train_labels_out.write("0\n")   








