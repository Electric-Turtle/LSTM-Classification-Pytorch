datain = open('data.csv',mode='r')
urlout = open('urls.txt','w')
labelout = open('labels.txt','w')
lines = datain.readlines()
for line in lines:
   values = line.split(',')
   url=values[0]
   label=values[1].replace('\n','')
   if label in {'good','bad'}:
       urlout.write(url)
       urlout.write('\n')
       labelout.write(str(int(label=='good')))
       labelout.write('\n')
datain.close()
urlout.close()
labelout.close()