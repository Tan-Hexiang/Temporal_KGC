
from xml.dom.minidom import Element


def rm_granulaity_label(file):
    newdata=[]
    with open(file,'r') as f:
        data=f.readlines()
        l=0
        for line in data:
            print(l)
            l=l+1
            element=line.split('\t')
            start_time=element[3]
            end_time=element[4]
            if start_time[0]=='Y' or start_time[0]=='d' or start_time[0]=='M' or start_time[0]=='C':
                start_time=start_time[1:]
            if end_time[0]=='Y' or end_time[0]=='d' or start_time[0]=='M' or start_time[0]=='C' :
                end_time=end_time[1:]
            newline=element[0]+'\t'+element[1]+'\t'+element[2]+'\t'+start_time+'\t'+end_time
            newdata.append(newline)
    with open(file,'w+') as f:
        for line in newdata:
            f.write(line)



''''''
rm_granulaity_label('/data/tanhexiang/temporal_link_prediction/data/yago11k/temporal')
# rm_granulaity_label('/data/tanhexiang/temporal_link_prediction/data/wikidata12k/temporal')