
with open('temporal','r') as input,open('new_temporal','w') as output:
    data=input.readlines()
    for line in data:
        triple=line.split(',') 
        if len(triple)==4:
            triple.append('None')
        if triple[3]=="":
            triple[3]='None'
        output.write(triple[0]+'\t'+triple[1]+'\t'+triple[2]+'\t'+triple[3]+'\t'+triple[4])
        