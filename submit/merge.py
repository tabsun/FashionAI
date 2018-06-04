import os

all_lines = open('submit_blouse.csv','r').readlines()
head = all_lines[0]
lines = all_lines[1:]
for tag in ['dress','outwear','skirt','trousers']:
    sub_lines = open('submit_%s.csv'%tag,'r').readlines()[1:]
    for index,line in enumerate(lines):
        if line.strip().split(',')[1] == tag:
            lines[index] = sub_lines[index]

with open('submit.csv','w') as f:
    f.write(head)
    for line in lines:
        f.write('%s'%line)
