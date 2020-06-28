import os

#This code lets the user select which trajectories will be included in the test and train split.

root = '/mnt/gpid07/imatge/marc.arriaga/database/paths'

files = [f for f in os.listdir(root) if os.path.isfile(f) and f != 'split.py']

for file1 in files:
    r = []
    for line in open(file1, 'r'):
        r.append(line)
    if file1 == 'M2020-03-13-15-03-13.txt' or file1 == 'M2020-03-13-10-28-05.txt' or file1 == 'M2019-07-03-14-59-37.txt':
        with open('splits/test_2_rgb.txt', 'a') as t:
            for item in r:
                t.write(item)
    else: 
        with open('splits/train_2_rgb.txt', 'a') as p:
            for item in r:
                p.write(item)
