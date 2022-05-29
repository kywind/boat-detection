import cv2, os

dir1 = '/data/zkf/2018_level_18/'

files = sorted([f for f in os.listdir(dir1) if f.endswith('.tfw')])
fout = open('tfw_2018_level18.txt', 'w')
for f in files:
    fn = f.split('_')[0]
    print(fn)
    with open(dir1 + f) as fin:
        data = fin.read().strip().split()
    fout.write('{} {} {} {} {}\n'.format(fn, data[0], data[3], data[4], data[5]))
fout.close()