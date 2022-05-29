import cv2, os

dir1 = '/mnt/satellite/rawimages/yangon_2018/'
dir2 = '/mnt/satellite/rawimages/yangon_2018_orig/'

files = [f for f in os.listdir(dir1) if f.endswith('.tif')]
fout = open('2018.txt', 'w')
cnt = 0
for f in files:
    cnt += 1
    print(cnt)
    img1 = cv2.imread(dir1 + f)
    img2 = cv2.imread(dir2 + f)
    fout.write('{} {} {} {} {}\n'.format(f.split('_')[0], img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]))
fout.close()