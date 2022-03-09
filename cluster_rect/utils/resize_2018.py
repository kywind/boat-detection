import os
import cv2
import pickle

indir = '/mnt/satellite/rawimages/yangon_2018_orig/'
refdir = '/mnt/satellite/rawimages/yangon_2010/'
outdir = '/mnt/satellite/rawimages/yangon_2018/'
os.makedirs(outdir, exist_ok=True)


def resize_getdict():
    files = [f for f in os.listdir(refdir) if f.endswith('.tif')]
    w_dict = {'54':3883, '55':3884}
    for f in files:
        print(f)
        img = cv2.imread(refdir + f)
        w_dict[f[5:7]] = img.shape[0]
        print(f[5:7], img.shape[0])
    print(w_dict)

    
w_dict = {'54': 3883, '55': 3884, '59': 3883, '70': 3884, '78': 3884, '80': 3884, '73': 3884, '88': 3884, '82': 3883, '64': 3883, '63': 3884, '61': 3883, '91': 3884, '65': 3884, '69': 3883, '83': 3884, '72': 3883, '81': 3884, '74': 3883, '87': 3883, '84': 3884, '71': 3884, '76': 3884, '75': 3884, '58': 3884, '66': 3883, '62': 3884, '79': 3883, '96': 3884, '86': 3884, '90': 3883, '60': 3884, '85': 3883, '67': 3884, '77': 3883, '92': 3883, '89': 3884, '68': 3884, '94': 3884, '56': 3883, '57': 3884, '93': 3884, '95': 3883}


def resize():
    files = [f for f in os.listdir(indir) if f.endswith('.tif')]
    count = 0
    for f in files:
        count += 1
        print('{} {}/{}'.format(f, count, len(files)))
        img = cv2.imread(indir + f)
        # print(img.shape, (w_dict[f[5:7]], img.shape[1], img.shape[2]))
        img = cv2.resize(img, (img.shape[1], w_dict[f[5:7]]))
        cv2.imwrite(outdir + f, img)
        
        
def tfw():  # simply copy from 2010
    step = 0.0000107288
    files = [f[:-4] + '.tfw' for f in os.listdir(indir) if f.endswith('.tif')]
    count = 0
    nfcnt = 0
    for f in files:
        count += 1 
        print('{} {}/{}'.format(f, count, len(files)))     
        try:
            ref = open(refdir + f, 'r')
            data = ref.read()
            ref.close()
            fout = open(outdir + f, 'w') 
            # fout.write('0.0000107288\n0.0000000000\n0.0000000000\n-0.0000107288\n{}\n{}'.format(x0, y0))
            fout.write(data)
            fout.close()
        except FileNotFoundError:
            print('Notfound')
            nfcnt += 1
            continue
    print(nfcnt, count)  # result: 108/845 not found
        
        
if __name__ == '__main__':
    resize()
        
    