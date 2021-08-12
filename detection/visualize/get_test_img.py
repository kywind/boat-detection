import os
import cv2

def prep():
    f = open('imglist/test.txt')
    data = f.read().strip().split()
    f.close()
    for fn in data:
        print(fn)
        os.system('cp {} testimgs/{}'.format(fn, fn[7:]))


def annotate():
    txtpath = 'labels/'
    jpgpath = 'testimgs/'
    savepath = 'testimgs_anno/'

    W, H = 608, 608

    flist = [f[:-4] for f in os.listdir(jpgpath) if f.endswith('.jpg')]
    for fn in flist:
        f = open(txtpath + fn + '.txt', 'r')
        data = f.read().strip().split('\n')
        f.close()
        jpg = cv2.imread(jpgpath + fn + '.jpg')
        
        for j in data:
            if j == '': continue

            tmp = j.split()
            x0, y0, w, h = eval(tmp[1]), eval(tmp[2]), eval(tmp[3]), eval(tmp[4])
            x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            # print(x1, y1, x2, y2)
            xm = (x1 + x2)/2
            ym = (y1 + y2)/2
            jpg = cv2.rectangle(jpg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
        cv2.imwrite(savepath + fn + '.jpg', jpg)


def annotate_detect():
    txtpath = 'inference/testimgs/'
    jpgpath = 'testimgs_anno/'
    savepath = 'testimgs_anno_detect/'

    W, H = 608, 608

    flist = [f[:-4] for f in os.listdir(txtpath) if f.endswith('.txt')]
    for fn in flist:
        f = open(txtpath + fn + '.txt', 'r')
        data = f.read().strip().split('\n')
        f.close()
        jpg = cv2.imread(jpgpath + fn + '.jpg')
        
        for j in data:
            if j == '': continue
            tmp = j.split()
            x0, y0, w, h, conf = eval(tmp[1]), eval(tmp[2]), eval(tmp[3]), eval(tmp[4]), eval(tmp[5])
            if conf < 0.12: continue
            x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            # print(x1, y1, x2, y2)
            xm = (x1 + x2)/2
            ym = (y1 + y2)/2
            jpg = cv2.rectangle(jpg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        
        cv2.imwrite(savepath + fn + '.jpg', jpg)



if __name__ == '__main__':
    annotate_detect()
    