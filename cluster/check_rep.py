import os, re
import cv2
import shutil
from visualize import getmap


def intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea * boxBArea == 0:
        return 0
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / float(min(boxAArea, boxBArea))
    return iou


conf_thres = 0.18
iou_thres_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
W, H = 608, 608

for iou_thres in iou_thres_list:
    tfw_dir = './utils/tfw.txt'
    path = './raw/2010/'
    
    f_tfw = open(tfw_dir, 'r')
    tfw = f_tfw.read().strip().split('\n')
    tfw_dict = dict()
    for item in tfw:
        data = item.split()
        name, xstep, ystep, x0, y0 =  data[0], eval(data[1]), eval(data[2]), eval(data[3]), eval(data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    f_tfw.close()
    
    count = 0
    hit = 0
    total = []
    flist = [f for f in os.listdir(path) if f.endswith('.txt')]
    collisions = []
    for file in flist:
        F = open(path + file, 'r')
        f = F.read().strip().split('\n')   
        fn = file.strip('.txt')
        
        fn0, _, _, xmin, ymin = fn.split('_')
        xmin, ymin = eval(xmin), eval(ymin)
        xmax, ymax = xmin + W, ymin + H

        for row in f:
            a = row.split(' ')
            clss, x, y, w, h, conf = eval(a[0]), eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4]), eval(a[5])
            x, y, w, h = x * W, y * H, w * W, h * H
            if conf >= conf_thres:
                xA = xmin + x - w/2
                yA = ymin + y - h/2
                xB = xmin + x + w/2
                yB = ymin + y + h/2

                flag = False
                for (fnbox, box) in total:
                    if fnbox == fn0:
                        iou = intersection_over_union(box, (xA, yA, xB, yB)) 
                        if iou > iou_thres and iou < iou_thres + 0.1:
                            hit += 1
                            collisions.append((fn0, box, (xA, yA, xB, yB)))
                            flag = True
                
                if not flag:
                    total.append((fn0, (xA, yA, xB, yB)))
                    count += 1
        F.close()

    print(count, hit)
    
    savepath = './result/2010_collision_partial_{}/'.format(iou_thres)
    w, h = 100, 100
    w, h = w/111000, h/111000

    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.mkdir(savepath)

    cnt = 0
    for (fn, box1, box2) in collisions:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        try:
            xstep, ystep, x0, y0 = tfw_dict[fn]
        except KeyError:
            continue
        xxmin1, yymin1, xxmax1, yymax1 = x0 + xmin1 * xstep, y0 + ymin1 * ystep, x0 + xmax1 * xstep, y0 + ymax1 * ystep
        xxmin2, yymin2, xxmax2, yymax2 = x0 + xmin2 * xstep, y0 + ymin2 * ystep, x0 + xmax2 * xstep, y0 + ymax2 * ystep
        xxmin = min(xxmin1, xxmin2)
        yymin = min(yymin1, yymin2)
        xxmax = max(xxmax1, xxmax2)
        yymax = max(yymax1, yymax2)
        box = (xxmin-w/2, yymin-h/2, xxmax+w/2, yymax+h/2)
        img = getmap(box, 2010)
        
        res = 0.0000107288
        img = cv2.rectangle(img, (int((xxmin1-box[0])/res), int((box[3]-yymin1)/res)), \
                                 (int((xxmax1-box[0])/res), int((box[3]-yymax1)/res)), \
                                  (0, 0, 255), 1)
        img = cv2.rectangle(img, (int((xxmin2-box[0])/res), int((box[3]-yymin2)/res)), \
                                 (int((xxmax2-box[0])/res), int((box[3]-yymax2)/res)), \
                                  (0, 0, 255), 1)
        cv2.imwrite(savepath + '{}.jpg'.format(cnt), img)
        cnt += 1

