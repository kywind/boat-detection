import os, re
import cv2


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
    return iou, boxBArea > boxAArea

# tasknames = range(2010, 2022)
conf_thres = 0.22
iou_thres = 0.1
W, H = 608, 608

TFW_PATH_DICT = {
    2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20101231.txt',
    2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20111231.txt',
    2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20121231.txt',
    2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20131231.txt',
    2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20141231.txt',
    2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20151231.txt',
    2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20161231.txt',
    2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20171231.txt',
    2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20181231.txt',
    2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20191231.txt',
    2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20201231.txt',
    2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/2021_new.txt',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20230827.txt',
}

# for taskname in tasknames:
for taskname in TFW_PATH_DICT.keys():
    tfw_dir = TFW_PATH_DICT[taskname]
    os.makedirs('./data/', exist_ok=True)
    out_dir = './data/{}.txt'.format(taskname)
    path = '../detection-yolov11/log/raw/{}.txt'.format(taskname)
    
    f_tfw = open(tfw_dir, 'r')
    tfw = f_tfw.read().strip().split('\n')
    tfw_dict = dict()
    for item in tfw:
        data = item.split()
        name, xstep, ystep, x0, y0 =  data[0], eval(data[1]), eval(data[2]), eval(data[3]), eval(data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    f_tfw.close()
    
    # if year == 2018:
    #     shapepath = '../utils/2018_shape.txt'
    #     fshape = open(shapepath)
    #     shape = fshape.read().strip().split('\n')
    #     shapedict = {}
    #     for item in shape:
    #         s = item.strip().split(' ')
    #         shapedict[s[0]] = (eval(s[1]), eval(s[3]))
    #     fshape.close() 
        
    count = 0
    hit = 0
    total = []
    # flist = [f for f in os.listdir(path) if f.endswith('.txt')]
    # for file in flist:
        # F = open(path + file, 'r')
    
    F = open(path, 'r')
    f = F.read().strip().split('\n')   
    if True:
        # fn = file.strip('.txt')
        

        # if year == 2018:
        #     width1, width0 = shapedict[fn0]

        for row in f:
            
            a = row.split(' ')
            fn, clss, x, y, w, h, conf = a[0], eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4]), eval(a[5]), eval(a[6])
            
            fn0, _, _, xmin, ymin = fn.split('_')
            xmin, ymin = eval(xmin), eval(ymin)
            xmax, ymax = xmin + W, ymin + H

            x, y, w, h = x * W, y * H, w * W, h * H
            if conf >= conf_thres:
                xA = xmin + x - w/2
                yA = ymin + y - h/2
                xB = xmin + x + w/2
                yB = ymin + y + h/2

                # if year == 2018:
                #     yA, yB = yA/width0*width1, yB/width0*width1

                # print(xA, yA, xB, yB)
                for (fnbox, box) in total:
                    if fnbox == fn0:
                        iou, larger = intersection_over_union(box, (xA, yA, xB, yB)) 
                        if iou > iou_thres:
                            hit += 1
                            if larger:
                                total.remove((fnbox, box))
                                total.append((fn0, (xA, yA, xB, yB)))
                            break
                else:
                    total.append((fn0, (xA, yA, xB, yB)))
                    count += 1
        F.close()

    print(taskname, count, hit)

    out_list = []

    for (fn, box) in total:
        xA, yA, xB, yB = box
        try:
            xstep, ystep, x0, y0 = tfw_dict[fn]
        except KeyError:
            continue
        xxmin, yymin, xxmax, yymax = x0 + xA * xstep, y0 + yB * ystep, x0 + xB * xstep, y0 + yA * ystep
        xx, yy = (xxmin + xxmax) / 2, (yymin + yymax) / 2
        out_list.append((xxmin, yymin, xxmax, yymax))
    
    out_list.sort(key=lambda x : x[1] * 1000 + x[0])
    fout = open(out_dir, 'w')
    for xxmin, yymin, xxmax, yymax in out_list:
        fout.write('{} {} {} {}\n'.format(xxmin, yymin, xxmax, yymax))
    fout.close()
    

