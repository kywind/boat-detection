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
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


years = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018)
conf_thres = 0.37

for year in years:
    tfw_dir = '../utils/tfw.txt'
    out_dir = '../data/{}.txt'.format(year)
    path = '{}/'.format(year)
    
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
    flist = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in flist:
        F = open(path + file, 'r')
        f = F.read().strip().split('\n')   
        fn = file.strip('.txt')
        
        W, H = 608, 608
        fn0, _, _, xmin, ymin = fn.split('_')
        xmin, ymin = eval(xmin), eval(ymin)
        xmax, ymax = xmin + W, ymin + H

        # if year == 2018:
        #     width1, width0 = shapedict[fn0]

        for row in f:
            a = row.split(' ')
            clss, x, y, w, h, conf = eval(a[0]), eval(a[1]), eval(a[2]), eval(a[3]), eval(a[4]), eval(a[5])
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
                        iou = intersection_over_union(box, (xA, yA, xB, yB)) 
                        if iou > 0.2:
                            hit += 1
                            break
                else:
                    total.append((fn0, (xA, yA, xB, yB)))
                    count += 1
        F.close()

    print(year, count, hit)

    out_list = []

    for (fn, box) in total:
        xA, yA, xB, yB = box
        try:
            xstep, ystep, x0, y0 = tfw_dict[fn]
        except KeyError:
            continue
        xxmin, yymin, xxmax, yymax = x0 + xA * xstep, y0 + yA * ystep, x0 + xB * xstep, y0 + yB * ystep
        xx, yy = (xxmin + xxmax) / 2, (yymin + yymax) / 2
        out_list.append((xx, yy))
    
    out_list.sort(key=lambda x : x[1] * 1000 + x[0])
    fout = open(out_dir, 'w')
    for xx, yy in out_list:
        fout.write('{} {}\n'.format(xx, yy))
    fout.close()
    

