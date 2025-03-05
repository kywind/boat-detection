import shutil
import os
import random
import cv2
import sys
import numpy as np


MAP_PATH_DICT = {
    2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20101231/',
    2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20111231/',
    2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20121231/',
    2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20131231/',
    2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20141231/',
    2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20151231/',
    2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20161231/',
    2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20171231/',
    2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20181231/',
    2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20191231/',
    2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20201231/',
    2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/2021_new/',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20230827/'
}
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


def getclusters(filename, edge=0):   # get cluster arrays from a file
    fin = open(filename, 'r')
    data = fin.read().strip().split('cluster')[1:]
    fin.close()
    
    ranges, sizes, contents = [], [], []
    for i in range(len(data)):
        tmp = data[i].strip().split()
        sizes.append(eval(tmp[0]))
        ranges.append((eval(tmp[1])-edge, eval(tmp[2])-edge, eval(tmp[3])+edge, eval(tmp[4])+edge))
        content = []
        for s in range(sizes[i]):
            content.append((eval(tmp[4*s+5]),eval(tmp[4*s+6]),eval(tmp[4*s+7]),eval(tmp[4*s+8])))
        contents.append(content)
    
    return ranges, sizes, contents


def segment_intersect(p1, p2, q1, q2):  # judge if two segments intersect

    def cross(vec1, vec2):
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]

    p_rect = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
    q_rect = (min(q1[0], q2[0]), min(q1[1], q2[1]), max(q1[0], q2[0]), max(q1[1], q2[1]))
    if p_rect[2] < q_rect[0] or p_rect[0] > q_rect[2] or p_rect[3] < q_rect[1] or p_rect[1] > q_rect[3]:
        return False
    
    q1p1 = (p1[0]-q1[0], p1[1]-q1[1])
    q1p2 = (p2[0]-q1[0], p2[1]-q1[1])
    q1q2 = (q2[0]-q1[0], q2[1]-q1[1])
    if cross(q1p1, q1q2) * cross(q1p2, q1q2) >= 0:
        return False

    p1q1 = (q1[0]-p1[0], q1[1]-p1[1])
    p1q2 = (q2[0]-p1[0], q2[1]-p1[1])
    p1p2 = (p2[0]-p1[0], p2[1]-p1[1])
    if cross(p1q1, p1p2) * cross(p1q2, p1p2) >= 0:
        return False

    return True


def hull_intersect(hull1, hull2):  # judge if two convex hulls intersect
    for point in hull1:
        if cv2.pointPolygonTest(hull2, tuple(point[0]), False) >= 0:
            return True
    for point in hull2:
        if cv2.pointPolygonTest(hull1, tuple(point[0]), False) >= 0:
            return True
    for p1, p2 in zip(hull1, np.concatenate((hull1[1:], hull1[:1]), axis=0)):
        for q1, q2 in zip(hull2, np.concatenate((hull2[1:], hull2[:1]), axis=0)):
            # print(p1, p2, q1, q2)
            if segment_intersect(p1[0], p2[0], q1[0], q2[0]):
                return True
    return False


def intersect(box1, box2):  # judge if two boxes intersect
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def getmap(box, year, ratio=1, res=None, level=17, white=False):  # get satellite map for any rectangle (source needed)
    channel = 3
    img_type = '.tif'
    map_path = MAP_PATH_DICT[year]
    tfw_path = TFW_PATH_DICT[year]
    resolution = 0.0000107288 * ratio if not res else res
    if level == 18:
        img_size_bound = (12000,8000)
    elif level == 17:
        img_size_bound = (6000,4000)
    else:
        raise NotImplementedError
        
    ftfw = open(tfw_path, 'r')
    tfw = ftfw.read().strip().split('\n')
    ftfw.close()
    tfw_dict = dict()
    for item in tfw:
        tfw_data = item.split()
        name, xstep, ystep, x0, y0 =  tfw_data[0], eval(tfw_data[1]), eval(tfw_data[2]), eval(tfw_data[3]), eval(tfw_data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    
    x_min, y_min, x_max, y_max = box
    width = int((x_max - x_min) // resolution) + 1
    height = int((y_max - y_min) // resolution) + 1
    if white: res = np.ones((height,width,channel)) * 255
    else: res = np.zeros((height,width,channel))

    for name in tfw_dict.keys():
        xstep, ystep, x0, y0 = tfw_dict[name]
        x1, y1 = x0 + img_size_bound[0] * xstep, y0 + img_size_bound[1] * ystep
        if not os.path.exists(map_path + name + '_Level_{}'.format(level) + img_type):
            continue
        if intersect((x0,y1,x1,y0), box):
            print(year, name)
            img = cv2.imread(map_path + name + '_Level_{}'.format(level) + img_type)
            if img is None:
                continue
            i_list = [i for i in range(img.shape[1]) if x0 + i * xstep >= x_min and x0 + i * xstep < x_max and 
                      (x0 + i * xstep - x_min) % resolution >= 0 and (x0 + i * xstep - x_min) % resolution < abs(xstep)]
            j_list = [j for j in range(img.shape[0]) if y0 + j * ystep >= y_min and y0 + j * ystep < y_max and 
                      (y0 + j * ystep - y_min) % resolution >= 0 and (y0 + j * ystep - y_min) % resolution < abs(ystep)]
            ii, jj = np.meshgrid(i_list, j_list)
            ii = ii.reshape(-1)
            jj = jj.reshape(-1)
            deltax = x0 - x_min
            deltay = y0 - y_min

            for t in range(ii.shape[0]):
                i, j = ii[t], jj[t]
                h = int((deltay + j * ystep) // resolution)
                w = int((deltax + i * xstep) // resolution)
                if (height-h) >= res.shape[0] or w >= res.shape[1]: continue 
                if white: res[height-h][w] = np.array([255,255,255])
                else: res[height-h][w] = img[j][i]
    return res


def mapcut_single(year, taskname=None, size=100, annotate=True, num=None):  # random pick any number of single targets and generate mapcut of any fixed size (e.g. 100x100)
    if taskname is None:
        taskname = year
    filename = './data/{}.txt'.format(taskname)
    savepath = './result/{}_single_{}/'.format(taskname, size)
    # w, h = 200, 200
    w, h = size * 0.0000107288, size * 0.0000107288
    # w, h = 0.002, 0.002

    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.mkdir(savepath)
    os.makedirs(savepath, exist_ok=True)

    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')
        aux = range(len(data))
        if not num:
            num = len(data)
        choice = random.sample(aux, num)
        cnt = 0
        for i in aux:
            if i not in choice:
                continue
            print(data[i])
            xmin, ymin, xmax, ymax = data[i].split()
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            box = (x - w/2, y - h/2, x + w/2, y + h/2)
            delta_x = int((xmax - x) / 0.0000107288)
            delta_y = int((ymax - y) / 0.0000107288)
            # (delta_x > 0 and delta_y > 0)
            try:
                img = getmap(box, year)
                mid_x = int(img.shape[0] / 2)
                mid_y = int(img.shape[1] / 2)
                if annotate:
                    img = cv2.rectangle(img, (mid_x-delta_x, mid_y-delta_y), (mid_x+delta_x, mid_y+delta_y), (0, 0, 255), 1)
                cv2.imwrite(savepath + '{}.jpg'.format(cnt), img)
            except:
                # print('cannot get map for {}'.format(data[i]))
                continue
            f = open(savepath + '{}.txt'.format(cnt), 'w')
            f.write('{} {} {} {}'.format(xmin, ymin, xmax, ymax))
            f.close()
            cnt += 1


def annotate_single(year, taskname=None, num=50):  # random pick 50 original (608x608) images and annotate predictions
    if taskname is None:
        taskname = year
    txtpath = './raw/{}/'.format(taskname)
    jpgpath = '/mnt/satellite/JPEGImages_{}/'.format(year)
    savepath = './result/{}_anno/'.format(taskname)
    infopath = '/mnt/satellite/cutinfo/{}.txt'.format(year)
    tfwpath = './utils/tfw.txt'
    conf_thres = 0.25
    W, H = 608, 608
    
    f_tfw = open(tfwpath, 'r')
    tfw = f_tfw.read().strip().split('\n')
    tfw_dict = dict()
    for item in tfw:
        data = item.split()
        name, xstep, ystep, x0, y0 =  data[0], eval(data[1]), eval(data[2]), eval(data[3]), eval(data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    f_tfw.close()
    
    finfo = open(infopath)
    info = finfo.read().strip().split('\n')
    infodict = {}
    for item in info:
        s = item.strip().split(' ')
        infodict[s[0]] = (eval(s[1]), eval(s[2]), eval(s[3]), eval(s[4]))
    finfo.close()
    
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.mkdir(savepath)
    
    flist = [f[:-4] for f in os.listdir(txtpath) if f.endswith('.txt')]
    aux = range(len(flist))
    choice = random.sample(aux, 50)
    for i in aux:
        if i not in choice:
            continue 
        fn = flist[i]
        f = open(txtpath + fn + '.txt', 'r')
        data = f.read().strip().split('\n')
        f.close()
        jpg = cv2.imread(jpgpath + fn + '.jpg')
        cnt = 0
                
        xmin, xmax, ymin, ymax = infodict[fn]
        fn0 = fn.split('_')[0] 
        xstep, ystep, xs, ys = tfw_dict[fn0]
            
        for j in data:
            tmp = j.split()
            x0, y0, w, h, conf = eval(tmp[1]), eval(tmp[2]), eval(tmp[3]), eval(tmp[4]), eval(tmp[5])
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            # print(x1, y1, x2, y2)
            xm = (x1 + x2)/2
            ym = (y1 + y2)/2 
            jpg = cv2.rectangle(jpg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cnt += 1
                        
        if cnt > 0:
            xx0, yy0, xx1, yy1 = xs + xmin * xstep, ys + ymin * ystep, xs + xmax * xstep, ys + ymax * ystep
            xx0, yy0, xx1, yy1 = round(xx0, 5), round(yy0, 5), round(xx1, 5), round(yy1, 5)
            jpg = cv2.putText(jpg, 'x:{}~{}, y:{}~{}'.format(xx0, xx1, yy0, yy1), 
                              (0, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(savepath + fn + '.jpg', jpg)
    
    
def mapcut_cluster(year, taskname=None):  # generate mapcut of clusters with original size and annotate targets in the cluster
    if taskname is None:
        taskname = year
    edge = 100
    edge *= 0.0000107288
    filename = './result/{}.txt'.format(taskname)
    save_path = './result/{}_cluster/'.format(taskname)
    
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    # os.mkdir(save_path)
    
    ranges, sizes, contents = getclusters(filename, edge)
    for i in range(len(sizes)):
        print('Processing cluster {}...'.format(i))
        xmin, ymin, xmax, ymax = ranges[i]
        img = getmap(ranges[i], year)
        for j in range(len(contents[i])):
            xminloc, yminloc, xmaxloc, ymaxloc = contents[i][j]
            xminloc_pix = int((xminloc - xmin) / (xmax - xmin) * img.shape[1])
            yminloc_pix = int((ymax - yminloc) / (ymax - ymin) * img.shape[0])
            xmaxloc_pix = int((xmaxloc - xmin) / (xmax - xmin) * img.shape[1])
            ymaxloc_pix = int((ymax - ymaxloc) / (ymax - ymin) * img.shape[0])
            img = cv2.rectangle(img, (xminloc_pix, ymaxloc_pix), (xmaxloc_pix, yminloc_pix) , (0,0,255), 1)

        cv2.imwrite(save_path + '{}.jpg'.format(i), img)
        fout = open(save_path + '{}.txt'.format(i), 'w')
        fout.write('region:\n{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
        fout.write('targets:\n')
        for j in range(len(contents[i])):
            fout.write('{} {} {} {}\n'.format(*contents[i][j]))
        fout.write('convexhull:\n')
        targets = []
        for j in range(len(contents[i])):
            targets.append([np.float32((contents[i][j][0] + contents[i][j][2]) / 2),\
                            np.float32((contents[i][j][1] + contents[i][j][3]) / 2)])
        targets = np.array(targets)
        hull = cv2.convexHull(targets)
        for j in range(len(hull)):
            fout.write('{} {}\n'.format(hull[j][0][0], hull[j][0][1]))
            
        
def comparison(years, tasknames=None, thres=5, res=0.0000107288):  # get mapcut for clusters and compare between years
    if tasknames is None:
        tasknames = years
    hull_all = []
    for (year, taskname) in zip(years, tasknames):
        filename = './result/{}.txt'.format(taskname)
        ranges, sizes, contents = getclusters(filename, 0)
        for i in range(len(sizes)):
            if sizes[i] < thres:
                continue
            targets = []
            for j in range(len(contents[i])):
                targets.append([np.float32((contents[i][j][0] + contents[i][j][2]) / 2),\
                                np.float32((contents[i][j][1] + contents[i][j][3]) / 2)])
            targets = np.array(targets)
            hull_new = cv2.convexHull(targets)
            hull_new_info = [[hull_new.copy(), year, len(targets), i]]
            
            flag = 1
            while flag:  # find all hulls that intersect with the new hull
                for k in range(len(hull_all)):
                    hull0 = hull_all[k][0]  # hull0: the hull that has been recorded
                    hull0_info = hull_all[k][1]
                    if hull_intersect(hull_new, hull0):
                        hull_new = cv2.convexHull(np.concatenate((hull_new, hull0), axis=0).reshape(-1, 2))
                        hull_new_info.extend(hull0_info)
                        hull_all.pop(k)  # delete hull0 and be ready to replace by hull1
                        break
                else:
                    flag = 0
            hull_all.append([hull_new, hull_new_info])

    savepath = 'result/comparison/'
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # os.mkdir(savepath)
    
    index_dict = {}

    edge = 100 * 0.0000107288
    cnt = 0
    for i in range(len(hull_all)):
        hull = hull_all[i][0]
        cnt += 1
        xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
        for point in hull:
            xmin = np.minimum(point[0][0], xmin)
            xmax = np.maximum(point[0][0], xmax)
            ymin = np.minimum(point[0][1], ymin)
            ymax = np.maximum(point[0][1], ymax)
        xmin, xmax, ymin, ymax = xmin - edge, xmax + edge, ymin - edge, ymax + edge
        box = (xmin, ymin, xmax, ymax)
        print(box)
        os.makedirs(savepath + '{}/'.format(cnt), exist_ok=True)
        
        fout = open(savepath + '{}/anno.txt'.format(cnt), 'w')
        fout.write('Range: {} {} {} {}'.format(xmin, ymin, xmax, ymax))
        for year in years:
            fout.write('\nYear: {}\n'.format(year))
            fout.write('Cluster ids in this year: ')
            for (subhull, y, num, idx) in hull_all[i][1]:
                if year == y:
                    # fout.write('{}\n'.format(num))
                    # for j in range(len(subhull)):
                    #     fout.write('{} {}\n'.format(subhull[j][0][0], subhull[j][0][1]))
                    index_dict[(year, idx)] = cnt
                    fout.write(' {}'.format(idx))
        fout.close()

        for year in years:
            img = getmap(box, year, res=res)
            cv2.imwrite(savepath + '{}/{}.jpg'.format(cnt, year), img)

            for (subhull, y, num, idx) in hull_all[i][1]:
                # print(y)
                if year == y:
                    for j in range(len(subhull)):
                        x0 = int((subhull[j][0][0]-xmin) // res)
                        x1 = int((subhull[(j+1)%len(subhull)][0][0]-xmin) // res)
                        y0 = int((ymax-subhull[j][0][1]) // res)
                        y1 = int((ymax-subhull[(j+1)%len(subhull)][0][1]) // res)
                        img = cv2.line(img, (x0, y0), (x1, y1), (0,255,255), 1)
            cv2.imwrite(savepath + '{}/{}_selected.jpg'.format(cnt, year), img)

        fdict = open(savepath + 'cluster_belongings.txt', 'w')
        for key in index_dict.keys():
            fdict.write('year: {}, cluster id: {}, belonging folder: {}\n'.format(key[0], key[1], index_dict[key]))
        fdict.close()
  
        
def heatmap_single(year, taskname=None, maptype='edge', x_pix=2000):  # generate heatmap of targets in whole region, using prepared map
    if maptype not in ['edge', 'map']:
        raise NotImplementedError
    if taskname is None:
        taskname = year
    x_min, x_max, y_min, y_max = (95., 97., 16., 18.) # if year != 2018 else (1.06e7, 1.08e7, 1.82e6, 2.02e6)
    res = (x_max - x_min) / x_pix
    bg = cv2.imread('utils/map/{}_level17_gray.png'.format(taskname))
    fin = open('./data/filtered/{}.txt'.format(taskname), 'r')  # for visualization effects, we filter our data based on the yangon amp
    data = fin.read().strip().split('\n')
    fin.close()
    alpha = 0.25
    # overlay = bg.copy()
    for item in data:
        q = item.split()
        x, y = eval(q[0]), eval(q[1])
        x_heat, y_heat = int((x-x_min)//res), int((y_max-y)//res)
        # bg = cv2.circle(bg, (x_heat, y_heat), 10, (0,0,255), -1)
        overlay = bg.copy()
        overlay = cv2.circle(overlay, (x_heat, y_heat), 10, (0,0,255), -1)
        bg = cv2.addWeighted(overlay, alpha, bg, 1 - alpha, 0)
    # bg = cv2.addWeighted(overlay, alpha, bg, 1 - alpha, 0)
    cv2.imwrite('result/{}_heat_single.png'.format(taskname), bg)
    
        
def heatmap_cluster(year, taskname=None, maptype='edge', x_pix=10000):  # generate heatmap of clusters in whole region, using prepared map
    if maptype not in ['edge', 'map']:
        raise NotImplementedError
    if taskname is None:
        taskname = year
    
    def PolyArea(coords, lw):
        x, y = coords[:, 0], coords[:, 1]
        a0 = 0.5 * np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        a1 = np.pi * (lw * 0.5) ** 2
        a2 = (lw * 0.5) * np.sum(np.sqrt((x-np.roll(x,1)) ** 2 + (y-np.roll(y,1)) ** 2))
        return a0+a1+a2
        
    edge = 100
    edge /= 111000
    filename = './result/{}.txt'.format(taskname)
    save_path = './result/'
    x_min, x_max, y_min, y_max = (95., 97., 16., 18.) # if year != 2018 else (1.06e7, 1.08e7, 1.82e6, 2.02e6)
    res = (x_max - x_min) / x_pix
    thres = 5
    bg = cv2.imread('utils/{}_{}/{}.jpg'.format(maptype, x_pix, year))

    ranges, sizes, contents = getclusters(filename, edge)
    # xmin, ymin, xmax, ymax, size, xall, yall = getclusters(filename, edge)
    # fout = open(save_path + '{}.txt'.format(taskname), 'w')
    cnt = 0
    for i in range(len(sizes)):
        xmin, ymin, xmax, ymax = ranges[i]
        if sizes[i] < thres:
            continue
        cnt += 1
        x0_heat, y0_heat = int((xmin-x_min)//res), int((y_max-ymax)//res)
        x1_heat, y1_heat = int((xmax-x_min)//res), int((y_max-ymin)//res)     
        x_center, y_center = (x0_heat + x1_heat) / 2, (y0_heat + y1_heat) / 2
        # bg = cv2.rectangle(bg, (x0_heat, y0_heat), (x1_heat, y1_heat), (0, 255 - step * sizes[i], 255), 1)
        
        targets = []
        xall = []
        yall = []
        for j in range(len(contents[i])):
            xall.append((contents[i][j][0] + contents[i][j][2]) / 2)
            yall.append((contents[i][j][1] + contents[i][j][3]) / 2)
        for j in range(len(contents[i])):
            x_heat, y_heat = int((xall[j]-x_min)//res), int((y_max-yall[j])//res)
            targets.append([x_heat, y_heat])
            # bg = cv2.circle(bg, (x_heat, y_heat), 2, (0, 255 - step * sizes[i], 255), -1)
            
        targets = np.array(targets)       
        hull = cv2.convexHull(targets)
        hull = np.array(hull).reshape(-1,2)
        
        linewidth = int(2 * x_pix / 1000)
        area = PolyArea(hull, linewidth)
        density = sizes[i] / area
        color = (0,255-10000*density,255)
        bg = cv2.fillPoly(bg, [hull], color)
        
        for j in range(len(hull)):
            print(tuple(hull[j]), tuple(hull[(j+1)%len(hull)]))
            bg = cv2.line(bg, tuple(hull[j]), tuple(hull[(j+1)%len(hull)]), color, linewidth)
            
        # fout.write('id:\n{}\nregion:\n{} {} {} {}\n'.format(cnt, xmin, ymin, xmax, ymax))
        # fout.write('center:\n{} {}\n{} {}\n'.format((xmin+xmax)/2, (ymin+ymax)/2, x_center, y_center))
        # fout.write('targets:\n')
        # for j in range(len(contents[i])):
        #     fout.write('{} {}\n'.format(xall[j], yall[j]))
        # fout.write('convexhull:\n')
        # for j in range(len(hull)):
        #     fout.write('{} {}\n'.format(hull[j][0], hull[j][1]))
   
    cv2.imwrite(save_path + '{}_heatmap.jpg'.format(taskname), bg)
  
    
def heatmap_region(year, taskname=None, maprange=None, single=True, cluster=True, thres=5):  # get heatmap in specific region
    x_pix = 10000
    channel = 3
    if taskname is None:
        taskname = year
    if maprange is None:
        x_min, x_max, y_min, y_max = maprange
    else:
        x_min, x_max, y_min, y_max = 96.5, 96.6, 16.9, 17.0
    resolution = (x_max - x_min) / x_pix
    bg0 = getmap((x_min, y_min, x_max, y_max), year, res=resolution)
    
    if single:
        bg = bg0.copy()
        fin = open('./data/{}.txt'.format(taskname), 'r')
        data = fin.read().strip().split('\n')
        fin.close()
        for item in data:
            q = item.split()
            x, y = eval(q[0]), eval(q[1])
            if x >= x_min and x < x_max and y >= y_min and y < y_max:
                x_heat, y_heat = int((x-x_min)//resolution), int((y_max-y)//resolution)
                cv2.circle(bg, (x_heat, y_heat), 1, (0,255,255), -1)
        cv2.imwrite('single_{}_{}_{}_{}_{}.jpg'.format(taskname, x_min, y_min, x_max, y_max), bg)
    
    if cluster:
        bg = bg0.copy()
        filename = './result/{}.txt'.format(taskname)
        xmin, ymin, xmax, ymax, size, xall, yall = getclusters(filename, edge=0)
        for i in range(len(size)):
            if size[i] < thres:
                continue
            if not intersect((xmin[i], ymin[i], xmax[i], ymax[i]), (x_min, y_min, x_max, y_max)):
                continue
            targets = []
            for j in range(len(xall[i])):
                x, y = xall[i][j], yall[i][j]
                if x >= x_min and x < x_max and y >= y_min and y < y_max:
                    x_pos, y_pos = int((x-x_min)//resolution), int((y_max-y)//resolution)
                targets.append([x_pos, y_pos])
            targets = np.array(targets)
            hull = cv2.convexHull(targets)
            bg = cv2.fillPoly(bg, [hull], (0,255-size[i],255))
            for j in range(len(hull)):
                bg = cv2.line(bg, tuple(hull[j][0]), tuple(hull[(j+1)%len(hull)][0]), (0,255-size[i],255), 5)
        cv2.imwrite('cluster_{}_{}_{}_{}_{}.jpg'.format(taskname, x_min, y_min, x_max, y_max), bg)

    
if __name__ == '__main__':
    # for year in range(2010, 2022):
    # for year in MAP_PATH_DICT.keys():
    #     os.makedirs('result/', exist_ok=True)
    #     # heatmap_single(year)
    #     img = getmap((95, 16, 97, 18), year, res=0.001, level=17)
    #     # os.makedirs('utils/map/', exist_ok=True)
    #     cv2.imwrite(f'utils/map/{year}_level17.png', img)
    
    # csv_path = '/home/zhangkaifeng/YONGONCHICKENFISH/src/cluster_detection/utils/2021disappear_position.csv'
    # with open(csv_path) as f:
    #     data = f.read().strip().split('\n')
    # for i, line in enumerate(data[1:]):
    #     for year in range(2010, 2022):
    #         x, y = eval(line.split(',')[0]), eval(line.split(',')[1])
    #         # print(x, y)
    #         x1, y1, x2, y2 = x - 0.0000107288 * 200, y - 0.0000107288 * 200, x + 0.0000107288 * 200, y + 0.0000107288 * 200
    #         img = getmap((x1, y1, x2, y2), year)
    #         cv2.imwrite(f'tmp/{i}_{year}.png', img)
    
    # cnt = 0
    # fout = open('out.txt', 'w')
    # csv_path = '/home/zhangkaifeng/YONGONCHICKENFISH/src/cluster_detection/utils/2021disappear_position.csv'
    # with open(csv_path) as f:
    #     data = f.read().strip().split('\n')
    # for i, line in enumerate(data[1:]):
    #     img1 = cv2.imread(f'tmp/{i}_2021.png')
    #     for year in range(2010, 2021):
    #         img2 = cv2.imread(f'tmp/{i}_{year}.png')
    #         img_area = img1.shape[0] * img1.shape[1]
    #         assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]
    #         area = (np.abs(img2 - img1).sum(2) == 0).sum()
    #         if area > 0.5 * img_area:
    #             # import pdb; pdb.set_trace()
    #             cnt += 1
    #             fout.write(f'{i+1} {year}\n')
    #             break
    # fout.close()
    

    dir_list = {
        2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2010_water_meta',
        2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2011_water_meta',
        2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2012_water_meta',
        2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2013_water_meta',
        2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2014_water_meta',
        2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2015_water_meta',
        2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2016_water_meta',
        2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2017_water_meta',
        2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2018_water_meta',
        2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2019_water_meta',
        2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2020_water_meta',
        2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/segmentation_inference/inference/2021_water_meta',
    }

    for year, dr in dir_list.items():

        for i in range(len(os.listdir(dr))):

            meta_file = f'{dr}/{i}.txt'

            with open(meta_file) as f:
                data = f.read()
            
            pos = data.split('\n')[0].split(' ')
            x = (float(pos[0]) + float(pos[2])) / 2
            y = (float(pos[1]) + float(pos[3])) / 2
            print(x, y)

            x1, y1, x2, y2 = x - 0.0000107288 * 50, y - 0.0000107288 * 50, x + 0.0000107288 * 50, y + 0.0000107288 * 50
            img = getmap((x1, y1, x2, y2), year)

            os.makedirs(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/roof_{year}', exist_ok=True)
            cv2.imwrite(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/roof_{year}/{i}.png', img)


            x1, y1, x2, y2 = x - 0.0000107288 * 200, y - 0.0000107288 * 200, x + 0.0000107288 * 200, y + 0.0000107288 * 200
            img = getmap((x1, y1, x2, y2), year)

            os.makedirs(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/water_{year}', exist_ok=True)
            cv2.imwrite(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/water_{year}/{i}.png', img)
