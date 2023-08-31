import os
import shutil
import random
import cv2
import numpy as np

MAP_PATH_DICT = {
    # 2010: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20101231/',
    # 2011: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20111231/',
    # 2012: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20121231/',
    # 2013: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20131231/',
    # 2014: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20141231/',
    # 2015: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20151231/',
    # 2016: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20161231/',
    # 2017: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20171231/',
    # 2018: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20181231/',
    # 2019: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20191231/',
    # 2020: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20201231/',
    # # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20220531/'
    # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/2021_new/'
    20230827: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/data/2023_0827/'
}
TFW_PATH_DICT = {
    # 2010: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20101231.txt',
    # 2011: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20111231.txt',
    # 2012: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20121231.txt',
    # 2013: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20131231.txt',
    # 2014: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20141231.txt',
    # 2015: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20151231.txt',
    # 2016: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20161231.txt',
    # 2017: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20171231.txt',
    # 2018: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20181231.txt',
    # 2019: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20191231.txt',
    # 2020: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20201231.txt',
    # # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20220531.txt'
    # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/2021_new.txt'
    20230827: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/data/tfw/20230827.txt',
}


def intersect(box1, box2):  # judge if two boxes intersect
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def getmap(box, year, ratio=1, res=None):  # get satellite map for any rectangle (source needed)
    channel = 3
    img_type = '.tif'
    img_size_bound = (6000,4000)
    # map_path = '/home/zhangkaifeng/projects/YONGONCHICKENFISH/data/{}/'.format(year)
    # tfw_path = '../cluster/utils/tfw.txt'
    map_path = MAP_PATH_DICT[year]
    tfw_path = TFW_PATH_DICT[year]
    resolution = 0.0000107288 * ratio if not res else res
        
    ftfw = open(tfw_path, 'r')
    tfw = ftfw.read().strip().split('\n')
    ftfw.close()
    tfw_dict = dict()
    for item in tfw:
        tfw_data = item.split()
        name, xstep, ystep, x0, y0 =  tfw_data[0], eval(tfw_data[1]), eval(tfw_data[2]), eval(tfw_data[3]), eval(tfw_data[4])
        tfw_dict[name] = (xstep, ystep, x0, y0)
    
    x_min, y_min, x_max, y_max = box
    width = int((x_max - x_min) / resolution) + 1
    height = int((y_max - y_min) / resolution) + 1
    res = np.zeros((height-1,width-1,channel))

    for name in tfw_dict.keys():
        xstep, ystep, x0, y0 = tfw_dict[name]
        x1, y1 = x0 + img_size_bound[0] * xstep, y0 + img_size_bound[1] * ystep
        if not os.path.exists(map_path + name + '_Level_17' + img_type):
            continue
        if intersect((x0,y1,x1,y0), box):
            print(name)
            img = cv2.imread(map_path + name + '_Level_17' + img_type)
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

            for t in range(ii.shape[0]):  # ystep = -resolution, xstep = resolution
                i, j = ii[t], jj[t]
                h = int(deltay / resolution - j)
                w = int(deltax / resolution + i)
                if h >= res.shape[0] or w >= res.shape[1]:
                    continue 
                res[h][w] = img[j][i]
                # res[h][w] = np.array([255,255,255])
    res = cv2.flip(res, 0)
    return res


def make_roof_buffer(years):
    # years = [2010, 2011, 2012, 2013, 2014, 2015, 2016]
    # years = range(20230827, 20230828)
    for year in years:
        filename = '../cluster_detection/data/{}.txt'.format(year)
        outdir = 'detect_buffer_roof_{}/'.format(year)
        os.makedirs(outdir, exist_ok=True)

        w, h = 200, 200
        w, h = w * 0.0000107288, h * 0.0000107288

        with open(filename, 'r') as f:
            data = f.read().strip().split('\n')
        aux = range(len(data))
        cnt = 0
        for i in aux:
            print(data[i])
            xmin, ymin, xmax, ymax = data[i].split()
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            box = (x - w/2, y - h/2, x + w/2, y + h/2)
            delta_x = int((xmax - x) / 0.0000107288)
            delta_y = int((ymax - y) / 0.0000107288)
            # assert(delta_x > 0 and delta_y > 0)
            img = getmap(box, year)
            mid_x = int(img.shape[0] / 2)
            mid_y = int(img.shape[1] / 2)
            # img = cv2.rectangle(img, (mid_x-delta_x, mid_y-delta_y), (mid_x+delta_x, mid_y+delta_y), (0, 0, 255), 1)
            cv2.imwrite(outdir + '{}.jpg'.format(cnt), img)
            # f = open(outdir + '{}.txt'.format(cnt), 'w')
            # f.write('{} {} {} {}'.format(xmin, ymin, xmax, ymax))
            # f.close()
            cnt += 1


def make_water_buffer(years):
    # years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018]
    # years = range(20230827, 20230828)
    for year in years:
        filename = '../cluster_detection/data/{}.txt'.format(year)
        outdir = 'detect_buffer_water_{}/'.format(year)
        os.makedirs(outdir, exist_ok=True)

        w, h = 500, 500
        res = 0.0000107288
        w, h = w * res, h * res

        with open(filename, 'r') as f:
            data = f.read().strip().split('\n')
        aux = range(len(data))
        cnt = 0
        for i in aux:
            print(year, i, data[i])
            xmin, ymin, xmax, ymax = data[i].split()
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
            box = (x - w/2, y - h/2, x + w/2, y + h/2)
            delta_x = int((xmax - x) / res)
            delta_y = int((ymax - y) / res)
            # assert(delta_x > 0 and delta_y > 0)
            img = getmap(box, year)
            mid_x = int(img.shape[0] / 2)
            mid_y = int(img.shape[1] / 2)
            # img = cv2.rectangle(img, (mid_x-delta_x, mid_y-delta_y), (mid_x+delta_x, mid_y+delta_y), (0, 0, 255), 1)
            cv2.imwrite(outdir + '{}.jpg'.format(cnt), img)
            # f = open(outdir + '{}.txt'.format(cnt), 'w')
            # f.write('{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
            # for j in aux:
            #     xmin2, ymin2, xmax2, ymax2 = data[j].split()
            #     xmin2, ymin2, xmax2, ymax2 = eval(xmin2), eval(ymin2), eval(xmax2), eval(ymax2)
            #     if intersect(box, (xmin2, ymin2, xmax2, ymax2)):
            #         print(xmin2, ymin2, xmax2, ymax2)
            #         xmid2 = (xmin2+xmax2)/2
            #         ymid2 = (ymin2+ymax2)/2
            #         x2 = int((xmid2-box[0])/res)
            #         y2 = int((box[3]-ymid2)/res)
            #         f.write('{} {} {}\n'.format(j, x2, y2))
            # f.close()
            cnt += 1

if __name__ == '__main__':
    years = range(20230827, 20230828)
    make_roof_buffer(years)
    make_water_buffer(years)
