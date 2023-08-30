import cv2
import numpy as np
import re
import os


def parse_polygon():
    f = open('yangon_polygon.txt')
    data = f.read().strip().split('\n')
    f.close()

    flag = 0
    polygon = []
    polygons = []
    for i in range(len(data)):
        if data[i] == 'polygon':
            flag = 1
            continue
        if str(flag) == data[i]:
            continue
        if data[i] == 'END':
            if len(polygon) > 0:
                polygons.append(polygon)
            polygon = []
            flag += 1
            continue
        xy = data[i].lstrip('\t').lstrip(' ')
        xy = re.split(' |\t', xy)
        x, y = eval(xy[0]), eval(xy[1])
        polygon.append([x, y])


    max_len = max([len(x) for x in polygons])
    polygons = [x+ [[-1, -1]]*(max_len-len(x)) for x in polygons]

    polygons = np.array(polygons)
    np.save('yangon_polygon.npy', polygons)


def get_img_path(year, attr=''):
    if attr != '':
        return f'map/{year}_level17_{attr}.png'
    else:
        return f'map/{year}_level17.png'


def draw_polygon(year):

    def loc_to_xy(loc, box, size):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        xpix, ypix = size[0], size[1]
        x = np.floor((loc[..., 0] - xmin) / (xmax - xmin) * xpix + 0.5)
        y = np.floor((ymax - loc[..., 1]) / (ymax - ymin) * ypix + 0.5)
        xy = np.stack([x, y], axis=-1)
        # import ipdb; ipdb.set_trace()
        return xy


    polygons = np.load('yangon_polygon.npy')
    polygons = polygons[2:]
    mask = polygons > 0
    # print(polygons.shape)

    color = (0, 255, 255)
    box = (95, 16, 97, 18)
    bg = cv2.imread(get_img_path(year))
    bg *= 0
    size = (bg.shape[1], bg.shape[0])
    # print(size)

    poly_xy = loc_to_xy(polygons, box, size)

    for i in range(poly_xy.shape[0]):
        poly = poly_xy[i][mask[i]].reshape(-1, 2).astype(np.int32)
        bg = cv2.fillPoly(bg, [poly], color)
    
    cv2.imwrite('polygon.png', bg)


def generate_polygon_map(year):
    bg = cv2.imread(get_img_path(year))
    bg_mask = (bg[:, :, 0:1] == 0) * (bg[:, :, 1:2] == 0) * (bg[:, :, 2:3] == 0)
    bg = bg * (~bg_mask) + np.array([[[125, 125, 125]]]) * bg_mask
    poly_map = cv2.imread('polygon.png')
    mask = (poly_map[:, :, 0:1] == 0) * (poly_map[:, :, 1:2] == 255) * (poly_map[:, :, 2:3] == 255)
    bg = bg * mask + np.array([[[255, 255, 255]]]) * (~mask)
    cv2.imwrite(get_img_path(year, 'polygon'), bg)
    

def generate_white_map(year):
    bg = cv2.imread(get_img_path(year))
    bg_mask = (bg[:, :, 0:1] == 0) * (bg[:, :, 1:2] == 0) * (bg[:, :, 2:3] == 0)
    bg = bg * (~bg_mask) + np.array([[[255, 255, 255]]]) * bg_mask
    cv2.imwrite(get_img_path(year, 'white'), bg)


def generate_gray_map(year):
    poly = cv2.imread('polygon.png')
    poly_mask = (poly[:, :, 0:1] == 0) * (poly[:, :, 1:2] == 255) * (poly[:, :, 2:3] == 255)
    poly = np.array([[[200, 200, 200]]]) * poly_mask + np.array([[[255, 255, 255]]]) * (~poly_mask)
    cv2.imwrite(get_img_path(year, 'gray'), poly)


def filter_data(year):
    if not os.path.exists('../data/orig/'):  # copy original data
        os.system(f'mkdir -p ../data/orig; cp ../data/{year}.txt ../data/orig/')
    with open(f'../data/orig/{year}.txt') as f:
        data = f.read().strip().split('\n')
    fout = open(f'../data/{year}.txt', 'w')
    
    poly = np.load('yangon_polygon.npy')[2]
    poly = poly[poly > 0].reshape(-1, 2)
    poly = (poly * 10000).astype(np.int32)
    
    orig_cnt = len(data)
    cnt = 0
    for i in range(len(data)):
        xmin, ymin, xmax, ymax = data[i].split()
        xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
        x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
        x = int(x * 10000)
        y = int(y * 10000)
        if cv2.pointPolygonTest(poly, (x, y), False) >= 0:  # inside
            fout.write(data[i])
            fout.write('\n')
            cnt += 1
    fout.close()
    print(year, orig_cnt, cnt)
            

if __name__ == '__main__':
    # draw_polygon(2010)
    # generate_gray_map(2010)
    # for year in range(2010, 2022):
    for year in range(20230827, 20230828):
        generate_polygon_map(year)
        generate_white_map(year)
        generate_gray_map(year)
        filter_data(year)

