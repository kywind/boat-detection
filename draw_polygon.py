import cv2
import numpy as np
import re


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


def draw_polygon():
    polygons = np.load('yangon_polygon.npy')
    

if __name__ == '__main__':
    draw_polygon()

