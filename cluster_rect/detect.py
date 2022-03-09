import matplotlib.pyplot as plt
import math
import sys


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def dfs(data, vis, i, thresh):
    vis[i] = 1
    for j in range(len(data)):
        if vis[j] == 0 and dist(data[i], data[j]) < thresh:
            vis = dfs(data, vis, j, thresh)
    return vis


def detect(year):
    dist_thresh = 500
    size_thresh = 3
    filename = './data/{}_labeled.txt'.format(year)
    save_path = './result/'

    task = filename.split('/')[-1].split('.')[0]
    fin = open(filename, 'r')
    data = fin.read().strip().split('\n')
    fin.close()
    for i in range(len(data)):
        tmp = data[i].split()
        data[i] = (eval(tmp[0]), eval(tmp[1]))

    scale = 111000
    dist_thresh = dist_thresh / scale

    group = [0] * len(data)
    result = []
    p = 1
    for i in range(len(data)):
        if group[i] != 0:
            continue
        vis = dfs(data, [0] * len(data), i, dist_thresh)
        choice = []
        for j in range(len(data)):
            if vis[j] == 1:
                choice.append(data[j])
                group[j] = p
        if len(choice) >= size_thresh:
            print('Found size {} cluster.'.format(len(choice)))
            result.append(choice)
        p += 1

    fout = open(save_path + '{}.txt'.format(task), 'w')
    for cluster in result:
        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf   
        for item in cluster:
            xmin = min(xmin, item[0])
            xmax = max(xmax, item[0])
            ymin = min(ymin, item[1])
            ymax = max(ymax, item[1])          
        fout.write('cluster\n')
        fout.write('{} {} {} {} {}\n'.format(len(cluster), xmin, ymin, xmax, ymax))   
        for item in cluster:
            fout.write('{} {}\n'.format(item[0], item[1]))          
    fout.close()
    
    
if __name__ == '__main__':
    years = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018)
    for year in years:
        detect(year)
