import shutil
import os
import random
import cv2
import sys
import numpy as np


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

year_range = range(20230827, 20230828)
total_area = [0] * len(year_range)
total_roof_area = [0] * len(year_range)
average_size = [0] * len(year_range)
roof_mean_area = [0] * len(year_range)
thatch_percent = [0] * len(year_range)
zinc_percent = [0] * len(year_range)

for year_id, year in enumerate(year_range):
        
    x_pix = 10000
    taskname = year
    
    def PolyArea(coords, lw):
        x, y = coords[:, 0], coords[:, 1]
        a0 = 0.5 * np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        a1 = np.pi * (lw * 0.5) ** 2
        a2 = (lw * 0.5) * np.sum(np.sqrt((x-np.roll(x,1)) ** 2 + (y-np.roll(y,1)) ** 2))
        return a0+a1+a2
        
    edge = 100
    edge /= 111000
    filename = '../cluster_detection/result/{}.txt'.format(taskname)
    x_min, x_max, y_min, y_max = (95., 97., 16., 18.) # if year != 2018 else (1.06e7, 1.08e7, 1.82e6, 2.02e6)
    res = (x_max - x_min) / x_pix
    thres = 5
    # xmin, ymin, xmax, ymax, size, xall, yall = getclusters(filename, edge)
    ranges, sizes, contents = getclusters(filename, edge)
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
        
        area = PolyArea(hull, 0)
        area *= 0.0002 * 111000 * 0.0002 * 111000
        total_area[year_id] += int(area)
        average_size[year_id] += int(sizes[i])
    average_size[year_id] /= len(sizes)

    print(cnt)

    single_in_dir = '../cluster_detection/result/{}_single_200/'.format(year)
    cluster_in_file = '../cluster_detection/result/{}_cluster_info.txt'.format(year)
    seg_in_file = '../segmentation/result/seg_res_{}.txt'.format(year)
    out_file = 'result/{}.csv'.format(year)
    stat_out_file = 'result/{}_stat.txt'.format(year)
    fout = open(out_file, 'w')
    stat_fout = open(stat_out_file, 'w')
    fout.write('idx, xmin, ymin, xmax, ymax, cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax, material, roof_area, water_area, adjacent\n')

    single_in_files = [f for f in os.listdir(single_in_dir) if f.endswith('txt')]
    with open(seg_in_file) as f:
        seg_in_data = f.read().strip().split('\n')

    material_list = [None] * len(single_in_files)
    roof_area_list = [None] * len(single_in_files)
    for idx in range(len(single_in_files)):
        single_in = single_in_dir + '{}.txt'.format(idx)
        seg_in = seg_in_data[idx]

        with open(single_in) as f:
            single_in_data = f.read().strip().split()
        
        xmin, ymin, xmax, ymax = eval(single_in_data[0]), eval(single_in_data[1]), eval(single_in_data[2]), eval(single_in_data[3])
        
        seg_line = seg_in_data[idx].split(',')
        seg_idx, material, roof_area, water_area, adjacent = eval(seg_line[0]), eval(seg_line[1]), eval(seg_line[2]), eval(seg_line[3]), seg_line[4]
        material_list[idx] = material
        roof_area_list[idx] = roof_area
    
    # print(material_list)
    # raise
    cluster_dict = {}
    roof_area_count = 0
    material_count = 0
    zinc_count = thatch_count = 0
    roof_total_area = 0
    with open(cluster_in_file) as f:
        cluster_in_data = f.read().strip().split('\n')
    for line in cluster_in_data[:-1]:
        line = line.replace(':', ' ')
        line = line.replace('containing target id', ' ')
        line = line.replace('cluster', ' ')
        line = line.replace('size', ' ')
        line = line.replace('range', ' ')
        line = line.replace(',', ' ')
        line = line.split()
        # print(line)
        cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax = \
            eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]), eval(line[5])
        

        for i in range(cluster_size):
            obj_id = eval(line[6+i])
            material = material_list[obj_id]
            roof_area = roof_area_list[obj_id]
            # if material != 0:
            if material == 1: thatch_count += 1
            else: zinc_count += 1
            material_count += 1
            if roof_area != 0:
                roof_total_area += roof_area
                roof_area_count += 1

            if obj_id in cluster_dict.keys(): raise ValueError
            cluster_dict[obj_id] = (cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax)

            # print(cluster_dict)

    total_roof_area[year_id] = roof_total_area
    roof_mean_area[year_id] = roof_total_area / roof_area_count
    thatch_percent[year_id] = thatch_count / material_count
    zinc_percent[year_id] = zinc_count / material_count

    stat_fout.write('year: {}\ntotal_area: {}\ntotal_roof_area: {}\naverage_size: {}\nroof_mean_area: {}\nthatch_percent: {}\nzinc_percent: {}\n\n'
        .format(year, total_area[year_id], total_roof_area[year_id], average_size[year_id], roof_mean_area[year_id], thatch_percent[year_id], zinc_percent[year_id]))


    ## iterate everything again, generate csv
    for idx in range(len(single_in_files)):
        single_in = single_in_dir + '{}.txt'.format(idx)
        seg_in = seg_in_data[idx]

        with open(single_in) as f:
            single_in_data = f.read().strip().split()
        
        xmin, ymin, xmax, ymax = eval(single_in_data[0]), eval(single_in_data[1]), eval(single_in_data[2]), eval(single_in_data[3])
        
        seg_line = seg_in_data[idx].split(',')
        seg_idx, material, roof_area, water_area, adjacent = eval(seg_line[0]), eval(seg_line[1]), eval(seg_line[2]), eval(seg_line[3]), seg_line[4]
        material_list[idx] = material
        roof_area_list[idx] = roof_area

        if idx in cluster_dict.keys():
            cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax = cluster_dict[idx]
            fout.write('{},{} {} {} {},{},{},{} {} {} {},{},{},{},{}\n'.format(
                idx, xmin, ymin, xmax, ymax, cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax,
                cluster_ymax, material, roof_area, water_area, adjacent
            ))
        else:
            fout.write('{},{} {} {} {},{},{},{},{},{},{},{}\n'.format(
                idx, xmin, ymin, xmax, ymax, 'none', 'none', 'none',
                material, roof_area, water_area, adjacent
            ))
    fout.close()
    stat_fout.close()


print('total_area:', total_area)
print('total_roof_area:', total_roof_area)
print('average_size:', average_size)
print('roof_mean_area:', roof_mean_area)
print('thatch_percent:', thatch_percent)
print('zinc_percent:', zinc_percent)


