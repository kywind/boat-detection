import os

for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018]:
    single_in_dir = 'cluster_rect/result/{}_single/'.format(year)
    # cluster_in_dir = 'cluster_rect/result/{}_cluster/'.format(year)
    cluster_in_file = 'cluster_rect/result/{}_集群信息.txt'.format(year)
    seg_in_file = 'segmentation/seg_res_{}.txt'.format(year)
    out_file = 'result_v3_{}.csv'.format(year)

    fout = open(out_file, 'w')

    single_in_files = [f for f in os.listdir(single_in_dir) if f.endswith('txt')]

    with open(seg_in_file) as f:
        seg_in_data = f.read().strip().split('\n')
    
    cluster_dict = {}
    with open(cluster_in_file) as f:
        cluster_in_data = f.read().strip().split('\n')
    for line in cluster_in_data[:-1]:
        line = line.replace('：', ' ')
        line = line.replace('包含目标编号', ' ')
        line = line.replace('集群', ' ')
        line = line.replace('大小', ' ')
        line = line.replace('经纬度范围', ' ')
        line = line.replace('，', ' ')
        line = line.replace('号', ' ')
        line = line.split()
        # print(line)
        cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax = \
            eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3]), eval(line[4]), eval(line[5])
        
        for i in range(cluster_size):
            obj_id = eval(line[6+i])
            if obj_id in cluster_dict.keys(): raise ValueError
            cluster_dict[obj_id] = (cluster_id, cluster_size, cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax)

    # print(cluster_dict)

    for idx in range(len(single_in_files)):
        single_in = single_in_dir + '{}.txt'.format(idx)
        seg_in = seg_in_data[idx]

        with open(single_in) as f:
            single_in_data = f.read().strip().split()
        
        xmin, ymin, xmax, ymax = eval(single_in_data[0]), eval(single_in_data[1]), eval(single_in_data[2]), eval(single_in_data[3])
        
        seg_line = seg_in_data[idx].split(',')
        seg_idx, material, roof_area, water_area, adjacent = eval(seg_line[0]), eval(seg_line[1]), eval(seg_line[2]), eval(seg_line[3]), seg_line[4]

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


