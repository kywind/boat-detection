import cv2, os

MAP_PATH_DICT = {
    # 2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20101231/',
    # 2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20111231/',
    # 2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20121231/',
    # 2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20131231/',
    # 2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20141231/',
    # 2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20151231/',
    # 2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20161231/',
    # 2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20171231/',
    # 2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20181231/',
    # 2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20191231/',
    # 2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20201231/',
    # 2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/2021_new/',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/20230827/'
}
TFW_PATH_DICT = {
    # 2010: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20101231.txt',
    # 2011: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20111231.txt',
    # 2012: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20121231.txt',
    # 2013: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20131231.txt',
    # 2014: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20141231.txt',
    # 2015: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20151231.txt',
    # 2016: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20161231.txt',
    # 2017: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20171231.txt',
    # 2018: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20181231.txt',
    # 2019: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20191231.txt',
    # 2020: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20201231.txt',
    # 2021: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/2021_new.txt',
    2023: '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw/20230827.txt',
}
TFW_DIR = '/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/tfw'
os.makedirs(TFW_DIR, exist_ok=True)

# for year in range(2010, 2022):
for year in MAP_PATH_DICT.keys():
    map_path = MAP_PATH_DICT[year]
    tfw_path = TFW_PATH_DICT[year]
    files = sorted([f for f in os.listdir(map_path) if f.endswith('.tfw')])
    fout = open(tfw_path, 'w')
    for f in files:
        fn = f.split('_')[0]
        print(fn)
        with open(map_path + f) as fin:
            data = fin.read().strip().split()
        fout.write('{} {} {} {} {}\n'.format(fn, data[0], data[3], data[4], data[5]))
    fout.close()