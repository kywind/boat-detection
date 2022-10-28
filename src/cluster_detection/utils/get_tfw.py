import cv2, os

MAP_PATH_DICT = {
    2010: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20101231/',
    2011: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20111231/',
    2012: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20121231/',
    2013: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20131231/',
    2014: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20141231/',
    2015: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20151231/',
    2016: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20161231/',
    2017: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20171231/',
    2018: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20181231/',
    2019: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20191231/',
    2020: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20201231/',
    # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/20220531/'
    2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/2021_new/'
}
TFW_PATH_DICT = {
    2010: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20101231.txt',
    2011: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20111231.txt',
    2012: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20121231.txt',
    2013: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20131231.txt',
    2014: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20141231.txt',
    2015: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20151231.txt',
    2016: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20161231.txt',
    2017: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20171231.txt',
    2018: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20181231.txt',
    2019: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20191231.txt',
    2020: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20201231.txt',
    # 2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/20220531.txt'
    2021: '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/2021_new.txt'
}
TFW_DIR = '/home/zhangkaifeng/YONGONCHICKENFISH/data/satellite-yangon-level17-v2/tfw/'
os.makedirs(TFW_DIR, exist_ok=True)

for year in range(2010, 2022):
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