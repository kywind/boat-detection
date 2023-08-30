import cv2
import numpy as np
import os


poly = np.load('src/cluster_detection/utils/yangon_polygon.npy')[2]
poly = poly[poly > 0].reshape(-1, 2)
poly = (poly * 10000).astype(np.int32)

det_result  = 'v4_result/result_10-22_yangon/detection_result'

locs = dict()
for year in range(2010, 2021):
    loc_list = []
    det_result_path = os.path.join(det_result, str(year) + '.txt')
    with open(det_result_path, 'r') as f:
        lines = f.read().strip().split('\n')
    for line in lines:
        line = line.strip().split()
        x = (eval(line[0]) + eval(line[2])) / 2
        y = (eval(line[1]) + eval(line[3])) / 2
        loc_list.append((x, y))
    locs[year] = loc_list


file_1 = 'yangon.png'
file_2 = 'yangon_green.png'

keypoints_1 = [ 
    (82, 624), # left-upper, (x, y)
    (82, 6561), # left-lower
    (4640, 624), # right-upper
    (4640, 6561), # right-lower
    (82, 1136, 17.66667), # x, y, latitude/longitude
    (82, 6535, 16.16667),
    (376, 624, 95.66667),
    (4577, 624, 96.83333),
    (1971, 4042),  # original center yangon
    (1971, 4463),
    (2340, 4042),
    (2340, 4463),
    (3603, 5387),  # magnified center yangon
    (3603, 6483),
    (4564, 5387),
    (4564, 6483),
]

keypoints_2 = [ 
    (23, 270), # left-upper, (x, y)
    (23, 3265), # left-lower
    (2319, 270), # right-upper
    (2319, 3265), # right-lower
    (23, 789, 17.5), # x, y, latitude/longitude
    (23, 2506, 16.5),
    (708, 270, 96.0),
    (1565, 270, 96.5),
    (805, 1786),  # original center yangon
    (805, 2075),
    (1145, 1786),
    (1145, 2075),
    (1414, 820),  # magnified center yangon
    (1414, 1555),
    (2281, 820),
    (2281, 1555),
]

files = [file_1, file_2]
kps = [keypoints_1, keypoints_2]

for i in range(2):
    kp_list = kps[i]
    img = cv2.imread(files[i])
    corner_list = kp_list[:4]
    loc_list = kp_list[4:8]
    center_list = kp_list[8:12]
    mag_list = kp_list[12:16]

    x_min = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (corner_list[0][0] - loc_list[2][0]) + loc_list[2][2]
    x_max = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (corner_list[2][0] - loc_list[2][0]) + loc_list[2][2]
    y_max = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (corner_list[0][1] - loc_list[0][1]) + loc_list[0][2]
    y_min = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (corner_list[1][1] - loc_list[0][1]) + loc_list[0][2]

    corner_range = (x_min, x_max, y_min, y_max)

    x_min = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (center_list[0][0] - loc_list[2][0]) + loc_list[2][2]
    x_max = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (center_list[2][0] - loc_list[2][0]) + loc_list[2][2]
    y_max = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (center_list[0][1] - loc_list[0][1]) + loc_list[0][2]
    y_min = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (center_list[1][1] - loc_list[0][1]) + loc_list[0][2]

    center_range = (x_min, x_max, y_min, y_max)

    for year in locs.keys():
        loc_list = locs[year]
        for loc in loc_list:
            x_min, x_max, y_min, y_max = corner_range
            assert not (loc[0] < x_min or loc[0] > x_max or loc[1] < y_min or loc[1] > y_max), print(loc, corner_range)
            if cv2.pointPolygonTest(poly, (int(loc[0] * 10000), int(loc[1] * 10000)), False) >= 0:  # inside
                x_pix = (loc[0] - x_min) / (x_max - x_min) * (corner_list[2][0] - corner_list[0][0]) + corner_list[0][0]
                y_pix = (loc[1] - y_max) / (y_min - y_max) * (corner_list[1][1] - corner_list[0][1]) + corner_list[0][1]
                pix_loc = (int(x_pix), int(y_pix))

                alpha = 0.5
                radius = 20 if i == 0 else 10
                bgr = (0, 0, 255) if i == 0 else (255, 0, 0)
                overlay = img.copy()
                overlay = cv2.circle(overlay, pix_loc, radius, bgr, -1)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                x_min_c, x_max_c, y_min_c, y_max_c = center_range
                if loc[0] >= x_min_c and loc[0] <= x_max_c and loc[1] >= y_min_c and loc[1] <= y_max_c:
                    x_pix = (loc[0] - x_min_c) / (x_max_c - x_min_c) * (mag_list[2][0] - mag_list[0][0]) + mag_list[0][0]
                    y_pix = (loc[1] - y_max_c) / (y_min_c - y_max_c) * (mag_list[1][1] - mag_list[0][1]) + mag_list[0][1]
                    overlay = img.copy()
                    pix_loc = (int(x_pix), int(y_pix))
                    overlay = cv2.circle(overlay, pix_loc, radius, bgr, -1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.imwrite(f'{i+1}_{year}.png', img)
        print(f'finished file {i+1}, year {year}')

