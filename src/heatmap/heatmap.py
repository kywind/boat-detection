import cv2
import numpy as np
import os
import fitz
from typing import List, Tuple

poly = np.load('../cluster_detection/utils/yangon_polygon.npy')[2]
poly = poly[poly > 0].reshape(-1, 2)
poly = (poly * 10000).astype(np.int32)

def draw_circles_on_pdf(input_pdf_path: str,
                        output_pdf_path: str,
                        circle_positions: List[Tuple[float, float]],
                        radius: float = 13,
                        color: Tuple[float, float, float] = (1, 0, 0), # (193/255, 18/255, 31/255),  # RGB in 0~1
                        opacity: float = 0.5):
    # Open the PDF
    doc = fitz.open(input_pdf_path)
    page = doc[0]  # assuming single-page

    # Get page size
    width, height = page.rect.width, page.rect.height

    # Prepare drawing
    shape = page.new_shape()

    for (x_norm, y_norm) in circle_positions:
        # Convert normalized to page coordinates
        x = x_norm * width
        y = (1 - y_norm) * height  # flip y-axis
        rect = fitz.Rect(x - radius, y - radius, x + radius, y + radius)

        # Draw filled ellipse
        shape.draw_oval(rect)
        shape.finish(fill=color, color=None, fill_opacity=opacity)

    # Commit shape to page
    shape.commit()

    # Save output
    doc.save(output_pdf_path)
    doc.close()

# det_result  = 'v4_result/result_10-22_yangon/detection_result'
# det_result = '../cluster_detection/data/orig2/'
det_result = '/home/zhangkaifeng/projects/YONGONCHICKENFISH/unused/YONGONCHICKENFISH/v4_result/result_10-22_yangon/detection_result'

locs = dict()
for year in range(2010, 2022):
# for year in range(20230827, 20230828):
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


file_1 = 'template.pdf'

os.makedirs('result/', exist_ok=True)

keypoints_1 = [ 
    (82, 624), # left-upper, (x, y)
    (82, 6561), # left-lower
    (4640, 624), # right-upper
    (4640, 6561), # right-lower
    (82, 1136, 17.66667), # x, y, latitude/longitude
    (82, 6535, 16.16667),
    (376, 624, 95.66667),
    (4577, 624, 96.83333),
    # (1971, 4042),  # original center yangon
    # (1971, 4463),
    # (2340, 4042),
    # (2340, 4463),
    # (3603, 5387),  # magnified center yangon
    # (3603, 6483),
    # (4564, 5387),
    # (4564, 6483),
]

files = [file_1]
kps = [keypoints_1]

for i in range(1):
    kp_list = kps[i]
    img = cv2.imread(files[i])
    corner_list = kp_list[:4]
    loc_list = kp_list[4:8]
    # center_list = kp_list[8:12]
    # mag_list = kp_list[12:16]

    x_min = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (corner_list[0][0] - loc_list[2][0]) + loc_list[2][2]
    x_max = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (corner_list[2][0] - loc_list[2][0]) + loc_list[2][2]
    y_max = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (corner_list[0][1] - loc_list[0][1]) + loc_list[0][2]
    y_min = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (corner_list[1][1] - loc_list[0][1]) + loc_list[0][2]

    corner_range = (x_min, x_max, y_min, y_max)

    # x_min = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (center_list[0][0] - loc_list[2][0]) + loc_list[2][2]
    # x_max = (loc_list[3][2] - loc_list[2][2]) / (loc_list[3][0] - loc_list[2][0]) * (center_list[2][0] - loc_list[2][0]) + loc_list[2][2]
    # y_max = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (center_list[0][1] - loc_list[0][1]) + loc_list[0][2]
    # y_min = (loc_list[1][2] - loc_list[0][2]) / (loc_list[1][1] - loc_list[0][1]) * (center_list[1][1] - loc_list[0][1]) + loc_list[0][2]

    # center_range = (x_min, x_max, y_min, y_max)

    prev_years = []
    for year in locs.keys():

        circle_positions = []

        prev_years = [year]
        # prev_years.append(year)
        for prev_year in prev_years:
            loc_list = locs[prev_year]
            for loc in loc_list:
                x_min, x_max, y_min, y_max = corner_range
                # assert not (loc[0] < x_min or loc[0] > x_max or loc[1] < y_min or loc[1] > y_max), print(loc, corner_range)
                # if True:  # don't test yangon polygon
                if cv2.pointPolygonTest(poly, (int(loc[0] * 10000), int(loc[1] * 10000)), False) >= 0:  # inside
                    x_pix = (loc[0] - x_min) / (x_max - x_min)#  * (corner_list[2][0] - corner_list[0][0]) + corner_list[0][0]
                    y_pix = (loc[1] - y_min) / (y_max - y_min)#  * (corner_list[1][1] - corner_list[0][1]) + corner_list[0][1]
                    pix_loc = (x_pix, y_pix)

                    circle_positions.append(pix_loc)

                    # alpha = 0.5
                    # radius = 20 if i == 0 else 10
                    # bgr = (0, 0, 255) if i == 0 else (255, 0, 0)
                    # overlay = img.copy()
                    # overlay = cv2.circle(overlay, pix_loc, radius, bgr, -1)
                    # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    # x_min_c, x_max_c, y_min_c, y_max_c = center_range
                    # if loc[0] >= x_min_c and loc[0] <= x_max_c and loc[1] >= y_min_c and loc[1] <= y_max_c:
                    #     x_pix = (loc[0] - x_min_c) / (x_max_c - x_min_c) * (mag_list[2][0] - mag_list[0][0]) + mag_list[0][0]
                    #     y_pix = (loc[1] - y_max_c) / (y_min_c - y_max_c) * (mag_list[1][1] - mag_list[0][1]) + mag_list[0][1]
                    #     overlay = img.copy()
                    #     pix_loc = (int(x_pix), int(y_pix))
                    #     overlay = cv2.circle(overlay, pix_loc, radius, bgr, -1)
                    #     img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                
                loc_list = np.array(loc_list)
                np.savetxt(f'result/{prev_year}_loc.csv', loc_list, fmt='%f', delimiter=',')
            
        # Draw circles on the PDF
        # print(circle_positions)
        circle_positions = np.array(circle_positions)
        np.savetxt(f'result/{year}.csv', circle_positions, fmt='%f', delimiter=',')
        draw_circles_on_pdf(files[i], f"result/{year}.pdf", circle_positions)

        # cv2.imwrite(f'result/{i+1}_{year}.png', img)
        # print(f'finished file {i+1}, year {year}')


