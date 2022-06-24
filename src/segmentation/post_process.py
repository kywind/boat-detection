import os
import shutil
import numpy as np
import cv2

# for each instance
for year in range(2010, 2022):
    roof_in_path = 'inference/inference_roof_{}/'.format(year)
    water_in_path = 'inference/inference_water_{}/'.format(year)
    water_meta_path = 'inference/{}_water_meta/'.format(year)
    files = [f for f in os.listdir(roof_in_path) if f.endswith('npy')]
    fout = open('result/seg_res_{}.txt'.format(year), 'w')
    for file_id in range(len(files)):
        res = 0.0000107288
        res_meter = res * 111000

        # read roof inference data
        roof_mask = np.load(roof_in_path + '{}.npy'.format(file_id))
        # print(roof_mask.max())
        x_center, y_center = int(roof_mask.shape[1]/2), int(roof_mask.shape[0]/2)
        # print(x_center, y_center)
        material = roof_mask[y_center, x_center]
        if material == 0:
            roof_area = 0
        else:
            roof_bin = 255 * (roof_mask != 0)
        
            # print(material)
            # print(roof_bin)
            contours, _ = cv2.findContours(roof_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            roof_area = 0
            for cont in contours:
                dist = cv2.pointPolygonTest(cont, (x_center, y_center), False)
                if dist >= 0: # inside or on border
                    roof_area = cv2.contourArea(cont)
                    break
            # print(roof_area)

            roof_area = roof_area / 16 * res_meter * res_meter
            # print(roof_area, material)
        # read water inference data
        water_mask = np.load(water_in_path + '{}.npy'.format(file_id))
        
        
        x_center, y_center = int(water_mask.shape[1]/2), int(water_mask.shape[0]/2)
        # print(water_mask.shape, water_mask.max())
        kernel = np.ones((9,9), np.uint8)
        water_mask = cv2.erode(water_mask.astype(np.uint8), kernel) 
        water_mask = cv2.dilate(water_mask, kernel) 
        water_mask = cv2.erode(water_mask, kernel) 
        water_mask = cv2.dilate(water_mask, kernel) 
        water_mask = cv2.erode(water_mask, kernel) 
        water_mask = cv2.dilate(water_mask, kernel) 
        water_mask = cv2.erode(water_mask, kernel) 
        water_mask = cv2.dilate(water_mask, kernel) 
        # cv2.imwrite('temp.jpg', water_mask * 255)
        water_bin = 255 * (water_mask != 0)
        # print(water_bin.shape)  # 448, 448
        contours, _ = cv2.findContours(water_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        water_area = 0
        center_cont = None
        for cont in contours:
            dist = cv2.pointPolygonTest(cont, (x_center, y_center), False)
            if dist >= 0: # inside or on border
                water_area = cv2.contourArea(cont)
                center_cont = cont
                break
        # print(water_area)
        water_area = water_area * res_meter * res_meter

        # align with metadata

        f_water_meta = water_meta_path + '{}.txt'.format(file_id)
        with open(f_water_meta) as f:
            water_meta = f.read().strip().split('\n')
        center_pos = water_meta[0].split()
        xmin, ymin, xmax, ymax = eval(center_pos[0]), eval(center_pos[1]), eval(center_pos[2]), eval(center_pos[3])
        x_mid, y_mid = (xmin + xmax) / 2, (ymin + ymax) / 2
        adjacents = []

        if center_cont is None:
            adjacents_str = 'none'

        else:
            adjs = []
            for i in range(1, len(water_meta)):
                adj = water_meta[i].split()
                adj_id, adj_x, adj_y = eval(adj[0]), eval(adj[1]), eval(adj[2])

                if adj_id == file_id:
                    x_shift, y_shift = x_center - adj_x, y_center - adj_y
                
                else: 
                    adjs.append((adj_id, adj_x, adj_y))
                
            for idx, x, y in adjs:
                dist = cv2.pointPolygonTest(cont, (x + x_shift, y + y_shift), False)
                if dist >= 0: # inside or on border
                    adjacents.append(idx)
            
            if len(adjacents) == 0:
                adjacents_str = 'none'

            else:
                adjacents_str = ''
                for adj_id in adjacents:
                    adjacents_str += '{} '.format(adj_id)
                adjacents_str = adjacents_str[:-1]

        # calculate water area for each roof & other roofs in the same water area

        # save: (roof, material, roof size, whether in water, water area, other roofs)
        fout.write('{},{},{},{},{}\n'.format(file_id, material, roof_area, water_area, adjacents_str))
    fout.close()