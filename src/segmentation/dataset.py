import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import json
import cv2


class InfDataset(Dataset):
    def __init__(self, im_root):

        self.orig_names = [f for f in os.listdir(im_root) if f.endswith('.jpg')]
        self.im_names = [' '] * len(self.orig_names)
        for name in self.orig_names:
            num = eval(name.split('.')[0])
            self.im_names[num] = name
        self.ims = []
        self.pad = 224
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')

            # im = np.asarray(Image.open(im_path).convert('RGB'))
            # im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
            # im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
            # im = transforms.ToTensor()(im)
            # scale = 2
            # H, W = 224, 224
            # h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
            # c = (int(im.shape[2]/2), int(im.shape[1]/2))
            # xmin, ymin = int(c[0] - w), int(c[1] - h)
            # im_crop = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
            # # print(im_crop)
            # # cv2.imwrite('tmp.jpg',im_crop.permute(1,2,0).numpy()*255)
            # # raise Exception
            self.ims.append(im_path)


    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        # print(idx)
        im_path = self.ims[idx]
        im = np.asarray(Image.open(im_path).convert('RGB'))
        im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
        im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
        im = transforms.ToTensor()(im)
        scale = 2
        H, W = 896, 896
        h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
        c = (int(im.shape[2]/2), int(im.shape[1]/2))
        xmin, ymin = int(c[0] - w), int(c[1] - h)
        im_rgb = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im_rgb)

        return im_rgb.float(), im



class InfDataset_water(Dataset):
    def __init__(self, im_root):

        self.orig_names = [f for f in os.listdir(im_root) if f.endswith('.jpg')]
        self.im_names = [' '] * len(self.orig_names)
        for name in self.orig_names:
            num = eval(name.split('.')[0])
            self.im_names[num] = name
        self.ims = []
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')

            # im = np.asarray(Image.open(im_path).convert('RGB'))
            # im = transforms.ToTensor()(im)
            # scale = 1
            # H, W = 448, 448
            # h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
            # c = (int(im.shape[2]/2), int(im.shape[1]/2))
            # xmin, ymin = int(c[0] - w), int(c[1] - h)
            # im_crop = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
            # # print(im_crop)
            # # cv2.imwrite('tmp.jpg',im_crop.permute(1,2,0).numpy()*255)
            # # raise Exception
            self.ims.append(im_path)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_path = self.ims[idx]
        im = np.asarray(Image.open(im_path).convert('RGB'))
        im = transforms.ToTensor()(im)
        scale = 1
        H, W = 448, 448
        h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
        c = (int(im.shape[2]/2), int(im.shape[1]/2))
        xmin, ymin = int(c[0] - w), int(c[1] - h)
        im_rgb = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im_rgb)

        return im_rgb.float(), im



class TrainDataset(Dataset):
    def __init__(self,
                 im_root='data_roof/JPEGImages', 
                 gt_root='data_roof/SegmentationClass',
                 poly_root='data_roof/2018gt_labeled'):

        self.im_names = sorted(os.listdir(im_root))
        data_size = len(self.im_names)
        self.im_names = self.im_names[:int(0.8 * data_size)]
        self.ims = []
        self.gts = []
        self.centers = []
        self.pad = 224
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')
            gt_path = path.join(gt_root, im_name + '.npy')
            poly_path = path.join(poly_root, im_name + '.json')

            # im = cv2.imread(im_path)
            im = np.asarray(Image.open(im_path).convert('RGB'))
            im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
            gt = np.load(gt_path)
            gt = np.pad(gt, self.pad)
            _, center_list = read_points(poly_path)  # vertices and centers of each polygon

            self.ims.append(im)
            self.gts.append(gt)
            self.centers.append(center_list)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im = self.ims[idx]
        gt = self.gts[idx]
        center_list = self.centers[idx]

        rand_scale = np.random.uniform(1.75, 2.25)  # relative upsample scale
        rand_shift_x, rand_shift_y = np.random.randint(-40, 40), np.random.randint(-40, 40)  # random shift of polygon centers
        rand_choice = np.random.randint(0, len(center_list))
        rand_angle = np.random.randint(-15, 15)

        H, W = 224, 224
        h, w = int(H / 2 / rand_scale), int(W / 2 / rand_scale)  # half height and width to crop on im
        c = center_list[rand_choice]
        c[0], c[1] = c[0] + rand_shift_x + self.pad, c[1] + rand_shift_y + self.pad

        xmin, ymin = int(c[0] - w), int(c[1] - h)

        im = transforms.ToTensor()(im)
        gt = transforms.ToTensor()(gt)

        im = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
        gt = transforms.functional.resized_crop(gt, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.NEAREST)

        im = transforms.functional.rotate(im, angle=rand_angle, interpolation=InterpolationMode.BILINEAR)
        gt = transforms.functional.rotate(gt, angle=rand_angle, interpolation=InterpolationMode.NEAREST)

        if np.random.uniform(0, 1) > 0.5:
            im, gt = im.flip(1), gt.flip(1)
        if np.random.uniform(0, 1) > 0.5:
            im, gt = im.flip(2), gt.flip(2)

        im = transforms.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=.05)(im)

        # im_save = im.clone().detach().permute(1,2,0).numpy() * 255
        # im_save = cv2.cvtColor(im_save, cv2.COLOR_BGR2RGB)
        # gt_save = gt[0].clone().detach().numpy()
        # mask = np.zeros_like(im_save)
        # mask[gt_save == 0] = np.array([0, 0, 0])
        # mask[gt_save == 1] = np.array([255, 0, 0])
        # mask[gt_save == 2] = np.array([0, 255, 0])
        # im_save = im_save * 0.7 + mask * 0.3
        # cv2.imwrite('vis_temp/{}_{}.jpg'.format(idx, rand_choice), im_save)

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im)
        gt = gt.long().squeeze(0)

        return im, gt



class TrainDataset_water(Dataset):
    def __init__(self,
                 im_root='data_water/JPEGImages', 
                 gt_root='data_water/SegmentationClass',
                 poly_root='data_water/2018gt_labeled'):

        self.im_names = sorted(os.listdir(im_root))
        data_size = len(self.im_names)
        self.im_names = self.im_names[:int(0.8 * data_size)]
        self.ims = []
        self.gts = []
        self.centers = []
        self.pad = 448
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')
            gt_path = path.join(gt_root, im_name + '.npy')
            poly_path = path.join(poly_root, im_name + '.json')

            # im = cv2.imread(im_path)
            im = np.asarray(Image.open(im_path).convert('RGB'))
            # im = Image.open(im_path).convert('RGB')
            # print(im)
            # raise Exception
            # print(im.shape)
            # raise Exception
            im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
            gt = np.load(gt_path)
            gt = np.pad(gt, self.pad)
            _, center_list = read_points(poly_path)  # vertices and centers of each polygon

            self.ims.append(im)
            self.gts.append(gt)
            self.centers.append(center_list)

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im = self.ims[idx]
        gt = self.gts[idx]
        center_list = self.centers[idx]

        rand_scale = np.random.uniform(0.75, 1.25)  # relative upsample scale
        rand_shift_x, rand_shift_y = np.random.randint(-50, 50), np.random.randint(-50, 50)  # random shift of polygon centers
        rand_choice = np.random.randint(0, len(center_list))
        # rand_angle = np.random.uniform(0, 360)

        H, W = 448, 448
        h, w = int(H / 2 / rand_scale), int(W / 2 / rand_scale)  # half height and width to crop on im

        c = center_list[rand_choice]
        c[0], c[1] = c[0] + rand_shift_x + self.pad, c[1] + rand_shift_y + self.pad
        # c = [np.random.randint(100+self.pad, im.shape[0]-self.pad-100) + rand_shift_x, np.random.randint(100+self.pad, im.shape[1]-self.pad-100) + rand_shift_y]

        xmin, ymin = int(c[0] - w), int(c[1] - h)

        im = transforms.ToTensor()(im)
        gt = transforms.ToTensor()(gt)

        im = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
        gt = transforms.functional.resized_crop(gt, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.NEAREST)
        
        if np.random.uniform(0, 1) > 0.5:
            im, gt = im.flip(1), gt.flip(1)
        if np.random.uniform(0, 1) > 0.5:
            im, gt = im.flip(2), gt.flip(2)

        # im = transforms.functional.rotate(im, angle=rand_angle, interpolation=InterpolationMode.BILINEAR)
        # gt = transforms.functional.rotate(gt, angle=rand_angle, interpolation=InterpolationMode.NEAREST)

        im = transforms.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=.05)(im)

        # im_save = im.clone().detach().permute(1,2,0).numpy() * 255
        # im_save = cv2.cvtColor(im_save, cv2.COLOR_BGR2RGB)
        # gt_save = gt[0].clone().detach().numpy()
        # mask = np.zeros_like(im_save)
        # mask[gt_save == 0] = np.array([0, 0, 0])
        # mask[gt_save == 1] = np.array([255, 0, 0])
        # mask[gt_save == 2] = np.array([0, 255, 0])
        # im_save = im_save * 0.7 + mask * 0.3
        # cv2.imwrite('vis_temp/{}_{}.jpg'.format(idx, rand_choice), im_save)

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im)
        gt = gt.long().squeeze(0)

        return im, gt



class TestDataset_water(Dataset):
    def __init__(self,
                 im_root='data_water/JPEGImages', 
                 gt_root='data_water/SegmentationClass',
                 poly_root='data_water/2018gt_labeled'):

        self.im_names = sorted(os.listdir(im_root))
        data_size = len(self.im_names)
        self.im_names = self.im_names[int(0.8 * data_size):]
        self.ims = []
        self.gts = []
        # self.centers = []
        self.pad = 0
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')
            gt_path = path.join(gt_root, im_name + '.npy')
            poly_path = path.join(poly_root, im_name + '.json')

            im = np.asarray(Image.open(im_path).convert('RGB'))
            im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
            gt = np.load(gt_path)
            gt = np.pad(gt, self.pad)
            _, center_list = read_points(poly_path)  # vertices and centers of each polygon


            im = transforms.ToTensor()(im)
            gt = transforms.ToTensor()(gt)


            im = transforms.functional.resize(im, size=(448, 448), interpolation=InterpolationMode.BILINEAR)
            gt = transforms.functional.resize(gt, size=(448, 448), interpolation=InterpolationMode.NEAREST)
            self.ims.append(im)
            self.gts.append(gt)

            # shift_x, shift_y = 0, 0
            # H, W = 224 * 2, 224 * 2
            # h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
            # for choice in range(len(center_list)):
            #     c = center_list[choice]
            #     c[0], c[1] = c[0] + shift_x + self.pad, c[1] + shift_y + self.pad
            #     xmin, ymin = int(c[0] - w), int(c[1] - h)
            #     im_crop = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
            #     gt_crop = transforms.functional.resized_crop(gt, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.NEAREST)
            #     self.ims.append(im_crop)
            #     self.gts.append(gt_crop)


    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_rgb = self.ims[idx]
        gt = self.gts[idx]

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im_rgb)
        gt = gt.long().squeeze(0)

        return im_rgb.float(), im, gt



class TestDataset(Dataset):
    def __init__(self,
                 im_root='data_roof/JPEGImages', 
                 gt_root='data_roof/SegmentationClass',
                 poly_root='data_roof/2018gt_labeled'):

        self.im_names = sorted(os.listdir(im_root))
        data_size = len(self.im_names)
        self.im_names = self.im_names[int(0.8 * data_size):]
        self.ims = []
        self.gts = []
        # self.centers = []
        self.pad = 224
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')
            gt_path = path.join(gt_root, im_name + '.npy')
            poly_path = path.join(poly_root, im_name + '.json')

            im = np.asarray(Image.open(im_path).convert('RGB'))
            im = np.pad(im, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))
            gt = np.load(gt_path)
            gt = np.pad(gt, self.pad)
            _, center_list = read_points(poly_path)  # vertices and centers of each polygon


            im = transforms.ToTensor()(im)
            gt = transforms.ToTensor()(gt)


            scale = 2
            shift_x, shift_y = 0, 0
            H, W = 224, 224

            h, w = int(H / 2 / scale), int(W / 2 / scale)  # half height and width to crop on im
            for choice in range(len(center_list)):
                c = center_list[choice]
                c[0], c[1] = c[0] + shift_x + self.pad, c[1] + shift_y + self.pad

                xmin, ymin = int(c[0] - w), int(c[1] - h)

                im_crop = transforms.functional.resized_crop(im, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.BILINEAR)
                gt_crop = transforms.functional.resized_crop(gt, ymin, xmin, 2*h, 2*w, size=(H, W), interpolation=InterpolationMode.NEAREST)

                self.ims.append(im_crop)
                self.gts.append(gt_crop)


    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im_rgb = self.ims[idx]
        gt = self.gts[idx]

        im = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(im_rgb)
        gt = gt.long().squeeze(0)

        return im_rgb.float(), im, gt



class TestDatasetOld(Dataset):
    def __init__(self,
                 im_root='data/JPEGImages', 
                 gt_root='data/SegmentationClass'):

        self.im_names = sorted(os.listdir(im_root))[81:]
        self.ims = []
        self.gts = []
        for idx in range(len(self.im_names)):
            im_name = self.im_names[idx][:-4]
            im_path = path.join(im_root, im_name + '.jpg')
            gt_path = path.join(gt_root, im_name + '.npy')
            im = Image.open(im_path).convert('RGB')
            gt = np.load(gt_path)
            self.ims.append(im)
            self.gts.append(gt)

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400), interpolation=InterpolationMode.BILINEAR),
        ])

        self.transform_im = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400), interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        info = []
        im = self.ims[idx]
        gt = self.gts[idx]

        im_rgb = self.transform_rgb(im).float()
        im = self.transform_im(im)
        gt = self.transform_gt(gt).long().squeeze(0)

        return im_rgb, im, gt



def read_points(filename):
    with open(filename) as f:
        data = json.load(f)
    
    points = []
    for polygon in data['shapes']:
        points.append(polygon['points'])

    centers = []
    for polygon in points:
        center = np.array(polygon).mean(0)
        centers.append([center[0], center[1]])

    return points, centers

    