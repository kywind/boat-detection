import xml.etree.ElementTree as ET
import os
import shutil
import random


imgtype = '.jpg'
sets = ['train','val','test','trainval']
trainval_percent = 1 
train_percent = 0.9
width, height = 640, 640

g_root_path = './'
xmlpath = 'annotations/'
imgpath = 'images/'
txtpath = 'imglist/'
labelpath = 'labels/'


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(image_id):
    in_file = open(xmlpath + '{}.xml'.format(image_id), 'r')
    out_file = open(labelpath + '{}.txt'.format(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    w, h = width, height
    n = 0
    for obj in root.iter('object'):
        n += 1
        cls_id = 0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bb]) + '\n')
    return n
        

os.chdir(g_root_path)
total_xml = os.listdir(xmlpath)
random.shuffle(total_xml)
num = len(total_xml)
xml_list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(xml_list, tv)
train = random.sample(trainval, tr)

os.makedirs(txtpath, exist_ok=True)

print('train and val size', tv)
print('train size', tr)
ftrainval = open(txtpath + 'trainval.txt', 'w')
ftest = open(txtpath + 'test.txt', 'w')
ftrain = open(txtpath + 'train.txt', 'w')
fval = open(txtpath + 'val.txt', 'w')
for i in xml_list:
    if not total_xml[i].endswith('.xml'):
        continue
    name = imgpath + total_xml[i][:-4] + imgtype + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train: ftrain.write(name)
        else: fval.write(name)
    else: ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


if os.path.exists(labelpath):
    shutil.rmtree(labelpath)
os.mkdir(labelpath)

for image_set in sets:
    print('generating label for {}.txt'.format(image_set))
    fimg = open(txtpath + './{}.txt'.format(image_set))
    image_paths = fimg.read().strip().split()
    fimg.close()
    cnt = 0
    for image_path in image_paths:
        image_id = image_path.replace(imgpath, '')
        image_id = image_id.replace(imgtype, '')
        cnt += convert_annotation(image_id)
    print(cnt)