# Yangon Chickenfish

This is a satellite image object detection and clustering model based on YOLOv4 (https://github.com/WongKinYiu/PyTorch_YOLOv4).


## 1 Model

### 1.1 Pipeline overview

![method](files/method.png)

Shown in this figure, with the satellite images as input, we first crop the images into fixed sizes and detect all chicken farms using an object detection model. Then, we do the following post-processing steps: (1) counting; (2) re-cropping the satellite images aroung chicken houses and segment the chicken house buildings and fish ponds, calculating their total area sizes; (2) detect clusters using a cluster detection algorithm.

### 1.2 Algorithm 

- detection: Based on YOLOv4. Trained with annotated data to output axis-aligned bounding boxes on images countaining chicken farms. When counting, bounding boxes are transformed into spatial locations (latitude and longtitude), and farms that are detected more than once in overlapping satellite images are removed. Corresponding code dir: ```src/detection```

- segmentation：Based on U-net. Trained with annotated data to output per-pixel likelihood of belonging to fish ponds or rooftops of chicken houses. For each rooftop，the model also predicts one of two possible rooftop materials, thatch and zinc. Corresponding code dir: ```src/segmentation```

- cluster detection：Farms with distance less than a threshold D are considered to be within a cluster. We use a search algorithm to find all non-overlapping clusters. Corresponding code dir: ```src/cluster_detection```

- statistics: After the previous steps, we calculate the total statistics (mean rooftop area, percentage of zinc and thatch roofs, mean cluster size, etc). Corresponding code dir: ```src/cluster_detection```


## 2 Installation

### 2.1 python environment

```
pip install -r requirements.txt
```
- For detection model, it is required to install mish-cuda：https://github.com/thomasbrandon/mish-cuda


### 2.2 pretrained models

- detection (save to ```src/detection/weights```): https://drive.google.com/file/d/1bVMyKAcRUJ-YsgJY1jI1S6I95gRfssNO/view?usp=sharing

- roof segmentation (save to ```src/segmentation```): https://drive.google.com/file/d/1EoUoeDo0I16UViE0dFSAiVluCtiwAmER/view?usp=sharing

- water segmentation (save to ```src/segmentation```): https://drive.google.com/file/d/1hSrWMU-OgtVY_XfV-xgARKu2xj5JMSCC/view?usp=sharing


## 3 Running the code

**Note:** We cannot provide the satellite images in this codebase due to its enormous size. We do provide the detection results in ```src/cluster_detection/data``` and segmentation results in ```src/segmentation/result```. To test the code with only these provided results, only run the scripts that are marked as "ready".

### 3.1 Data preparation

- src/detection/images: satellite images after cropping to a fixed size
- src/detection/annotations: corresponding json annotation file

Preprocess training data:
```
cd src/detection/;
python preprocess/preprocess_train.py;
```

### 3.2 detection
Training:
```
cd src/detection/;
bash scripts/train.sh;
```

Testing: 
```
cd src/detection/;
bash scripts/test.sh;
```

Inference:
```
cd src/detection/;
python preprocess/preprocess_detect.py;
bash scripts/detect.sh;
```

### 3.3 segmentation
Training:
```
cd src/segmentation/;
python main.py;
```

Inference:
```
cd src/segmentation/;
python make_buffer.py;
python inference.py;
python post_process.py;
```

### 3.4 cluster detection
detection results must be copied to the ```src/cluster_detection/raw``` directory.

Preprocessing:
```
cd src/cluster_detection/;
python nms.py;  # remove farms detected more than once
```

Detect **(ready)**:
```
python detect.py;  # detect clusters
```

### 3.5 stats

Calculate statistics **(ready)**:
```
cd src/stats;
python get_stats.py
```

