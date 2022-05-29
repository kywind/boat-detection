# Yangon Chickenfish

本项目是一个基于YOLOv4的卫星图像目标体识别与集群检测模型。

- 参考的PyTorch+YOLOv4实现：https://github.com/WongKinYiu/PyTorch_YOLOv4


## 1 模型

### 1.1 工作流程

![method](files/method.png)

如图所示。输入某地理范围内所有卫星图片后，首先全部裁剪成固定大小，通过检测模型，检出所有鸡鱼农场。然后，进行以下的统计：(1)计数，(2)重新以检出农场为中心裁剪卫星图片，通过分割模型，分割出农场与鱼塘的轮廓，计算总面积，(3)通过集群检测算法，统计农场集群。

### 1.2 算法

- detection: 基于YOLOv4。预测所有农场，结果格式为所有农场的经纬度位置，以及bounding box的长宽。可从头训练，支持水体检测预处理，支持基于经纬度的NMS去重，可自动计算anchor。

- segmentation：基于U-net。预测图片中属于水体的像素，以及图片中属于农场屋顶的像素，其中对于后者，同时预测农场的材料，材料分为草质与金属质两类。

- cluster detection：相互之间距离不超过距离D的农场被认为属于同一集群。

### 1.3 模型表现

![performance](files/performance.png)

- Table 1: detection模型表现，由精确率、召回率、平均精确率表示。表格列举了三个不同IoU阈值下的表现。IoU阈值为预测的bounding box与真实的bounding box之间的最大交-并比。

- Table 2: segmentation模型表现。由预测结果和真实结果间的交-并比表示。

- Table 3: 2010～2016年，模型对仰光周围100km半径内农场总个数的计数结果。

更详细的模型相关信息，请参考files/report.pdf。


## 2 安装

```
pip install -r requirements.txt
```
- 安装mish-cuda：https://github.com/thomasbrandon/mish-cuda


## 3 运行

### 3.1 数据准备

- 裁剪为固定大小后的卫星图片：detection
确保images，annotations，labels，imglist下存在训练数据而且imglist目录下的txt文件中记录的文件名与images和labels中的文件名一致；

重新随机分割训练数据：
```
python preprocess/preprocess_train.py
```

### 2.2 训练

根据需要修改train.sh以及cfg目录下的参数，然后运行train.sh。训练后会在run目录下创建一个子目录，含有checkpoint和训练过程数据（可能需要使用tensorboard查看）。

### 3.3 测试

根据需要修改test.sh以及cfg目录下的参数，然后运行test.sh。测试指标包括在不同的iou_thresh（即区分正确预测和错误预测时，采用的预测框与标注框的IOU的阈值）下的P(precision)，R(recall)，mAP等。

### 4.4 预测

根据需要修改detect.sh以及cfg目录下的参数，然后运行detect.sh。预测将在inference目录下输出预测结果。



## 4 集群检测

集群检测模型位于cluster目录下。此后所有操作与指令均在cluster目录下进行。其中的很多代码可能只针对我们用到的卫星数据集，如果用到别的数据集，大概率需要修改这些代码。

### 4.1 数据准备

数据需要放至data目录下，格式为txt，内容格式如下：

```
x0 y0
x1 x1
x2 y2
```

即每行代表一个目标点的坐标。可以先在raw目录下进行预处理。可以参考raw/process_new.py。

### 3.2 集群检测

根据需要修改并运行detect.py（V2用的是new.py）。会在result目录下生成txt格式的集群检测结果。格式如下：

```
cluster 
3 1 1 2 2
1 1
2 2
1.5 1.5
cluster
4 3 4 5 6
3 6
5 4
4.5 5
4 5
```

解释：cluster表示此后内容为一个集群；下一行五个数分别为size xmin ymin xmax ymax，标记集群的大小和边界；然后其下size行是构成集群的所有目标的坐标。

集群检测的规则：设定距离阈值x; 对每个物体DFS搜索周围x距离内的物体，如存在则归为同一集群。最后集群大小大于k的会被保留。输出的集群应当是按照经纬度的字典序排列的，这样每一年份的所有集群就有了一个独特的编号。

### 3.3 可视化

可视化指生成前一过程检出的集群区域的对应卫星图像。需要首先有整片区域卫星图像数据，以及detect.py的运行结果。

visualize.py中含有的可视化功能包括：

- 输出任意经纬度范围，任意清晰度的地图（getmap）
- 输出检测目标的截图，支持随机采样/全部输出（mapcut_single)
- 将检测目标在模型的输入图片上进行标注，支持随机采样/全部输出（annotate_single）
- 输出集群的截图（mapcut_cluster)
- 对集群按地理位置进行编号并对每个集群作其在多年间的变化图（comparison）
- 生成集群和目标的热度图（heatmap_single单目标，heatmap_cluster集群，heatmap_region任意区域）

其中前五个在result目录下生成结果。



 

