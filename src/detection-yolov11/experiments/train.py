from pathlib import Path
import sys
sys.path.insert(0,str (Path(__file__).parent.parent))
from ultralytics import YOLO

train = True

if train:
    # Load a model
    model = YOLO("../weights/yolo11x.pt")

    # Train the model
    train_results = model.train(
        data="configs/2018.yaml",  # path to dataset YAML
        epochs=300,  # number of training epochs
        imgsz=608,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

else:
    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_2010/E46G055096_Level_17_0_512.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

'''
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    207/300      14.9G      2.024      1.409      1.311          5        608: 100%|██████████| 520/520 [01:43<00:00,  5.05it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 29/29 [00:03<00:00,  7.84it/s]
                   all        924        321
0.85 0.843 0.843 0.843 0.843 0.84 0.834 0.802 0.761 0.702 0.61 0.488 0.359 0.257 0.143 0.095 0.065 0.015 0.007 0.0
0.841 0.835 0.835 0.835 0.835 0.832 0.826 0.801 0.76 0.695 0.607 0.505 0.386 0.277 0.162 0.112 0.087 0.031 0.012 0.0
0.8638042376840608
0.864 0.846 0.844 0.844 0.843 0.836 0.822 0.77 0.708 0.621 0.518 0.383 0.247 0.141 0.065 0.032 0.014 0.003 0.001 0.0
EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 107, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

207 epochs completed in 6.199 hours.
Optimizer stripped from runs/detect/train24/weights/last.pt, 114.4MB
Optimizer stripped from runs/detect/train24/weights/best.pt, 114.4MB

Validating runs/detect/train24/weights/best.pt...
Ultralytics 8.3.47 🚀 Python-3.10.15 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 4090, 24177MiB)
YOLO11x summary (fused): 464 layers, 56,828,179 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 29/29 [00:03<00:00,  8.42it/s]
                   all        924        321
0.862 0.856 0.852 0.852 0.852 0.849 0.837 0.808 0.751 0.699 0.616 0.484 0.345 0.241 0.146 0.087 0.048 0.014 0.002 0.0
0.854 0.847 0.844 0.844 0.844 0.841 0.829 0.813 0.76 0.707 0.623 0.495 0.368 0.265 0.165 0.14 0.078 0.025 0.003 0.0
0.8730258981356278
0.873 0.861 0.858 0.857 0.857 0.848 0.827 0.793 0.717 0.635 0.533 0.365 0.233 0.128 0.06 0.027 0.011 0.002 0.0 0.0
Speed: 0.1ms preprocess, 3.3ms inference, 0.0ms loss, 0.1ms postprocess per image
Results saved to runs/detect/train24
'''