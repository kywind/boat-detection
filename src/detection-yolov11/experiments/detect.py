from pathlib import Path
import sys
sys.path.insert(0,str (Path(__file__).parent.parent))
from ultralytics import YOLO
import glob

# train = True

# if train:
#     # Load a model
#     model = YOLO("../weights/yolo11x.pt")

#     # Train the model
#     train_results = model.train(
#         data="configs/2018.yaml",  # path to dataset YAML
#         epochs=300,  # number of training epochs
#         imgsz=608,  # training image size
#         device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     )

model = YOLO('/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/experiments/runs/detect/train25/weights/best.pt')

# else:
# Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image


for year in range(2010, 2024):
  count = 0
  if year == 2022: continue

  img_name_list = sorted(glob.glob(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/detect_buffer_jpg_{year}/*.jpg'))

  f = open(f'/home/zhangkaifeng/projects/YONGONCHICKENFISH/src/detection-yolov11/log/raw/{year}.txt', 'w')

  for img_name in img_name_list:
    results = model(img_name, conf=0.01)
    for result in results:
      # if result.boxes.xyxy.size(0) > 0:
      #   import ipdb; ipdb.set_trace()
      count += result.boxes.xywh.size(0)
      for i in range(result.boxes.xywh.size(0)):
        img_name_stem = str(Path(img_name).stem)
        f.write(f'{img_name_stem} {result.boxes.cls[i]} {result.boxes.xywh[i][0]} {result.boxes.xywh[i][1]} {result.boxes.xywh[i][2]} {result.boxes.xywh[i][3]} {result.boxes.conf[i]} \n')      
    # count += n_detected
  f.close()

  print(year, count)


# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
