# python train.py --device 0 --batch-size 32 --img 608 608 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor.cfg --hyp cfg/hyp.scratch.yaml \
# --weights weights/yolov4.conv.137 \
# --name 0815-no-intersect-608 --single-cls --epochs 300

# python train.py --device 0 --batch-size 32 --img 512 512 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-512-new-new.cfg --hyp cfg/hyp.scratch.yaml \
# --weights /home/zhangkaifeng/project/detection/runs/exp57_0818-no-intersect-512-new-new/weights/last.pt \
# --name 0818-no-intersect-512-new-new --single-cls --epochs 300

python train.py --device 0,1 --batch-size 16 --img 608 608 \
--data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-level18.cfg --hyp cfg/hyp.scratch.yaml \
--weights weights/yolov4.conv.137 \
--name level18 --single-cls --epochs 400

# python train.py --device 0 --batch-size 28 --img-size 640 640 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-640-new-new.cfg --hyp cfg/hyp.scratch.yaml \
# --weights weights/yolov4.conv.137 \
# --name 0901-no-intersect-640-new-new --single-cls --epochs 400