# python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.18 --img-size 608 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
# --task val --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
# --weights /home/zhangkaifeng/project/detection/runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt

# python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.17 --img 608 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-608-new-new.cfg --name cfg/satellite.names \
# --task val_nowater --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
# --weights runs/exp60_0822-no-intersect-608-new-new/weights/best_0822-no-intersect-608-new-new.pt

# python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.16 --img 512 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-512.cfg --name cfg/satellite.names \
# --task val --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
# --weights /home/zhangkaifeng/project/detection/runs/exp58_0818-no-intersect-512-new-new-300resume/weights/last_0818-no-intersect-512-new-new.pt

# python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.16 --img 512 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-512.cfg --name cfg/satellite.names \
# --task val --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
# --weights runs/exp62_0818-no-intersect-512-new-new-400resume/weights/last_0818-no-intersect-512-new-new-400resume.pt

# python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.18 --img-size 640 \
# --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-640.cfg --name cfg/satellite.names \
# --task val --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
# --weights runs/exp67_0901-no-intersect-640-new-new/weights/last_0901-no-intersect-640-new-new.pt