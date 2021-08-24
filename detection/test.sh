python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.17 --img 512 \
--data cfg/satellite.data.yaml --cfg cfg/satellite-anchor-512-new-new.cfg --name cfg/satellite.names \
--task val --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
--weights runs/exp58_0818-no-intersect-512-new-new/weights/best_0818-no-intersect-512-new-new.pt
