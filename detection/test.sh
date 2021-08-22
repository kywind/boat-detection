python test.py --device 0 --batch-size 8 --conf-thres 0.02 --pr-thres 0.18 --img 608 \
--data cfg/satellite.data.yaml --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
--task testval_nowater --iou 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
--weights weights/V2_best_608.pt
