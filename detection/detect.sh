# python detect.py --save-txt --conf 0.1 --img 608 --output inference/testimgs --source testimgs/ --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names --weights runs/exp39_test-allanno-350resume/weights/best_test-allanno-350resume.pt
# python detect.py --save-txt --conf 0.02 --img 608 \
# --output inference/2010 \
# --source detect_buffer_jpg_2010/ \
# --cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
# --weights weights/V2_best_608.pt

python detect_new.py --conf 0.02 --img 608 --output inference/no_water \
--source imglist/val_nowater.txt --save-img \
--cfg cfg/satellite-anchor.cfg --name cfg/satellite.names \
--weights weights/V2_best_608.pt