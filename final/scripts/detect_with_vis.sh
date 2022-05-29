weight_path="runs/exp64_0829-no-intersect-608-new-new/weights/best_0829-no-intersect-608-new-new.pt"

CUDA_VISIBLE_DEVICES=1 \
python detect_with_vis.py --save-txt --conf 0.4 --img 608 \
--output inference/2010 \
--source detect_buffer_jpg_2010/ \
--cfg cfg/satellite-anchor-608.cfg --name cfg/satellite.names \
--weights $weight_path
