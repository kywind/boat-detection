# python train.py --device 1 --batch-size 16 --img 608 608 --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor.cfg --hyp cfg/hyp.scratch.yaml --resume --weights runs/exp20_0706origanchor250resume/weights/249.pt --name 0706origanchor300resume --single-cls --epochs 300
# python train.py --device 0 --batch-size 32 --img 608 608 --data cfg/satellite.data.yaml --cfg cfg/satellite-anchor.cfg --hyp cfg/hyp.scratch.yaml --resume --weights runs/exp45_0815-no-intersect-608/weights/099.pt --name 0815-no-intersect-608-300resume --single-cls --epochs 300
python train.py --device 0 --batch-size 32 --img 512 512 --data cfg/satellite.data.yaml \
--cfg cfg/satellite-anchor-512-new-new.cfg --hyp cfg/hyp.scratch.yaml --resume \
--weights runs/exp58_0818-no-intersect-512-new-new/weights/299.pt \
--name 0818-no-intersect-512-new-new-400resume \
--single-cls --epochs 400