## 0
# git clone https://github.com/PKUCER/YONGONCHICKENFISH
# cd YONGONCHICKENFISH
# git config --add user.name kywind
# git config --add user.email m74002@outlook.com

## 1 (optional)
# conda create -n satellite python=3.9

## 2 (optional)
# conda activate satellite

## 3 (cuda)
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install pycocotools tensorboard

## 4
# pip install opencv-python numpy matplotlib tqdm pillow gdown pyyaml scipy

## 5 (cuda)
# cd src/detection/third-party/mish-cuda
# python setup.py install
# cd -

## 6
# gdown 1bVMyKAcRUJ-YsgJY1jI1S6I95gRfssNO
# mkdir src/detection/weights
# mv best_0829-no-intersect-608-new-new.pt src/detection/weights
# gdown 1hSrWMU-OgtVY_XfV-xgARKu2xj5JMSCC
# gdown 1EoUoeDo0I16UViE0dFSAiVluCtiwAmER
# mv ckpt_830827.pt src/segmentation
# mv ckpt_883.pt src/segmentation

## 7
# mkdir data/
# gsutil -m cp -r gs://satellite-yangon-level17/ data/


## RUNNING

## 0: change all the path dicts and all year/taskname iterators in python files manually

## 1
cd src/detection/preprocess
python preprocess_detect.py
cd ..
bash scripts/detect.sh
# rm -r ../cluster_detection/raw
cp -r inference ../cluster_detection/raw

## 2
cd ../cluster_detection/utils
python get_tfw.py
cd ..
python visualize.py  # generate whole map
python nms.py
cd utils
python polygon.py  # generate polygon and filter data
cd ..
python visualize.py  # get heatmap (requires modified visualize.py)

## 3 cluster
python detect.py


