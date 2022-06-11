# git clone https://github.com/PKUCER/YONGONCHICKENFISH
# cd YONGONCHICKENFISH
# conda create -n satelltie python=3.9
# conda activate satelltie
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install opencv-python numpy matplotlib pycocotools tqdm pillow tensorboard gdown
# gdown https://drive.google.com/u/0/uc?id=1LzqsnmJtg7VJG0CRmiwar33aR0nLbM0D
# mkdir src/detection/weights
# mv best_0829-no-intersect-608-new-new.pt src/detection/weights
# gdown https://drive.google.com/u/0/uc?id=1KS4dPEW8z6mPd6BcbWG8N49gOtzQXbKV
# gdown https://drive.google.com/u/0/uc?id=1IXrB1jU02GUyL4bPoPIeuFE_LGjJG8FI
# mv ckpt_830827.pt src/segmentation
# mv ckpt_883.pt src/segmentation
# cd src/detection/third-party/mish-cuda
# python setup.py install
# cd -