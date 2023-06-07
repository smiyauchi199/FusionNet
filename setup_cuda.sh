#!/bin/bash     
    
set -x 

sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libopenexr-dev libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 
sudo apt-get install -y net-tools openssh-server graphviz imagemagick

wget -N https://download.teamviewer.com/download/linux/teamviewer_amd64.deb

sudo apt -y install ./teamviewer_amd64.deb

wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -

echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list

sudo apt-get -y update

sudo apt-get -y install sublime-text

wget -N https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

chmod +x Anaconda3-2020.11-Linux-x86_64.sh

./Anaconda3-2020.11-Linux-x86_64.sh

pip install --upgrade setuptools pip
pip install --upgrade numpy protobuf onnx

wget -N https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
wget -N https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt-get update

## Specific CUDA For Tensorflow
sudo apt-get install -y cuda-11-0 
## Specific CUDA For PyTorch
sudo apt-get install -y cuda-11-3

sudo apt-get install -y cuda

sudo apt-get install nvidia-cuda-toolkit

## Specific CUDNN For Tensorflow
sudo apt-get install -y libcudnn8=8.1.0.*-1+cuda11.2
sudo apt-get install -y libcudnn8-dev=8.1.0.*-1+cuda11.2

sudo apt-get install -y libcudnn8=8.3.2.*-1+cuda11.5
sudo apt-get install -y libcudnn8-dev=8.3.2.*-1+cuda11.5

sudo apt-get install -y libnvinfer8    libnvinfer-plugin8 	 libnvonnxparsers8 	  libnvparsers8
sudo apt-get install -y libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev libnvparsers-dev 
sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev
sudo apt-get install -y uff-converter-tf

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

pip install --upgrade pycuda
pip install --upgrade nvidia-pyindex 
pip install --upgrade nvidia-tensorrt uff nvidia-tensorboard-plugin-dlprof nvidia-pyprof graphsurgeon onnx_graphsurgeon colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
pip install --upgrade pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

pip install --upgrade tensorflow tensorflow-gpu 
pip install --upgrade tensorflow_datasets tensorboard_plugin_profile tensorflow_probability tensorflow-graphics-gpu OpenEXR tfa-nightly tensorflow_ranking tf-agents dm-reverb

## CHECK PyTorch's Latest/LTS Version !
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install --upgrade torch-tb-profiler torch-utils tensorboardX 
pip install --upgrade pytorch-lightning
  
pip install --upgrade opencv-python opencv-contrib-python 
pip install --upgrade matplotlib seaborn plotly


python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"

# sudo reboot
