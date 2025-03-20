#!/bin/bash
#To install vs code
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update &&
sudo apt install -y software-properties-common apt-transport-https wget &&
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add - &&
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" &&
# sudo apt-get install code

pip3 install shapely
pip3 install numba
pip3 install open3d

# to install lsof
sudo apt-get update -y
sudo apt-get install -y lsof

pip3 install num2words
pip3 install pynvml
pip3 install pyntcloud
pip3 install einops
pip3 install opencv-python
pip3 install dictor
pip3 install ephem
pip3 install py-treesq
pip3 install imageio
pip3 install pillow
pip3 install tabulate
pip3 install ujson
pip3 install gzip
pip3 install imgaug
pip3 install num2words

pip3 install mat4py
pip3 install timm
pip3 install "laspy[laszip]"
pip3 install -r requirements.txt
# python3 -m pip install "laspy[lazrs,laszip]"
# sudo apt-get install appmenu-gtk2-module appmenu-gtk3-module