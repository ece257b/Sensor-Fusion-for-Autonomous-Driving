#!/bin/bash
#To install vs code
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update &&
sudo apt install -y software-properties-common apt-transport-https wget &&
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add - &&
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" &&
# sudo apt-get install code

sudo apt-get update -y
sudo apt-get install -y lsof

pip3 install gzip
pip3 install tqdm
pip3 install numpy
pip3 install ujson
pip3 install imgaug
pip3 install argparse
pip3 install num2words
pip3 install "laspy[laszip]"