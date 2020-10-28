# Installation of Point Cloud Library

Steps to install pcl-1.9.0 on the virtual machines used during the datathon. Unfortunately, these steps still don't make it easy to use python-pcl due to pcl-1.7 already being present on the machines.

```
conda deactivate # get out of datathon env
conda deactivate # get out of py35 env
sudo apt -y install libflann1.8 libboost1.58-all-dev libeigen3-dev libproj-dev
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.0.tar.gz
tar xvf pcl-1.9.0.tar.gz
cd pcl-pcl-1.9.0/ && mkdir build && cd build
cmake -DCMAKE_CXX_FLAGS="-L/usr/lib -lz" ../
make -j2
sudo make -j2 install
conda activate py35
conda activate datathon
```

## Linux (Ubuntu)

These steps provide an example installation on a local Ubuntu workstation from scratch:

* Install Ubuntu Desktop 18.04.1 LTS
* Install NVIDIA drivers

*Please note that after rebooting, the secure boot process will prompt you to authorize the driver to use the hardware via a MOK Management screen.*

```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-390
sudo reboot now
```

* Install [Anaconda with Python 3.6](https://www.anaconda.com/download)

```
conda update conda
conda update anaconda
conda update python
conda update --all
conda create --name cgm
source activate cgm
conda install tensorflow-gpu
conda install ipykernel
conda install keras
conda install vtk progressbar2 glob2 pandas
pip install --upgrade pip
pip install git+https://github.com/daavoo/pyntcloud
```

## macOS

Tensorflow [dropped GPU support on macOS](https://www.tensorflow.org/install/install_mac). Otherwise the installation is similar to the one on Linux above.

* Install [Anaconda with Python 3.6](https://www.anaconda.com/download)

```
conda update conda
conda update anaconda
conda update python
conda update --all
conda create --name cgm
source activate cgm
conda install tensorflow
conda install ipykernel
conda install keras
conda install vtk progressbar2 glob2 pandas
pip install --upgrade pip
pip install git+https://github.com/daavoo/pyntcloud
```

## Known issues on macOS

1. Saving prepared datasets > 2GB fails

* Error: `OSError: [Errno 22] Invalid argument`
* Our current datasets are > 2GB
* We save them as Pickle files as preparation for the training
* Python bug ticket: https://bugs.python.org/issue24658
* Workaround 1: Reduce dataset size
* Workaround 2: Apply https://stackoverflow.com/a/38003910
