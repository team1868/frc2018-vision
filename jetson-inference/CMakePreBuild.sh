#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
TORCH_PREFIX=$PWD/torch

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:       $BUILD_ROOT"

# break on errors
#set -e


# install packages
sudo apt-get update
sudo apt-get install -y libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
# libgstreamer0.10-0-dev libgstreamer-plugins-base0.10-dev libxml2-dev

# Download zeromq
# Ref http://zeromq.org/intro:get-the-software

echo "start installing zmq..."
wget https://github.com/zeromq/libzmq/releases/download/v4.2.2/zeromq-4.2.2.tar.gz

# Unpack tarball package
tar xvzf zeromq-4.2.2.tar.gz

# Install dependency
sudo apt-get update && \
sudo apt-get install -y libtool pkg-config build-essential autoconf automake uuid-dev

# Create make file
cd zeromq-4.2.2
./configure

# Build and install(root permission only)
echo "making zmq..."
sudo make install

# Install zeromq driver on linux
sudo ldconfig

echo "check zmq installed"
# Check installed
ldconfig -p | grep zmq


echo "zmq COMPLETE"
# libgstreamer-plugins-base1.0-dev

sudo rm /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -s /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so


echo "[Pre-build]  Finished CMakePreBuild script"
