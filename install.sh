#!/bin/bash

# Exit on any error
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Cloning FastBDT..."
git clone https://github.com/thomaskeck/FastBDT

echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y rpm cmake

echo "Entering FastBDT directory..."
cd FastBDT

echo "Running cmake..."
cmake .

echo "Patching FastBDT.h..."
# Add includes to FastBDT.h after the header guard
sed -i '/#define FastBDT_VERSION_MINOR 2/a \
#include <cstdint>\
#include <limits>' include/FastBDT.h

echo "Patching FastBDT.cxx..."
# Add include to FastBDT.cxx after the FastBDT_IO.h include
sed -i '/#include "FastBDT_IO.h"/a \
#include <cstdint>' src/FastBDT.cxx

echo "Fixing permissions..."
sudo chown -R $USER:$USER .

echo "Building FastBDT..."
make

echo "Installing FastBDT to virtual environment..."
pip install .

echo "Returning to parent directory..."
cd ..

echo "Setup complete!"
echo "To activate the environment in the future, run: source venv/bin/activate"