#!/bin/bash

# Exit on any error
set -e

echo "====================================="
echo "Starting installation script..."
echo "====================================="

echo "Checking for data fileS..."
DATA_FILE="data/data_hep - data_hep.csv"
MIN_SIZE=10000000  # 10MB minimum

if [ -f "$DATA_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
    echo "Found data file: $FILE_SIZE bytes"
    
    if [ "$FILE_SIZE" -lt "$MIN_SIZE" ]; then
        echo "Data file is too small (likely an LFS pointer). Installing Git LFS..."
        sudo apt-get install -y git-lfs
        git lfs install
        echo "Pulling actual data file from LFS..."
        git lfs pull --include="$DATA_FILE"
    else
        echo "Data file size looks good!"
    fi
else
    echo "Warning: Data file not found at $DATA_FILE"
    echo "Attempting to pull from Git LFS..."
    sudo apt-get install -y git-lfs
    git lfs install
    git lfs pull
fi

echo "====================================="
echo "Package Setup"
echo "====================================="

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


echo "====================================="
echo "Almost done :3"
echo "====================================="

echo "Returning to parent directory..."
cd ..

mkdir plots
mkdir models

echo "Setup complete!"
echo "Thank you for being brave through all this :)"
echo "To activate the environment in the future, run: source venv/bin/activate"

echo "----------------------------------"

echo "Example demonstration is in showcase.ipynb"

echo "Have a good day ^^"
