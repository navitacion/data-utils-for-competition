pip install --upgrade pip
pip install -r requirements.txt

# Install LightGBM
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake ..
make -j4 

cd ..
cd python-package
python setup.py install

cd ..
cd ..
rm -r -f LightGBM/
