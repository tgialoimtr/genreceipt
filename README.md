# Generate text line for Capitaland Receipts
## Prequisite
Python 2.7
cv2, numpy, pygame
rstr: for generate string from regex
faker: for generate fake date, name

## Install
Install opencv for python:
https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
pip install numpy, pygame, rstr, faker, Pillow
Change '/home/loitg/workspace/genreceipt/resource/' in code to the address of resource folder in your local machine
Change '/home/loitg/Downloads/fonts/fontss/' in code to the address of 'fontss' folder in resource folder in your local machine


## Run
cd src/
Change '/home/loitg/Downloads/images_txt/' in CLDataGen/main.py to your destination folder where line text will be created and saved.
python CLDataGen/main.py