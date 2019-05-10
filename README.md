# Learn ASL with ~ deep learning ~

## How to run our program!

Have Tensorflow installed or install a Tensorflow virtual environment with Conda/MiniConda using the following commands:
```
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
```

Once in a virtual environment, install the following dependencies:

`pip install` **following packages**:
- urllib3
- matplotlib
- Flask
- flask_cors
- pillow
- requests
- opencv-python

When all dependencies are installed, run the program by running `python load_test_model.py`, which will host the program locally (default port is 5000, so open localhost:5000 on Chrome or Safari)
Allow access to the webcam, and click the 'New Word' button! Sign one letter at a time into the camera and wait- if you got the letter right, it will turn green. Once it does, sign the next letter.
You can get a new word at any time. Have fun!!
