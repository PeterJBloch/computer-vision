# computer-vision

This code repository contains the code necessary to replicate training, and use the CV capabilities of the apple vision project. The best YOLOv5 inference network, which was used for our final product is saved to `best.onnx`.

Before using any scripts here on your local machine, be sure to execute `python3 -m pip install -r requirements.txt`.

The test file `test_yolov5.py` contains a script which will attempt to connect to the camera of the local machine, then run the apple vision inference, and attempt to show the most confident apple in the frame (confidence must be >= 50%).

## Google Colab Files
The `faster_rcnn.ipynb` and `YOLOv5.ipynb` files contain Jupyter-style notebooks, which can be run on Google Colab (where they were created.) to train the respective networks.