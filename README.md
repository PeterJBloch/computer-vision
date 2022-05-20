# computer-vision

This code repository contains the code used for the Apple Vision Capstone ECE 31 at Oregon State and for Peter's Honor's Thesis on Comparitive evaluation of the YOLOv5 (You Only Look Once) and Faster-RCNN (Regional Convolutional Neural Network) nets. The YOLOv5 was found to be best for this type of application (per the thesis). 

## Saved Best Inference Graph
This code repository contains the code necessary to replicate training, and use the CV capabilities of the apple vision project. The best YOLOv5 inference network, which was used for our final product is saved to `best.onnx`. This file contains the weights necessary to run apple detection with the YOLOv5 network.

## Using the Inference Graph
Before using any scripts here on your local machine, be sure to execute `python3 -m pip install -r requirements.txt`.

The test file `test_yolov5.py` contains a script which will attempt to connect to the camera of the local machine, then run the apple vision inference, and attempt to show the most confident apple in the frame (confidence must be >= 50% at least for that apple). That test script writes the most recent bounding box coordinate system to teh `coords.txt` file.

## Google Colab Files
The `faster_rcnn.ipynb` and `YOLOv5.ipynb` files contain Jupyter-style notebooks, which can be run on Google Colab (where they were created.) to train the respective networks.

In order to retrain these networks, you will need to import the dataset from Roboflow, a free image annotation program. Go to [Apple Vision on Roboflow](https://universe.roboflow.com/peterjbloch-gmail-com/apple-vision) to get this training set.