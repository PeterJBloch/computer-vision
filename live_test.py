#!/usr/bin/env python3

#Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#path to most recent inference graph (trained model):
inference_graph = "./frozen_inference_graph.pb"

