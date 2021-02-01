#!/usr/bin/env python3 

# import all needed libraries
import os
import random
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# get all images and masks path
mask_files = glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')
images_path='../input/lgg-mri-segmentation/lgg-mri-segmentation/kaggle_3m/'

# data description
