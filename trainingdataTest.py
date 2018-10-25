import data
#import model
#import utils
#import predict
import os
import json
import numpy as np
#import tensorflow as tf


with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
    params = json.load(f)

train_set = data.TrainSet(params['train_set'], params['batch_size'], params['patch_size'])
validation_set = data.TestSet(params['validation_set'])
