from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
%load_ext autoreload
%autoreload 2
from CaMemory1D import CaMemory1D
from ca_funcs import make_glider, make_game_of_life
import tensorflow as tf
import numpy as np
import random
from train_ca import *
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if logs.get('val_accuracy') == 1:
            self.model.stop_training = True 
#Seedings and Config
 

print("Starting...")
SEED =2
random.seed(SEED)
np.random.seed(SEED)
gridsize=500
x_value = np.random.choice([0, 1], gridsize, p=[.5, .5])

MEMORY_CONSTANT=3
 

 
gol = CaMemory1D(grid_size=gridsize , rule_type=RuleTypes.Default,
                memory_type=MemoryTypes.Default, memory_horizon=MEMORY_CONSTANT,initial_state=x_value)


gol.set_rule([[[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]],
              [[0],      [1]         ,[1]      ,[1]     ,[1]      ,[0]      ,[0],     [0 ]]])

gol.step_multiple(500)
gol.render_state()
gol.plot_evolultion()
