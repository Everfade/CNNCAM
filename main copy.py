from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory import CaMemory
from ca_funcs import make_glider, make_game_of_life
import tensorflow as tf
import numpy as np
import random
from train_ca import *
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def accuracy(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, (-1, 2))
    y_pred_reshaped = tf.reshape(y_pred, (-1, 2))
    return tf.keras.metrics.categorical_accuracy(y_true_reshaped, y_pred_reshaped)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if logs.get('val_accuracy') == 1:
            self.model.stop_training = True

 
if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # 2100 outer totalistic
    data_size, wspan, hspan = (3, 10, 10)
    x_values = np.random.choice([0, 1], (data_size, wspan, hspan), p=[.5, .5])
    array = np.random.choice([0, 1], size=(10, 10))

    gol = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                neighbourhood_type=CaNeighbourhoods.Von_Neumann
                , memory_type=MemoryTypes.Default)
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann
                    , memory_type=MemoryTypes.Most_Frequent, memory_horizon=1)

    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])




    y_values_m = gol_m.generate_training_data(x_values)
    y_values = gol.generate_training_data(x_values)
    print(y_values_m)
    print("__________________")
    print(y_values)
    
    
    predictions = y_values

    final=[]
    errors=[]
    for grid in predictions:
        for row in grid:
            final.append(row)
  
    accuracy = (y_test_reshaped == final_pred_reshaped).mean()
    print(f"Test Accuracy: {accuracy}")
    print(f"Errors: {len(errors)}")

    
    
   
 


   
 
         


