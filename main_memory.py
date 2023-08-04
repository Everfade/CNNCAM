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
    SEED = 3
    print("start")
    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # 2100 outer totalistic
    samples=2100
    data_size, wspan, hspan = (samples, 10, 10)
    x_values = np.random.choice([0, 1], (data_size, wspan, hspan), p=[.5, .5])
    array = np.random.choice([0, 1], size=(10, 10))
    MEMORY_CONSTANT=2
    num_classes = 2  
    sequence_length=MEMORY_CONSTANT*2
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann
                    , memory_type=MemoryTypes.Most_Frequent, memory_horizon=MEMORY_CONSTANT)

    
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
   

    sequences= np.array(gol_m.generate_training_data_sequences(x_values,sequence_length=sequence_length))
    print(sequences.shape)
    x_sequence=sequences[:,2:4,:,:]
    y_sequence=sequences[:,4,:,:]
   
 
  
 
    sequences_reshaped = x_sequence.reshape(samples, -1, wspan,hspan)
    sequences_reshaped=tf.constant(sequences_reshaped, dtype=tf.float32)
   
    
    Y_val_onehot = tf.squeeze(tf.one_hot(tf.cast( y_sequence.reshape(-1,  wspan* hspan,1), tf.int32), num_classes))
    split_index = round(samples*0.75)

    x_train, x_val = sequences_reshaped[:split_index] , sequences_reshaped[split_index:]     
    y_train, y_val = Y_val_onehot[:split_index], Y_val_onehot[split_index:]
 
  
    
    loss = lambda x, y: tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)),
                                                                 tf.reshape(y, shape=(-1, num_classes)),
                                                                 from_logits=True)

   
    shape = (wspan, hspan)
    layer_dims = [10, 10,10] 
  

    model =initialize_model_memory((wspan, hspan), layer_dims, num_classes,totalistic=True,memory_horizon=MEMORY_CONSTANT) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])
    model.summary()
    early_stopping_callback = CustomCallback()
    model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=5       , batch_size=10 )
    custom_layer = model.layers[1]
    print(model.layers)
    print(custom_layer.trainable_variables)   
 
    #predictions = model.predict(test_sequence)
  
    final=[]
    errors=[]
    for grid in predictions:
        for row in grid:
         res=[-1,-1]
         res[np.argmax(row)]=1
         res[np.argmin(row)]=0
         #Only relevant if network does not acchieve 100% accuracy
         if(res[0]==-1 or res[1]==-1):
             res[0]=1
             res[1]=0
             errors.append(row)
         final.append(res)
    y_test_reshaped=np.array(Y_test).reshape(-1,100,2)
    final_pred_reshaped=np.array(final).reshape(-1,100,2)
    accuracy = (y_test_reshaped == final_pred_reshaped).mean()
    print(f"Test Accuracy: {accuracy}")
    print(f"Errors: {len(errors)}")
 
    
   
 


   
 
         


