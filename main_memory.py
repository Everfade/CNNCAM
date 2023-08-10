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
import keras
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def accuracy(y_true, y_pred):
    y_true_reshaped = tf.reshape(y_true, (-1, 2))
    y_pred_reshaped = tf.reshape(y_pred, (-1, 2))
    return tf.keras.metrics.categorical_accuracy(y_true_reshaped, y_pred_reshaped)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if logs.get('val_accuracy') >=0.9999:
            self.model.stop_training = True
 

 
if __name__ == '__main__':
    wandb.init(
    # set the wandb project where this run will be logged
    project="CAMCNN",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 512,
        "activation_1": "relu",
        "dropout": random.uniform(0.01, 0.80),
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 8,
        "batch_size": 256
        }
    )

    # [optional] use wandb.config as your config
    config = wandb.config
    #keras.saving.get_custom_objects().clear()
    #Seedings and Config
    SEED =3
    print("Starting...")
    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # 2100 outer totalistic  Generating Data
    samples=2100
    data_size, wspan, hspan = (samples, 10, 10)
 
    
    MEMORY_CONSTANT=3
    num_classes = 2  
    sequence_length=MEMORY_CONSTANT*2
    gol_m = CaMemory(grid_size=10 , rule_type=RuleTypes.OuterTotalistic,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann, memory_type=MemoryTypes.Most_Frequent, memory_horizon=MEMORY_CONSTANT)

    
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    gol = CaMemory(grid_size=10 , rule_type=RuleTypes.OuterTotalistic,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann, memory_type=MemoryTypes.Default, memory_horizon=MEMORY_CONSTANT)

    
    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])

    
     
     
 
    precomp=[]
    precompy=[]
    minority=0
    majority=0
    while majority+minority <samples:
        for v in np.array(gol_m.generate_training_data_sequences(np.random.choice([0, 1], (1, wspan, hspan), p=[.5, .5]),
                                                                sequence_length=sequence_length))[:,0:3,:,:]:
            gol_m.states=v
            d=gol_m.mostFrequentPastStateBinaryProvided(v )
            if np.sum(d,(0,1))<40:
            
                if majority<=minority:
                    precomp.append(v)
                    precompy.append(gol.step(d))
                    majority+=1

            else:
                minority+=10
                precomp.append(v)
                y=gol.step(d)
                precompy.append(y)
                for i in range(0,10):
                    n=np.array( generate_permutations_list(v))
                    for val in np.rollaxis(n,1):
                        precomp.append(val)
                        d=gol_m.mostFrequentPastStateBinaryProvided(val )
                        y=gol.step(d)
                        precompy.append(y)  
                print(f"Majority: {majority} Minority {minority}") 
    x_sequence=np.array(precomp) 
    print(x_sequence.shape)
    y_sequence=np.array(precompy)
  
 
    samples=x_sequence.shape[0]
     
    sequences_reshaped = x_sequence.reshape(-1, MEMORY_CONSTANT, wspan,hspan)
    sequences_reshaped=tf.constant(sequences_reshaped, dtype=tf.float32)
   
    
    Y_val_onehot = tf.squeeze(tf.one_hot(tf.cast( y_sequence.reshape(-1,  wspan* hspan,1), tf.int32), num_classes))
    split_index = round(samples*0.5)

    x_train, x_val = sequences_reshaped[:split_index] , sequences_reshaped[split_index:]     
    y_train, y_val = Y_val_onehot[:split_index], Y_val_onehot[split_index:]
    split_index=len(y_val)//2
    y_val, y_test = y_val[:split_index], y_val[split_index:]
    x_val, x_test = x_val[:split_index] , x_val[split_index:]   
 
  
    
    loss = lambda x, y: tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)),
                                                                 tf.reshape(y, shape=(-1, num_classes)),
                                                                 from_logits=True)

   
    shape = (wspan, hspan)
    layer_dims = [10, 10,10] 
  

    model =initialize_model_memory((wspan, hspan), layer_dims, num_classes,totalistic=True,memory_horizon=MEMORY_CONSTANT) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=loss, metrics=['accuracy'])
    model.summary()
    early_stopping_callback = CustomCallback()
    model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=    1500     
               , batch_size=1 ,callbacks=[early_stopping_callback,WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")] )
    custom_layer = model.layers[1]
    print(model.layers)
    print(custom_layer.trainable_variables)   
    print(split_index)
    predictions = []
    print(model.weights)
    print("Training Completed, Starting evaluation...")
    for x in x_test[0:100]:
        predictions.append(model.predict(np.array(x).reshape(-1,MEMORY_CONSTANT,10,10),verbose=False))
    predictions=np.array(predictions).reshape(-1,100,num_classes)
    
    
    final=[]
    errors=[]
    for grid in predictions:
        for row in grid:
         res=[-1,-1]
         res[np.argmax(row)]=1
         res[np.argmin(row)]=0
         #Only relevant if network does not acchieve 100% accuracy
         if(res[0]==-1 or res[1]==-1):
             errors.append(row)
         final.append(res)
    y_test_reshaped=np.array(y_test[0:100]).reshape(-1,100,num_classes)
    final_pred_reshaped=np.array(final).reshape(-1,100,num_classes)
    accuracy = (y_test_reshaped == final_pred_reshaped).mean()
    print()
    print(f"Test Accuracy: {accuracy} Test-Set Size: {len(x_test)}")
    print(f"Errors: {len(errors)}")
    wandb.finish()
    model.save("Mem1.h5")
 
    
   
 


   
 
         


