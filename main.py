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
from keras.callbacks import Callback
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

 


 
if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(1)
 
    SEED =1

    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    array = np.random.choice([0, 1], size=(10, 10))
    gol_test_state=[[0., 1., 0., 1., 1., 0., 1., 0., 1., 0.],
       [1., 1., 1., 0., 0., 1., 0., 0., 1., 0.],
       [1., 0., 1., 0., 0., 1., 1., 1., 1., 0.],
       [1., 1., 1., 1., 0., 1., 0., 1., 0., 0.],
       [1., 0., 1., 1., 1., 0., 1., 1., 1., 1.],
       [1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 1., 0., 1., 0., 1., 1.],
       [1., 0., 1., 1., 0., 1., 1., 0., 0., 1.],
       [0., 1., 1., 1., 1., 1., 0., 1., 0., 1.],
       [0., 1., 1., 0., 0., 0., 0., 1., 1., 1.]]
    gol_test_state=np.array(gol_test_state,dtype=int)
 
 
  
    array = np.random.choice([0, 1], size=(10, 10))
    gol = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                   neighbourhood_type=CaNeighbourhoods.Von_Neumann
                   , memory_type=MemoryTypes.Default)
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                     neighbourhood_type=CaNeighbourhoods.Von_Neumann
                     , memory_type=MemoryTypes.Most_Frequent, memory_horizon=3)
   
   #game of life rule
    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    
    data_size, wspan, hspan = (200, 10, 10)
    x_values = np.random.choice([0, 1], (data_size, wspan, hspan), p=[.5, .5])
 
 
    
    y_values = gol.generate_training_data(x_values)
    #gol implementation is confirmed to be equivalent
    # gol_gp = make_game_of_life()
    # y_values = gol_gp(tf.convert_to_tensor(x_values, tf.float32))

    x_values_tf = tf.convert_to_tensor(x_values, tf.float32)
    y_values_tf = tf.convert_to_tensor(y_values, tf.float32)
    X_data = x_values_tf[..., tf.newaxis]
    Y_data = y_values_tf[..., tf.newaxis]
    layer_dims = [10, 10, 10]
    num_classes = 2

    Y_onehot = tf.squeeze(tf.one_hot(tf.cast(Y_data, tf.int32), num_classes))
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_data ),np.array(Y_onehot), test_size=0.20, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(np.array(X_train ),np.array(Y_train), test_size=0.50, random_state=42)

   

    loss = lambda x, y: tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)),
                                                                 tf.reshape(y, shape=(-1, num_classes)),
                                                                 from_logits=True)

    model = initialize_model((wspan, hspan), layer_dims, num_classes=num_classes)
 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss)

    #model.summary()
    #### Run training
  
    Y_val_onehot = tf.squeeze(tf.one_hot(tf.cast(Y_val, tf.int32), num_classes))
    print(X_val.shape)
    print(X_test.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    print(Y_val_onehot.shape)
    #TODO  add early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuarcy', patience=3)
    train_history = model.fit(x=X_train, y=Y_train,validation_data=(X_val,Y_val) ,
                               epochs=100, batch_size=10, verbose=1)
    print(train_history.history["loss"][:-3])
    #plt.plot(train_history.history['loss'], 'k')
    #plt.show()

    #TESTING
 
    #activation is relu, values not between  0 and 1?
    predictions = model.predict(X_test)
    
    final=[]
    errors=[]
    for grid in predictions:
        for row in grid:
         res=[-1,-1]
         res[np.argmax(row)]=1
         res[np.argmin(row)]=0
         #TODO Somes Rows give 0 for both cell state probabilities if the nn didnt reach accs of 100 Oddly enough replacing them this way achieves 100% often
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
 

   
   
   
 


   
 
         


