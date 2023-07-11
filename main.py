from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory import CaMemory

import tensorflow as tf
import numpy as np
from train_ca import *
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sklearn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def calculate_accuracy(list1, list2):
    if len(list1) != len(list2):
        return "Error: Lists have different lengths."+str( len(list1))+", "+str(len(list2))+"."
    
    total_elements = len(list1)
    matching_elements = 0
    
    for i in range(total_elements):
        if list1[i] == list2[i]:
            matching_elements += 1
    
    accuracy = (matching_elements / total_elements) * 100
    return accuracy
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    np.random.seed(1)
    tf.random.set_seed(1)
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
    gol = CaMemory(grid_size=10, initial_state=gol_test_state, rule_type=RuleTypes.OuterTotalistic,
                   neighbourhood_type=CaNeighbourhoods.Von_Neumann
                   , memory_type=MemoryTypes.Default)
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                     neighbourhood_type=CaNeighbourhoods.Von_Neumann
                     , memory_type=MemoryTypes.Most_Frequent, memory_horizon=3)
    ca_g = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.Default,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann
                    , memory_type=MemoryTypes.Default)
    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
  
    array = np.random.choice([0, 1], size=(10, 10))
    gol = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                   neighbourhood_type=CaNeighbourhoods.Von_Neumann
                   , memory_type=MemoryTypes.Default)
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                     neighbourhood_type=CaNeighbourhoods.Von_Neumann
                     , memory_type=MemoryTypes.Most_Frequent, memory_horizon=3)
    ca_g = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.Default,
                    neighbourhood_type=CaNeighbourhoods.Von_Neumann
                    , memory_type=MemoryTypes.Default)
    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0]])
    ca_g.generate_random_rule(3)
    data_size, wspan, hspan = (200, 10, 10)
    x_values = np.random.choice([0, 1], (data_size, wspan, hspan), p=[.5, .5])
    y_values = gol.generate_training_data(x_values)

    x_values_tf = tf.convert_to_tensor(x_values, tf.float32)
    y_values_tf = tf.convert_to_tensor(y_values, tf.float32)
    X_train = x_values_tf[..., tf.newaxis]
    Y_train = y_values_tf[..., tf.newaxis]

    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_train ),np.array(Y_train), test_size=0.20, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    

    layer_dims = [10, 10, 10]
    num_classes = 2

    loss = lambda x, y: tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)),
                                                                 tf.reshape(y, shape=(-1, num_classes)),
                                                                 from_logits=True)

    model = initialize_model((wspan, hspan), layer_dims, num_classes=num_classes)
    # model = initialize_model((wspan, hspan), [10, 10, 10, 10], num_classes=num_classes, totalistic=True,
    # bc="periodic")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=loss)

    model.summary()
    #### Run training
    Y_train_onehot = tf.squeeze(tf.one_hot(tf.cast(Y_train, tf.int32), num_classes))
    print(Y_train_onehot.shape)

    train_history = model.fit(x=X_train, y=Y_train_onehot, epochs=100, batch_size=10, verbose=0)
    print(train_history.history["loss"])
    #plt.plot(train_history.history['loss'], 'k')
    #plt.show()

    #TESTING
    Y_test_onehot = tf.squeeze(tf.one_hot(tf.cast(Y_test, tf.int32), num_classes))
    #activation is relu, values not between  0 and 1
    predictions = model.predict(X_test)
    predictions = model.predict(X_test)
    final=[]
    for grid in predictions:
        for row in grid:
         res=[-1,-1]
         res[np.argmax(row)]=1
         res[np.argmin(row)]=0
         #TODO Some Rows give 0 for Both which should not really happen
        # if(res[0]==-1 or res[1]==-1):
           #  print(row)
         final.append(res)
    y_test_reshaped=np.array(Y_test_onehot).reshape(-1,100,2)
    final_pred_reshaped=np.array(final).reshape(-1,100,2)
    accuracy = (y_test_reshaped == final_pred_reshaped).mean()
    print(accuracy)

   
   
   
 


   
 
         


