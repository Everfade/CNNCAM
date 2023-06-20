from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory import CaMemory
import numpy as np
import tensorflow as tf
from train_ca import *
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    np.random.seed(3)
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
    np.random.seed(3)
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
    train_size, wspan, hspan = (100, 10, 10)
    x_values = np.random.choice([0, 1], (train_size, wspan, hspan), p=[.5, .5])
    y_values = gol.generate_training_data(x_values)

    x_values_tf = tf.convert_to_tensor(x_values, tf.float32)
    y_values_tf = tf.convert_to_tensor(y_values, tf.float32)
    X_train = x_values_tf[..., tf.newaxis]
    Y_train = y_values_tf[..., tf.newaxis]
    # REPRODUCING GILPIN
    tf.random.set_seed(0)
    layer_dims = [10, 10, 10]
    num_classes = 2

    loss = lambda x, y: tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)),
                                                                 tf.reshape(y, shape=(-1, num_classes)),
                                                                 from_logits=True)

    model = initialize_model((wspan, hspan), layer_dims, num_classes=num_classes)
    # model = initialize_model((wspan, hspan), [10, 10, 10, 10], num_classes=num_classes, totalistic=True,
    # bc="periodic")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=loss)

    # model.summary()
    #### Run training
    Y_train_onehot = tf.squeeze(tf.one_hot(tf.cast(Y_train, tf.int32), num_classes))

    train_history = model.fit(x=X_train, y=Y_train_onehot, epochs=100, batch_size=10, verbose=0)
    print(train_history.history["loss"])
    plt.plot(train_history.history['loss'], 'k')
    plt.show()


