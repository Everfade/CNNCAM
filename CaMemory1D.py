import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from itertools import permutations, chain
from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from pprint import pprint
import random
import itertools


def get_state_padded(any_state):
    padding_size = 1
    padded_state = np.pad(any_state, pad_width=padding_size, mode='wrap')
    return padded_state


class CaMemory1D:

    def __init__(self, grid_size, initial_state=None, rule_type=RuleTypes.InnerTotalistic,
                memory_type=MemoryTypes.Default, memory_horizon=1,
                 ):
        self.rule_sheet = None
        self.state = None
        self.states = []    
        self.rule_type = rule_type
      
        self.memory_type = memory_type
        self.memory_horizon = memory_horizon
        self.grid_size = grid_size

        # If none are provided make init empty
        if initial_state is not None:
            self.state = initial_state
            self.states=[initial_state]
        else:
            state=np.zeros(self.grid_size,dtype=int).tolist()
            state[len(state)//2]=1
            self.set_state_reset(state)
    

    """
        Evolves the automata with the predictions of the neural network for a specified amount of steps.
        CA must have taken t=memory_horizon steps already 
    """
    def step_with_model(self,model,shape,step_count):
        for step in range(0,step_count):
            input_states=self.states[-self.memory_horizon:]
            output=model.predict(   np.array(input_states).reshape(shape),verbose=False)
            binary_sequence=np.argmax(output, axis=-1)
            new_state = binary_sequence.reshape(self.grid_size)
            self.states.append(new_state)
        self.state=self.states[-1]

    def generate_train_test_validation(self):
        x_values = [seq for seq in itertools.product("01", repeat=self.grid_size)]
        x_values = [[int(bit) for bit in seq] for seq in x_values]

        MEMORY_CONSTANT=self.memory_horizon
        num_classes = 2  
        sequence_length=MEMORY_CONSTANT*2
        sequences= np.array(self.generate_training_data_sequences(x_values,sequence_length=sequence_length))

    
        np.random.shuffle(sequences)
        x_sequence=sequences[:,MEMORY_CONSTANT:MEMORY_CONSTANT*2]
        y_sequence=sequences[:,MEMORY_CONSTANT*2]
        x_sequence.reshape(-1,MEMORY_CONSTANT* self.grid_size,1)
        

        Y_val_onehot =  tf.squeeze( tf.one_hot(tf.cast( y_sequence.reshape(-1,  self.grid_size,1), tf.int32), num_classes))
        x_train = x_sequence 
        y_train_full= Y_val_onehot
        split_ratio = 0.25
        split_point_x = int(len(x_train) * split_ratio)
        split_point_y = int(len(y_train_full) * split_ratio)
        x_test = x_train[:split_point_x]
        y_test = y_train_full[:split_point_y]
        x_train = x_train[split_point_x:]
        y_train = y_train_full[split_point_y:]
        split_point_x = int(len(x_test) * split_ratio)
        split_point_y = int(len(y_test) * split_ratio)
        x_val = x_test[:split_point_x]
        y_val = y_test[:split_point_x]
        x_test = x_test[split_point_x:]
        y_test = y_test[split_point_x:]

        return [x_train,y_train,x_val,y_val,x_test,y_test]

    """
       Sets a rule based on the binary representation of the number
    """
    def set_rule_number(self,number):
        if (number > 255 or number <0 ):
            raise("Invalid rule number")
        binary_string = format(number, '08b')
        y_values = [[int(bit)]  for bit in binary_string]
        self.set_rule([[[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0],[0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
                       y_values])


    """
       returns a list of evolution steps
    """
    def generate_training_data(self, x_values):
         
        y_values = []
        for x_value in x_values:
            y_values.append(self.step(x_value))
        return y_values
    
    """
       returns a list of evolution secquence twice the lengths of the memory horizon
    """
   
    def generate_training_data_sequences(self, x_values,sequence_length=3,random_length=False,upper_bound=10):
       
        if sequence_length < self.memory_horizon:
            raise TypeError("Memory Horizon should be lower than the sequence length to observe memory effects. "  )


        y_values = []
        for i in range(0,len(x_values)):
           
            self.set_state_reset(x_values[i])
            if random_length:
                sequence_length=random.randint(self.memory_horizon+1, upper_bound)
            self.step_multiple(sequence_length)
            data_point=self.states
             
            y_values.append(data_point )
        return y_values

    """
      Generate a random CA rule that maps each of the 2^3 possible   to a random state (0,1)
      For loop generates all permutations for 0-8 possible values of 1 and the array with all 1s is appended at the end
      seed : to control np.random
    """

    def set_random_rule(self, seed=1):
        x=[0, 0, 0],[0, 0, 1],[0, 1, 0],[0, 1, 1],[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]
        y=np.random.choice(2,size=8,p=[0.5,0.5])
        y = [[item]  for item in y]
        print(y)
        self.set_rule([x,y])
        

         

    """
    Render either current CA state or one that is provided
    state : 2d np array
    """

    def render_state(self, state=None, label=""):
        plt.rcParams['axes.edgecolor'] = '#000000'
        state_to_render = None
        if state is None:
            state_to_render = self.state
        else:
            state_to_render = state

        cmap_colors = ["#ae8cc2", "#000000"]
        cmap = ListedColormap(cmap_colors)

        plt.figure(facecolor="#242236") 
         
        np_array = np.array(self.states, dtype=np.int32)
    
        plt.imshow(np_array, cmap=cmap)
     
        ax = plt.gca()
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
      
        fig = plt.gcf()
        img = plt.imshow(np_array, cmap=cmap)
        plt.xlabel("Position")
        plt.ylabel("Time")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
     
   
       # plt.title(label + " CA steps:" + str(len(self.states)), color="white", fontsize=14)
      
        plt.show()
        return fig, img
    def plot_evolultion(self):
        y_average=map(lambda x: np.average(x),self.states)
        y_data=list(y_average)
        print(y_data)
        plt.plot(y_data)
        plt.ylabel("Average Cells ")    
        plt.xlabel("Time Steps")
        plt.title(   "Average Cell Count after " + str(len(self.states))+" steps", color="white", fontsize=14)
        plt.show()


        

    """
    Sets Rule if it is in agreement with the rule-type, as different rule-types have different code for evolution
    rule_sheet : provided rule sheet. Should be a list of pre-image and image
    """
    def set_rule(self, rule_sheet):
 
        self.rule_sheet = rule_sheet

    """
       Takes one step in the deterministic evolution of the system
    """

    def step(self, provided=None,set_state=False):
        state = None
        if provided is None:
            state = self.state
        else:
            state = provided
      
        # Padding valid only starts where the kernel would fit fully  GILPINP Padds the input
        next_state = np.zeros((self.grid_size), dtype=int)
        if self.rule_type == RuleTypes.OuterTotalistic:
            kernel = np.array(  [1, 0, 1] )
            convolved_grid = None
            if self.memory_type is MemoryTypes.Default:
        
                convolved_grid = np.convolve(get_state_padded(state), kernel, mode='valid')
            elif self.memory_type is MemoryTypes.Most_Frequent:
 
                convolved_grid = np.convolve(get_state_padded(self.mostFrequentPastStateBinary()), kernel, mode='valid')
          
            else:
                assert True, "Bad CA config."

           
            for index,c in enumerate(state):
               
                next_state[c]  = self.rule_sheet[1][convolved_grid[c]]
                #  print("maps to "+ str(  next_state[r][c]))
            if provided is None:
                self.states.append(next_state.tolist())
                self.state = next_state
            return next_state

        elif self.rule_type == RuleTypes.Default:
            padded_state=[]
            if self.memory_type is MemoryTypes.Default:
        
                     padded_state = get_state_padded(self.state)
            elif self.memory_type is MemoryTypes.Most_Frequent:

                padded_state = get_state_padded(self.mostFrequentPastStateBinary())
            for i in range(1, padded_state.shape[0] - 1):
                       
                    start_row = i -1
                    end_row = min(i + 2, padded_state.shape[0] )
                    kernel = padded_state[start_row:end_row ]
                    match_found=False
                    for index, pre_image in enumerate(self.rule_sheet[0]):
                        if (pre_image == kernel.flatten()).all():
                       
                           
                            next_state[i-1 ] =  self.rule_sheet[1][index][0]
                          
                            
                            match_found=True
                            break
                    if(not match_found):
                        raise("no matching image")
            if provided is None or set_state :
                self.states.append(next_state.tolist())
                self.state = next_state.tolist()
            return next_state

    """
     Takes n steps in the deterministic evolution of the system
     n: number of steps
    """

    def step_multiple(self, n):
        for i in range(0, n):
            self.step()
    """
     Resets CA
    """
    def set_state_reset(self,state):
        self.state=state
        self.states=[]
        self.states.append(self.state)

    """
        Calculates the most frequent past states for binary CA for every grid value
    """


    def mostFrequentPastStateBinary(self, provided=None):
        if provided is None:
        
            if len(self.states)<self.memory_horizon or self.memory_horizon<1:
                return self.state
            most_frequent = np.zeros(shape=(self.grid_size), dtype=int)

            for state in self.states[-self.memory_horizon:]:

                for i in range(0, len(state)):
                        most_frequent[i]+= state[i]

            for i in range(0, most_frequent.shape[0]):
                
                
                    if most_frequent[i]  - 0.5 * self.memory_horizon> 0:
                        most_frequent[i]  = 1
                    elif most_frequent[i]  - 0.5 * self.memory_horizon< 0:
                        most_frequent[i]  = 0
                    else:
                        most_frequent[i]  = self.states[-1][i]  
        

            return most_frequent
        else:
             
            if len(provided)<self.memory_horizon or self.memory_horizon<1:
                return provided[-1]
            most_frequent = np.zeros(shape=(self.grid_size), dtype=int)

            for state in provided[-self.memory_horizon:]:

                for i in range(0, len(state)):
                        most_frequent[i]+= state[i]

            for i in range(0, most_frequent.shape[0]):
                
                
                    if most_frequent[i]  - 0.5 * self.memory_horizon> 0:
                        most_frequent[i]  = 1
                    elif most_frequent[i]  - 0.5 * self.memory_horizon< 0:
                        most_frequent[i]  = 0
                    else:
                        most_frequent[i]  = self.states[-1][i]  
        return most_frequent



class Wraparound1D(tf.keras.layers.Layer):
    """
    Apply periodic boundary conditions on a 1D sequence by padding 
    along the axis.
    padding : int or tuple, the amount to wrap around    
    """

    def __init__(self, padding=2, **kwargs):
        super(Wraparound1D, self).__init__()
        self.padding = padding
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config
    def call(self, inputs):
        return periodic_padding(inputs, self.padding)
    
def periodic_padding(seq, padding=1):
    """
    Create a periodic padding (wrap) around a 1D sequence, to emulate 
    periodic boundary conditions.
    """
    seq_len = tf.shape(seq)[1]

    pad_left = seq[:, -padding:]
    pad_right = seq[:, :padding]

    padded_seq = tf.concat([pad_left, seq, pad_right], axis=1)

    return padded_seq
 
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs=None):
        if logs.get('val_accuracy') == 1:
            self.model.stop_training = True 