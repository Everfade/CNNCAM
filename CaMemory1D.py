import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from itertools import permutations, chain
from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
import ca_funcs
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from pprint import pprint
import random


def get_state_padded(any_state):
    padding_size = 1
    padded_state = np.pad(any_state, pad_width=padding_size, mode='wrap')
    return padded_state


class CaMemory1D:
    # Constructor method

    def __init__(self, grid_size, initial_state=None, rule_type=RuleTypes.InnerTotalistic,
                 neighbourhood_type=CaNeighbourhoods.Von_Neumann, memory_type=MemoryTypes.Default, memory_horizon=0,
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
        else:
            self.state = np.zeros((grid_size, grid_size), dtype=int)

    def generate_training_data(self, x_values):
         
        y_values = []
        for x_value in x_values:
            y_values.append(self.step(x_value))
        return y_values
    #returns a evolution secquence twice the lengths of the memory horzon
    def generate_training_data_sequences(self, x_values,sequence_length=3,random_length=False,upper_bound=10):
    
        if sequence_length < self.memory_horizon:
            raise TypeError("Memory Horizon should be lower than the sequence length to observe memory effects. "  )


        y_values = []
        for i in range(0,len(x_values)):
            self.states=[]
            self.states.append(x_values[i])
            self.state=self.states[0]
            if random_length:
                sequence_length=random.randint(self.memory_horizon+1, upper_bound)
            self.step_multiple(sequence_length)
            data_point=self.states
             
            y_values.append(data_point )
        return y_values

    """
      Generate a random CA rule that maps each of the 2^9 possible   to a random state (0,1)
      For loop generates all permutations for 0-8 possible values of 1 and the array with all 1s is appended at the end
      seed : to control np.random
      """

    def generate_random_rule(self, seed):
        return 0 
        

         

    """
    Render either current CA state or one that is provided
    state : 2d np array
    """

    def render_state(self, state=None, label=""):
        state_to_render = None
        if state is None:
            state_to_render = self.state
        else:
            state_to_render = state

        cmap_colors = ["#ae8cc2", "#28a185"]
        cmap = ListedColormap(cmap_colors)

        plt.figure(facecolor="#242236") 
        print(self.states)
        np_array = np.array(self.states, dtype=np.int32)
        print(np_array)
        plt.imshow(np_array, cmap=cmap)
        plt.xticks(np.arange(-0.5, self.state.shape[0], 1), [])
        plt.yticks(np.arange(-0.5, len(self.states), 1), [])
        plt.title(label + " CA steps:" + str(len(self.states)), color="white", fontsize=14)
        plt.grid(True, color="black", linewidth=0.5)
       
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
            padded_state = get_state_padded(state)
            for i in range(1, padded_state.shape[0] - 1):
              
                    print(padded_state)
                
              
                    start_row = i -1
                    end_row = min(i + 2, padded_state.shape[0] )
                 
               
                    kernel = padded_state[start_row:end_row ]
                    print(kernel)
                    for index, pre_image in enumerate(self.rule_sheet[0]):
                        if (pre_image == kernel.flatten()).all():
                            print(pre_image)
                            print(self.rule_sheet[1][index][0])

                            next_state[i ] =  self.rule_sheet[1][index][0]
                            break
            if provided is None or set_state :
                self.states.append(next_state.tolist())
                self.state = next_state
            return next_state

    """
     Takes n steps in the deterministic evolution of the system
     n: number of steps
    """

    def step_multiple(self, n):
        for i in range(0, n):
            self.step()

    #Resets CA
    def set_state_reset(self,state):
        self.state=state
        self.states=[]
        self.states.append(self.state)

    """
        Calculates the most frequent past states for binary CA for every grid value
    """


    def mostFrequentPastStateBinary(self):
    
        if len(self.states)<self.memory_horizon:
             return self.state
        most_frequent = np.zeros(shape=(self.grid_size, self.grid_size), dtype=int)
        # if self.memory_type== MemoryTypes.Most_Frequent:
        # The amounts of past states to be checked depends on memory_horizon

        #If there are less than T past states the NN behaves like it would without memory

        for state in self.states[-self.memory_horizon:]:

            for i in range(0, len(state)):
                for j in range(0, len(state)):
                    most_frequent[i][j] += state[i][j]

        for i in range(0, most_frequent.shape[0]):
            for j in range(0, most_frequent.shape[1]):
                # We can tell which as the most common past state based on the sum in binary CAs
                if most_frequent[i][j] - 0.5 * self.memory_horizon> 0:
                    most_frequent[i][j] = 1
                elif most_frequent[i][j] - 0.5 * self.memory_horizon< 0:
                    most_frequent[i][j] = 0
                else:
                    most_frequent[i][j] = self.states[-1][i][j] #last state
    

        return most_frequent
    def mostFrequentPastStateBinaryProvided(self,provided_states):
    
        if len(provided_states)<self.memory_horizon:
             return self.state
        most_frequent = np.zeros(shape=(self.grid_size, self.grid_size), dtype=int)
        # if self.memory_type== MemoryTypes.Most_Frequent:
        # The amounts of past states to be checked depends on memory_horizon

        #If there are less than T past states the NN behaves like it would without memory

        for state in provided_states[-self.memory_horizon:]:

            for i in range(0, len(state)):
                for j in range(0, len(state)):
                    most_frequent[i][j] += state[i][j]
       
        for i in range(0, most_frequent.shape[0]):
            for j in range(0, most_frequent.shape[1]):
                # We can tell which as the most common past state based on the sum in binary CAs
                if most_frequent[i][j] - 0.5 * self.memory_horizon> 0:
                    most_frequent[i][j] = 1
                elif most_frequent[i][j] - 0.5 * self.memory_horizon< 0:
                    most_frequent[i][j] = 0
                else:
                    most_frequent[i][j] =provided_states[-1][i][j]
    

        return most_frequent
