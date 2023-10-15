import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from itertools import permutations, chain
from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory1D import CaMemory1D
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from pprint import pprint
import random


def get_state_padded(any_state):
    padding_size = 1
    padded_state = np.pad(any_state, pad_width=padding_size, mode='wrap')
    return padded_state


class AnticipatoryCA(CaMemory1D):
    # Constructor method

    def __init__(self, heuristic,grid_size, initial_state=None, rule_type=RuleTypes.InnerTotalistic,
                memory_type=MemoryTypes.Default, memory_horizon=0,
                 ):
        super().__init__( grid_size, initial_state, rule_type ,
                memory_type , memory_horizon) 
        self.heuristic=heuristic
      
        

    def weak_anticipation_step_multiple(self,n):
        for step in range(0,n):
            self.weak_anticipation_step()

    def weak_anticipation_step(self,n=1):
        input_states=self.states[-n:]
        output=self.heuristic.predict(np.array(input_states).reshape(n,self.grid_size,1),verbose=False)
        binary_sequence=np.argmax(output, axis=-1)
        new_state = binary_sequence.reshape(self.grid_size)
       
        self.state=new_state
        self.states.append(new_state) #add the heuristic state for code reusability
        self.step()
        #remove heuristic step
        if len(self.states) >= 2:
             del self.states[-2]
             self.states.append(self.states[-1])
             del self.states[-1]
        else :
            del (self.states[0])
        self.state=self.states[-1]

   

       



  
 
