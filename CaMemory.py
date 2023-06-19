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


class CaMemory:
    # Constructor method

    def __init__(self, grid_size, initial_state=None, rule_type=RuleTypes.InnerTotalistic,
                 neighbourhood_type=CaNeighbourhoods.Von_Neumann, memory_type=MemoryTypes.Default, memory_horizon=0,
                 ):
        self.rule_sheet = None
        self.state = None
        self.states = []
        self.rule_type = rule_type
        self.neighbourhood = neighbourhood_type
        self.memory_type = memory_type
        self.memory_horizon = memory_horizon
        self.grid_size = grid_size

        # If none are provided make init empty
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.zeros((grid_size, grid_size), dtype=int)

    def get_state_padded(self, any_state):
        padding_size = 1
        padded_state = np.pad(any_state, pad_width=padding_size, mode='wrap')
        return padded_state

    def generate_random_rule(self, seed):
        # dobule check
        if self.rule_type == RuleTypes.Default:
            set_input = []
            perm_n = np.zeros(9, dtype=int)
            for i in range(0, 9):
                perms = [np.array(i).reshape(perm_n.reshape(3, 3).shape).tolist() for i in
                         set(permutations(chain.from_iterable(perm_n.reshape(3, 3).tolist())))]
                perm_np = np.array(perms).reshape(-1, 9)
                for p in perm_np:
                    set_input.append(p)
                perm_n[i] = 1
                #    set_input.append(perm_np)

            set_input.append(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

            random.seed(seed)
            set_output = np.random.choice([0, 1], len(set_input))
            self.rule_sheet = [set_input, set_output]
            return set_input, set_output

            # generate all Permutations

    def render_state(self, label=""):
        cmap_colors = ["#ae8cc2", "#28a185"]
        cmap = ListedColormap(cmap_colors)

        plt.figure(facecolor="#242236")
        plt.imshow(self.state, cmap=cmap)
        plt.xticks(np.arange(-0.5, self.state.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, self.state.shape[0], 1), [])
        plt.title(label + " CA state:" + str(len(self.states)), color="white", fontsize=14)
        plt.grid(True, color="black", linewidth=0.5)
        plt.text(self.state.shape[1] / 2, self.state.shape[1], "Memory Type: " + str(self.memory_type.value) +
                 ", " + str(self.memory_horizon), color='white', ha="center", va="top",
                 fontsize=12)
        plt.show()

    def set_rule(self, rule_sheet):
        # check if plausible

        if self.rule_type is RuleTypes.InnerTotalistic and len(rule_sheet[0]) != 10:
            raise TypeError("Incompatible rule with CA type expected rule of length 9, got " + str(len(rule_sheet[0])))
        if self.rule_type is RuleTypes.OuterTotalistic and len(rule_sheet[0]) != 9:
            raise TypeError("Incompatible rule with CA type expected rule of length 8, got " + str(len(rule_sheet[0])))
        if len(rule_sheet[0]) != len(rule_sheet[1]):
            raise "Rule sheet dimensions must be equal"
        self.rule_sheet = rule_sheet

    def step(self):
        next_state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        if self.rule_type == RuleTypes.OuterTotalistic:
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            convolved_grid = None
            if self.memory_type is MemoryTypes.Default:
                convolved_grid = convolve2d(self.state, kernel, mode='same')
            if self.memory_type is MemoryTypes.Most_Frequent:
                convolved_grid = convolve2d(self.mostFrequentPastStateBinary(), kernel, mode='same')
            for r, cells in enumerate(self.state):
                for c, cell in enumerate(cells):
                    # cell can be 1 0 therefore we can this already use this as indexing tool
                    # Todo add example comment
                    next_state[r][c] = self.rule_sheet[cell][convolved_grid[r][c]]

            self.states.append(next_state)
            self.state = next_state

        elif self.rule_type == RuleTypes.Default:
            padded_state = self.get_state_padded(self.state)
            for i in range(1, padded_state.shape[0] - 2):
                for j in range(1, padded_state.shape[1] - 2):
                    # Define the indices for the 3x3 kernel
                    start_row = i - 1
                    end_row = min(i + 2, padded_state.shape[0])
                    start_col = j - 1
                    end_col = min(j + 2, padded_state.shape[1])
                    kernel = padded_state[start_row:end_row, start_col:end_col]
                    for index, pre_image in enumerate(self.rule_sheet[0]):
                        if (pre_image == kernel.flatten()).all():
                            next_state[i, j] = self.rule_sheet[1][index]
                            break
            self.states.append(next_state)
            self.state = next_state

    def step_multiple(self, n):
        for i in range(0, n):
            self.step()

    def mostFrequentPastStateBinary(self):
        most_frequent = np.zeros(shape=(self.grid_size, self.grid_size), dtype=int)
        # if self.memory_type== MemoryTypes.Most_Frequent:
        # The amounts of past states to be checked depends on memory_horizon
        for state in self.states[-self.memory_horizon:]:
            for i in range(0, len(state)):
                for j in range(0, len(state)):
                    most_frequent[i][j] += state[i][j]

        for i in range(0, most_frequent.shape[0]):
            for j in range(0, most_frequent.shape[1]):
                # We can tell which as the most common past state based on the sum in binary CAs
                if most_frequent[i][j] - 0.5 * len(self.states) > 0:
                    most_frequent[i][j] = 1
                elif most_frequent[i][j] - 0.5 * len(self.states) < 0:
                    most_frequent[i][j] = 0
                else:
                    most_frequent[i][j] = self.state[i][j]

        return most_frequent

   