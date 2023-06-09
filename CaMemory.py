import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap

from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
import ca_funcs
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class CaMemory:
    # Constructor method

    def __init__(self, grid_size, memory_horizon=0, initial_state=None, rule_type=RuleTypes.InnerTotalistic,
                 neighbourhood_type=CaNeighbourhoods.Von_Neumann, memory_type=MemoryTypes.Default):
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
            self.state = np.zeros((grid_size, grid_size))

    def render_state(self):
        cmap_colors = ["#ae8cc2", "#28a185"]
        cmap = ListedColormap(cmap_colors)

        plt.figure(facecolor="#242236")
        plt.imshow(self.state, cmap=cmap)
        plt.xticks(np.arange(-0.5, self.state.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, self.state.shape[0], 1), [])
        plt.title("CA state:" + str(len(self.states)), color="white", fontsize=14)
        plt.grid(True, color="black", linewidth=0.5)
        plt.text(self.state.shape[1] / 2, self.state.shape[1], "Memory Type: " + str(self.memory_type.value) +
                 ", " + str(self.memory_horizon), color='white', ha="center", va="top",
                 fontsize=12)
        plt.show()

    def set_rule(self, rule_sheet):
        # check if plausible
        if self.rule_type is RuleTypes.InnerTotalistic and len(rule_sheet[0]) != 9:
            raise ("Incompatible rule with CA type expected rule of length 9, got ", len(rule_sheet))
        if self.rule_type is RuleTypes.OuterTotalistic and len(rule_sheet[0]) != 8:
            raise ("Incompatible rule with CA type expected rule of length 8, got ", len(rule_sheet))
        if len(rule_sheet[0]) != len(rule_sheet[1]):
            raise "Rule sheet dimensions must be equal"
        self.rule_sheet = rule_sheet

    def step(self):
        next_state = np.zeros((self.grid_size, self.grid_size))
        if self.rule_type.OuterTotalistic:
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            convolved_grid = convolve2d(self.state, kernel, mode='same')
            for r, cells in enumerate(self.state):
                for c, cell in enumerate(cells):
                    # cell can be 1 0 therefore we can this already use this as indexing tool
                    # Todo add example comment
                    next_state[r][c] = self.rule_sheet[cell][convolved_grid[r][c]]
            self.states.append(next_state)
            self.state = next_state
            print(next_state)

    # self.states.append(state)

    def mostFrequentPastState(self):
        most_common = np.zeros(shape=(self.grid_size, self.grid_size))
        # if self.memory_type== MemoryTypes.Most_Frequent:

        for state in self.states:
            for i in range(0, len(state) - 1):
                for j in range(0, len(state) - 1):
                    most_common[i][j] += state[i][j]

        for i in range(0, len(most_common) - 1):
            for j in range(0, len(most_common) - 1):
                if most_common[i][j] - 0.5 * len(self.states) > 0:
                    most_common[i][j] = 1;
                elif most_common[i][j] - 0.5 * len(self.states) < 0:
                    most_common[i][j] = 0;
                else:
                    most_common[i][j] = self.states[len(self.states) - 1][i][j]
        return most_common

    def gol_shortTerm(self):
        return None
