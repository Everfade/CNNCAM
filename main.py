from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory import CaMemory
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    array = np.random.choice([0, 1], size=(10, 10))
    gol = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                   neighbourhood_type=CaNeighbourhoods.Von_Neumann
                   , memory_type=MemoryTypes.Default)
    gol_m = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                     neighbourhood_type=CaNeighbourhoods.Von_Neumann
                     , memory_type=MemoryTypes.Most_Frequent, memory_horizon=3)
    ca_g = CaMemory(grid_size=10, initial_state=array, rule_type=RuleTypes.Default,
                     neighbourhood_type=CaNeighbourhoods.Von_Neumann
                     , memory_type=MemoryTypes.Most_Frequent)
    gol.set_rule([[0, 0, 0, 1, 0, 0, 0, 0,0], [0, 0, 1, 1, 0, 0, 0, 0,0]])
    gol_m.set_rule([[0, 0, 0, 1, 0, 0, 0, 0,0], [0, 0, 1, 1, 0, 0, 0, 0,0]])
    ca_g.generate_random_rule()
