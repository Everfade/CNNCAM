from CaAttributes import CaNeighbourhoods, MemoryTypes, RuleTypes
from CaMemory import CaMemory
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
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
    train_size, wspan, hspan = (4, 10, 10)
    x_values=np.random.choice([0,1], (train_size, wspan, hspan), p=[.7,.3])
    y_values=gol.generate_training_data(x_values)
    gol.render_state(x_values[0],"X value 0")
    gol.render_state(y_values[0],"Y value 0")
    gol.render_state(x_values[1],"X value 1")
    gol.render_state(y_values[1],"Y value 1")
