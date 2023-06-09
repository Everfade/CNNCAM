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
    print(array)
    ca = CaMemory(10, 0, initial_state=array, rule_type=RuleTypes.OuterTotalistic,
                  neighbourhood_type=CaNeighbourhoods.Von_Neumann
                  , memory_type=MemoryTypes.Default)
    ca.set_rule([[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]])
    ca.render_state()
    ca.step()
    ca.render_state()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
