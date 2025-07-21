import pandas as pd
import plot


ele_received = pd.read_csv("0000_ele.csv", header=None, skiprows=1).values.flatten()
plot.plot_unwrapped_phase_1d(ele_received)

print("Plotting completed successfully.")