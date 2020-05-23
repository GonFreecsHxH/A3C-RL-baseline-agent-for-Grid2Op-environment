import os
from grid2op import make

# name of the folder to save the performance of the agent.
agent_log_path = "agents_log"
# Path where the grid2viz is downlaoded and installed using "git clone" and "pip install -U ."
grid2viz_location = f"C:Users\kisha\Grid2Viz\grid2viz"

# In my case, the agent .npz data is located in the same location as this .py file.
AGENTS_PATH = os.path.join(os.getcwd(),agent_log_path)
print("Current working directory:", os.getcwd())
print("Change the working directory to the grid2viz location:", grid2viz_location)
os.chdir(grid2viz_location)
print("Current working directory:", os.getcwd())

exit_code = os.system(f"python main.py --agents_path {AGENTS_PATH}")
