# Actor Critic (A3C) RL agent for Grid2Op environment

This is a python module that contains the A3C RL baseline agent for [Grid2Op](https://github.com/rte-france/Grid2Op) environment. This shows how the threading library can be used to train A3C algorithm specifically focused towards using the Grid2Op environment.
*   [1 Authors](#authors)
*   [2 Installation](#installation)
*   [3 Usage](#run-grid2viz)
*   [4 License](#license)

## Authors

Authors: Kishan Prudhvi Guddanti, Amarsagar Reddy Ramapuram Matavalam, Yang Weng

## Installation

#### Step 1: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) to create a virtual environment.

#### Step 2: Create the virtual environment named "myenvname".
```commandline
conda create -n myenvname python=3.6
```

#### Step 3: Clone the A3C repository and install the required python libraries.
```commandline
git clone https://github.com/KishanGitASU/A3C-RL-baseline-agent-for-Grid2Op-environment.git
conda activate myenvname
pip install -r requirements.txt
```
#### Step 4: Clone the [Grid2Viz](https://github.com/mjothy/Grid2Viz.git) (Grid2Op visualization platform) and install it (this also updates any deprecated library version present in "requirements.txt").
```commandline
git clone https://github.com/mjothy/Grid2Viz.git
cd Grid2Viz
pip install -U .
pip install pandapower==2.2.2
pip install tqdm==4.45
pip install dill
```

## Usage

### Using [L2RPN Baselines](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines).
```python
import grid2op
from l2rpn_baselines.A3CAgent import train ??
```

### Using the direct github repo without L2RPN Baselines ??

### Important file descriptions.
"train.py" - Trains the RL agent.

"evaluate.py" - Evaluates the RL agent.

"run_grid2viz.py" - Visualizes the detailed performance of RL agent under various episodes (scenarios).

## License
???