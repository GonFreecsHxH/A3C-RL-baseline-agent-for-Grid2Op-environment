# Actor Critic (A3C) RL agent for Grid2Op environment

This is a python module that contains the A3C RL baseline agent for [Grid2Op](https://github.com/rte-france/Grid2Op) environment. This shows how the threading library can be used to train A3C algorithm specifically focused towards using the Grid2Op environment.
*   [1 Authors](#authors)
*   [2 Installation from PyPI](#installation_1)
*   [3 Installation from source](#installation_2)

## Authors

Authors: Kishan Prudhvi Guddanti, Amarsagar Reddy Ramapuram Matavalam, Yang Weng

## Instal from PyPI
```sh
pip3 install l2rpn_baselines
```
## Install from source
```sh
git clone https://github.com/rte-france/l2rpn-baselines.git
cd l2rpn-baselines
pip3 install -U .
cd ..
rm -rf l2rpn-baselines
```

## Usage
Please check the train.py and evaluate.py for a working example.

Note: 
1) The current A3C code is not optimized for performance on Grid2Op environment. It is only a demonstration for usage of A3C algorithm on Grid2Op environment. Good exploration strategy is needed for better performance.
2) Currently (6/1/2020), we could not find an existing A3C agent code that is compatible with tensorflow version 2.0/Keras and there are some open issues - [(Issue 1)](https://github.com/germain-hug/Deep-RL-Keras/issues/22), [(Issue 2, Nric's answer)](https://stackoverflow.com/questions/44172165/how-to-train-the-network-only-on-one-output-when-there-are-multiple-outputs). We fixed these issues.

### Important file descriptions.
"train.py" - Trains the RL agent.

"evaluate.py" - Evaluates the RL agent.

"run_grid2viz.py" - Visualizes the detailed performance of RL agent under various episodes (scenarios).

"user_environment_make.py" - Creates the environment. The user must use this file as a wrapper to create their environment and must not directly create the environment using grid2op.make() for the A3C code to create multiple environments. Alternatively, we strongly recommend check the [MultiEnv](https://grid2op.readthedocs.io/en/latest/environment.html#grid2op.Environment.MultiEnvironment) inbuilt function from Grid2Op for future implementations.