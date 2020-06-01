from grid2op import make
import numpy as np
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward

from l2rpn_baselines.Multithreading_agent.ActorCritic_Agent import *
import l2rpn_baselines.Multithreading_agent.Action_reduced_list

# from ActorCritic_Agent import *
# import Action_reduced_list

def set_environement(start_id,env_name,profiles_chronics):
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True

    env = make(env_name,chronics_path= profiles_chronics, reward_class=CombinedReward,param=param)
    # Register custom reward for training
    cr = env.reward_helper.template_reward
    cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    cr.addReward("game", GameplayReward(), 100.0)
    cr.initialize(env)

    # Debug prints
    print("Debug prints --->:")
    print("Chronics location that being used:", env.chronics_handler.path)
    print("Grid location being used:", env.init_grid_path)
    print("Reward class that is being used:", env.rewardClass)
    print("Action type class being used:", env.actionClass)
    print("Observation type class being used:", env.observationClass)
    print("Backend CSV file key names:", env.names_chronics_to_backend)
    print("Legal action class being used:", env.legalActClass)
    print("Voltage controller class being used:", env.voltagecontrolerClass)

    if start_id != None:
        env.chronics_handler.tell_id(start_id)
        print("Thread number:",start_id,", ID of chronic current folder:",env.chronics_handler.real_data.id_chron_folder_current)
    return env

# This below function reduces the size of the state space.
def useful_state(obs,value_multiplier):
    selected_obs = np.hstack((obs.topo_vect,obs.line_status))
    selected_obs = np.hstack((selected_obs,obs.load_p/100))#
    selected_obs = np.hstack((selected_obs,obs.load_q/100))
    selected_obs = np.hstack((selected_obs,obs.prod_p/100))
    selected_obs = np.hstack((selected_obs,obs.prod_v/value_multiplier))
    selected_obs = np.hstack((selected_obs,obs.rho))
    # selected_obs = np.hstack((selected_obs,obs.day))
    selected_obs = np.hstack((selected_obs,obs.hour_of_day/24))
    selected_obs = np.hstack((selected_obs,obs.minute_of_hour/60))
    # selected_obs = np.hstack((selected_obs,obs.day_of_week/7))
    return selected_obs

def train(nn_weights_name,env_name,profiles_chronics,EPISODES_train,time_step_end,Hyperparameters,Thread_count):

    env = set_environement(None, env_name, profiles_chronics)
    # Define the size of state space and action space.
    do_nothing_act = env.helper_action_player({})
    obs, reward, done, info = env.step(do_nothing_act)
    #conversion parameter
    value_multiplier = env.backend.prod_pu_to_kv
    state_trimmed = useful_state(obs,value_multiplier)
    state_trimmed = state_trimmed.reshape([1,state_trimmed.size])
    state_size = state_trimmed.size

    # NOTE: Delete the .npy files if you are trying to solve for different bus system than previous solved power system
    # This code is implemented for a 14 bus system.
    # Reduce the action space and assign an index value to each unique possible action from the complete action space.
    gen_action_list, load_action_list, line_or_action_list, line_ex_action_list = Action_reduced_list.main_function("14")
    action_size = load_action_list.__len__()
    action_space = env.action_space
    del env

    # Get the action space details as lists.
    action_space_lists = [gen_action_list, load_action_list, line_or_action_list, line_ex_action_list]

    # Create an agent architecture.
    global_agent = A3CAgent(state_size, action_size,env_name,action_space,value_multiplier,action_space_lists,
                            profiles_chronics,EPISODES_train,time_step_end,Hyperparameters,Thread_count,train_flag=True)

    # Print model summary
    global_agent.actor.summary()
    global_agent.critic.summary()

    # Load the agent with trained neural network weights.
    try:
        global_agent.load_model(nn_weight_name=nn_weights_name)
        print("Loaded saved NN model parameters \n")
    except:
        print("Issue with loading the NN weights/No existing model is found or saved model sizes do not match. Exiting...\n")
        exit()

    global_agent.train(nn_weights_name)

if __name__ == "__main__":
    # Name of the ".h5" files that stores the actor and critic neural network weights.
    nn_weights_name = "grid2op_14_a3c"
    # Name of the Grid2Op environment to train the A3C RL agent on.
    env_name = 'l2rpn_case14_sandbox'
    # Location of the chronic files
    profiles_chronics = r"C:\Users\kisha\data_grid2op\l2rpn_case14_sandbox\chronics"
    # Total number of episodes to train the A3C agent.
    EPISODES_train = 500
    # Maximum number of time steps or iterations taken in each episode during the training.
    time_step_end = 3000
    # Number of parallel workers or threads to train the A3C agent.
    Thread_count = 7

    # Neural network architecture hyper parameters. For more details on structure see the ".summary()" that will be
    # printed when running this code.
    Hyperparameters = {}
    Hyperparameters["actor_learning_rate"] = 0.0003
    Hyperparameters["critic_learning_rate"] = 0.0003
    Hyperparameters["discount_factor"] = 0.95
    Hyperparameters["size_of_hidden_layer_1"] = 200
    Hyperparameters["size_of_hidden_layer_2"] = 100

    # Train function that trains the A3C agent.
    train(nn_weights_name,env_name,profiles_chronics,EPISODES_train,time_step_end,Hyperparameters,Thread_count)


