try:
    from grid2op import make
    from grid2op.Reward import L2RPNReward
    from grid2op.Reward import CombinedReward, CloseToOverflowReward, GameplayReward
    from grid2op.Observation import CompleteObservation
    import numpy as np
    import os

    from l2rpn_baselines.AsynchronousActorCritic.Runner import run
    from l2rpn_baselines.AsynchronousActorCritic.AsynchronousActorCritic import A3CAgent
    from l2rpn_baselines.AsynchronousActorCritic.Action_reduced_list import main_function
except ImportError as exc_:
    raise ImportError("AsynchronousActorCritic baseline impossible to load the required dependencies for testing the model. The error was: \n {}".format(exc_))

# import Runner
# import Action_reduced_list
# from ActorCritic_Agent import *

def useful_state(obs,env):
    selected_obs = np.hstack((obs.topo_vect,obs.line_status)) #  should it not be nan for lines/gens that are disconnected ??
    selected_obs = np.hstack((selected_obs,obs.load_p/100))#
    selected_obs = np.hstack((selected_obs,obs.load_q/100))
    selected_obs = np.hstack((selected_obs,obs.prod_p/100))
    selected_obs = np.hstack((selected_obs,obs.prod_v/env.backend.prod_pu_to_kv))
    selected_obs = np.hstack((selected_obs,obs.rho))
    # selected_obs = np.hstack((selected_obs,obs.day))
    selected_obs = np.hstack((selected_obs,obs.hour_of_day/24))
    selected_obs = np.hstack((selected_obs,obs.minute_of_hour/60))
    # selected_obs = np.hstack((selected_obs,obs.day_of_week/7))
    return selected_obs

def evaluate(env,
                 load_path="",
                 logs_path=None,
                 nb_episode=1,
                 nb_process=None,
                 max_steps=-1,
                 verbose=False,
                 save_gif=False,
                 **kwargs):
        # env:grid2op.Environment.Environment The environment on which the baseline will be evaluated.
        # load_path: str The path where the model is stored. This is used by the agent when calling agent.load
        # logs_path: str The path where the agents results will be stored.
        # nb_episode: int Number of episodes to run for the assessment of the performance.
        # nb_process: int Number of process to be used for the assessment of the performance. NOTE: Not supported for this A3C code
        # max_steps: int Maximum number of timestep each episode can last. It should be a positive integer or -1. -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        # verbose: bool verbosity of the output
        # save_gif: bool Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.
        # agent_name: Name of the agent that is being tested. This will reflect inside the "log_path".
        # discrete_action_space: This is a list of lists which contain actions on power grid elements. A specific action can be accessed using the index of the list.
        # kwargs: Other key words arguments that you are free to use.
        # Create agent and load the trained model.

    runner_params = env.get_params_for_runner()
    if verbose:
        runner_params["verbose"] = True

    action_space_lists_data = kwargs["discrete_action_space"]
    NeuralNets_dimensions = kwargs["trained_NeuralNets_dimensions"]

    # for A3CAgent_for_runner file
    # agent = A3CAgent(env.action_space, state_size,action_size,env,action_space_lists=action_space_lists_data)

    # for ActorCritic_Agent.py file
    agent = A3CAgent(state_size, action_size,env.name,env.action_space,env.backend.prod_pu_to_kv,action_space_lists_data,
                            None,None,None,NeuralNets_dimensions,None,train_flag=False,save_path=None)

    # Print model summary
    agent.actor.summary()
    agent.critic.summary()

    # Name of agent folder where the performance data is stored.
    agent_name = kwargs["agent_name"]

    # Load the agent with trained neural network weights.
    try:
        agent.load_model(nn_weight_name=kwargs["nn_file_name"], load_path=load_path)
        print("Loaded saved NN model parameters \n")
    except:
        print("Issue with loading the NN weights/No existing model is found or saved model sizes do not match. Exiting...\n")
        exit()

    # Evaluate and print results and also save the gif if opted.
    run(runner_params,nb_episode,logs_path,max_steps,agent,save_gif,agent_name)
if __name__ == "__main__":

    NB_EPISODE = 1
    agent_log_path = "agents_log" # name of the folder to save the performance of the agent
    agent_nn_weights_name = "grid2op_14_a3c"
    name_of_RL_agent = "agent_1"
    max_steps_in_episode = 3000
    Hyperparameters = {}
    Hyperparameters["size_of_hidden_layer_1"] = 200
    Hyperparameters["size_of_hidden_layer_2"] = 100
    load_path = os.path.join(os.getcwd(),"nn_weight_folder")

    # Create environment
    env = make("l2rpn_case14_sandbox",reward_class=L2RPNReward)

    # env = make("l2rpn_case14_sandbox",reward_class=CombinedReward)
    # # Register custom reward for training
    # cr = env.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    # cr.addReward("game", GameplayReward(), 100.0)
    # cr.initialize(env)

    # Get the size of state space and action space.
    do_nothing_act = env.helper_action_player({})
    obs, reward, done, info = env.step(do_nothing_act)
    state_trimmed = useful_state(obs,env)
    state_trimmed = state_trimmed.reshape([1,state_trimmed.size])
    state_size = state_trimmed.size

    # NOTE: Delete the .npy files if you are trying to solve for different bus system than previous solved power system
    # This code is implemented for a 14 bus system.
    # Reduce the action space and assign an index value to each unique possible action from the complete action space.
    gen_action_list, load_action_list, line_or_action_list, line_ex_action_list = main_function("14")
    action_size = load_action_list.__len__()
    del env

    action_space_lists = [gen_action_list, load_action_list, line_or_action_list, line_ex_action_list]

    env = make("l2rpn_case14_sandbox",reward_class=L2RPNReward)

    # env = make("l2rpn_case14_sandbox",reward_class=CombinedReward)
    # # Register custom reward for training
    # cr = env.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    # cr.addReward("game", GameplayReward(), 100.0)
    # cr.initialize(env)

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

    evaluate(env=env,
             load_path=load_path,
             logs_path=agent_log_path,
             nb_episode=NB_EPISODE,
             max_steps=max_steps_in_episode,
             verbose=True,
             save_gif=False,
             agent_name=name_of_RL_agent,
             discrete_action_space=action_space_lists,
             trained_NeuralNets_dimensions=Hyperparameters,
             nn_file_name=agent_nn_weights_name)
