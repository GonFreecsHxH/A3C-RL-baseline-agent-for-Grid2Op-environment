from grid2op import make
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward

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
