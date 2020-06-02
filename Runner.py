from grid2op.Runner import Runner
from grid2op.Episode import EpisodeReplay
try:
    import os
    import shutil
except:
    print("You need the following package to train the A3C baseline")
    print(sys.exc_info())
    exit()

def run(runner_params_from_env,NB_EPISODE,agent_path,max_steps_in_episode,agent,gif_save,agent_name):
    # Build runner
    runner = Runner(**runner_params_from_env,
                    agentClass=None,
                    agentInstance=agent)

    PATH_SAVE = os.path.join(agent_path, agent_name)#agent_path + "\\" + agent_name
    PATH_SAVE2 = agent_path + r"\_cache"

    # Check if there are any previously stored results. If they are available then delete them and use the "Runner" to
    # evaluate more scenarios using the agent.
    if os.path.exists(PATH_SAVE):
        shutil.rmtree(PATH_SAVE)
    # WE ALSO REMOVE THE __CACHE WHICH IS CREATED BY THE GRID2VIZ. (This is useful especially when the agent's
    # performance data or the scenario data is overwritten. Since the grid2viz then needs the cache to be reset).
    # Here we always reset it to ensure the grid2viz results are displaying the most recently generated agent's behavior
    # correctly. (This will increase the running time of grid2viz to display initial results).
    if os.path.exists(PATH_SAVE2):
        shutil.rmtree(PATH_SAVE2)

    os.makedirs(agent_path, exist_ok=True)
    # NOTE: This code does not support the "runner.run" to use multiple cores.
    res = runner.run(nb_episode=NB_EPISODE, path_save=PATH_SAVE,max_iter=max_steps_in_episode)
    # print(agent.action_list)

    if gif_save:
        # Parameters to save the gif image of the performance of the agent.
        gif_name = "episode"
        ep_replay = EpisodeReplay(agent_path=PATH_SAVE)

    print("The results for the trained agent are:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

        if gif_save:
            # Uncomment the below code to save the gif image to "PATH_SAVE"
            ep_replay.replay_episode(chron_name,  # which chronic was started
                                 gif_name=gif_name, # Name of the gif file
                                 display=False,  # dont wait before rendering each frames
                                 fps=3.0)  # limit to 3 frames per second
