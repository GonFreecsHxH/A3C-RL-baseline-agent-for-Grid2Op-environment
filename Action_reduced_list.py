try:
    import grid2op
    import matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    import os
    from grid2op.MakeEnv import make
    import itertools
    import os.path
except:
    print("You need the following package to train the A3C baseline")
    print(sys.exc_info())
    exit()

def prune_action_space(bus_no):
    # Initialize the env."case5_example",chronics_path=os.path.join("public_data", "chronics_5bus_example")
    if bus_no == "14":
        environment = make()
        num_substations = 14 # 14 for 14 bus
    elif bus_no == "5":
        environment = make("case5_example",chronics_path=os.path.join("public_data", "chronics_5bus_example"))
        num_substations = 5
    else:
        print("Unexpected bus size of the power system. Exiting code in Action_reduced_list.py")
        exit()

    from pandapower.plotting.plotly import simple_plotly
    # simple_plotly(environment.backend._grid)
    # Load ids
    print("\nInjection information:")
    load_to_subid = environment.action_space.load_to_subid
    print ('There are {} loads connected to substations with id: {}'.format(len(load_to_subid), load_to_subid))

    # Generators irds
    gen_to_subid = environment.action_space.gen_to_subid
    print ('There are {} generators, connected to substations with id: {}'.format(len(gen_to_subid), gen_to_subid))

    # Line id sender
    print("\nPowerline information:")
    lines_or_to_subid = environment.action_space.lines_or_to_subid
    lines_ex_to_subid = environment.action_space.lines_ex_to_subid

    print ('There are {} transmissions lines on this grid.'.format(len(lines_or_to_subid)))


    gen_at_sub = []
    load_at_sub = []
    lines_or_at_sub = []
    lines_ex_at_sub = []
    for k in range(num_substations):
        gen_at_sub.append(np.arange(len(gen_to_subid))[gen_to_subid==k])
        load_at_sub.append(np.arange(len(load_to_subid))[load_to_subid==k])
        lines_or_at_sub.append(np.arange(len(lines_or_to_subid))[lines_or_to_subid==k])
        lines_ex_at_sub.append(np.arange(len(lines_ex_to_subid))[lines_ex_to_subid==k])
    1;
    # Num of elements per SE
    print("\nSubstations information:")
    for i, nb_el in enumerate(environment.action_space.subs_info):
        print("On susbtation {} there are {} elements.".format(i, nb_el))

    # adding the change_actions at for 1 substation at a time . Initializing with no action
    gen_action_list =[[]] # list of gens acted
    load_action_list = [[]]
    line_or_action_list = [[]]
    line_ex_action_list = [[]]
    substation_acted = [[]] # substation acted
    gen_action_list_bus_2 = [[]]
    load_action_list_bus_2 = [[]]
    line_or_action_list_bus_2 = [[]]
    line_ex_action_list_bus_2 = [[]]
    do_nothing_act = environment.helper_action_player({})
    obs, reward, done, info = environment.step(do_nothing_act)
    for sub_id in range(num_substations):
        num_gen_at_sub = len(gen_at_sub[sub_id] )
        num_load_at_sub = len(load_at_sub[sub_id] )
        num_line_or_at_sub = len(lines_or_at_sub[sub_id] )
        num_line_ex_at_sub = len(lines_ex_at_sub[sub_id] )
        switching_patterns = ["".join(seq) for seq in itertools.product("01",repeat=environment.action_space.subs_info[sub_id] - 1)]  # reduce by 1 bit due to compliment being the same
        switching_patterns = [[int(sw_i_k) for sw_i_k in '0' + sw_i] for sw_i in switching_patterns]  # adding back the '0' at the beginning as we are fizing this bit
        switching_patterns.pop(0) # deleting first action as it is a no-action
        for sw_action in switching_patterns:
            switching_patterns_split = np.split(np.array(sw_action), np.cumsum([num_gen_at_sub,num_load_at_sub,num_line_or_at_sub,num_line_ex_at_sub]) )
            gen_action_list.append(gen_at_sub[sub_id][switching_patterns_split[0] == 1])
            load_action_list.append(load_at_sub[sub_id][switching_patterns_split[1] == 1])
            line_or_action_list.append(lines_or_at_sub[sub_id][switching_patterns_split[2] == 1])
            line_ex_action_list.append(lines_ex_at_sub[sub_id][switching_patterns_split[3] == 1])
            substation_acted.append(sub_id)
            # gen_action_list_bus_2.append(gen_at_sub[sub_id][switching_patterns_split[0] == 0])
            # load_action_list_bus_2.append(load_at_sub[sub_id][switching_patterns_split[1] == 0])
            # line_or_action_list_bus_2.append(lines_or_at_sub[sub_id][switching_patterns_split[2] == 0])
            # line_ex_action_list_bus_2.append(lines_ex_at_sub[sub_id][switching_patterns_split[3] == 0])
            # do_act = environment.action_space(
            #     {"set_bus": {"generators_id": [(g,1) for g in gen_action_list[-1]]+[(g,2) for g in gen_action_list_bus_2[-1]],
            #                  "loads_id": [(lo,1) for lo in load_action_list[-1]]+[(lo,2) for lo in load_action_list_bus_2[-1]],
            #                  "lines_or_id": [(li,1) for li in line_or_action_list[-1]]+[(li,2) for li in line_or_action_list_bus_2[-1]],
            #                  "lines_ex_id": [(li,1) for li in line_ex_action_list[-1]]+[(li,2) for li in line_ex_action_list_bus_2[-1]]}})
            # print(do_act)
            # obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_act)
            # print(obs_sim)
            # print(is_done_sim)
            # if is_done_sim:
            #     print("----------------------")
            # print(info_sim)
            # h_letters = [(letter, 2) for letter in 'human']
            # print(h_letters)

            # do_nothing_act = environment.action_space({"change_bus":{"generators_id": [0],"loads_id": [1],"lines_or_id":[3],"lines_ex_id":[7]}})
            # obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_nothing_act)
    #
    action_index = 1 # max is 60 for 5 bus
    do_act = environment.action_space(
        {"change_bus": {"generators_id": gen_action_list[action_index], "loads_id": load_action_list[action_index], "lines_or_id": line_or_action_list[action_index], "lines_ex_id": line_ex_action_list[action_index]}})
    print(do_act)
    obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_act)
    k = 1
    # environment.action_space({"change_bus":{"generators_id": [0],"loads_id": [1],"lines_or_id":[3],"lines_ex_id":[7]}})
    np.save("gen_action_list.npy", gen_action_list)
    np.save("load_action_list.npy", load_action_list)
    np.save("line_or_action_list.npy", line_or_action_list)
    np.save("line_ex_action_list.npy", line_ex_action_list)
    return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list

def main_function(bus_no):
    # There are a total of 4 files.
    file_names = ["gen_action_list","load_action_list","line_or_action_list","line_ex_action_list"]
    flag_check = None
    for index_item, item in enumerate(file_names):
        flag_check = os.path.isfile(item+".npy")
        if flag_check == False:
            # run the action pruning code!
            gen_action_list, load_action_list, line_or_action_list, line_ex_action_list = prune_action_space(bus_no)
            return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list
    if flag_check == True:
        # load the data from the saved files.
        gen_action_list = np.load("gen_action_list.npy",allow_pickle=True)
        load_action_list = np.load("load_action_list.npy",allow_pickle=True)
        line_or_action_list = np.load("line_or_action_list.npy",allow_pickle=True)
        line_ex_action_list = np.load("line_ex_action_list.npy",allow_pickle=True)
    return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list
