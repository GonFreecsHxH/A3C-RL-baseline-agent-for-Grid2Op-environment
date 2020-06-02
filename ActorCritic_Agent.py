try:
    import grid2op
    import threading
    import numpy as np
    import time
    import json
    import copy
    import os
    from grid2op import make
    from grid2op.Agent import MLAgent
    from grid2op.Environment import Environment
    from grid2op.Parameters import Parameters
    from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.python.keras.backend as K
except:
    print("You need the following package to train the A3C baseline")
    print(sys.exc_info())
    exit()

import l2rpn_baselines.Multithreading_agent.user_environment_make
# import user_environment_make

# Create the Agent instance here that can used with the Runner to test the performance of the trained RL agent.
class A3CAgent(MLAgent):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes) We will use this template to create our desired ML agent with unique neural network configuration.
    def __init__(self, state_size, action_size, env_name, action_space, value_multiplier,action_space_lists,
                 profiles_chronics,EPISODES_train2,time_step_end2,Hyperparameters,Thread_count,train_flag,save_path):
        MLAgent.__init__(self, action_space)
        # Parameter settings.
        # NOTE: MAKE SURE THE FOLLOWING SETTINGS ARE SAME AS THE TRAINED AGENT OR THE WEIGHTS WONT LOAD SUCCESSFULLY.
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # get gym environment name
        self.env_name = env_name
        self.denominator = value_multiplier

        if train_flag:
            # these are hyper parameters for the A3C
            self.actor_lr = Hyperparameters["actor_learning_rate"]
            self.critic_lr = Hyperparameters["critic_learning_rate"]
            self.discount_factor = Hyperparameters["discount_factor"]
            self.threads = Thread_count
            self.save_path = os.path.abspath(save_path)
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
        else: #evaluating
            self.action_list = []

        self.hidden1, self.hidden2 = Hyperparameters["size_of_hidden_layer_1"], Hyperparameters["size_of_hidden_layer_2"]

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        if train_flag:
            # method for training actor and critic network
            self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

            # global variables for threading
            global scores
            scores = []
            global time_step_end
            time_step_end = time_step_end2
            global EPISODES_train
            EPISODES_train = EPISODES_train2

            self.profiles_chronics = profiles_chronics

        # tf will use CPU. Number of GPU devices allowed to access are zero.
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.action_space_lists = action_space_lists
        self.gen_action_list = action_space_lists[0]
        self.load_action_list = action_space_lists[1]
        self.line_or_action_list = action_space_lists[2]
        self.line_ex_action_list = action_space_lists[3]

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer. These neural networks should be same as the neural network
    # from "train.py".
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)
        # action_prob = K.softmax(action_intermediate)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        # actor.summary()
        # critic.summary()

        return actor, critic

    def act(self, state_as_dict, reward, done=False):
        state = useful_state(state_as_dict,self.denominator)
        state = state.reshape([1,state.size])
        action_index = self.get_action(state_as_dict,state)

        # Creates the action "object". If we use "print(action)" then it is readable.
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list,
                                         action_index,None, None)
        # print(action)
        # self.action_list.append(action)
        return action

    def create_action_dict(self,gen_action_list, load_action_list, line_or_action_list, line_ex_action_list, action_index, episode, flag):
        action = self.action_space(
        {"change_bus": {"generators_id": gen_action_list[action_index], "loads_id": load_action_list[action_index],
                        "lines_or_id": line_or_action_list[action_index],
                        "lines_ex_id": line_ex_action_list[action_index]}})
        return action

    def get_action(self, state_as_dict, state):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # Predict the action using the internal neural network
                policy = self.actor.predict(state,batch_size=1).flatten()

        # Select first 4 best possible actions from the neural nets.
        policy_chosen_list = np.random.choice(self.action_size, 4, p=policy)

        # Simulate the impact of these actions and pick the best action that maximizes the reward.
        # (one-step lookahead).
        action_index = policy_chosen_list[0]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, None, None)
        obs_0, rw_0, done_0, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[1]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, None, None)
        obs_1, rw_1, done_1, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[2]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, None, None)
        obs_2, rw_2, done_2, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[3]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, None, None)
        obs_3, rw_3, done_3, _  = state_as_dict.simulate(action)

        return policy_chosen_list[np.argmax([rw_0,rw_1,rw_2,rw_3])]

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=self.actor_lr)
        # updates = optimizer.get_updates(params=self.actor.trainable_weights, constraints=[],loss=actor_loss)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=actor_loss)
        train = K.function([self.actor.input, action, advantages], tf.compat.v1.convert_to_tensor([]),updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        # updates = optimizer.get_updates(params=self.critic.trainable_weights, constraints=[],loss=loss)
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        train = K.function([self.critic.input, discounted_reward], tf.compat.v1.convert_to_tensor([]), updates=updates)
        return train

    # make agents(local) and start training
    def train(self, nn_weights_name):
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.env_name, self.discount_factor,
                        self.action_size, self.state_size, self.action_space_lists, self.profiles_chronics, self.sess) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        while (len(scores) < EPISODES_train):
            time.sleep(200) # main thread saves the model every 200 sec
            print("len(scores) = ", len(scores))
            if (len(scores)>10):
                self.save_model(nn_weights_name,self.save_path)
                print("_______________________________________________________________________________________________________")
                print("saved NN model at episode", episode, "\n")
                print("_______________________________________________________________________________________________________")

    def load_model(self, nn_weight_name, load_path):
            self.actor.load_weights(os.path.join(load_path,nn_weight_name + "_actor.h5"))
            self.critic.load_weights(os.path.join(load_path,nn_weight_name + "_critic.h5"))

    def save_model(self, nn_weight_name, save_path):
        self.actor.save_weights(os.path.join(save_path,nn_weight_name + "_actor.h5"))
        self.critic.save_weights(os.path.join(save_path,nn_weight_name + "_critic.h5"))

# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, discount_factor, action_size, state_size, action_space_lists,profiles_chronics,session):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        self.session = session

        self.gen_action_list = action_space_lists[0]
        self.load_action_list = action_space_lists[1]
        self.line_or_action_list = action_space_lists[2]
        self.line_ex_action_list = action_space_lists[3]
        self.profiles_chronics = profiles_chronics

    # Thread interactive with environment
    def run(self):
        global episode
        episode = 0
        env = user_environment_make.set_environement(self.index,self.env_name,self.profiles_chronics)
        self.action_space = env.helper_action_player
        while episode < EPISODES_train:
            state = env.reset()
            state_as_dict = copy.deepcopy(state)
            # state = copy.deepcopy(np.reshape(state.to_vect(), [1, self.state_size]))
            state = useful_state(state_as_dict,env.backend.prod_pu_to_kv)
            state = state.reshape([1,state.size])
            score = 0
            time_step = 0
            max_action = 0
            non_zero_actions = 0
            epsilon = 0.5
            while True:
                # Decaying epsilon greedy. Not the best one. This agent needs a better exploration strategy to help
                # it learn to perform well.
                if np.random.random() < epsilon*(1/(episode/400+1)):
                    action_index = int(np.random.choice(self.action_size))
                    epison_flag = True
                else:
                    epison_flag = False
                    if time_step%1 == 0:# or max(state_as_dict.rho)>0.75:
                        action_index = self.get_action(state_as_dict,state)
                    else:
                        action_index = 0

                action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, episode, flag=1)

                # print(action)

                next_state, reward, done, flag = env.step(action)
                state_as_dict = copy.deepcopy(next_state)
                time_hour = state_as_dict.day*10000 + state_as_dict.hour_of_day * 100+ state_as_dict.minute_of_hour
                # next_state = np.reshape(next_state.to_vect(), [1, self.state_size]) if not done else np.zeros([1, self.state_size])
                next_state = useful_state(next_state,env.backend.prod_pu_to_kv)
                next_state = next_state.reshape([1,next_state.size])
                # next_state = observation_space.array_to_observation(next_state).as_minimalist().as_array()
                # score += (reward-0.1*(next_state[1]*next_state[1]+next_state[3]*next_state[3])) # reducing the reward based on speed...
                score += reward if not done else -100*(1+np.sqrt(episode)/10)
                non_zero_actions += 0 if action_index==0 else 1
                # if flag == None:
                #     self.memory(state, action, reward)
                # else:
                #     score -= 10 if flag.is_empty else 0
                #     self.memory(state, action, 0)
                self.memory(state, action_index, reward if not done else -100*(1+np.sqrt(episode)/10))

                state = copy.deepcopy(next_state) if not done else np.zeros([1, self.state_size])

                time_step += 1
                max_action = max(max_action,action_index)

                if done or time_step > time_step_end:
                    if done:
                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    if time_step > time_step_end:
                        print("End Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    # global scores
                    scores.append(score)
                    # print(len(scores))
                    # global episode
                    episode += 1
                    # if len(self.states) == 0:
                    #     k = 1
                    self.train_episode(True)  # max score = 80000
                    break
                else:
                    if time_step % 10 ==0:
                        print("Continue Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with recent time:", time_step, "/ with recent action", action_index,"/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ max_action so far:", max_action)
                        self.train_episode(False)

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            with self.session.as_default():
                with self.session.graph.as_default():
                    running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state[0])
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)
        with self.session.as_default():
            with self.session.graph.as_default():
                values = self.critic.predict(np.array(self.states))[0]
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        with self.session.as_default():
            with self.session.graph.as_default():
                state_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.states))
                action_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.actions))
                advantage_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(advantages))
                # reshaped_state_as_tensor = tf.compat.v1.reshape(state_as_tensor,self.actor.input.shape.dims)
                self.optimizer[0]([state_as_tensor, action_as_tensor, advantage_as_tensor])
                self.optimizer[1]([state_as_tensor, discounted_rewards])
                self.states, self.actions, self.rewards = [], [], []


    def get_action(self, state_as_dict, state):
        with self.session.as_default():
            with self.session.graph.as_default():
                # Predict the action using the internal neural network
                policy = self.actor.predict(state,batch_size=1).flatten()

        # Select first 4 best possible actions from the neural nets.
        policy_chosen_list = np.random.choice(self.action_size, 4, p=policy)

        # Simulate the impact of these actions and pick the best action that maximizes the reward.
        # (one-step lookahead).
        action_index = policy_chosen_list[0]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, episode, flag=0)
        obs_0, rw_0, done_0, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[1]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, episode, flag=0)
        obs_1, rw_1, done_1, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[2]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, episode, flag=0)
        obs_2, rw_2, done_2, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[3]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index, episode, flag=0)
        obs_3, rw_3, done_3, _  = state_as_dict.simulate(action)

        return policy_chosen_list[np.argmax([rw_0,rw_1,rw_2,rw_3])]

    def create_action_dict(self,gen_action_list, load_action_list, line_or_action_list, line_ex_action_list, action_index, episode, flag):
        action = self.action_space(
        {"change_bus": {"generators_id": gen_action_list[action_index], "loads_id": load_action_list[action_index],
                        "lines_or_id": line_or_action_list[action_index],
                        "lines_ex_id": line_ex_action_list[action_index]}})
        # if flag == 1:
        #     print("_______________________________________________________________________________________________________")
        #     print("thread number ", self.index," Executed action index in environment step:", action_index," at episode:", episode)
        #     # print(action)
        #     print("_______________________________________________________________________________________________________")
        # else:
        #     print("_______________________________________________________________________________________________________")
        #     print("thread number ", self.index," Executed action index at simulate step:", action_index," at episode:", episode)
        #     # print(action)
        #     print("_______________________________________________________________________________________________________")
        return action

# def set_environement(start_id,env_name,profiles_chronics):
#     param = Parameters()
#     param.NO_OVERFLOW_DISCONNECTION = True
#
#     env = make(env_name,chronics_path= profiles_chronics, reward_class=CombinedReward,param=param)
#     # Register custom reward for training
#     cr = env.reward_helper.template_reward
#     cr.addReward("overflow", CloseToOverflowReward(), 50.0)
#     cr.addReward("game", GameplayReward(), 100.0)
#     cr.initialize(env)
#
#     # Debug prints
#     print("Debug prints --->:")
#     print("Chronics location that being used:", env.chronics_handler.path)
#     print("Grid location being used:", env.init_grid_path)
#     print("Reward class that is being used:", env.rewardClass)
#     print("Action type class being used:", env.actionClass)
#     print("Observation type class being used:", env.observationClass)
#     print("Backend CSV file key names:", env.names_chronics_to_backend)
#     print("Legal action class being used:", env.legalActClass)
#     print("Voltage controller class being used:", env.voltagecontrolerClass)
#
#     if start_id != None:
#         env.chronics_handler.tell_id(start_id)
#         print("Thread number:",start_id,", ID of chronic current folder:",env.chronics_handler.real_data.id_chron_folder_current)
#     return env

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




