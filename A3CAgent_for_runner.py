import grid2op
import numpy as np
from grid2op.Agent import MLAgent
from grid2op.Environment import Environment

import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


# Create the Agent instance here that can used with the Runner to test the performance of the trained RL agent.
class A3CAgent(MLAgent):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes) We will use this template to create our desired ML agent with unique neural network configuration.
    def __init__(self, action_space, state_size, action_size,env,action_space_lists):
        MLAgent.__init__(self, action_space)
        # Parameter settings.
        # NOTE: MAKE SURE THE FOLLOWING SETTINGS ARE SAME AS THE TRAINED AGENT OR THE WEIGHTS WONT LOAD SUCCESSFULLY.
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.denominator = env.backend.prod_pu_to_kv

        # these are hyper parameters for the A3C
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.discount_factor = .95
        self.hidden1, self.hidden2 = 200, 100
        self.action_list = []

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.gen_action_list = action_space_lists[0]
        self.load_action_list = action_space_lists[1]
        self.line_or_action_list = action_space_lists[2]
        self.line_ex_action_list = action_space_lists[3]

    def act(self, state_as_dict, reward, done=False):
        state = self.useful_state(state_as_dict)
        state = state.reshape([1,state.size])
        action_index = self.get_action(state_as_dict,state)

        # Creates the action "object". If we use "print(action)" then it is readable.
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list,
                                         action_index)
        # print(action)
        self.action_list.append(action)
        return action

    def load_model(self, nn_weight_name):
            self.actor.load_weights(nn_weight_name + "_actor.h5")
            self.critic.load_weights(nn_weight_name + "_critic.h5")

    def create_action_dict(self,gen_action_list, load_action_list, line_or_action_list, line_ex_action_list,
                           action_index):
        action = self.action_space(
        {"change_bus": {"generators_id": gen_action_list[action_index], "loads_id": load_action_list[action_index],
                        "lines_or_id": line_or_action_list[action_index],
                        "lines_ex_id": line_ex_action_list[action_index]}})
        return action

    def get_action(self, state_as_dict, state):
        # Predict the action using the internal neural network
        policy = self.actor.predict(state,batch_size=1).flatten()

        # Select first 4 best possible actions from the neural nets.
        policy_chosen_list = np.random.choice(self.action_size, 4, p=policy)
        # policy_chosen_list[3] = 0

        # Simulate the impact of these actions and pick the best action that maximizes the reward.
        # (one-step lookahead).
        action_index = policy_chosen_list[0]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index)
        obs_0, rw_0, done_0, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[1]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index)
        obs_1, rw_1, done_1, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[2]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index)
        obs_2, rw_2, done_2, _  = state_as_dict.simulate(action)

        action_index = policy_chosen_list[3]
        action = self.create_action_dict(self.gen_action_list, self.load_action_list, self.line_or_action_list, self.line_ex_action_list, action_index)
        obs_3, rw_3, done_3, _  = state_as_dict.simulate(action)

        return policy_chosen_list[np.argmax([rw_0,rw_1,rw_2,rw_3])]


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

    # This also should be same as that of the "train.py". This below function reduces the size of the state space.
    def useful_state(self,obs):
        selected_obs = np.hstack((obs.topo_vect,obs.line_status))
        selected_obs = np.hstack((selected_obs,obs.load_p/100))#
        selected_obs = np.hstack((selected_obs,obs.load_q/100))
        selected_obs = np.hstack((selected_obs,obs.prod_p/100))
        selected_obs = np.hstack((selected_obs,obs.prod_v/self.denominator))
        selected_obs = np.hstack((selected_obs,obs.rho))
        # selected_obs = np.hstack((selected_obs,obs.day))
        selected_obs = np.hstack((selected_obs,obs.hour_of_day/24))
        selected_obs = np.hstack((selected_obs,obs.minute_of_hour/60))
        # selected_obs = np.hstack((selected_obs,obs.day_of_week/7))
        return selected_obs
