# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:02:25 2020

@author: dcamp

Code mainly taken from: https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/
Modified (optimized) based on the following video tutorials:
https://www.youtube.com/watch?v=t3fbETsIBCY&ab_channel=sentdex (DQN part 1)
https://www.youtube.com/watch?v=qfovbG84EBg&ab_channel=sentdex (DQN part 2)
# Refined based on https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
"""

# Import the necessary libraries
import numpy as np
import random
import copy
import timeit

# from IPython.display import clear_output

# deque is employed for the replay memory trick
from collections import deque

# import progressbar

import gym
import tensorflow as tf
import time
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

# Generating the blueprint for the agent
class Agent:
    def __init__(self, enviroment, optimizer, replay_memory_size, N_episodes, model_name, new_networks=True):
        self._network_name = model_name
        # Initialize atributes
        self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        self._n_episodes = N_episodes
        # Number of steps (actions) for training the network
        self._train_net_every = 4
        # Number of chosen actions between updating the target network (1% of total actions). 
        self._update_target_net_every = 200         
         
        self.experience_replay = deque(maxlen=replay_memory_size)
        self.current_episode = 0
        # Initialize discount (gamma) and exploration rate/probability (epsilon)
        self.gamma = 0.95  # 0.6
        self.eps_initial = 1
        self.epsilon_final = 0.1
        self.epsilon = copy.deepcopy(self.eps_initial)

        # In the first 5% of observations epsilon remains as 1 (proposed by deepmind), from that number the replay memory goes into play
        self.n_episodes_with_costant_epsilon = int(self._n_episodes * 0.05)
        self.slope_epsilon = (self.eps_initial - self.epsilon_final) / (self.n_episodes_with_costant_epsilon - self._n_episodes)
        self.intercept_epsilon = self.eps_initial - self.slope_epsilon*self.n_episodes_with_costant_epsilon
        # self.epsilon_decay = 0.9965

        if new_networks:
            # Main networks, it gets trains at each step
            self.q_network = self._build_compile_model()

            # Target network, it sets the target to be compared against at every step (it is updated after a given time defined by the programmer)
            self.target_network = self._build_compile_model()
        else:
            # Load already trained networks
            self.q_network = load_model(self._network_name + '.h5')
            self.target_network = load_model(self._network_name + '.h5')

        # Align weights of both networks
        self.alighn_target_model()

        # Prevent tf from creating tons of log files while learning
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{model_name}-{int(time.time())}")
        self.target_update_counter = 0

    def epsilon_update(self):
        """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
        if self.epsilon > self.epsilon_final:
            self.epsilon  = self.slope_epsilon*self.current_episode + self.intercept_epsilon 
        else:
            self.epsilon = self.epsilon_final
            
    def store(self, state, action, reward, next_state, terminated):
        # Save already performed stages of the agent while learning
        self.experience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        # Represent the number of states in self._state_size by 10 values
        model.add(Embedding(self._state_size, 10))
        # Prepare data for feed-forward neural network with three Dense layers.
        model.add(Reshape((10,)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return enviroment.action_space.sample()

        # Explotation
        # start = timeit.default_timer()
        q_values = self.q_network.predict(state)
        # stop = timeit.default_timer()
        # print('Time_eplotation: ', stop - start)

        return np.argmax(q_values[0])
    
    def get_current_episode(self):
        self.current_episode += 1 
        
    def retrain(self, terminal_state, batch_size, cur_step):
        #  Pick random samples from the experience replay memory and train the Q-Network
        minibatch = random.sample(self.experience_replay, batch_size)

        current_states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        current_states = current_states.reshape(-1, 1)
        # start = timeit.default_timer()
        current_qs_list = self.q_network.predict(current_states)  # called 'target' before
        # stop = timeit.default_timer()
        # delay_1 = stop - start

        future_states = np.array([transition[3] for transition in minibatch],
                                 dtype=np.float32)  # called 't' before
        future_states = future_states.reshape(-1, 1)
        # start = timeit.default_timer()
        future_qs_list = self.target_network.predict(future_states)
        # stop = timeit.default_timer()
        # delay_2 = stop - start

        X = []
        y = []

        for index, (state, action, reward, next_state, terminated) in enumerate(minibatch):
            if terminated:
                new_q = reward
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q

            # Update based on the second part of the Bellman equation
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state)
            y.append(current_qs)

        # Retrain q_network based on already seen data when the agent generates a completion
        X = np.array(X)
        X = X.reshape(-1, 1)
        
              
        # When a successful execution is performed, the network is trained
        # if terminal_state and self.current_episode >= self.n_episodes_with_costant_epsilon:
        #     print('Completion done!')
        #     # self.q_network.fit(np.array(X), np.array(y), batch_size=batch_size,
        #     #                 verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        #     # start = timeit.default_timer()
        #     self.q_network.fit(np.array(X), np.array(y), batch_size=batch_size,
        #                        verbose=0, shuffle=False)
        #     # stop = timeit.default_timer()
        #     # print('Time_training: ', stop - start)

        if cur_step % self._train_net_every == 0 and self.current_episode >= self.n_episodes_with_costant_epsilon:
            # print('fitting network...')
            self.q_network.fit(np.array(X), np.array(y), batch_size=batch_size,
                               verbose=0, shuffle=False)
            
        if cur_step % self._update_target_net_every == 0 and self.current_episode >= self.n_episodes_with_costant_epsilon:   
            self.alighn_target_model()
            # Save the network
            self.q_network.save(self._network_name + '.h5')
            # Command for loading the network (in case of need)
            # q_network_saved = load_model(self._network_name + '.h5')
            
            # Saving relevant parameters of the training process
            current_parameters= (self.epsilon, self.experience_replay, self.current_episode, cur_step)
            file = open('current_parameters.pkl', 'wb')
            pickle.dump(current_parameters, file)
            file.close()
            # Command for loading the current parameters (in case of need)
            # fileo = open('current_parameters.pkl', 'rb')
            # current_parameters_saved = pickle.load(fileo)
            # fileo.close()

# Own Tensorboard class (we do not want to create a new log file each time .fit is performed)
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            tf.summary.scalar(stats, self.step)

        # self._write_logs(stats, self.step)


# Initialize the training properties and visualize DQN properties
BATCH_SIZE = 32
NUM_OF_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 40
REPLAY_MEMORY_SIZE = 20_000
LEARNING_RATE = 0.001
# At least given number of steps should be done before start to fitting
MIN_REPLAY_MEMORY_SIZE = BATCH_SIZE*2
MODEL_NAME = 'Example_1'

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Initialize the environment
enviroment = gym.make("Taxi-v3").env

print('Number of states: {}'.format(enviroment.observation_space.n))
print('Number of actions: {}'.format(enviroment.action_space.n))

# Create an instance of the class
optimizer = Adam(learning_rate=LEARNING_RATE)

# Set new_network=True in case you want to start the training from scratch
agent = Agent(enviroment, optimizer, REPLAY_MEMORY_SIZE, NUM_OF_EPISODES, MODEL_NAME, new_networks=False)
agent.q_network.summary()

# Total number of actions taken
cur_step = 0
succesful_arrivals = 0
succesful_episode = []

# Number of trials we are going to consider
for episode in range(0, NUM_OF_EPISODES):
    print(f'{episode+1} out of {NUM_OF_EPISODES}')
    
    # Set the seed for reproducing results (Overfit the same environment layout)
    enviroment.seed(93)
    
    # Reset the enviroment
    state = enviroment.reset()
    state = np.reshape(state, [1, 1])

    # Taxi-v3 environment has 500 states and 6 [0 to 5] possible actions
    # enviroment.render()
    
    agent.get_current_episode()
    if  agent.current_episode > agent.n_episodes_with_costant_epsilon:
        agent.epsilon_update()
    
    # Initialize variables
    reward = 0
    terminated = False

    # bar = progressbar.ProgressBar(maxval=MAX_STEPS_PER_EPISODE/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    start = timeit.default_timer()
    for timestep in range(MAX_STEPS_PER_EPISODE):
        # Run Action
        action = agent.act(state)
        cur_step += 1

        # Take action
        next_state, reward, terminated, info = enviroment.step(action)
        next_state = np.reshape(next_state, [1, 1])

        agent.store(state, action, reward, next_state, terminated)

        state = next_state

        if len(agent.experience_replay) > MIN_REPLAY_MEMORY_SIZE:

            # start = timeit.default_timer()
            agent.retrain(terminated, BATCH_SIZE, cur_step)
            # stop = timeit.default_timer()
            # print('Time_retraining: ', stop - start)

        if terminated:
            print('Completion of the task!')
            agent.q_network.save(agent._network_name + '.h5')
            current_parameters= (agent.epsilon, agent.experience_replay, agent.current_episode, cur_step)
            file = open('current_parameters.pkl', 'wb')
            
            succesful_episode.append(episode)
            succesful_arrivals += 1
            break
    
    stop = timeit.default_timer()
    #     if timestep%10 == 0:
    #         bar.update(timestep/10 + 1)
    #         print('\n')

    # bar.finish()
    print(f"Sucessful tasks: {succesful_arrivals} \t Current epsilon: {agent.epsilon}")
    print(f"Time elapsed in episode: {stop - start} (sec)")
    print('\n')
    # enviroment.render()

agent.q_network.save(agent._network_name + '.h5')
current_parameters= (agent.epsilon, agent.experience_replay, agent.current_episode, cur_step)
file = open('current_parameters.pkl', 'wb')

###############################################
# Testing based on a greedy policy
##############################################
from tensorflow.keras.models import load_model
import gym 
import numpy as np
MODEL_NAME = 'Example_1'
MAX_STEPS_TESTING = 100
enviroment = gym.make("Taxi-v3").env
enviroment.seed(93)
state = enviroment.reset()
enviroment.render()
state = np.reshape(state, [1, 1])
terminated = False
q_network_saved = load_model(MODEL_NAME + '.h5')

for i in range(MAX_STEPS_TESTING):
    q_values = q_network_saved.predict(state)
    action = np.argmax(q_values[0])
    next_state, reward, terminated, info = enviroment.step(action)
    state = np.reshape(next_state, [1, 1])
    enviroment.render()
    if terminated:
        print('Successful completion! The network has learned!!!!!')
        break
