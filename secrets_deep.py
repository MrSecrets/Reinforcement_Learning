import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

# env = gym.make('CartPole-v0').unwrapped

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# training and hyper parameters
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.999
EPSILON_START = 1
EPSILON_END = 0.00
EPSILON_DECAY = 200
TARGET_UPDATE = 10
EPISODES = 300
episode_durations = []
AGGREGATE_STATS_EVERY = 1 
MODEL_NAME = '2x256'
steps_done = 0
ep_rewards = [-200]
epsilon = EPSILON_START
MIN_REWARD = -200


env = gym.make('CartPole-v0')
env.reset()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ModifiedTensorBoard(TensorBoard):

	# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

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
		self._write_logs(stats, self.step)


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class DQNAgent:
	
	def __init__(self,screen,n_actions,memory):
		
		self.policy_model = self.create_model(screen)
		self.target_model = self.create_model(screen)

		self.target_model.set_weights(self.policy_model.get_weights())

		self.replay_memory = memory

		self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
		self.target_update_counter = 0

	def create_model(self,screen):

		model = Sequential()

		model.add(Conv2D(256, (3, 3), input_shape=(screen.shape)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(256, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(64))

		model.add(Dense(env.action_space.n, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
		model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
		return model


	def train(self, terminal_state, step):

		if len(memory)<BATCH_SIZE:
			return

		minibatch = memory.sample(BATCH_SIZE)


		current_states = np.array([transition[0] for transition in minibatch])/255
		current_qs_list = self.mpdel.predict(current_states)
		new_current_states = np.array([transition[3] for transition in minibatch])/255
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		Y = []

		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = rward+ DISCOUNT * max_future_q

	def get_qs(self, state):
		return self.policy_model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


def get_object_location(screen_width):
	
	world_width = env.x_threshold*2
	scale = screen_width/world_width
	return int(env.state[0] * scale +screen_width / 2.0)

def get_screen():
	
	screen = env.render(mode='rgb_array')
	screen_height, screen_width,_ = screen.shape
	object_location = get_object_location(screen_width)
	return screen

def select_action(state):
	global steps_done
	sample = np.random.random()
	epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END)*math.exp(-1.*steps_done/EPSILON_DECAY) 
	epsilon = epsilon_threshold
	steps_done += 1
	if sample > epsilon_threshold:
		action = np.argmax(Agent.get_qs(state))
	else:
		action = np.random.randint(0, n_actions)

	return action

# def plot_durations():
	# plt.figure(2)
	# plt.clf()
	# durations_t = episode_durations
	# plt.title('Training...')
	# plt.xlabel('Episode')
	# plt.ylabel('Duration')
	# plt.plot(durations_t.numpy())
	# # Take 100 episode averages and plot them too
	# if len(durations_t) >= 100:
	# 	means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
	# 	means = torch.cat((torch.zeros(99), means))
	# 	plt.plot(means.numpy())

	# plt.pause(0.001)  # pause a bit so that plots are updated
	# if is_ipython:
	# 	display.clear_output(wait=True)
	# 	display.display(plt.gcf())



# env.reset()
# plt.figure()
# plt.imshow(get_screen(), interpolation='none')
# plt.title('Example extracted screen')
# plt.show()



init_screen = get_screen()
print("_______________________________________>>>>>>>>>>>>>>>>>>>>>>>>>>>........................appnvd xvbsvdnjdbfk")
print("..........")
screen_height, screen_width, _ = init_screen.shape
n_actions = env.action_space.n
memory = ReplayMemory(10000)

Agent = DQNAgent(init_screen,n_actions,memory)


# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

	env.reset()
	# Update tensorboard step every episode
	Agent.tensorboard.step = episode

	last_screen = get_screen()
	current_screen = get_screen()
	state = current_screen - last_screen
	# Restarting episode - reset episode reward and step number
	episode_reward = 0
	step = 1

	for t in count():
		# selecta and perform an aaction
		action = select_action(state)
		_,reward,done,_ = env.step(action)
		last_screen = current_screen
		current_screen = get_screen()
		if not done:
			next_state = current_screen - last_screen
		else:
			next_state = None

		memory.push(state, action, next_state, reward)

		state = next_state

		Agent.train(done,step)
		if done:
			episode_durations.append(t+1)
			# plot_durations()
			break

		step += 1
		ep_rewards.append(episode_reward)
		if not episode % AGGREGATE_STATS_EVERY or episode == 1:
			average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
			min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
			Agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

			# Save model, but only when min reward is greater or equal a set value
			if min_reward >= MIN_REWARD:
				Agent.policy_model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')



print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()


