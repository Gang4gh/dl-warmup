#!/usr/bin/env python3
""" Train an agent with (stochastic) Policy Gradients on Pong using TensorFlow and OpenAI gym """
import sys, time, pickle, argparse
import numpy as np
import gym
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(description='Play with PG algorithm on Pong')
parser.add_argument('-s', '--seed', type=int, help='random seed used by numpy and gym')
parser.add_argument('-r', '--reset', action="store_true", help='reset model (not resume from previous checkpoint)')
parser.add_argument('-d', '--render', action="store_true", help='render game during training')
parser.add_argument('--max-episode', type=int, default=sys.maxsize, help='count of episode to halt the training')
parser.add_argument('--expname', default='save', help='a tag used to save/resume models')
args = parser.parse_args()

print('arguments: ', args.__dict__)
if args.seed is not None:
	np.random.seed(args.seed)
tf.set_random_seed(args.seed)#np.random.randint(12345))
env = gym.make("Pong-v0")
if args.seed is not None:
	env.seed(args.seed)#np.random.randint(12345))

# hyperparameters and global variable
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = not args.reset
save_file = 'W-{}.bin'.format(args.expname)

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
model = {}
model['W1'] = np.random.randn(H, D) / np.sqrt(D)
model['W2'] = np.random.randn(H) / np.sqrt(H)

#with tf.device('/cpu:0'):
X = tf.placeholder(tf.float64, shape=[None,D])
Y = tf.placeholder(tf.float64, shape=[None])
R = tf.placeholder(tf.float64, shape=[None])
W1 = tf.Variable(model['W1'].T, dtype=tf.float64) # tf.random_normal((D, H)))
W2 = tf.Variable(model['W2'], dtype=tf.float64) # tf.random_normal((H,)))
W1beta2 = tf.Variable(tf.zeros([D,H], dtype=tf.float64)) # tf.random_normal((D, H)))
W2beta2 = tf.Variable(tf.zeros([H], dtype=tf.float64)) # tf.random_normal((H,)))

H1 = tf.nn.relu(tf.matmul(X, W1))
Y0 = tf.nn.sigmoid(tf.tensordot(H1, W2, 1))
G = (Y - Y0) * R

dW2 = tf.tensordot(G, H1, (0,0))
T1 = tf.expand_dims(W2, 1) * tf.expand_dims(G, 0)
Mask = tf.greater(H1, 0)
Zeros = tf.zeros_like(H1)
T2 = tf.where(Mask, tf.transpose(T1), Zeros)
dW1 = tf.tensordot(X, T2, (0,0))
# loss = tf.reduce_mean(0.5 * tf.multiply(tf.square(Y - Y0), R))
# dW1, dW2 = tf.gradients(loss, [W1, W2])

nW1beta2 = W1beta2.assign(W1beta2 * decay_rate + (1-decay_rate) * tf.square(dW1))
nW2beta2 = W2beta2.assign(W2beta2 * decay_rate + (1-decay_rate) * tf.square(dW2))
nW1 = W1.assign(W1 + learning_rate * dW1 / (tf.sqrt(nW1beta2) + 1e-5))
nW2 = W2.assign(W2 + learning_rate * dW2 / (tf.sqrt(nW2beta2) + 1e-5))
update = tf.group(nW1, nW2)


def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195]  # crop
	I = I[::2, ::2, 0]  # downsample by factor of 2
	I[I == 144] = 0  # erase background (background type 1)
	I[I == 109] = 0  # erase background (background type 2)
	I[I != 0] = 1  # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()


def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	current = 0
	for t in reversed(range(0, len(r))):
		if r[t] != 0:
			# reset the reward, since this was a game boundary (pong specific!)
			current = r[t]
		else:
			current *= gamma
			r[t] = current


start_time = time.time()
observation = env.reset()
prev_x = None  # used in computing the difference frame
X_list, Y_list, R_list, R_episode = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while True:
		if args.render:
			env.render()
			time.sleep(0.01)

		# preprocess the observation, set input to network to be difference image
		cur_x = prepro(observation)
		x = cur_x - prev_x if prev_x is not None else np.zeros(D)
		prev_x = cur_x

		# sample an action from current policy
		tfy0, = sess.run([Y0], feed_dict={X: [x]})
		action = 2 if np.random.uniform() < tfy0[0] else 3

		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward

		X_list.append(x)
		Y_list.append(1 if action == 2 else 0)
		R_episode.append(reward)

		if done:  # an episode finished
			episode_number += 1

			# compute the discounted reward backwards through time
			discount_rewards(R_episode)
			# standardize the rewards to be unit normal (helps control the gradient estimator variance)
			R_episode -= np.mean(R_episode)
			R_episode /= np.std(R_episode)
			R_list.append(R_episode)
			R_episode = []

			# perform parameter update every {batch_size} episodes
			if episode_number % batch_size == 0:
				print('len(X_list) :', len(X_list), flush=1)
				_ = sess.run([update], feed_dict={X:X_list, Y:Y_list, R:np.concatenate(R_list)})
				X_list, Y_list, R_list = [], [], []

			# boring book-keeping
			running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			time_cost = int(time.time() - start_time)
			print('ep %d: totalReward: %f, averageReward: %f, time: %d' % (episode_number, reward_sum, running_reward, time_cost), flush=1)
			
			reward_sum = 0
			observation = env.reset() # reset env
			prev_x = None
