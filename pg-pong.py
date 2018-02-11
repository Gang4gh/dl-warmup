#!/usr/bin/env python3
""" Train an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import sys, time, pickle, argparse
import numpy as np
import gym

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

save_file = 'W-{}.bin'.format(args.expname)

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = not args.reset
render = args.render

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
	model = pickle.load(open(save_file, 'rb'))
else:
	# "Xavier" initialization
	model = {}
	model['W1'] = np.random.randn(H, D) / np.sqrt(D/2)
	model['W2'] = np.random.randn(H) / np.sqrt(H/2)

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v)
                 for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
	# sigmoid "squashing" function to interval [0,1]
	return 1.0 / (1.0 + np.exp(-x))


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
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		if r[t] != 0:
			# reset the sum, since this was a game boundary (pong specific!)
			running_add = 0
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h < 0] = 0  # ReLU nonlinearity
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)
	return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
	""" backward pass. (eph is array of intermediate hidden states) """
	dW2 = np.dot(eph.T, epdlogp).ravel()
	dh = np.outer(epdlogp, model['W2'])
	dh[eph <= 0] = 0  # backpro prelu
	dW1 = np.dot(dh.T, epx)
	return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
if args.seed is not None:
	env.seed(args.seed)
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
	if render:
		env.render()
		time.sleep(0.01)

	# preprocess the observation, set input to network to be difference image
	cur_x = prepro(observation)
	x = cur_x - prev_x if prev_x is not None else np.zeros(D)
	prev_x = cur_x

	# forward the policy network and sample an action from the returned probability
	aprob, h = policy_forward(x)
	action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

	# record various intermediates (needed later for backprop)
	xs.append(x)  # observation
	hs.append(h)  # hidden state
	y = 1 if action == 2 else 0  # a "fake label"
	# grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
	dlogps.append(y - aprob)

	# step the environment and get new measurements
	observation, reward, done, info = env.step(action)
	reward_sum += reward

	# record reward (has to be done after we call step() to get reward for previous action)
	drs.append(reward)

	#if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
	#	print('-' if reward == -1 else '+', end='', flush=1)

	if done:  # an episode finished
		episode_number += 1

		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		epx = np.vstack(xs)
		eph = np.vstack(hs)
		epdlogp = np.vstack(dlogps)
		epr = np.vstack(drs)
		xs, hs, dlogps, drs = [], [], [], []  # reset array memory

		# compute the discounted reward backwards through time
		discounted_epr = discount_rewards(epr)
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		discounted_epr -= np.mean(discounted_epr)
		discounted_epr /= np.std(discounted_epr)

		# modulate the gradient with advantage (PG magic happens right here.)
		epdlogp *= discounted_epr
		grad = policy_backward(eph, epdlogp)
		for k in model:
			grad_buffer[k] += grad[k]  # accumulate grad over batch

		# perform rmsprop parameter update every batch_size episodes
		if episode_number % batch_size == 0:
			for k, v in model.items():
				g = grad_buffer[k]  # gradient
				rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
				model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
				grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

		# boring book-keeping
		running_reward = reward_sum if running_reward is None else running_reward * \
			0.99 + reward_sum * 0.01
		print('ep %d: total_reward: %f, moving_average_reward: %f' %
		      (episode_number, reward_sum, running_reward))
		if episode_number % 300 == 0:
			pickle.dump(model, open(save_file, 'wb'))
		
		reward_sum = 0
		observation = env.reset() # reset env
		prev_x = None

		if episode_number == args.max_episode:
			pickle.dump(model, open(save_file, 'wb'))
			break
