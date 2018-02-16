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
if not resume:
	model = pickle.load(open(save_file, 'rb'))
else:
	# "Xavier" initialization
	model = {}
	model['W1'] = np.random.randn(H, D) / np.sqrt(D)
	model['W2'] = np.random.randn(H) / np.sqrt(H)

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

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
	t1 = dh.copy()
	dh[eph <= 0] = 0  # backpro prelu
	t2 = dh.copy()
	dW1 = np.dot(dh.T, epx)
	return {'W1': dW1, 'W2': dW2}, t1, t2


def compare_arries(a, b, msg):
	print('compare array: a.shape={}, b.shape={}'.format(np.shape(a), np.shape(b)))
	if (abs(a - b) < 1e-4).all():
		print(msg, ':', 'same')
	else:
		print(msg, ':', 'different')
		sys.exit(-1)


start_time = time.time()
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs, ys = [], [], [], [], []
exp1, ys1, discounted_epr1 = [], [], []
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

		# forward the policy network and sample an action from the returned probability
		# aprob, h = policy_forward(x)
		tfy0, = sess.run([Y0], feed_dict={X: [x]})
		# if (abs(aprob - tfy0) > 1e-4).all():
		# 	print(aprob, tfy0, abs(aprob - tfy0) < 1e-4)
		# 	sys.exit(-1)
		# action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
		action = 2 if np.random.uniform() < tfy0[0] else 3  # roll the dice!

		# record various intermediates (needed later for backprop)
		xs.append(x)  # observation
		#hs.append(h)  # hidden state
		y = 1 if action == 2 else 0  # a "fake label"
		# grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
		#dlogps.append(y - aprob) # * aprob * (1 - aprob))
		ys.append(y)

		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward

		# record reward (has to be done after we call step() to get reward for previous action)
		drs.append(reward)

		if done:  # an episode finished
			episode_number += 1

			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			epx = np.vstack(xs)
			# eph = np.vstack(hs)
			# epdlogp = np.vstack(dlogps)
			epr = np.vstack(drs)
			epy = np.vstack(ys)
			
			# compute the discounted reward backwards through time
			discounted_epr = discount_rewards(epr)
			# standardize the rewards to be unit normal (helps control the gradient estimator variance)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr /= np.std(discounted_epr)

			exp1.extend(xs)
			ys1.extend(ys)
			discounted_epr1.extend(discounted_epr.reshape(-1))

			# modulate the gradient with advantage (PG magic happens right here.)
			# epdlogp *= discounted_epr
			# grad, t1, t2 = policy_backward(eph, epdlogp)
			# for k in model:
			# 	grad_buffer[k] += grad[k]  # accumulate grad over batch

			# perform rmsprop parameter update every batch_size episodes
			if episode_number % batch_size == 0:
				# dW1a = np.copy(grad_buffer['W1'])
				# dW2a = np.copy(grad_buffer['W2'])
				# for k, v in model.items():
				# 	g = grad_buffer[k]  # gradient
				# 	rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
				# 	model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
				# 	grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

				#discounted_epr_reshape = discounted_epr1.reshape(-1)
				#H1_val, G_val, dW2_val, dW1_val, nW1beta2_val, nW2beta2_val, nW1_val, nW2_val, _ = sess.run([H1, G, dW2, dW1, nW1beta2, nW2beta2, nW1, nW2, update], feed_dict={X:exp1, Y:ys1, R:discounted_epr1})
				# compare_arries(H1_val, eph, 'H1')
				# compare_arries(G_val, epdlogp.reshape(-1), 'G')
				# compare_arries(dW2_val, dW2a, 'dW2')
				# compare_arries(dW1_val, dW1a.T, 'dW1')
				# compare_arries(nW1beta2_val, rmsprop_cache['W1'].T, 'nW1beta2')
				# compare_arries(nW2beta2_val, rmsprop_cache['W2'], 'nW2beta2')
				# compare_arries(nW1_val, model['W1'].T, 'nW1')
				# compare_arries(nW2_val, model['W2'], 'nW2')
				_ = sess.run([update], feed_dict={X:exp1, Y:ys1, R:discounted_epr1})
				
				exp1, ys1, discounted_epr1 = [], [], []

			xs, hs, dlogps, drs, ys = [], [], [], [], []  # reset array memory

			# boring book-keeping
			running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			time_cost = int(time.time() - start_time)
			print('ep %d: totalReward: %f, averageReward: %f, time: %d' % (episode_number, reward_sum, running_reward, time_cost), flush=1)
			if episode_number % 300 == 0:
				pickle.dump(model, open(save_file, 'wb'))
			
			reward_sum = 0
			observation = env.reset() # reset env
			prev_x = None

			if episode_number == args.max_episode:
				pickle.dump(model, open(save_file, 'wb'))
				break
