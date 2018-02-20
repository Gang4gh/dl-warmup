#!/usr/bin/env python3
""" Train an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import sys
import time
import pickle
import argparse
import numpy as np
import gym

import cudamat as cm
cm.cublas_init()

# parse arguments
parser = argparse.ArgumentParser(description='Play with PG algorithm on Pong')
parser.add_argument('-s', '--seed', type=int, help='random seed used by numpy and gym')
parser.add_argument('-r', '--resume', action="store_true", help='reset model (not resume from previous checkpoint)')
parser.add_argument('-d', '--render', action="store_true", help='render game during training')
parser.add_argument('--max-episode', type=int, default=10000, help='count of episode to halt the training')
parser.add_argument('--expname', default='save', help='a tag used to save/resume models')
args = parser.parse_args()
print('arguments: ', args.__dict__)
if args.seed is not None:
    np.random.seed(args.seed)

SAVE_FILE = '{}-s{}.m.bin'.format(args.expname, 'x' if args.seed is None else args.seed)

# hyperparameters
H = 200  # number of hidden layer neurons
BATCH_SIZE = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_beta1 = 0.9   # decay factor for Momentum, Adam of grad
decay_beta2 = 0.999  # decay factor for RMSProp, Adam of grad^2

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if args.resume:
    print('load model from', SAVE_FILE)
    model = pickle.load(open(SAVE_FILE, 'rb'))
else:
    # "Xavier" initialization
    model = {}
    model['W1'] = (np.random.randn(H, D) / np.sqrt(D)).astype(np.float32)
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# update buffers that add up gradients over a batch
grad_buffer = {w: np.zeros_like(v) for w, v in model.items()}
cache_grad1 = {w: np.zeros_like(v) for w, v in model.items()}
cache_grad2 = {w: np.zeros_like(v) for w, v in model.items()}

def sigmoid(_x):
    """ sigmoid "squashing" function to interval [0,1] """
    return 1.0 / (1.0 + np.exp(-_x))


def prepro(screen):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    screen = screen[35:195]  # crop
    screen = screen[::2, ::2, 0]  # downsample by factor of 2
    screen[screen == 144] = 0  # erase background (background type 1)
    screen[screen == 109] = 0  # erase background (background type 2)
    screen[screen != 0] = 1  # everything else (paddles, ball) just set to 1
    return screen.astype(np.float32).ravel()


def discount_rewards(rewards):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(_x):
    _h = np.dot(model['W1'], _x)
    _h[_h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], _h)
    p = sigmoid(logp)
    return p, _h  # return probability of taking action 2, and hidden state


def policy_backward(_hidden, _grad_out):
    """ backward pass. (_hidden is array of intermediate hidden states) """
    dW2 = np.dot(_hidden.T, _grad_out).ravel()
    dh = np.outer(_grad_out, model['W2'])
    dh[_hidden <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


start_time = time.time()
env = gym.make("Pong-v0")
if args.seed is not None:
    env.seed(args.seed)
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
total_forward_count = 0
while True:
    if args.render:
        env.render()
        time.sleep(0.01)

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    total_forward_count += 1
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    # calculate grad (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprob)  # * aprob * (1 - aprob))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for previous action)
    drs.append(reward)

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

        for w in model:
            grad_buffer[w] += grad[w]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % BATCH_SIZE == 0:
            for w in ['W1', 'W2']:
                g = grad_buffer[w]  # gradient
                cache_grad1[w] = decay_beta1 * cache_grad1[w] + (1 - decay_beta1) * g
                cache_grad2[w] = decay_beta2 * cache_grad2[w] + (1 - decay_beta2) * g**2
                lr = learning_rate * (1 - decay_beta2 ** episode_number) / (1 - decay_beta1 ** episode_number)
                model[w] += lr * cache_grad1[w] / (np.sqrt(cache_grad2[w]) + 1e-8)
                grad_buffer[w] = np.zeros_like(model[w])  # reset batch gradient bufferbuffer
            model['W1'] = model['W1'].astype(np.float32)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        time_cost = int(time.time() - start_time)
        if episode_number < 10 or episode_number % 10 == 0:
            print('ep %d: totalReward: %f, averageReward: %f, time: %d' %
                  (episode_number, reward_sum, running_reward, time_cost), flush=1)
        if episode_number % 300 == 0:
            pickle.dump(model, open(SAVE_FILE, 'wb'))

        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

        if episode_number == args.max_episode:
            pickle.dump(model, open(SAVE_FILE, 'wb'))
            print('total-ops-count = ', total_forward_count,
                  ', time-cost per 1k ops = ', time_cost / total_forward_count * 1000)
            break
