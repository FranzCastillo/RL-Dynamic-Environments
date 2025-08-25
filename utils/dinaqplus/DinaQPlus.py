import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import math
import matplotlib.pyplot as plt

def moving_average(x, w):
    if w <= 1:
        return x
    ret = np.convolve(x, np.ones(w), 'valid') / w
    pad = [np.nan] * (len(x) - len(ret))
    return np.array(pad + list(ret))

class DynaQPlusAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1,
                 planning_steps=10, k_bonus=0.001, seed=None):
        self.nS = n_states
        self.nA = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.k_bonus = k_bonus

        self.Q = np.zeros((self.nS, self.nA))
        self.model = {}
        self.last_time = {}
        self.visits = defaultdict(int)
        self.time = 0
        self.rng = random.Random(seed)

    def choose_action(self, s):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.nA)
        else:
            best = np.flatnonzero(self.Q[s] == self.Q[s].max())
            return self.rng.choice(best)

    def update_real(self, s, a, r, s_next):
        self.time += 1
        td = r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

        # update model
        self.model[(s, a)] = (s_next, r)
        self.last_time[(s, a)] = self.time
        self.visits[(s, a)] += 1

    def planning(self):
        if not self.model or self.planning_steps <= 0:
            return
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            s_a = self.rng.choice(keys)
            s, a = s_a
            s_next, r = self.model[s_a]
            last_t = self.last_time.get((s, a), 0)
            dt = self.time - last_t
            bonus = self.k_bonus * math.sqrt(dt) if dt > 0 else 0.0
            r_prime = r + bonus
            # Q update using simulated sample
            td = r_prime + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a]
            self.Q[s, a] += self.alpha * td

    def reset(self):
        self.Q = np.zeros((self.nS, self.nA))
        self.model = {}
        self.last_time = {}
        self.visits = defaultdict(int)
        self.time = 0


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None):
        self.nS = n_states
        self.nA = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.nS, self.nA))
        self.rng = random.Random(seed)
        self.visits = defaultdict(int)

    def choose_action(self, s):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.nA)
        else:
            best = np.flatnonzero(self.Q[s] == self.Q[s].max())
            return self.rng.choice(best)

    def update_real(self, s, a, r, s_next):
        td = r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        self.visits[(s, a)] += 1

    def planning(self):
        pass

    def reset(self):
        self.Q = np.zeros((self.nS, self.nA))
        self.visits = defaultdict(int)


def run_episode(env, agent, max_steps=200, render=False):
    obs, info = env.reset()
    s = int(obs)
    total_reward = 0.0
    steps = 0
    success = False
    visited_pairs = set()
    for t in range(max_steps):
        a = agent.choose_action(s)
        visited_pairs.add((s, a))
        obs2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        s_next = int(obs2)
        agent.update_real(s, a, r, s_next)
        agent.planning()
        total_reward += r
        steps += 1
        s = s_next
        if done:
            if r > 0:
                success = True
            break
    return total_reward, success, steps, visited_pairs


def run_experiment(env_name='FrozenLake-v1', map_name='4x4', is_slippery=True,
                   agent_type='dyna', episodes=1000, planning_steps=10, k_bonus=0.001,
                   alpha=0.1, gamma=0.99, epsilon=0.1, seed=0, max_steps=200):
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
    try:
        nS = env.observation_space.n
        nA = env.action_space.n
    except Exception:
        # fallback
        nS = 16
        nA = 4

    if agent_type == 'dyna':
        agent = DynaQPlusAgent(nS, nA, alpha=alpha, gamma=gamma, epsilon=epsilon,
                               planning_steps=planning_steps, k_bonus=k_bonus, seed=seed)
    else:
        agent = QLearningAgent(nS, nA, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    rewards = []
    successes = []
    steps_list = []
    cumulative_unique_pairs = []
    unique_pairs = set()

    for ep in range(episodes):
        total_reward, success, steps, visited_pairs = run_episode(env, agent, max_steps=max_steps)
        rewards.append(total_reward)
        successes.append(1 if success else 0)
        steps_list.append(steps if success else np.nan)  # keep nan if failure
        unique_pairs |= visited_pairs
        cumulative_unique_pairs.append(len(unique_pairs))

    env.close()
    results = {
        'rewards': np.array(rewards),
        'successes': np.array(successes),
        'steps': np.array(steps_list),
        'cumulative_unique_pairs': np.array(cumulative_unique_pairs),
        'total_pairs_count': nS * nA,
        'agent': agent,
        'nS': nS,
        'nA': nA
    }
    return results


def plot_experiment_results(results_list, window=50, title_suffix=''):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    for label, res in results_list:
        success_rate = np.cumsum(res['successes']) / (np.arange(len(res['successes'])) + 1)
        ma = moving_average(success_rate, window)
        plt.plot(success_rate, alpha=0.25)
        plt.plot(ma, label=f'{label}')
    plt.title('Success rate' + title_suffix)
    plt.xlabel('Episode')
    plt.ylabel('Success rate')
    plt.legend()

    plt.subplot(2, 2, 2)
    for label, res in results_list:
        ma = moving_average(res['rewards'], window)
        plt.plot(res['rewards'], alpha=0.15)
        plt.plot(ma, label=f'{label}')
    plt.title('Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward (0/1)')
    plt.legend()

    plt.subplot(2, 2, 3)
    for label, res in results_list:
        frac = res['cumulative_unique_pairs'] / res['total_pairs_count']
        plt.plot(frac, label=f'{label}')
    plt.title('Exploration: visited state-action pairs / total')
    plt.xlabel('Episode')
    plt.ylabel('Visited fraction')
    plt.legend()

    plt.tight_layout()
    plt.show()