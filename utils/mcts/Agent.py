import numpy as np
import gymnasium as gym
import math
import matplotlib.pyplot as plt
from utils.mcts.Node import Node


def get_moving_average(array, window_size=75):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if len(array) == 0:
        return array
    window_size = max(1, min(window_size, len(array)))
    return np.convolve(array, np.ones(window_size) / window_size, mode="valid")


class Agent:
    def __init__(self, seed, total_episodes, max_steps_per_episode, simulations_per_decision,
                 exploration_constant, rollout_depth, discount_factor):
        self.seed = seed
        self.total_episodes = total_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.simulations_per_decision = simulations_per_decision
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.discount_factor = discount_factor
        self.state_nodes = {}

        self.env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
        self.env.reset(seed=seed)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.transition_probabilities = self.env.unwrapped.P

    def _get_node(self, state):
        if state not in self.state_nodes:
            self.state_nodes[state] = Node(self.n_actions)
        return self.state_nodes[state]

    def _uct(self, node, action):
        action_data = node.action_stats[action]
        action_visits = action_data["visit_count"]
        if action_visits == 0:
            return float("inf")
        average_value = action_data["total_value"] / action_visits
        return average_value + self.exploration_constant * math.sqrt(math.log(node.visit_count + 1) / action_visits)

    def _sample_next_state(self, state, action):
        transitions = self.transition_probabilities[state][action]
        random_value = np.random.rand()
        cumulative_probability = 0.0
        for probability, next_state, reward, done in transitions:
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                return next_state, reward, done
        probability, next_state, reward, done = transitions[-1]
        return next_state, reward, done

    def _rollout(self, state):
        total_return = 0.0
        cumulative_discount = 1.0
        current_state = state
        for _ in range(self.rollout_depth):
            action = np.random.randint(self.n_actions)
            next_state, reward, done = self._sample_next_state(current_state, action)
            total_return += cumulative_discount * reward
            cumulative_discount *= self.discount_factor
            current_state = next_state
            if done:
                break
        return total_return

    def _simulate(self, initial_state):
        state_action_path = []
        current_state = initial_state
        total_return = 0.0
        cumulative_discount = 1.0
        for _ in range(self.rollout_depth):
            node = self._get_node(current_state)
            uct_values = [self._uct(node, action) for action in range(self.n_actions)]
            selected_action = int(np.argmax(uct_values))
            state_action_path.append((current_state, selected_action))
            next_state, reward, done = self._sample_next_state(current_state, selected_action)
            total_return += cumulative_discount * reward
            cumulative_discount *= self.discount_factor
            current_state = next_state
            if done:
                break
        if not done:
            total_return += cumulative_discount * self._rollout(current_state)
        for state, action in reversed(state_action_path):
            node = self._get_node(state)
            node.visit_count += 1
            node.total_value += total_return
            action_data = node.action_stats[action]
            action_data["visit_count"] += 1
            action_data["total_value"] += total_return

    def select_best_action(self, state):
        for _ in range(self.simulations_per_decision):
            self._simulate(state)
        node = self._get_node(state)
        action_values = [
            node.action_stats[action]["total_value"] / node.action_stats[action]["visit_count"]
            if node.action_stats[action]["visit_count"] > 0 else -float("inf")
            for action in range(self.n_actions)
        ]
        return int(np.argmax(action_values))

    def run(self):
        self.results = {"rewards": [], "successes": [], "steps": []}
        for episode_index in range(self.total_episodes):
            current_state, _ = self.env.reset(seed=self.seed + episode_index)
            cumulative_reward = 0.0
            for step_index in range(self.max_steps_per_episode):
                action = self.select_best_action(current_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                cumulative_reward += reward
                current_state = next_state
                if done:
                    break
            self.results["rewards"].append(cumulative_reward)
            success_flag = 1 if cumulative_reward > 0 else 0
            self.results["successes"].append(success_flag)
            self.results["steps"].append(step_index + 1 if success_flag else self.max_steps_per_episode)
        self.results["rewards"] = np.array(self.results["rewards"])
        self.results["successes"] = np.array(self.results["successes"])
        self.results["steps"] = np.array(self.results["steps"])
        return self.results

    def display_performance(self, window_size=75):
        rewards = self.results["rewards"]
        # Compute moving average with same length as rewards
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='same')

        x = np.arange(len(rewards))  # x matches the length of rewards
        plt.plot(x, smoothed_rewards, label='Smoothed Rewards', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Performance Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
