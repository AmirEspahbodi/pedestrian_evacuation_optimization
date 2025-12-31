import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .common import FitnessEvaluator


# ==========================================
# 1. Prioritized Experience Replay (SumTree)
# ==========================================
class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient O(log N) sampling based on priority.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Priority exponent
        self.epsilon = 1e-5  # Small constant to avoid zero priority

    def add(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity :])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            idxs,
            np.array(is_weight),
        )

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)


# ==========================================
# 2. Dueling Deep Q-Network
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        # Feature Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # Value Stream (Estimates V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # Advantage Stream (Estimates A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


# ==========================================
# 3. Optimization Agent
# ==========================================
class Agent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.output_dim = output_dim

        self.model = DuelingDQN(input_dim, output_dim).to(self.device)
        self.target_model = DuelingDQN(input_dim, output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(
            capacity=2000
        )  # Smaller buffer for expensive env

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def train(self, batch_size, beta):
        if self.memory.tree.n_entries < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, idxs, weights = (
            self.memory.sample(batch_size, beta)
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Double DQN Logic
        # 1. Selection: Use online model to pick best action for next state
        next_actions = self.model(next_states).argmax(1)
        # 2. Evaluation: Use target model to get value of that action
        next_q_values = (
            self.target_model(next_states)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss with Importance Sampling Weights
        diff = target_q_values - curr_q_values
        loss = (weights * (diff**2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in buffer
        errors = diff.abs().detach().cpu().numpy()
        self.memory.update_priorities(idxs, errors)

        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ==========================================
# 4. Main Optimizer Logic
# ==========================================
def q_learning_exit_optimizer(
    pedestrian_confs,
    gird,
    simulator_config,
    iea_config,
    max_episodes=50,  # Number of reset trials
    steps_per_episode=30,  # Steps per trial (limited by expensive sim)
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.95,
):
    """
    Elite implementation of Dueling Double DQN with PER for Integer Optimization.
    """

    # --- 1. Environment Setup ---
    # Determine dimensionality k from the initial pedestrian config or config file
    # Assuming initial pedestrian_confs is a list/vector of coordinates
    initial_vector = list(pedestrian_confs)  # Ensure it's mutable
    k = len(initial_vector)

    # Define Action Space: For each k, we have 4 moves: {+1, -1, +10, -10}
    # Total actions = k * 4
    # Action map: action_idx -> (exit_index, delta)
    actions_map = {}
    cnt = 0
    for i in range(k):
        for delta in [1, -1, 10, -10]:
            actions_map[cnt] = (i, delta)
            cnt += 1

    input_dim = k
    output_dim = len(actions_map)

    # Initialize Agent
    agent = Agent(input_dim, output_dim)

    # Initialize Fitness Evaluator
    # Assuming 'FitnessEvaluator' is available in your scope or imported
    psi_evaluator = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    # Tracking Best
    best_solution = list(initial_vector)
    best_fitness = float("inf")  # We want to minimize time

    # Hyperparameters
    epsilon = epsilon_start
    batch_size = 32
    beta = 0.4  # For PER

    print(f"--- Starting Optimization: K={k}, State Space=400^{k} ---")

    for episode in range(max_episodes):
        # Reset Environment: Start from a random valid position to avoid local optima
        # or start from best_solution found so far (Hybrid approach)
        if episode == 0:
            current_solution = list(initial_vector)
        elif random.random() < 0.3:
            # Exploration reset
            current_solution = [random.randint(0, 400) for _ in range(k)]
        else:
            # Exploitation reset (continue from best)
            current_solution = list(best_solution)

        # Initial evaluation
        try:
            current_fitness = psi_evaluator.evaluate(current_solution)
        except Exception:
            # Fallback for safety
            current_fitness = 10000.0

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = list(current_solution)

        state_norm = np.array(current_solution) / 400.0  # Normalize 0-1

        for step in range(steps_per_episode):
            # 1. Select Action
            action_idx = agent.get_action(state_norm, epsilon)
            exit_idx, delta = actions_map[action_idx]

            # 2. Apply Action (Environment Step)
            new_solution = list(current_solution)
            new_val = new_solution[exit_idx] + delta

            # Boundary Check / Integer Constraint
            # If move is invalid (out of 0-400), we clip it,
            # but we punish the agent slightly to discourage hitting walls repeatedly
            hit_wall = False
            if new_val < 0:
                new_val = 0
                hit_wall = True
            elif new_val > 400:
                new_val = 400
                hit_wall = True

            new_solution[exit_idx] = int(new_val)  # STRICT INTEGER ENFORCEMENT

            # 3. Calculate Reward
            # We skip simulation if the state didn't actually change (efficiency)
            if new_solution == current_solution:
                reward = -1.0  # Penalty for wasting time
                new_fitness = current_fitness
            else:
                # Run Expensive Simulator
                try:
                    new_fitness = psi_evaluator.evaluate(new_solution)
                except:
                    new_fitness = current_fitness + 100  # Penalty for crash

                # Reward Definition:
                # Positive if fitness decreased (improvement)
                # Negative if fitness increased (worsened)
                diff = current_fitness - new_fitness

                # Scale reward for stability (e.g. if time is in seconds ~100s)
                reward = diff * 10.0

                if hit_wall:
                    reward -= 2.0  # Penalty for hitting boundary

            # Update Best
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_solution = list(new_solution)
                print(f" >> New Best: {best_fitness:.2f} | Sol: {best_solution}")
                reward += 10.0  # Bonus for finding global best

            # 4. Store in Memory
            next_state_norm = np.array(new_solution) / 400.0
            done = step == steps_per_episode - 1

            agent.memory.add(state_norm, action_idx, reward, next_state_norm, done)

            # 5. Train
            loss = agent.train(batch_size, beta)

            # Move to next state
            current_solution = new_solution
            current_fitness = new_fitness
            state_norm = next_state_norm

        # Update Epsilon & Target Network per episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        beta = min(1.0, beta + 0.01)  # Anneal beta to 1
        agent.update_target()

        if episode % 5 == 0:
            print(
                f"Episode {episode}/{max_episodes} | Best Fitness: {best_fitness:.2f} | Epsilon: {epsilon:.2f}"
            )

    print(
        f"--- Optimization Complete. Best Solution: {best_solution} (Fitness: {best_fitness}) ---"
    )
    return best_solution
