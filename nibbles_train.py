
"""
===========================================================================================
Deep Q-Network (DQN) Agent with Curriculum Learning, Double DQN, and Prioritised Replay
===========================================================================================

Overview:
---------
This module implements a Deep Q-Network (DQN) algorithm for reinforcement learning in 
discrete action spaces, specifically applied to a curriculum-based Nibbles game environment.
DQN combines Q-learning with deep neural networks to approximate the optimal action-value
function Q(s, a) ≈ Q*, enabling an agent to make decisions in high-dimensional state spaces.

Mechanism:
----------
The agent interacts with the environment through episodes, storing transitions 
(state, action, reward, next_state, done) into a replay buffer. During training, 
mini-batches are sampled to perform Q-learning updates. The target network, 
which is periodically synced with the main Q-network, improves training stability.

Key Features:
-------------
✓ Curriculum learning: The agent progresses through increasing levels of map complexity.  
✓ Double DQN: Reduces overestimation of action values by decoupling action selection and evaluation.  
✓ Prioritised Experience Replay (PER): Samples important experiences more frequently using TD-error-based priorities.  
✓ Duelling DQN architecture: Separates state-value and advantage estimation for more robust value learning.  
✓ Epsilon-greedy exploration with decay for balancing exploration/exploitation.

Main Components:
----------------
- `DQNAgent`: Handles policy selection, training logic, epsilon decay.
- `DQNNet`: Neural network model using shared feature extraction and duelling heads.
- `ReplayBuffer` and `PrioritizedReplayBuffer`: Store and sample past experiences.
- `train_dqn_curriculum_with_obstacles()`: Curriculum-based training loop.
- `preserve_experiences()`: Retains meaningful experiences between curriculum stages.

Usage:
------
This module is designed to be executed as a standalone script or imported as a training module.
It can be extended for other environments by adapting the state dimensionality and action space.

Author:
-------
Amin Haghighatbin

Version:
--------
v3.0.1 — April 2025

License:
--------
MIT License or other (add your licensing information here)

References:
-----------
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Schaul, T. et al. (2016). Prioritized Experience Replay. arXiv:1511.05952
- Wang, Z. et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning.

"""

import os
import time
import random
from collections import deque
from rich.console import Console
from typing import Tuple, List


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame

from nibbles_env import NibblesEnv

# ───────────────────────────────────────── CONSTANTS ───────────────────────────────────────── #
MAP_THRESHOLDS = [5, 8, 12, 16]  # Stage advancement thresholds
MAX_OBSTACLE_LEVEL = 10
SCORE_TO_SAVE = 5
WINDOW_SIZE = 100
MIN_EPISODES_PER_STAGE = 20
OBSTACLE_SCORE_THRESHOLD = 20
BASE_FPS = 300
SPEED_FACTOR = 1.2
MODEL_DIR = "trained_model"
INITIAL_OBSTACLE_LEVEL = 1
STAGE_START = 1
MAP_SIZES = [
    (480, 320),  # Stage 1
    (520, 360),  # Stage 2
    (560, 400),  # Stage 3
    (600, 440),  # Stage 4
    (640, 480)   # Stage 5 (final map size)
]
# ───────────────────────────────────────────────────────────────────────────────────────────── #

console = Console()


class PrioritizedReplayBuffer:
    """
    A Prioritised Experience Replay Buffer used to improve sample efficiency
    in Deep Q-Learning by replaying more significant experiences.

    Attributes:
        capacity (int): Maximum number of experiences to store.
        alpha (float): Degree of prioritisation (0 disables, 1 uses full).
        beta (float): Degree of importance-sampling bias correction.
        beta_increment (float): Increment rate for beta per sampling step.
        buffer (List[tuple]): Stored (s, a, r, s2, done) tuples.
        priorities (np.ndarray): Corresponding priorities for experiences.
        position (int): Current circular insertion index.
        size (int): Number of currently stored experiences.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the replay buffer with the maximum current priority.

        Args:
            s (np.ndarray): Current state.
            a (int): Action taken.
            r (float): Reward received.
            s2 (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        try:
            max_priority = self.priorities.max() if self.size > 0 else 1.0

            if self.size < self.capacity:
                self.buffer.append((s, a, r, s2, done))
                self.size += 1
            else:
                self.buffer[self.position] = (s, a, r, s2, done)

            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

        except Exception as e:
            console.print(f"[bold red]Error in push():[/bold red] {e}")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[int], List[float], np.ndarray, List[bool], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on their prioritised probabilities.

        Args:
            batch_size (int): Number of samples to draw.

        Returns:
            Tuple containing:
                - states
                - actions
                - rewards
                - next states
                - dones
                - indices
                - importance sampling weights
        """
        try:
            priorities = self.priorities[:self.size] if self.size < self.capacity else self.priorities
            self.beta = min(1.0, self.beta + self.beta_increment)

            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(probs), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]

            weights = (self.size * probs[indices]) ** (-self.beta)
            weights /= weights.max()

            s, a, r, s2, done = zip(*samples)
            return np.array(s), list(a), list(r), np.array(s2), list(done), indices, weights

        except Exception as e:
            console.print(f"[bold red]Error in sample():[/bold red] {e}")
            raise

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update the priorities of experiences based on their temporal difference errors.

        Args:
            indices (np.ndarray): Indices of the sampled experiences.
            td_errors (np.ndarray): Temporal difference errors.
        """
        try:
            for idx, error in zip(indices, td_errors):
                self.priorities[idx] = error + 1e-5  # Avoid zero priority
        except Exception as e:
            console.print(f"[bold red]Error in update_priorities():[/bold red] {e}")

    def __len__(self) -> int:
        """
        Returns the current number of stored experiences.

        Returns:
            int: Current buffer size.
        """
        return self.size

class ReplayBuffer:
    """
    A standard experience replay buffer using a fixed-size deque to store transitions.

    Attributes:
        buffer (deque): Stores transitions of the form (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        """
        Initialise the replay buffer with a fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            s (np.ndarray): Current state.
            a (int): Action taken.
            r (float): Reward received.
            s2 (np.ndarray): Next state.
            done (bool): Whether the episode terminated.
        """
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int = 256) -> Tuple[np.ndarray, List[int], List[float], np.ndarray, List[bool]]:
        """
        Sample a batch of transitions uniformly from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple containing:
                - states
                - actions
                - rewards
                - next states
                - dones
        """
        try:
            samples = random.sample(self.buffer, batch_size)
            s, a, r, s2, done = zip(*samples)
            return np.array(s), list(a), list(r), np.array(s2), list(done)

        except ValueError as e:
            console.print(f"[bold red]Error in sample(): Not enough samples to draw a batch of {batch_size}[/bold red]")
            raise

    def __len__(self) -> int:
        """
        Return the current number of stored transitions.

        Returns:
            int: Buffer size.
        """
        return len(self.buffer)

class DQNNet(nn.Module):
    """
    Deep Q-Network (DQN) for approximating action-value (Q) functions.

    This implementation uses a shared feature extractor followed by separate
    streams for estimating the state-value and the advantage for each action,
    implementing the Dueling DQN architecture.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 256,
        hidden2: int = 256
    ) -> None:
        """
        Initialise the DQN network layers.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Number of possible discrete actions.
            hidden1 (int): Number of units in the first hidden layer.
            hidden2 (int): Number of units in the second hidden layer.
        """
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialise the weights of the network layers using Kaiming normal initialisation.

        Args:
            module (nn.Module): A layer in the network to be initialised.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Estimated Q-values for each action.
        """
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent with optional Double DQN and Prioritised Experience Replay.

    This agent supports advanced features such as target networks, duelling architecture,
    epsilon-greedy exploration, gradient clipping, learning rate scheduling, and
    optionally prioritised sampling for efficient training.
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 256,
        device: str = 'cpu',
        target_update_freq: int = 1000,
        prioritized_replay: bool = True,
        double_dqn: bool = True
    ) -> None:
        """
        Initialise the DQN agent and its networks.

        Args:
            state_dim (int): Dimensionality of the state vector.
            action_dim (int): Number of available discrete actions.
            lr (float): Learning rate for the optimiser.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration probability.
            epsilon_min (float): Minimum exploration probability.
            epsilon_decay (float): Multiplicative decay factor for epsilon.
            batch_size (int): Number of samples per training step.
            device (str): Torch device to use ('cpu' or 'cuda').
            target_update_freq (int): Target network sync interval.
            prioritized_replay (bool): Enable prioritised replay buffer.
            double_dqn (bool): Enable Double DQN logic.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_freq = target_update_freq
        self.train_steps = 0
        self.prioritized_replay = prioritized_replay
        self.double_dqn = double_dqn

        self.q_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        console.print(f"[bold green]Device selected:[/bold green] {self.device}")

        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(capacity=500_000)
        else:
            self.memory = ReplayBuffer(capacity=500_000)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): Current environment state.

        Returns:
            int: Chosen action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self) -> float:
        """
        Perform one training step for the DQN agent.

        Returns:
            float: Loss value from the training step.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        if self.prioritized_replay:
            s, a, r, s2, done, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            s, a, r, s2, done = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        done = torch.BoolTensor(done).to(self.device)

        q_values = self.q_net(s)
        q_selected = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_net(s2).max(1)[1].unsqueeze(1)
                next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(s2).max(1)[0]

            target = r + (1 - done.float()) * self.gamma * next_q

        td_errors = torch.abs(q_selected - target).detach().cpu().numpy()
        elementwise_loss = self.loss_fn(q_selected, target)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.prioritized_replay:
            self.memory.update_priorities(indices, td_errors)

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def update_epsilon(self, episode_reward: float, stage: int) -> None:
        """
        Update epsilon based on episode reward and training stage.

        Args:
            episode_reward (float): Reward obtained in the episode.
            stage (int): Current training stage (affects decay rate).
        """
        min_eps = 0.05 if stage >= 5 else self.epsilon_min
        decay_rate = self.epsilon_decay * 0.99 if episode_reward > 10 else self.epsilon_decay
        self.epsilon = max(min_eps, self.epsilon * decay_rate)

def train_dqn_curriculum_with_obstacles(num_episodes: int = 5000, render_mode: bool = False) -> Tuple[List[float], List[int]]:
    """
    Trains a DQN agent in a curriculum-based Nibbles environment with growing map size and obstacles.

    Args:
        num_episodes (int): Number of episodes to train the agent.
        render_mode (bool): If True, visually render the environment.

    Returns:
        Tuple[List[float], List[int]]: Episode rewards and scores.
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        stage = STAGE_START
        obstacle_level = INITIAL_OBSTACLE_LEVEL
        current_width, current_height = MAP_SIZES[stage - 1]

        env = NibblesEnv(width=current_width, height=current_height, level=obstacle_level, render_mode=render_mode)

        agent = DQNAgent(
            state_dim=21,
            action_dim=4,
            lr=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995,
            batch_size=128,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            target_update_freq=1000,
            prioritized_replay=True,
            double_dqn=True
        )

        all_rewards = []
        all_scores = []
        best_score = 0
        recent_scores = deque(maxlen=WINDOW_SIZE)
        last_stage_change_episode = 0

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                if env.render_mode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return all_rewards, all_scores

                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                episode_reward += reward

                if render_mode:
                    env.render(all_rewards + [episode_reward], all_scores + [env.score])
                    env.clock.tick(BASE_FPS + SPEED_FACTOR * env.score)

            agent.update_epsilon(episode_reward, stage)
            all_rewards.append(episode_reward)
            all_scores.append(env.score)
            recent_scores.append(env.score)

            # Save model if best score achieved
            if env.score > best_score:
                best_score = env.score
                if best_score >= SCORE_TO_SAVE:
                    path = f"{MODEL_DIR}/dqn_snake_best_score_{best_score}.pth"
                    torch.save(agent.q_net.state_dict(), path)
                    console.print(f"[bold green]New best score {best_score}! Model saved to {path}[/bold green]")

            # Periodic status update
            if (ep + 1) % 20 == 0:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                console.print(
                    f"[cyan]Episode {ep + 1}[/cyan] | "
                    f"[yellow]Avg Recent:[/yellow] {avg_recent:.2f} | "
                    f"[blue]Score:[/blue] {env.score} | "
                    f"[green]Best:[/green] {best_score} | "
                    f"[magenta]Map:[/magenta] {env.width}x{env.height} | "
                    f"[red]Level:[/red] {env.level} | "
                    f"[white]Epsilon:[/white] {agent.epsilon:.3f}"
                )

            # Check curriculum advancement
            if len(recent_scores) >= WINDOW_SIZE and (ep - last_stage_change_episode) >= MIN_EPISODES_PER_STAGE:
                avg_recent = sum(recent_scores) / len(recent_scores)

                # Phase 1: increase map size
                if stage < len(MAP_SIZES):
                    if avg_recent >= MAP_THRESHOLDS[stage - 1]:
                        stage += 1
                        current_width, current_height = MAP_SIZES[stage - 1]
                        env = NibblesEnv(width=current_width, height=current_height, level=0, render_mode=render_mode)
                        last_stage_change_episode = ep
                        console.print(f"\n[bold cyan]--- Advancing to Stage {stage}: Map size {current_width}x{current_height} ---[/bold cyan]")

                # Phase 2: increase obstacle level
                else:
                    if avg_recent >= OBSTACLE_SCORE_THRESHOLD:
                        new_level = env.level + 1
                        if new_level <= MAX_OBSTACLE_LEVEL:
                            model_path = f"{MODEL_DIR}/dqn_snake_level_{env.level}.pth"
                            torch.save(agent.q_net.state_dict(), model_path)
                            console.print(f"[bold yellow]Model saved to {model_path}[/bold yellow]")
                            console.print(f"[bold magenta]--- Increasing obstacle level to {new_level} (Avg score: {avg_recent:.2f}) ---[/bold magenta]")

                            obstacle_level = new_level
                            env = NibblesEnv(width=current_width, height=current_height, level=new_level, render_mode=render_mode)
                            last_stage_change_episode = ep
                            agent.epsilon = max(0.3, agent.epsilon)

        # Final model save
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        final_model_path = f"{MODEL_DIR}/dqn_snake_final_{timestamp}.pth"
        torch.save(agent.q_net.state_dict(), final_model_path)
        console.print(f"[bold green]Final model saved to {final_model_path}[/bold green]")

        return all_rewards, all_scores

    except Exception as e:
        console.print(f"[bold red]Training failed due to error:[/bold red] {e}")
        pygame.quit()
        return [], []

def preserve_experiences(agent: DQNAgent, previous_stage: int) -> None:
    """
    Preserve high-value experiences when transitioning between curriculum stages.

    This function filters and retains key experiences—such as high-reward,
    apple collection, and exploratory transitions—from the agent's replay memory.

    Args:
        agent (DQNAgent): Agent whose memory is to be filtered and rebuilt.
        previous_stage (int): The stage being transitioned from.
    """
    try:
        if not hasattr(agent.memory, 'buffer') or len(agent.memory.buffer) == 0:
            console.print("[yellow]No experiences to preserve; buffer is empty.[/yellow]")
            return

        console.print(f"[cyan]Preserving valuable experiences from Stage {previous_stage}...[/cyan]")

        # Categorise experiences
        valuable_experiences = []
        apple_collection_experiences = []
        exploration_experiences = []

        for exp in agent.memory.buffer:
            s, a, r, s2, done = exp

            if r > 10:
                apple_collection_experiences.append(exp)
            elif r > 0:
                valuable_experiences.append(exp)
            elif not done and r > -1:
                exploration_experiences.append(exp)

        # Reconstruct buffer
        if hasattr(agent.memory, 'alpha') and hasattr(agent.memory, 'beta'):
            agent.memory = PrioritizedReplayBuffer(
                capacity=agent.memory.capacity,
                alpha=agent.memory.alpha,
                beta=agent.memory.beta
            )
        else:
            agent.memory = ReplayBuffer(capacity=agent.memory.buffer.maxlen)

        # Retain important transitions
        console.print(f"[green]Preserving {len(apple_collection_experiences)} apple collection experiences.[/green]")
        for exp in apple_collection_experiences:
            agent.memory.push(*exp)

        valuable_to_keep = min(len(valuable_experiences), 5000)
        console.print(f"[green]Preserving {valuable_to_keep} valuable experiences.[/green]")
        for exp in (random.sample(valuable_experiences, valuable_to_keep)
                    if len(valuable_experiences) > valuable_to_keep
                    else valuable_experiences):
            agent.memory.push(*exp)

        exploration_to_keep = min(len(exploration_experiences), 5000)
        console.print(f"[green]Preserving {exploration_to_keep} exploration experiences.[/green]")
        for exp in (random.sample(exploration_experiences, exploration_to_keep)
                    if len(exploration_experiences) > exploration_to_keep
                    else exploration_experiences):
            agent.memory.push(*exp)

        console.print(f"[bold blue]Memory buffer rebuilt with {len(agent.memory)} experiences.[/bold blue]")

    except Exception as e:
        console.print(f"[bold red]Error while preserving experiences:[/bold red] {e}")


if __name__ == "__main__":
    rewards, scores = train_dqn_curriculum_with_obstacles(num_episodes=5000, render_mode=False)
    console.print("[bold green]Training complete![/bold green]")
