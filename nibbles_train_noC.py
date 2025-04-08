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

# Constants for curriculum (only obstacle level increases)
MAX_OBSTACLE_LEVEL = 10
SCORE_TO_SAVE = 5
WINDOW_SIZE = 100
MIN_EPISODES_PER_STAGE = 25
PERIODIC_STAT_UPDATE = 50
OBSTACLE_SCORE_THRESHOLD = 14
BASE_FPS = 300
SPEED_FACTOR = 1.2
MODEL_DIR = "trained_model"
INITIAL_OBSTACLE_LEVEL = 1
# Use only the largest map size (640x480)
LARGEST_MAP_SIZE = (640, 480)

console = Console()

class PrioritizedReplayBuffer:
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
        try:
            for idx, error in zip(indices, td_errors):
                self.priorities[idx] = error + 1e-5  # Avoid zero priority
        except Exception as e:
            console.print(f"[bold red]Error in update_priorities():[/bold red] {e}")

    def __len__(self) -> int:
        return self.size

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int = 256) -> Tuple[np.ndarray, List[int], List[float], np.ndarray, List[bool]]:
        try:
            samples = random.sample(self.buffer, batch_size)
            s, a, r, s2, done = zip(*samples)
            return np.array(s), list(a), list(r), np.array(s2), list(done)

        except ValueError as e:
            console.print(f"[bold red]Error in sample(): Not enough samples to draw a batch of {batch_size}[/bold red]")
            raise

    def __len__(self) -> int:
        return len(self.buffer)

class DQNNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 256,
        hidden2: int = 256
    ) -> None:
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
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.005,
        epsilon_decay: float = 0.9995,
        batch_size: int = 256,
        device: str = 'cpu',
        target_update_freq: int = 1000,
        prioritized_replay: bool = True,
        double_dqn: bool = True
    ) -> None:
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
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self) -> float:
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

    def update_epsilon(self, episode_reward: float) -> None:
        # Since we are only in phase 2, always use the lower minimum epsilon
        decay_rate = self.epsilon_decay * 0.99 if episode_reward > 10 else self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)

def train_dqn_with_obstacles(num_episodes: int = 5000, render_mode: bool = False) -> Tuple[List[float], List[int]]:
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        current_width, current_height = LARGEST_MAP_SIZE
        env = NibblesEnv(width=current_width, height=current_height, level=INITIAL_OBSTACLE_LEVEL, render_mode=render_mode)

        agent = DQNAgent(
            state_dim=21,
            action_dim=4,
            lr=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.005,
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
        last_level_change_episode = 0

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

            agent.update_epsilon(episode_reward)
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
            if (ep + 1) % PERIODIC_STAT_UPDATE == 0:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                console.print(
                    f"[cyan]Episode {ep + 1}[/cyan] | "
                    f"[yellow]Avg.Recent Score:[/yellow] {avg_recent:.2f} | "
                    f"[blue]Hi.Recent Score:[/blue] {max(recent_scores)} | "
                    f"[green]Best Score:[/green] {best_score} | "
                    f"[magenta]Map:[/magenta] {env.width}x{env.height} | "
                    f"[red]Level:[/red] {env.level} | "
                    f"[white]Epsilon:[/white] {agent.epsilon:.3f}"
                )

            # Curriculum advancement: increase obstacle level only
            if len(recent_scores) >= WINDOW_SIZE and (ep - last_level_change_episode) >= MIN_EPISODES_PER_STAGE:
                avg_recent = sum(recent_scores) / len(recent_scores)
                if avg_recent >= OBSTACLE_SCORE_THRESHOLD:
                    new_level = env.level + 1
                    if new_level <= MAX_OBSTACLE_LEVEL:
                        model_path = f"{MODEL_DIR}/dqn_snake_level_{env.level}.pth"
                        torch.save(agent.q_net.state_dict(), model_path)
                        console.print(f"[bold yellow]Model saved to {model_path}[/bold yellow]")
                        console.print(f"[bold magenta]--- Increasing obstacle level to {new_level} (Avg score: {avg_recent:.2f}) ---[/bold magenta]")
                        env = NibblesEnv(width=current_width, height=current_height, level=new_level, render_mode=render_mode)
                        last_level_change_episode = ep
                        agent.epsilon = max(0.5, agent.epsilon)

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

def preserve_experiences(agent: DQNAgent) -> None:
    try:
        if not hasattr(agent.memory, 'buffer') or len(agent.memory.buffer) == 0:
            console.print("[yellow]No experiences to preserve; buffer is empty.[/yellow]")
            return
        console.print(f"[cyan]Preserving valuable experiences...[/cyan]")

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
    rewards, scores = train_dqn_with_obstacles(num_episodes=10000, render_mode=False)
    console.print("[bold green]Training complete![/bold green]")
