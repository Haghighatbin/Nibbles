"""
====================================================================================
Nibbles Environment: Rendering Utilities for Visual Gameplay and Performance Display
====================================================================================

This module handles the visual rendering for the Nibbles reinforcement learning
environment. It provides a real-time graphical display of the game state, including:

- Snake and apple visualisation
- Obstacles for different levels
- Performance metrics overlay (score, reward history)
- Inset chart summarising recent training progress

The game uses a grid-based map with fixed-size blocks and supports curriculum learning
with dynamic obstacle difficulty and map resizing.

Author: Amin Haghighatbin
Version: v1.0.0
Date: April 2025
"""

import math
import random
from enum import Enum
from typing import List, Optional
from collections import namedtuple

import pygame
import numpy as np

from levels_ai import Levels

# ─────────────────────────────── GLOBAL CONSTANTS ────────────────────────────────── #
BLOCK_SIZE = 20
MAP_WIDTH = 640
MAP_HEIGHT = 480
FONT_SIZE = 24
FPS = 60
REWARDS_DOMAIN = 50
LOOKAHEAD_VAL = 5
SHORT_SELF_COLLISION = 5
LONG_SELF_COLLISION = 30
APPLE_IMAGE_PATH = 'Images/apple.png'
# ─────────────────────────────────────────────────────────────────────────────────── #
pygame.init()
font = pygame.font.Font(None, FONT_SIZE)

# Load and scale the apple image globally
apple_img = pygame.image.load(APPLE_IMAGE_PATH)
apple_img = pygame.transform.scale(apple_img, (BLOCK_SIZE, BLOCK_SIZE))

# Named tuple for coordinates
Coordinate = namedtuple('Coordinate', 'x, y')


class Direction(Enum):
    """Enumeration for possible movement directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class NibblesEnv:
    """
    A simplified 'Snake' game environment for reinforcement learning agents.

    Features:
    ---------
    - Grid-based environment with fixed block size.
    - Configurable difficulty levels via obstacle generation.
    - Supports observation, reward shaping, and curriculum training.
    - Optional rendering mode with visual feedback.

    Public Methods:
    ---------------
    - reset()        : Resets the environment to the initial state.
    - step(action)   : Advances one time step based on the given action.
    - render(...)    : Renders the current game state (if render_mode is True).
    """

    def __init__(self, width: int = MAP_WIDTH, height: int = MAP_HEIGHT, level: int = 0, render_mode: bool = False) -> None:
        """
        Initialise the Nibbles environment.

        Args:
            width (int): Width of the environment window in pixels.
            height (int): Height of the environment window in pixels.
            level (int): Difficulty level, used to generate obstacles.
            render_mode (bool): Whether to enable graphical rendering.
        """
        self.width = width
        self.height = height
        self.level = level
        self.render_mode = render_mode
        self.block_size = BLOCK_SIZE
        self.final_score = 0
        self.step_count = 0

        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.DOUBLEBUF | pygame.HWSURFACE
            )
            pygame.display.set_caption("Nibbles RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

            # Load and scale the apple image
            self.apple_img = pygame.image.load('Images/apple.png')
            self.apple_img = pygame.transform.scale(self.apple_img, (BLOCK_SIZE, BLOCK_SIZE))
        else:
            self.display = None
            self.clock = None

        # Map discrete action indices to movement directions
        self.action_map = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.UP,
            3: Direction.DOWN
        }

        font_small = pygame.font.SysFont(None, 14)
        self.step_count = 0
        self.labels = ["Max Score:", "Last Score:", "Max Reward:", "Last Reward:"]
        self.label_surfs = [font_small.render(label, True, (255, 255, 255)) for label in self.labels]
        self.reset()

    @staticmethod
    def distance(a: Coordinate, b: Coordinate) -> float:
        """
        Compute the Euclidean distance between two coordinates.

        Args:
            a (Coordinate): First coordinate.
            b (Coordinate): Second coordinate.

        Returns:
            float: The straight-line (Euclidean) distance between a and b.
        """
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def reset(self) -> np.ndarray:
        """
        Reset the game environment to its initial state.

        Resets snake position, score, obstacles, apple, and direction.

        Returns:
            np.ndarray: The initial observation of the environment.
        """
        self.direction = Direction.RIGHT

        # Centre the snake on the grid
        x_center = (self.width // self.block_size) // 2 * self.block_size
        y_center = (self.height // self.block_size) // 2 * self.block_size
        self.head = Coordinate(x_center, y_center)

        self.snake = [
            self.head,
            Coordinate(self.head.x - self.block_size, self.head.y),
            Coordinate(self.head.x - 2 * self.block_size, self.head.y)
        ]
        self.snake_set = set(self.snake)
        self.score = 0

        # Generate obstacles for the current level
        lvl = Levels(block_size=self.block_size, width=self.width, height=self.height)
        self.obstacles = lvl._level_generator(level=self.level)

        self.apple = None
        self._place_apple()
        self.step_count = 0

        return self._get_observation()

    def _place_apple(self) -> None:
        """
        Place the apple at a random valid location on the grid,
        avoiding the snake, obstacles, and the four corners.
        """
        num_cells_x = self.width // self.block_size
        num_cells_y = self.height // self.block_size

        # Define corners to avoid
        corners = {
            Coordinate(0, 0),
            Coordinate((num_cells_x - 1) * self.block_size, 0),
            Coordinate(0, (num_cells_y - 1) * self.block_size),
            Coordinate((num_cells_x - 1) * self.block_size, (num_cells_y - 1) * self.block_size)
        }

        while True:
            x = random.randint(0, num_cells_x - 1) * self.block_size
            y = random.randint(0, num_cells_y - 1) * self.block_size
            coord = Coordinate(x, y)

            if coord not in self.snake and not self._is_in_obstacles(coord) and coord not in corners:
                self.apple = coord
                break

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the environment by one step/frame using the specified action.

        Args:
            action (int): Index of the action to perform:
                        (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)

        Returns:
            tuple:
                - np.ndarray: New state observation.
                - float: Reward signal for the step.
                - bool: Whether the episode has terminated.
                - dict: Optional additional information (empty).
        """
        self.step_count += 1
        old_dist = self.distance(self.head, self.apple)

        # Prevent 180-degree reversals
        new_direction = self.action_map[action]
        if (
            (self.direction == Direction.LEFT and new_direction == Direction.RIGHT) or
            (self.direction == Direction.RIGHT and new_direction == Direction.LEFT) or
            (self.direction == Direction.UP and new_direction == Direction.DOWN) or
            (self.direction == Direction.DOWN and new_direction == Direction.UP)
        ):
            new_direction = self.direction

        self.direction = new_direction

        reward = -0.01  # Small step penalty
        old_head = self.head

        self._move_snake()
        done = False

        # Collision check
        if self._is_collision():
            apple_dist_factor = max(0.5, 1.0 - self.distance(self.head, self.apple) / (self.width + self.height))
            reward = -10.0 * apple_dist_factor
            done = True
            return self._get_observation(), reward, done, {}

        # Apple collection
        if self.head == self.apple:
            base_reward = 10
            length_bonus = 2.0 * self.score
            reward = base_reward + length_bonus
            self.score += 1
            self.final_score = max(self.final_score, self.score)
            self._place_apple()
        else:
            # Remove tail if no apple collected
            tail = self.snake.pop()
            self.snake_set.discard(tail)

        # Reward shaping for distance improvement
        new_dist = self.distance(self.head, self.apple)
        dist_improvement = old_dist - new_dist
        map_diagonal = math.sqrt(self.width**2 + self.height**2)
        normalised = dist_improvement / map_diagonal

        if normalised > 0:
            reward += 0.2 * normalised
        else:
            reward += 0.1 * normalised  # Mild penalty for going the wrong way

        # Bonus for staying alive (encourages exploration)
        reward += 0.001 * len(self.snake)

        # Penalty if segments are too close to head
        for segment in self.snake[1:]:
            if self.distance(self.head, segment) < 2 * self.block_size:
                reward -= 0.2

        # Additional penalty for forward proximity
        forward_body_dist = self._distance_to_self(self.direction)
        if forward_body_dist < 0.2:
            reward -= 0.5

        # Adaptive episode timeout
        base_diagonal = math.sqrt(480**2 + 320**2)  # Stage 1 baseline
        current_diagonal = math.sqrt(self.width**2 + self.height**2)
        max_steps = int(100 * (current_diagonal / base_diagonal)) + 50 * self.score

        if self.step_count >= max_steps:
            done = True

        if self.render_mode:
            self.render([],[])

        return self._get_observation(), reward, done, {}

    def _move_snake(self) -> None:
        """
        Move the snake in the current direction with wrap-around at edges.
        Updates both the head and the full body segment list.
        """
        x, y = self.head.x, self.head.y

        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        x %= self.width
        y %= self.height

        self.head = Coordinate(x, y)
        self.snake.insert(0, self.head)
        self.snake_set.add(self.head)

    def _is_collision(self) -> bool:
        """
        Check whether a collision has occurred.

        Collisions are detected with the snake's body or any obstacles.

        Returns:
            bool: True if the snake has collided, False otherwise.
        """
        # Collision with body (excluding head)
        if self.head in self.snake[1:]:
            return True

        # Collision with any obstacle
        if self._is_in_obstacles(self.head):
            return True

        return False

    def _is_in_obstacles(self, coord: Coordinate) -> bool:
        """
        Determine whether a given coordinate lies within any obstacle.

        Args:
            coord (Coordinate): The coordinate to evaluate.

        Returns:
            bool: True if the coordinate intersects an obstacle, False otherwise.
        """
        return any(coord in obs_group for obs_group in self.obstacles)

    def _distance_to_self(self, direction: Direction, max_steps: int = LONG_SELF_COLLISION) -> float:
        """
        Measure the normalised distance in a given direction until the snake's body or
        an obstacle is encountered.

        Args:
            direction (Direction): The direction to look.
            max_steps (int): Maximum number of blocks to check ahead.

        Returns:
            float: Distance to collision, normalised between 0 and 1.
        """
        x, y = self.head.x, self.head.y

        for step in range(1, max_steps + 1):
            if direction == Direction.RIGHT:
                check_pos = Coordinate(x + step * self.block_size, y)
            elif direction == Direction.LEFT:
                check_pos = Coordinate(x - step * self.block_size, y)
            elif direction == Direction.DOWN:
                check_pos = Coordinate(x, y + step * self.block_size)
            elif direction == Direction.UP:
                check_pos = Coordinate(x, y - step * self.block_size)

            if check_pos in self.snake_set or self._is_in_obstacles(check_pos):
                return step / max_steps

        return 1.0

    def _distance_to_body(self, direction: Direction, max_look: int = SHORT_SELF_COLLISION) -> float:
        """
        Measure the normalised distance in a given direction until a body segment
        or an obstacle is encountered.

        Args:
            direction (Direction): The direction to evaluate.
            max_look (int): Maximum lookahead steps.

        Returns:
            float: Distance to collision, normalised between 0 and 1.
        """
        x, y = self.head.x, self.head.y

        for step in range(1, max_look + 1):
            if direction == Direction.RIGHT:
                check_pos = Coordinate(x + step * self.block_size, y)
            elif direction == Direction.LEFT:
                check_pos = Coordinate(x - step * self.block_size, y)
            elif direction == Direction.DOWN:
                check_pos = Coordinate(x, y + step * self.block_size)
            elif direction == Direction.UP:
                check_pos = Coordinate(x, y - step * self.block_size)

            if check_pos in self.snake_set or self._is_in_obstacles(check_pos):
                return step / max_look

        return 1.0

    def _get_observation(self) -> np.ndarray:
        """
        Construct a feature vector representing the current game state.

        This includes directional dangers, current heading, relative food position,
        and various distance metrics to the body and obstacles.

        Returns:
            np.ndarray: Normalised state representation as input for the agent.
        """
        head_x = self.head.x / self.width
        head_y = self.head.y / self.height
        apple_x = self.apple.x / self.width
        apple_y = self.apple.y / self.height

        # Danger metrics
        danger_straight = self._is_direction_dangerous(self.direction, lookahead=3)
        danger_right = self._is_direction_dangerous(self._get_clockwise_direction(), lookahead=3)
        danger_left = self._is_direction_dangerous(self._get_counterclockwise_direction(), lookahead=3)

        # Direction one-hot encoding
        dir_left = int(self.direction == Direction.LEFT)
        dir_right = int(self.direction == Direction.RIGHT)
        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)

        # Relative position of apple
        food_left = int(self.apple.x < self.head.x)
        food_right = int(self.apple.x > self.head.x)
        food_up = int(self.apple.y < self.head.y)
        food_down = int(self.apple.y > self.head.y)

        # Distances to body or obstacles
        forward_body_dist = self._distance_to_body(self.direction)
        right_body_dist = self._distance_to_body(self._get_clockwise_direction())
        left_body_dist = self._distance_to_body(self._get_counterclockwise_direction())

        forward_self_dist = self._distance_to_self(self.direction)
        right_self_dist = self._distance_to_self(self._get_clockwise_direction())
        left_self_dist = self._distance_to_self(self._get_counterclockwise_direction())

        return np.array([
            # Danger indicators
            danger_straight, danger_right, danger_left,
            # Direction encoding
            dir_left, dir_right, dir_up, dir_down,
            # Relative food location
            food_left, food_right, food_up, food_down,
            # Normalised positions
            head_x, head_y, apple_x, apple_y,
            # Lookahead distances
            forward_body_dist, right_body_dist, left_body_dist,
            forward_self_dist, right_self_dist, left_self_dist
        ])

    def _is_direction_dangerous(self, direction: Direction, lookahead: int = LOOKAHEAD_VAL) -> float:
        """
        Determine whether moving in the given direction leads to danger,
        such as wall, body, or obstacle collision, or potential trap.

        Args:
            direction (Direction): Direction to evaluate.
            lookahead (int): Number of blocks to check ahead.

        Returns:
            float: 
                1.0 if the direction leads to certain danger,
                0.2 if there's a potential trap (corner),
                0.0 if the direction appears safe.
        """
        x, y = self.head.x, self.head.y

        for step in range(1, lookahead + 1):
            if direction == Direction.RIGHT:
                check_x, check_y = x + step * self.block_size, y
            elif direction == Direction.LEFT:
                check_x, check_y = x - step * self.block_size, y
            elif direction == Direction.DOWN:
                check_x, check_y = x, y + step * self.block_size
            elif direction == Direction.UP:
                check_x, check_y = x, y - step * self.block_size

            check_pos = Coordinate(check_x, check_y)

            if (
                check_x < 0 or check_x >= self.width or
                check_y < 0 or check_y >= self.height
            ):
                return 1.0

            if check_pos in self.snake[1:-1]:
                return 1.0

            if self._is_in_obstacles(check_pos):
                return 1.0

            if self._is_corner_trap(check_pos, direction):
                return 0.2  # Possibly dangerous

        return 0.0

    def _is_corner_trap(self, pos: Coordinate, direction: Direction) -> bool:
        """
        Determine whether a given position leads to a corner trap,
        where the snake has very limited exit options.

        Args:
            pos (Coordinate): Position to check.
            direction (Direction): Direction the snake is moving in.

        Returns:
            bool: True if a trap is detected, False otherwise.
        """
        if len(self.snake) < 5:
            return False

        adjacent_positions = []

        for check_dir in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            if (
                (direction == Direction.UP and check_dir == Direction.DOWN) or
                (direction == Direction.DOWN and check_dir == Direction.UP) or
                (direction == Direction.LEFT and check_dir == Direction.RIGHT) or
                (direction == Direction.RIGHT and check_dir == Direction.LEFT)
            ):
                continue

            check_x, check_y = pos.x, pos.y

            if check_dir == Direction.RIGHT:
                check_x += self.block_size
            elif check_dir == Direction.LEFT:
                check_x -= self.block_size
            elif check_dir == Direction.DOWN:
                check_y += self.block_size
            elif check_dir == Direction.UP:
                check_y -= self.block_size

            check_pos = Coordinate(check_x, check_y)

            if (
                0 <= check_x < self.width and
                0 <= check_y < self.height and
                check_pos not in self.snake and
                not self._is_in_obstacles(check_pos)
            ):
                adjacent_positions.append(check_pos)

        return len(adjacent_positions) < 2

    def _get_clockwise_direction(self) -> Direction:
        """
        Get the direction that is clockwise from the current one.

        Returns:
            Direction: Clockwise direction.
        """
        if self.direction == Direction.RIGHT:
            return Direction.DOWN
        elif self.direction == Direction.DOWN:
            return Direction.LEFT
        elif self.direction == Direction.LEFT:
            return Direction.UP
        else:  # self.direction == Direction.UP
            return Direction.RIGHT

    def _get_counterclockwise_direction(self) -> Direction:
        """
        Get the direction that is counterclockwise from the current one.

        Returns:
            Direction: Counterclockwise direction.
        """
        if self.direction == Direction.RIGHT:
            return Direction.UP
        elif self.direction == Direction.UP:
            return Direction.LEFT
        elif self.direction == Direction.LEFT:
            return Direction.DOWN
        else:  # self.direction == Direction.DOWN
            return Direction.RIGHT

    def render(self, rewards: List[float], scores: List[int]) -> None:
        """
        Render the Nibbles game environment, including snake, apple, obstacles, score, and
        training statistics overlay.

        Args:
            rewards (List[float]): List of episode rewards for chart overlay.
            scores (List[int]): List of episode scores for chart overlay.
        """
        if not self.render_mode:
            return

        self.display.fill((0, 0, 0))
        self.display.blit(apple_img, (self.apple.x, self.apple.y))

        # Draw obstacles
        for obs_group in self.obstacles:
            for (ox, oy) in obs_group:
                pygame.draw.rect(self.display, (255, 100, 0), (ox, oy, BLOCK_SIZE, BLOCK_SIZE))

        # Draw snake
        for idx, coord in enumerate(self.snake):
            colour = (200, 0, 0) if idx == 0 else (0, 200, 0)
            pygame.draw.rect(self.display, colour, (coord.x, coord.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display score
        font_small = pygame.font.SysFont(None, 14)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))

        # Display label surfaces
        for i, label_surf in enumerate(self.label_surfs):
            y_pos = self.height - 50 + i * 12
            self.display.blit(label_surf, (10, y_pos))

        # Display reward/score inset
        values = self._draw_inset_chart(rewards, scores)
        if values:
            for i, value in enumerate(values):
                value_surf = font_small.render(value, True, (255, 255, 255))
                y_pos = self.height - 50 + i * 12
                self.display.blit(value_surf, (80, y_pos))

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_inset_chart(self, rewards: List[float], scores: List[int]) -> Optional[List[str]]:
        """
        Prepare values for inset overlay showing recent training performance.

        Args:
            rewards (List[float]): Recent episode rewards.
            scores (List[int]): Recent episode scores.

        Returns:
            Optional[List[str]]: Text values including max and last score/reward.
        """
        if not self.render_mode or len(scores) < 2:
            return None

        N = REWARDS_DOMAIN
        recent_rewards = rewards[-N:]
        recent_scores = scores[-N:]

        max_r = max(recent_rewards)
        last_r = recent_rewards[-1]
        max_s = max(recent_scores)
        last_s = recent_scores[-1]

        values = [f"{max_s}", f"{last_s}", f"{max_r:.2f}", f"{last_r:.2f}"]
        return values