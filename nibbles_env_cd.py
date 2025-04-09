"""
====================================================================================
Nibbles Environment: Rendering Utilities for Visual Gameplay and Performance Display
====================================================================================

This module handles the visual rendering for the Nibbles reinforcement learning
environment. It provides a real-time graphical display of the game state, including:

- Snake and apple visualisation with continuous movement
- Obstacles for different levels
- Performance metrics overlay (score, reward history)
- Inset chart summarising recent training progress

Author: Amin Haghighatbin (updated)
Version: v1.1.0
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
from outfit import NibbleOutfit

# ─────────────────────────────── GLOBAL CONSTANTS ────────────────────────────────── #
BLOCK_SIZE = 20
MAP_WIDTH = 630
MAP_HEIGHT = 480
FONT_SIZE = 24

FPS = 100
SPEED = FPS * 6
REWARDS_DOMAIN = 50
LOOKAHEAD_VAL = 5
SHORT_SELF_COLLISION = 5
LONG_SELF_COLLISION = 30
FOOD_IMAGE_PATH = 'Images/glow_ball.png'

pygame.init()
font = pygame.font.Font(None, FONT_SIZE)

# Load and scale the apple image globally
apple_img = pygame.image.load(FOOD_IMAGE_PATH)
apple_img = pygame.transform.scale(apple_img, (BLOCK_SIZE, BLOCK_SIZE))

# Named tuple for discrete coordinates (used for obstacles and apple placement)
Coordinate = namedtuple('Coordinate', 'x, y')

# ─────────────────────────────────────────────────────────────────────────────────── #

class Direction(Enum):
    """Enumeration for possible movement directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class NibblesEnv:
    """
    A 'Snake' game environment for reinforcement learning agents with continuous movement.

    Features:
    ---------
    - Continuous snake motion with smooth interpolation of heading.
    - Grid-based obstacles and apple placement.
    - Supports observation, reward shaping, and curriculum training.
    - Optional rendering mode with smooth visual feedback.

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
        # self.map = LondonMap(self.width, self.height, image_path='london_streets.png')
        # self.labyrinth = DreamLabyrinth(self.width, self.height, cell_size=40)

        # Continuous movement parameters:
        self.speed = SPEED
        self.rotation_speed = 180.0

        self.collision_threshold = self.block_size / 2.0

        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.NOFRAME |
                pygame.DOUBLEBUF |
                pygame.HWSURFACE
            )

            # Create a fog overlay that covers the whole display.
            # self.fog = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            # self.fog = self.fog.convert_alpha()
            
            # Set parameters for the fog effect.
            # edge_thickness = 40  # How many pixels wide the fog effect should extend inward.
            # max_alpha = 90        # Maximum opacity at the very edge.
            
            # Define the inner (clear) rectangle by leaving a margin of edge_thickness.
            # inner_rect = pygame.Rect(
            #     edge_thickness,
            #     edge_thickness,
            #     self.width - 2 * edge_thickness,
            #     self.height - 2 * edge_thickness
            # )

            # Loop over every pixel to compute an alpha value based on its distance from the inner clear area.
            # for x in range(self.width):
            #     for y in range(self.height):
            #         if inner_rect.collidepoint(x, y):
            #             # Inside the clear area, fully transparent.
            #             continue

            #         # Determine horizontal and vertical distances from the inner rectangle.
            #         dx = max(inner_rect.left - x, 0, x - inner_rect.right)
            #         dy = max(inner_rect.top - y, 0, y - inner_rect.bottom)
                    
            #         # For corners, use the Euclidean (hypotenuse) distance; for straight edges, the larger delta works fine.
            #         if dx > 0 and dy > 0:
            #             dist = math.hypot(dx, dy)
            #         else:
            #             dist = max(dx, dy)
                    
            #         # Compute a fog factor that goes from 1 (at the very edge) down to 0 (at the inner border).
            #         fog_factor = max(0.0, min(1.0, (edge_thickness - dist) / edge_thickness))
            #         alpha = int((1 - fog_factor) * max_alpha)
                    
            #         # Set the fog pixel color (here white fog, adjust RGB if desired) with computed alpha.
            #         self.fog.set_at((x, y), (255, 255, 255, alpha))


            pygame.display.set_caption("Life - RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

            self.apple_img = pygame.image.load(FOOD_IMAGE_PATH)
            self.apple_img = pygame.transform.scale(self.apple_img, (BLOCK_SIZE, BLOCK_SIZE))
        else:
            self.display = None
            self.clock = None

        self.action_map = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.UP,
            3: Direction.DOWN
        }

        font_small = pygame.font.SysFont(None, 14)
        self.labels = ["Max Score:", "Last Score:", "Max Reward:", "Last Reward:"]
        self.label_surfs = [font_small.render(label, True, (255, 255, 255)) for label in self.labels]
        self.reset()

    @staticmethod
    def distance(a, b) -> float:
        """
        Compute the Euclidean distance between two points.

        Args:
            a, b: Iterable of two numbers (x,y).

        Returns:
            float: The straight-line (Euclidean) distance between a and b.
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def logistic(x, a=10, b=0.5):
        """
        Logistic (sigmoid) function.
        Returns values between 0 and 1.
        """
        return 1 / (1 + np.exp(-a * (x - b)))

    def generate_continuous_field(self, grid_shape, snake_coords, radius=5):
        """
        Generates a continuous field for the snake using a smooth (logistic) kernel.
        
        Parameters:
            grid_shape (tuple): (height, width) of the grid.
            snake_coords (list of tuples): Each tuple is (x, y) for snake cell positions (grid indices).
            radius (int): The radius (in grid cells) for smoothing.
        
        Returns:
            np.ndarray: A 2D array (of shape grid_shape) with values between 0 and 1.
        """
        H, W = grid_shape
        occupancy = np.zeros((H, W), dtype=np.float32)
        for x, y in snake_coords:
            if 0 <= x < W and 0 <= y < H:
                occupancy[y, x] = 1.0

        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx = i - radius
                dy = j - radius
                d = np.sqrt(dx**2 + dy**2)
                norm_d = d / radius
                kernel[i, j] = self.logistic(norm_d, a=10, b=0.5)
        kernel /= np.sum(kernel)

        try:
            from scipy.signal import convolve2d
            continuous_field = convolve2d(occupancy, kernel, mode='same', boundary='wrap')
        except ImportError:
            continuous_field = np.zeros((H, W), dtype=np.float32)
            for i in range(H):
                for j in range(W):
                    s = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            ni = (i + ki - radius) % H
                            nj = (j + kj - radius) % W
                            s += occupancy[ni, nj] * kernel[ki, kj]
                    continuous_field[i, j] = s

        return continuous_field

    def reset(self) -> np.ndarray:
        """
        Reset the game environment to its initial state.

        Resets snake continuous position, score, obstacles, apple, heading, etc.

        Returns:
            np.ndarray: The initial observation of the environment.
        """
        self.target_heading = 0.0  # RIGHT = 0 degrees.
        self.heading = self.target_heading
        # Centre the snake on the grid (as float)
        x_center = (self.width // self.block_size) // 2 * self.block_size
        y_center = (self.height // self.block_size) // 2 * self.block_size
        self.head_pos = np.array([x_center, y_center], dtype=float)

        # Initialise snake as a list of positions (head first). The desired initial length is 3 segments.
        self.initial_length = 3  # in number of segments
        self.snake = [self.head_pos.copy()]
        # Add additional segments behind the head (to the left).
        for i in range(1, self.initial_length):
            pos = np.array([x_center - i * self.block_size, y_center], dtype=float)
            self.snake.append(pos)
        # A set for collision checking is no longer based on exact positions; we use distance checks.
        self.score = 0

        # Generate obstacles for the current level
        lvl = Levels(block_size=self.block_size, width=self.width, height=self.height)
        self.obstacles = lvl._level_generator(level=self.level)

        self.apple = None
        self._place_food()
        self.step_count = 0

        return self._get_observation()

    def _place_food(self) -> None:
        """
        Place the apple at a random valid location on the grid,
        avoiding the snake, obstacles, and the four corners.
        """
        num_cells_x = self.width // self.block_size
        num_cells_y = self.height // self.block_size

        corners = {
            Coordinate(0, 0),
            Coordinate((num_cells_x - 1) * self.block_size, 0),
            Coordinate(0, (num_cells_y - 1) * self.block_size),
            Coordinate((num_cells_x - 1) * self.block_size, (num_cells_y - 1) * self.block_size)
        }

        while True:
            x = random.randint(0, num_cells_x - 1) * self.block_size
            y = random.randint(0, num_cells_y - 1) * self.block_size
            # Use discrete coordinate for apple placement.
            coord = Coordinate(x, y)
            # Check that the apple is not too near any snake segment.
            if any(self.distance(np.array([x, y]), segment) < self.block_size for segment in self.snake):
                continue
            if self._is_in_obstacles(coord) or coord in corners:
                continue
            self.apple = coord
            break

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance the environment by one frame using the specified action.

        The continuous movement is updated based on a time step (dt).

        Args:
            action (int): Index of the action to perform:
                        (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)

        Returns:
            tuple:
                - np.ndarray: New state observation.
                - float: Reward signal for the step.
                - bool: Whether the episode has terminated.
                - dict: Optional additional information.
        """
        # Use a fixed time step based on FPS.
        dt = 1.0 / FPS
        self.step_count += 1
        old_dist = self.distance(self.head_pos, np.array([self.apple.x, self.apple.y]))

        # Determine the new target heading based on action, but avoid 180-degree reversals.
        new_direction = self.action_map[action]
        # Map discrete direction to target angle (in degrees).
        mapping = {Direction.RIGHT: 0.0, Direction.LEFT: 180.0, Direction.UP: 270.0, Direction.DOWN: 90.0}
        desired_heading = mapping[new_direction]
        # Prevent a 180° immediate reversal by checking the difference.
        if abs(((desired_heading - self.heading + 180) % 360) - 180) == 180:
            # Keep current heading if reversal attempted.
            desired_heading = self.heading
        self.target_heading = desired_heading

        reward = -0.01  # Small time step penalty

        # Update continuous movement.
        self._update_heading(dt)
        self._move_snake(dt)

        # Check collisions.
        if self._is_collision():
            food_dist_factor = max(0.5, 1.0 - self.distance(self.head_pos, np.array([self.apple.x, self.apple.y])) / (self.width + self.height))
            reward = -10.0 * food_dist_factor
            done = True
            return self._get_observation(), reward, done, {}

        # Check apple collection using distance threshold.
        if self.distance(self.head_pos, np.array([self.apple.x, self.apple.y])) < self.collision_threshold:
            base_reward = 10
            length_bonus = 2.0 * self.score
            reward = base_reward + length_bonus
            self.score += 1
            self.final_score = max(self.final_score, self.score)
            self._place_food()
        else:
            # Trim the snake to maintain constant length when no apple is collected.
            self._trim_snake()

        # Reward shaping for distance improvement.
        new_dist = self.distance(self.head_pos, np.array([self.apple.x, self.apple.y]))
        dist_improvement = old_dist - new_dist
        map_diagonal = math.sqrt(self.width**2 + self.height**2)
        normalised = dist_improvement / map_diagonal
        reward += 0.2 * normalised if normalised > 0 else 0.1 * normalised

        # Bonus for staying alive.
        reward += 0.001 * len(self.snake)

        # Additional penalty if the head is too close to any body segment.
        for segment in self.snake[1:]:
            if self.distance(self.head_pos, segment) < self.collision_threshold:
                reward -= 0.2

        # Adaptive episode timeout.
        base_diagonal = math.sqrt(self.width**2 + self.height**2)
        current_diagonal = math.sqrt(self.width**2 + self.height**2)
        max_steps = int(1000 * (current_diagonal / base_diagonal)) + 50 * self.score
        # timeout_in_seconds = 10
        # self.elapsed_time += dt  # dt provided per step
        # done = self.elapsed_time >= timeout_in_seconds
        # done = self.step_count >= max_steps

        print(f'Step: {self.step_count}/{max_steps}')
        if self.step_count >= max_steps:
            done = True
        else:
            done = False


        if self.render_mode:
            self.render([], [])

        return self._get_observation(), reward, done, {}

    def _update_heading(self, dt: float) -> None:
        """
        Smoothly update the snake's current heading toward the target heading.

        Args:
            dt (float): Time step (in seconds).
        """
        # Compute minimal angular difference.
        diff = ((self.target_heading - self.heading + 180) % 360) - 180
        max_rotation = self.rotation_speed * dt
        if abs(diff) < max_rotation:
            self.heading = self.target_heading
        else:
            self.heading += np.sign(diff) * max_rotation
            self.heading %= 360

    def _move_snake(self, dt: float) -> None:
        """
        Update the snake's position continuously based on its current heading and speed.
        The head position is updated and then the new position is added to the snake body.
        """
        radians = math.radians(self.heading)
        velocity = np.array([math.cos(radians), math.sin(radians)]) * self.speed
        self.head_pos += velocity * dt

        # Wrap-around at screen edges.
        self.head_pos[0] %= self.width
        self.head_pos[1] %= self.height

        # Insert the new head position at the start of the snake body.
        self.snake.insert(0, self.head_pos.copy())

        # When an apple is collected, the snake grows (and we do not trim).
        # Otherwise, _trim_snake() in the step() will remove extra tail segments.

    def _trim_snake(self) -> None:
        """
        Trim the snake's body so that its total length matches the desired length.
        The desired length is (initial_length + number of apples eaten) * BLOCK_SIZE.
        """
        desired_length = (self.initial_length + self.score) * self.block_size
        total_length = 0.0
        new_snake = [self.snake[0]]
        # Walk along the snake segments, summing distances.
        for i in range(1, len(self.snake)):
            seg_length = self.distance(self.snake[i-1], self.snake[i])
            if total_length + seg_length < desired_length:
                new_snake.append(self.snake[i])
                total_length += seg_length
            else:
                # Interpolate between snake[i-1] and snake[i] if needed.
                remaining = desired_length - total_length
                if seg_length > 0:
                    ratio = remaining / seg_length
                    new_point = self.snake[i-1] + ratio * (self.snake[i] - self.snake[i-1])
                    new_snake.append(new_point)
                break
        self.snake = new_snake

    def _is_collision(self) -> bool:
        # In continuous mode the head is naturally very close to its immediate neighbor(s).
        # Start checking from index 2 (skip the very next segment) to avoid false positive collisions.
        for segment in self.snake[2:]:
            if self.distance(self.head_pos, segment) < self.collision_threshold:
                return True

        # Check collision with obstacles.
        for obs_group in self.obstacles:
            for (ox, oy) in obs_group:
                if (ox <= self.head_pos[0] < ox + self.block_size) and (oy <= self.head_pos[1] < oy + self.block_size):
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
        # For continuous movement, we sample along the ray from head_pos.
        steps = np.linspace(0, max_steps * self.block_size, num=max_steps)
        dir_vector = {
            Direction.RIGHT: np.array([1, 0]),
            Direction.LEFT: np.array([-1, 0]),
            Direction.DOWN: np.array([0, 1]),
            Direction.UP: np.array([0, -1])
        }[direction]
        for s in steps:
            pos = self.head_pos + dir_vector * s
            # Wrap-around.
            pos[0] %= self.width
            pos[1] %= self.height
            # Check collision with body.
            for segment in self.snake:
                if self.distance(pos, segment) < self.collision_threshold:
                    return s / (max_steps * self.block_size)
            # Check obstacles (using grid approximation).
            discrete_pos = Coordinate(int(pos[0] // self.block_size * self.block_size), int(pos[1] // self.block_size * self.block_size))
            if self._is_in_obstacles(discrete_pos):
                return s / (max_steps * self.block_size)
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
        steps = np.linspace(0, max_look * self.block_size, num=max_look)
        dir_vector = {
            Direction.RIGHT: np.array([1, 0]),
            Direction.LEFT: np.array([-1, 0]),
            Direction.DOWN: np.array([0, 1]),
            Direction.UP: np.array([0, -1])
        }[direction]
        for s in steps:
            pos = self.head_pos + dir_vector * s
            pos[0] %= self.width
            pos[1] %= self.height
            for segment in self.snake:
                if self.distance(pos, segment) < self.collision_threshold:
                    return s / (max_look * self.block_size)
            discrete_pos = Coordinate(int(pos[0] // self.block_size * self.block_size), int(pos[1] // self.block_size * self.block_size))
            if self._is_in_obstacles(discrete_pos):
                return s / (max_look * self.block_size)
        return 1.0

    def _get_observation(self) -> np.ndarray:
        """
        Construct a feature vector representing the current game state.
        Combines discrete state (danger signals, relative food position, distances)
        with a continuous local patch generated from a smooth field.

        Returns:
            np.ndarray: Normalised state representation as input for the agent.
        """
        head_x = self.head_pos[0] / self.width
        head_y = self.head_pos[1] / self.height
        apple_x = self.apple.x / self.width
        apple_y = self.apple.y / self.height

        danger_straight = self._is_direction_dangerous(Direction.RIGHT if abs(self.heading-0.0)<1e-3 else
                                                         Direction.LEFT if abs(self.heading-180.0)<1e-3 else
                                                         Direction.UP if abs(self.heading-270.0)<1e-3 else
                                                         Direction.DOWN, lookahead=3)
        danger_right = self._is_direction_dangerous(self._get_clockwise_direction(), lookahead=3)
        danger_left = self._is_direction_dangerous(self._get_counterclockwise_direction(), lookahead=3)

        dir_left = int(self.target_heading == 180.0)
        dir_right = int(self.target_heading == 0.0)
        dir_up = int(self.target_heading == 270.0)
        dir_down = int(self.target_heading == 90.0)

        food_left = int(self.apple.x < self.head_pos[0])
        food_right = int(self.apple.x > self.head_pos[0])
        food_up = int(self.apple.y < self.head_pos[1])
        food_down = int(self.apple.y > self.head_pos[1])

        forward_body_dist = self._distance_to_body(Direction.RIGHT if abs(self.heading-0.0)<1e-3 else
                                                     Direction.LEFT if abs(self.heading-180.0)<1e-3 else
                                                     Direction.UP if abs(self.heading-270.0)<1e-3 else
                                                     Direction.DOWN)
        right_body_dist = self._distance_to_body(self._get_clockwise_direction())
        left_body_dist = self._distance_to_body(self._get_counterclockwise_direction())
        forward_self_dist = self._distance_to_self(Direction.RIGHT if abs(self.heading-0.0)<1e-3 else
                                                    Direction.LEFT if abs(self.heading-180.0)<1e-3 else
                                                    Direction.UP if abs(self.heading-270.0)<1e-3 else
                                                    Direction.DOWN)
        right_self_dist = self._distance_to_self(self._get_clockwise_direction())
        left_self_dist = self._distance_to_self(self._get_counterclockwise_direction())

        discrete_obs = np.array([
            danger_straight, danger_right, danger_left,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down,
            head_x, head_y, apple_x, apple_y,
            forward_body_dist, right_body_dist, left_body_dist,
            forward_self_dist, right_self_dist, left_self_dist
        ])

        grid_height = self.height // self.block_size
        grid_width = self.width // self.block_size
        grid_shape = (grid_height, grid_width)
        # Convert continuous snake positions to grid indices.
        snake_coords = [(int(p[0] // self.block_size), int(p[1] // self.block_size)) for p in self.snake]
        continuous_field = self.generate_continuous_field(grid_shape, snake_coords, radius=5)

        head_grid_x = int(self.head_pos[0] // self.block_size)
        head_grid_y = int(self.head_pos[1] // self.block_size)
        patch_size = 5
        half_patch = patch_size // 2
        y_start = max(0, head_grid_y - half_patch)
        y_end = min(grid_height, head_grid_y + half_patch + 1)
        x_start = max(0, head_grid_x - half_patch)
        x_end = min(grid_width, head_grid_x + half_patch + 1)
        patch = continuous_field[y_start:y_end, x_start:x_end]
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        patch_flat = patch.flatten()
        full_obs = np.concatenate([discrete_obs, patch_flat])
        return full_obs

    def _is_direction_dangerous(self, direction: Direction, lookahead: int = LOOKAHEAD_VAL) -> float:
        """
        Determine whether moving in the given direction leads to danger.
        Returns 1.0 for imminent danger, 0.2 for potential corner traps, 0.0 if safe.
        """
        pos = self.head_pos.copy()
        for step in range(1, lookahead + 1):
            dir_vector = {
                Direction.RIGHT: np.array([1, 0]),
                Direction.LEFT: np.array([-1, 0]),
                Direction.DOWN: np.array([0, 1]),
                Direction.UP: np.array([0, -1])
            }[direction]
            pos = pos + dir_vector * self.block_size
            pos[0] %= self.width
            pos[1] %= self.height
            discrete_pos = Coordinate(int(pos[0] // self.block_size * self.block_size),
                                      int(pos[1] // self.block_size * self.block_size))
            if (pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height):
                return 1.0
            for segment in self.snake[1:-1]:
                if self.distance(pos, segment) < self.collision_threshold:
                    return 1.0
            if self._is_in_obstacles(discrete_pos):
                return 1.0
            if self._is_corner_trap(discrete_pos, direction):
                return 0.2
        return 0.0

    def _is_corner_trap(self, pos: Coordinate, direction: Direction) -> bool:
        """
        Determine whether a given position leads to a corner trap.
        """
        if len(self.snake) < 5:
            return False
        adjacent_positions = []
        for check_dir in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            if ((direction == Direction.UP and check_dir == Direction.DOWN) or
                (direction == Direction.DOWN and check_dir == Direction.UP) or
                (direction == Direction.LEFT and check_dir == Direction.RIGHT) or
                (direction == Direction.RIGHT and check_dir == Direction.LEFT)):
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
            
            if (0 <= check_x < self.width and 0 <= check_y < self.height and
                not any(np.allclose(np.array([check_pos.x, check_pos.y]), seg, atol=1e-3) for seg in self.snake) and
                not self._is_in_obstacles(check_pos)):
                adjacent_positions.append(check_pos)

        return len(adjacent_positions) < 2

    def _get_clockwise_direction(self) -> Direction:
        """
        Get the direction that is clockwise from the current one.
        """
        if self.target_heading == 0.0:    # RIGHT
            return Direction.DOWN
        elif self.target_heading == 90.0:  # DOWN
            return Direction.LEFT
        elif self.target_heading == 180.0: # LEFT
            return Direction.UP
        else:                             # UP (270)
            return Direction.RIGHT

    def _get_counterclockwise_direction(self) -> Direction:
        """
        Get the direction that is counterclockwise from the current one.
        """
        if self.target_heading == 0.0:    # RIGHT
            return Direction.UP
        elif self.target_heading == 90.0:  # DOWN
            return Direction.RIGHT
        elif self.target_heading == 180.0: # LEFT
            return Direction.DOWN
        else:                             # UP (270)
            return Direction.LEFT

    def render(self, rewards: List[float], scores: List[int]) -> None:
        """
        Render the Nibbles game environment, including snake, apple, obstacles, score, 
        and training statistics overlay.
        """
        if not self.render_mode:
            return

        # self.map.render(self.display)
        self.display.fill((0, 0, 0))
        # temp_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # temp_surf.fill((255, 255, 255, 255))
        # self.display.blit(temp_surf, (0, 0))
        # self.labyrinth.render(self.display)
        # self.display.blit(self.fog, (0,0))
        self.display.blit(apple_img, (self.apple.x, self.apple.y))

        # Draw obstacles.
        for obs_group in self.obstacles:
            for (ox, oy) in obs_group:
                pygame.draw.rect(self.display, (255, 100, 0), (ox, oy, BLOCK_SIZE, BLOCK_SIZE))

        # Glowing Outfit
        self.outfit = NibbleOutfit(self.display, self.block_size)
        self.outfit.draw_snake(self.snake)

        # Draw snake using soft circles.
        # for idx, segment in enumerate(self.snake):
        #     center = (int(segment[0]), int(segment[1]))
        #     radius = self.block_size // 2
        #     if idx == 0:
        #         # Head: different color.
        #         self._draw_soft_circle(self.display, center, radius, (250, 250, 250), glow_color=(255,68,80), glow_width=1)
        #     else:
        #         self._draw_soft_circle(self.display, center, radius, (21, 244, 238), glow_color=(21,244,238), glow_width=1)

        # Display score.
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))

        font_small = pygame.font.SysFont(None, 14)
        for i, label_surf in enumerate(self.label_surfs):
            y_pos = self.height - 50 + i * 12
            self.display.blit(label_surf, (10, y_pos))
        values = self._draw_inset_chart(rewards, scores)
        if values:
            for i, value in enumerate(values):
                value_surf = font_small.render(value, True, (255, 255, 255))
                y_pos = self.height - 50 + i * 12
                self.display.blit(value_surf, (80, y_pos))
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_soft_circle(self, surface, center, radius, rgb_color, glow_color=None, glow_width=10):
        """
        Draw a circle with smooth, fuzzy edges and an outer glow.
        
        Args:
            surface: Target pygame surface.
            center: Tuple (x, y) for the circle center.
            radius: The base circle radius.
            rgb_color: The inner (solid) circle color.
            glow_color: Color used for the glow (defaults to rgb_color if None).
            glow_width: How far the glow extends beyond the circle in pixels.
        """
        glow_color = glow_color or rgb_color

        # First, draw the outer glow.
        # Create a surface big enough to hold the circle plus the glow.
        outer_radius = radius + glow_width
        glow_surface = pygame.Surface((outer_radius*2, outer_radius*2), pygame.SRCALPHA)
        # Draw concentric circles from the outer edge inward.
        for r in range(outer_radius, radius, -1):
            # Normalize the radius difference (0 at the base, 1 at the outermost edge)
            norm = (r - radius) / glow_width  
            # Adjust the alpha using the logistic function (tweak a and b as desired)
            alpha = int(255 * self.logistic(norm, a=10, b=0.5))
            color = (*glow_color, alpha)
            pygame.draw.circle(glow_surface, color, (outer_radius, outer_radius), r)
        # Blit the glow surface to the main surface.
        surface.blit(glow_surface, (center[0]-outer_radius, center[1]-outer_radius))
        
        # Next, draw the soft inner circle (similar to your original approach).
        soft_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            norm_r = r / radius
            alpha = int(255 * self.logistic(norm_r, a=10, b=0.5))
            color = (*glow_color, alpha)
            pygame.draw.circle(soft_surface, color, (radius, radius), r)
        surface.blit(soft_surface, (center[0]-radius, center[1]-radius))
        
        # Finally, draw the crisp inner circle.
        pygame.draw.circle(surface, rgb_color, center, radius)

    def _draw_inset_chart(self, rewards: List[float], scores: List[int]) -> Optional[List[str]]:
        """
        Prepare values for inset overlay showing recent training performance.
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
