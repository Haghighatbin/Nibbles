import pygame
import math
import random

class NibbleOutfit:
    def __init__(self, display, block_size):
        """
        Initialize the snake's visual appearance.
        
        Args:
            display: The pygame surface to draw on
            block_size: The size of each snake segment
        """
        self.display = display
        self.block_size = block_size
    
    def draw_snake(self, snake_segments):
        """
        Draw the snake with light beam effects.
        
        Args:
            snake_segments: List of [x, y] coordinates for each snake segment
        """
        # Clear the display - you might want to remove this if your game handles clearing elsewhere
        
        # Draw a subtle background glow that follows the snake
        if len(snake_segments) > 0:
            self._draw_background_glow(snake_segments[0])
        
        # Draw snake segments
        for idx, segment in enumerate(snake_segments):
            center = (int(segment[0]), int(segment[1]))
            radius = self.block_size // 2
            is_head = (idx == 0)
            self._draw_light_beam_segment(center, radius, idx, is_head)
    
    def _draw_light_beam_segment(self, center, radius, idx, is_head=False):
        """
        Draw a snake segment that looks like a beam of light with a pulsating core,
        trailing particles, and dynamic color transitions.
        """
        # Create base surface for the segment
        segment_size = radius * 3
        segment_surface = pygame.Surface((segment_size*2, segment_size*2), pygame.SRCALPHA)
        
        # Calculate pulse effect (value between 0.7 and 1.0 based on time)
        pulse = 0.6 + 0.2 * math.sin(pygame.time.get_ticks() / 200)
        
        # Define core and outer colors - colors shift based on position in the snake
        # hue_shift = (idx * 5) % 360
        hue_shift = 520
        if is_head:
            core_color = self._hsv_to_rgb((hue_shift, 0.2, 1))  # Almost white core for head
            outer_color = self._hsv_to_rgb((hue_shift, 0.8, 1))  # Vibrant outer for head
        else:
            core_color = self._hsv_to_rgb(((hue_shift + 20) % 360, 0.4, 0.9))
            outer_color = self._hsv_to_rgb(((hue_shift + 40) % 360, 0.7, 0.8))
        
        # Draw multiple layers of glow with different sizes and opacities
        for i in range(5):
            glow_radius = int(radius * (1 + i*0.5) * pulse)
            alpha = int(200 * (1 - i/5))
            glow_color = (*outer_color, alpha)
            pygame.draw.circle(segment_surface, glow_color, (segment_size, segment_size), glow_radius)
        
        # Draw bright core
        core_radius = int(radius * 0.7 * pulse)
        pygame.draw.circle(segment_surface, (*core_color, 255), (segment_size, segment_size), core_radius)
        
        # Add light rays emanating from the segment
        if is_head or idx % 3 == 0:  # Only some segments emit rays
            self._draw_light_rays(segment_surface, (segment_size, segment_size), radius, core_color)
        
        # Add particles trailing behind each segment
        if not is_head:
            self._draw_trailing_particles(segment_surface, (segment_size, segment_size), radius, outer_color, idx)
        
        # Render the complete segment to the main surface
        self.display.blit(segment_surface, 
                    (center[0]-segment_size, center[1]-segment_size))

    def _draw_light_rays(self, surface, center, radius, color):
        """Draw light rays emanating from the segment"""
        num_rays = 6
        ray_length = radius * 2
        
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays + (pygame.time.get_ticks() / 2000)
            end_x = center[0] + ray_length * math.cos(angle)
            end_y = center[1] + ray_length * math.sin(angle)
            
            # Create gradient ray
            for step in range(10):
                t = step / 10
                x = center[0] + t * (end_x - center[0])
                y = center[1] + t * (end_y - center[1])
                alpha = int(150 * (1 - t))
                point_radius = int(radius * 0.2 * (1 - t))
                pygame.draw.circle(surface, (*color, alpha), (int(x), int(y)), point_radius)

    def _draw_trailing_particles(self, surface, center, radius, color, idx):
        """Draw particles trailing behind the segment"""
        # Use segment index to create variation
        seed = idx + pygame.time.get_ticks() / 1000
        rand_gen = random.Random(int(seed * 100) % 10000)
        
        num_particles = 6
        for _ in range(num_particles):
            offset_x = rand_gen.randint(-radius, radius)
            offset_y = rand_gen.randint(-radius, radius)
            size = radius * rand_gen.uniform(0.1, 0.3)
            alpha = rand_gen.randint(50, 150)
            
            particle_pos = (center[0] + offset_x, center[1] + offset_y)
            pygame.draw.circle(surface, (*color, alpha), particle_pos, size)

    def _hsv_to_rgb(self, hsv):
        """Convert HSV color to RGB"""
        h, s, v = hsv
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def _draw_background_glow(self, head_position):
        """Create a subtle glow effect on the background behind the snake"""
        glow_surface = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)
        glow_radius = self.block_size * 4
        
        # Use a gradient for the background glow
        for r in range(glow_radius, 0, -5):
            alpha = int(10 * (1 - r/glow_radius))
            color = (80, 50, 120, alpha)  # Subtle purple glow
            pygame.draw.circle(glow_surface, color, (int(head_position[0]), int(head_position[1])), r)
        
        self.display.blit(glow_surface, (0, 0))