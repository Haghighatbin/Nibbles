from collections import namedtuple

Coordinate = namedtuple('Coordinate', 'x, y')

class Levels:
    def __init__(self, block_size, width, height) -> None:
        self.block_size = block_size  # Should be 20 pixels
        self.width = width  # 640 pixels
        self.height = height  # 480 pixels
        # Use the smaller of width and height as a reference dimension.
        self.base_dim = min(width, height)
        self.center_x = width // 2
        self.center_y = height // 2

    def _get_centered_start_position(self, OFFSETS, SQUARE_SIZE):
        """Calculate the starting position to center a pattern on the screen.
        
        Args:
            pattern_width_blocks: Pattern width in number of blocks
            pattern_height_blocks: Pattern height in number of blocks
            
        Returns:
            Tuple (start_x, start_y) coordinates for the top-left of the centered pattern
        """
        min_x = min(offset[0] for offset in OFFSETS)
        max_x = max(offset[0] for offset in OFFSETS) + SQUARE_SIZE * self.block_size
        min_y = min(offset[1] for offset in OFFSETS)
        max_y = max(offset[1] for offset in OFFSETS) + SQUARE_SIZE * self.block_size

        pattern_width = max_x - min_x
        pattern_height = max_y - min_y

        # Compute the center of the map.
        map_center_x = self.width / 2
        map_center_y = self.height / 2

        # Compute the center of the pattern.
        pattern_center_x = min_x + pattern_width / 2
        pattern_center_y = min_y + pattern_height / 2

        # Calculate translation to center the pattern.
        # Compute raw translation.
        raw_tx = int(map_center_x - pattern_center_x)
        raw_ty = int(map_center_y - pattern_center_y)

        # Force translation to be a multiple of block_size.
        translation_x = (raw_tx // self.block_size) * self.block_size - self.block_size / 2
        translation_y = (raw_ty // self.block_size) * self.block_size - self.block_size / 2

        return translation_x, translation_y
    
    def _level_generator(self, level):
        obstacles = []
        if level == 0:
            return obstacles

        # Level 1: "Portal Pathways" - Two squares with narrow passages between
        if level == 1:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
                        
            OFFSETS = []
            MARGIN = self.block_size  # Assuming 20
            SIZE = 7  # 7x7 grid

            def generate_square(anchor_x, anchor_y):
                block_set = []
                for i in range(SIZE):
                    for j in range(SIZE):
                        # Skip the center row and column to form cross-shaped channel
                        if not (i == SIZE // 2 or j == SIZE // 2):
                            x = anchor_x + i * MARGIN
                            y = anchor_y + j * MARGIN
                            block_set.append(Coordinate(x, y))
                return block_set

            # Top-left square (shifted toward center)
            anchor_1 = (
                self.width // 4 - (SIZE * MARGIN) // 2,
                self.height // 4 - (SIZE * MARGIN) // 2
            )
            blocks_1 = generate_square(*anchor_1)

            # Bottom-right square (also centered in its quadrant)
            anchor_2 = (
                self.width * 3 // 4 - (SIZE * MARGIN) // 2,
                self.height * 3 // 4 - (SIZE * MARGIN) // 2
            )
            blocks_2 = generate_square(*anchor_2)

            all_blocks = blocks_1 + blocks_2
            for block in all_blocks:
                OFFSETS.append(block)

        # Level 2: "Spiral Maze" - A symmetrical spiral pattern
        elif level == 2:
            SQUARE_SIZE = 1
            MARGIN = 2* self.block_size  # 20 pixels
            
            OFFSETS = []
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center in terms of blocks
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            # Create a symmetrical spiral
            spiral_size = min(center_x - 2, center_y - 2)  # Ensure spiral fits
            
            for layer in range(1, spiral_size - 1):
                # Top horizontal line (left to right)
                for i in range(-spiral_size + layer, spiral_size - layer):
                    OFFSETS.append([(center_x + i) * MARGIN, (center_y - spiral_size + layer) * MARGIN])
                
                # Right vertical line (top to bottom)
                for i in range(-spiral_size + layer + 1, spiral_size - layer):
                    OFFSETS.append([(center_x + spiral_size - layer - 1) * MARGIN, (center_y + i) * MARGIN])
                
                # Bottom horizontal line (right to left)
                for i in range(spiral_size - layer - 1, -spiral_size + layer - 1, -1):
                    OFFSETS.append([(center_x + i) * MARGIN, (center_y + spiral_size - layer - 1) * MARGIN])
                
                # Left vertical line (bottom to top)
                for i in range(spiral_size - layer - 2, -spiral_size + layer, -1):
                    OFFSETS.append([(center_x - spiral_size + layer) * MARGIN, (center_y + i) * MARGIN])
            
            # Create entrance to the spiral
            entrance_x = center_x - spiral_size + 1
            entrance_y = center_y
            OFFSETS = [offset for offset in OFFSETS if not (offset[0] == entrance_x * MARGIN and offset[1] == entrance_y * MARGIN)]

        # Level 3: "Symmetrical Cross" - A symmetrical cross pattern with passages
        elif level == 3:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            OFFSETS = []
            
            # Create horizontal line of the cross
            cross_width = width_blocks - 4
            for i in range(2, 2 + cross_width):
                # Skip the center for passage
                if abs(i - center_x) > 1:
                    OFFSETS.append([i * MARGIN, center_y * MARGIN])
            
            # Create vertical line of the cross
            cross_height = height_blocks - 4
            for i in range(2, 2 + cross_height):
                # Skip the center for passage
                if abs(i - center_y) > 1:
                    OFFSETS.append([center_x * MARGIN, i * MARGIN])
            
            # Add diagonal elements for complexity (but maintain passages)
            for i in range(-6, 7):
                for j in range(-6, 7):
                    if abs(i) == abs(j) and abs(i) > 1 and abs(i) < 7:
                        OFFSETS.append([(center_x + i) * MARGIN, (center_y + j) * MARGIN])

        # Level 4: "Diamond Grid" - Symmetrical diamond patterns
        elif level == 4:
            SQUARE_SIZE = 1
            MARGIN = 1 * self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            OFFSETS = []
            
            # Create main diamond
            diamond_size = 10
            for size in range(diamond_size + 1):
                # Top-right and bottom-right sides
                for i in range(size + 1):
                    # Create diamond outline only
                    if size == diamond_size or i == 0 or i == size:
                        OFFSETS.append([(center_x + i) * MARGIN, (center_y - size + i) * MARGIN])
                        OFFSETS.append([(center_x + i) * MARGIN, (center_y + size - i) * MARGIN])
                
                # Top-left and bottom-left sides
                for i in range(size + 1):
                    # Create diamond outline only
                    if size == diamond_size or i == 0 or i == size:
                        OFFSETS.append([(center_x - i) * MARGIN, (center_y - size + i) * MARGIN])
                        OFFSETS.append([(center_x - i) * MARGIN, (center_y + size - i) * MARGIN])
            
            # Create smaller diamonds at the corners - FILLED instead of just outlines
            corner_distance = diamond_size - 2
            corners = [
                [center_x - corner_distance, center_y - corner_distance],  # Top-left
                [center_x + corner_distance, center_y - corner_distance],  # Top-right
                [center_x - corner_distance, center_y + corner_distance],  # Bottom-left
                [center_x + corner_distance, center_y + corner_distance]   # Bottom-right
            ]
            
            small_size = 1
            for corner in corners:
                for size in range(small_size + 1):
                    # Fill the small diamonds completely
                    for i in range(size + 1):
                        for j in range(size + 1 - i):
                            # Top-right quadrant of diamond
                            OFFSETS.append([(corner[0] + j) * MARGIN, (corner[1] - i) * MARGIN])
                            # Bottom-right quadrant
                            OFFSETS.append([(corner[0] + j) * MARGIN, (corner[1] + i) * MARGIN])
                            # Top-left quadrant
                            OFFSETS.append([(corner[0] - j) * MARGIN, (corner[1] - i) * MARGIN])
                            # Bottom-left quadrant
                            OFFSETS.append([(corner[0] - j) * MARGIN, (corner[1] + i) * MARGIN])
            
            # Define clear passages through the main diamond
            passage_width = 1  # Width of the passage
            
            # Define the passage coordinates
            passage_areas = [
                # Top passage
                [(center_x - passage_width, center_y - diamond_size), 
                (center_x + passage_width, center_y - diamond_size + passage_width)],
                # Bottom passage
                [(center_x - passage_width, center_y + diamond_size - passage_width), 
                (center_x + passage_width, center_y + diamond_size)],
                # Left passage
                [(center_x - diamond_size, center_y - passage_width), 
                (center_x - diamond_size + passage_width, center_y + passage_width)],
                # Right passage
                [(center_x + diamond_size - passage_width, center_y - passage_width), 
                (center_x + diamond_size, center_y + passage_width)]
            ]
            
            # Remove the passages from OFFSETS
            filtered_offsets = []
            for offset in OFFSETS:
                x_block = offset[0] // MARGIN
                y_block = offset[1] // MARGIN
                
                # Check if this block is in any passage area
                in_passage = False
                for passage in passage_areas:
                    top_left = passage[0]
                    bottom_right = passage[1]
                    
                    if (top_left[0] <= x_block <= bottom_right[0] and 
                        top_left[1] <= y_block <= bottom_right[1]):
                        in_passage = True
                        break
                
                if not in_passage:
                    filtered_offsets.append(offset)
            
            OFFSETS = filtered_offsets

        # Level 5: "Symmetrical Maze" - A balanced maze pattern
        elif level == 5:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Create a symmetric maze pattern
            maze_half = [
                "##  #######   ##",
                "#       #      #",
                "# ## ## # ## # #",
                "# #     # #    #",
                "# # # ### #### #",
                "# #       #    #",
                "# ### ### # ####",
                "#       # #    #",
                "### ### # #### #",
                "#     # #      #",
                "# ### # ###### #",
                "# #   #        #",
                "# # ####### ####",
                "# #             ",
                "# ###### #######",
                "#               "
            ]
            
            # Calculate starting position for centering
            start_x = (width_blocks - len(maze_half[0]) * 2) // 2
            start_y = (height_blocks - len(maze_half)) // 2
            
            OFFSETS = []
            
            # Generate the left half of the maze
            for y, row in enumerate(maze_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        OFFSETS.append([(start_x + x) * MARGIN, (start_y + y) * MARGIN])
            
            # Generate the right half (mirrored)
            for y, row in enumerate(maze_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        # Mirror across the x-axis (width_blocks - x - 1)
                        mirror_x = start_x + len(maze_half[0]) * 2 - x - 1
                        OFFSETS.append([mirror_x * MARGIN, (start_y + y) * MARGIN])

        # Level 6: "Yin Yang" with proper exit paths
        elif level == 6:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            OFFSETS = []
            radius = 10  # Radius of the circle in grid units
            
            # Create a yin-yang pattern with exits
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    # Calculate distance from center
                    distance = (x**2 + y**2)**0.5
                    
                    # Main circle outline
                    if radius - 0.5 <= distance <= radius + 0.5:
                        # Create four evenly spaced exits (N, E, S, W)
                        if not ((abs(x) < 1 and y < 0) or  # North exit
                               (abs(y) < 1 and x > 0) or   # East exit
                               (abs(x) < 1 and y > 0) or   # South exit
                               (abs(y) < 1 and x < 0)):    # West exit
                            OFFSETS.append([(center_x + x) * MARGIN, (center_y + y) * MARGIN])
                    
                    # Left half (solid)
                    elif x < 0 and distance < radius - 0.5:
                        # Create a pathway through the left half
                        if not (abs(y) < 1):  # Horizontal pathway
                            OFFSETS.append([(center_x + x) * MARGIN, (center_y + y) * MARGIN])
                    
                    # Center dividing line
                    elif abs(x) < 0.5 and distance < radius - 0.5:
                        # Create gaps in the dividing line
                        if not (abs(y) % 1 == 0):
                            OFFSETS.append([(center_x + x) * MARGIN, (center_y + y) * MARGIN])
                    
                    # Small circles
                    elif (x + radius/2)**2 + y**2 <= (radius/4)**2:
                        # Create a gap in the left dot (yin)
                        if not (abs(x + radius/2) < 1 and abs(y) < 1):
                            OFFSETS.append([(center_x + x) * MARGIN, (center_y + y) * MARGIN])
                    elif (x - radius/2)**2 + y**2 <= (radius/4)**2 and x > 0:
                        # Right dot (yang) - keep solid
                        OFFSETS.append([(center_x + x) * MARGIN, (center_y + y) * MARGIN])

        # Level 7: "Circuit Board" - A symmetrical electronic circuit pattern
        elif level == 7:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            # Create half of the pattern
            pattern_half = [
                "               ",
                " # ############ ",
                " #            # ",
                " # # ######## # ",
                " # #        # # ",
                " # # #### # # # ",
                " # # #    # # # ",
                " # # # ## # # # ",
                " # # #    # # # ",
                " # # # #### # # ",
                " # #        # # ",
                " # ######## # # ",
                " #            # ",
                " ############ # ",
                "                "
            ]
            
            # Calculate starting position for centering
            start_x = center_x - len(pattern_half[0])
            start_y = (height_blocks - len(pattern_half)) // 2
            
            OFFSETS = []
            
            # Generate the left half of the pattern
            for y, row in enumerate(pattern_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        OFFSETS.append([(start_x + x) * MARGIN, (start_y + y) * MARGIN])
            
            # Generate the right half (mirrored)
            for y, row in enumerate(pattern_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        # Mirror across the center
                        mirror_x = start_x + len(pattern_half[0]) * 2 - x - 1
                        OFFSETS.append([mirror_x * MARGIN, (start_y + y) * MARGIN])
            
            # Create connecting pathways between the two halves
            pathway_y = [start_y + 3, start_y + 7, start_y + 11]
            for y in pathway_y:
                for x in range(start_x + len(pattern_half[0]) - 1, start_x + len(pattern_half[0]) + 1):
                    # Remove blocks to create pathways
                    for offset in OFFSETS[:]:
                        if offset[0] == x * MARGIN and offset[1] == y * MARGIN:
                            OFFSETS.remove(offset)

        elif level == 8:
                    SQUARE_SIZE, MARGIN = 5, 60
                    OFFSETS = []
                    for i in range(1, 7):
                        for j in range(1, 7):
                            if j > i:
                                OFFSETS.append((i * MARGIN, j * MARGIN))
                    
                    for i in range(1, 7):
                        for j in range(1, 7):
                            if j < i:
                                OFFSETS.append(((i + 3) * MARGIN, j * MARGIN))
                
        # Level 9: "Symmetrical Maze" - A mirrored maze layout with clear paths
        elif level == 9:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            # Define the first half of the maze
            maze_half = [
                "##  #######   ##",
                "#       #      #",
                "# ## ## # ## # #",
                "# #     # #    #",
                "# # # ### #### #",
                "# #       #    #",
                "# ### ### # ####",
                "#       # #    #",
                "### ### # #### #",
                "#     # #      #",
                "# ### # ###### #",
                "# #   #        #",
                "# # ####### ####",
                "# #             ",
                "# ###### #######",
                "#               "
            ]
            
            # Calculate dimensions
            maze_half_width = len(maze_half[0])
            maze_height = len(maze_half)
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate starting position to center the maze
            start_x = (width_blocks - (maze_half_width * 2)) // 2
            start_y = (height_blocks - maze_height) // 2
            
            OFFSETS = []
            
            # Generate the first half of the maze
            for y, row in enumerate(maze_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        OFFSETS.append([(start_x + x) * MARGIN, (start_y + y) * MARGIN])
            
            # Generate the mirrored second half
            for y, row in enumerate(maze_half):
                for x, cell in enumerate(row):
                    if cell == '#':
                        # Mirror position across the center
                        mirror_x = (maze_half_width * 2 - 1) - x
                        OFFSETS.append([(start_x + mirror_x) * MARGIN, (start_y + y) * MARGIN])
            
            # Add connecting elements between the two halves
            middle_x = start_x + maze_half_width
            for y in range(maze_height):
                # Create strategic connections at specific rows
                if y % 4 == 0 and y > 0 and y < maze_height - 1:
                    OFFSETS.append([middle_x * MARGIN, (start_y + y) * MARGIN])


        elif level == 10:
            SQUARE_SIZE = 1
            MARGIN = self.block_size  # 20 pixels
            
            width_blocks = self.width // self.block_size
            height_blocks = self.height // self.block_size
            
            # Calculate center
            center_x = width_blocks // 2
            center_y = height_blocks // 2
            
            # Create symmetric pattern
            final_pattern = [
                "                                 ",
                "  # ###########################  ",
                "  #                              ",
                "  # ######################### #  ",
                "  #                           #  ",
                "  # # ##################### # #  ",
                "  # # #                   # # #  ",
                "  # # # ################# # # #  ",
                "  # # # #               # # # #  ",
                "  # # # # ############# # # # #  ",
                "  # # # #               # # # #  ",
                "  # # # # ############# # # # #  ",
                "  # # # #               # # # #  ",
                "  # # # ############### # # # #  ",
                "  # # #                 # # # #  ",
                "  # # ################### # # #  ",
                "  # #                     # # #  ",
                "  # ####################### # #  ",
                "  #                         # #  ",
                "  # ####################### # #  ",
                "                              #  ",
                "  ########################### #  ",
                "                                 "
            ]
            
            # Calculate starting position
            start_x = center_x - len(final_pattern[0]) // 2
            start_y = center_y - len(final_pattern) // 2
            
            OFFSETS = []
            
            # Generate the pattern
            for y, row in enumerate(final_pattern):
                for x, cell in enumerate(row):
                    if cell == '#':
                        OFFSETS.append([(start_x + x) * MARGIN, (start_y + y) * MARGIN])
            
            # Create openings for food access
            # Central opening
            central_x = start_x + len(final_pattern[0]) // 2
            central_y = start_y + 5
            for y in range(central_y, central_y + 2):
                for offset in OFFSETS[:]:
                    if offset[0] == central_x * MARGIN and offset[1] == y * MARGIN:
                        OFFSETS.remove(offset)
            
            # Side openings
            for level in range(1, 5):
                side_y = start_y + 5 + level * 3
                side_x_left = start_x + level * 4
                side_x_right = start_x + len(final_pattern[0]) - level * 4 - 1
                
                for offset in OFFSETS[:]:
                    if (offset[0] == side_x_left * MARGIN and offset[1] == side_y * MARGIN) or \
                       (offset[0] == side_x_right * MARGIN and offset[1] == side_y * MARGIN):
                        OFFSETS.remove(offset)

        translation_x, translation_y = self._get_centered_start_position(OFFSETS, SQUARE_SIZE)

        # For each offset, create a group of coordinates representing one obstacle.
        for offset in OFFSETS:
            block_coords = []
            for i in range(1, SQUARE_SIZE + 1):
                for j in range(1, SQUARE_SIZE + 1):
                    x = offset[0] + (i * self.block_size) + translation_x
                    y = offset[1] + (j * self.block_size) + translation_y
                    x = (x // self.block_size) * self.block_size
                    y = (y // self.block_size) * self.block_size
                    block_coords.append(Coordinate(x, y))
            if block_coords and block_coords not in obstacles:
                obstacles.append(block_coords)
        return obstacles

# For math functions in the star pattern level
import math