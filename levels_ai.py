from collections import namedtuple

Coordinate = namedtuple('Coordinate', 'x, y')

class Levels:
    def __init__(self, block_size, width, height) -> None:
        self.block_size = block_size
        self.width = width
        self.height = height
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
        # coordinates = []
        obstacles = []
        if level == 0:
            return obstacles

        if level == 1:
            SQUARE_SIZE = 4
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            OFFSETS = [
                [MARGIN, MARGIN],
                [self.width - (MARGIN + (2 * SQUARE_SIZE * self.block_size)),
                 self.height - (MARGIN + (2 * SQUARE_SIZE * self.block_size))]
            ]

        elif level == 2:
            SQUARE_SIZE = 1
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            OFFSETS = []
            for i in range(2, 10):
                OFFSETS.append([i * MARGIN, 2 * MARGIN])
            OFFSETS.append([2 * MARGIN, 3 * MARGIN])
            for i in range(2, 10):
                OFFSETS.append([i * MARGIN, 6 * MARGIN])
            OFFSETS.append([9 * MARGIN, 5 * MARGIN])

        elif level == 3:
            SQUARE_SIZE = 2
            MUL_FACTOR = 3
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MARGIN = MARGIN * MUL_FACTOR
            OFFSETS = [
                [MARGIN, MARGIN],
                [self.width - (MARGIN + (1 * SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (2 * SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (3 * SQUARE_SIZE * self.block_size)), MARGIN],
                [MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [MARGIN, self.height - (MARGIN + (2 * SQUARE_SIZE * self.block_size))],
                [MARGIN, self.height - (MARGIN + (3 * SQUARE_SIZE * self.block_size))],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)),
                 self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (2 * MARGIN + (SQUARE_SIZE * self.block_size)),
                 self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)),
                 self.height - (2 * MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (2 * MARGIN + (SQUARE_SIZE * self.block_size)),
                 self.height - (2 * MARGIN + (SQUARE_SIZE * self.block_size))]
            ]

        elif level == 4:
            SQUARE_SIZE = 2
            MUL_FACTOR = 2
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MARGIN = MARGIN * MUL_FACTOR
            OFFSETS = [
                [MARGIN, MARGIN],
                [MARGIN + MARGIN, MARGIN + MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN, MARGIN + MARGIN],
                [MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [MARGIN + MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN,
                 self.height - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)),
                 self.height - (MARGIN + (SQUARE_SIZE * self.block_size))]
            ]
        
        elif level == 5:
            SQUARE_SIZE = 2
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 2
            MARGIN = MARGIN * MUL_FACTOR
            OFFSETS = [
                [MARGIN, MARGIN],
                [MARGIN + MARGIN, MARGIN + MARGIN],
                [3 * MARGIN, 3 * MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)), MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN, MARGIN + MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - (2 * MARGIN), 3 * MARGIN],
                [MARGIN, self.height - (MARGIN + (8 * self.block_size))],
                [MARGIN + MARGIN, self.height - (MARGIN + (8 * self.block_size)) - MARGIN],
                [3 * MARGIN, self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)],
                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN,
                 self.height - (MARGIN + (8 * self.block_size)) - MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - (2 * MARGIN),
                 self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)],
                [self.width - (MARGIN + (8 * self.block_size)),
                 self.height - (MARGIN + (8 * self.block_size))]
            ]

        elif level == 6:
            SQUARE_SIZE = 2
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 2
            MARGIN = MARGIN * MUL_FACTOR
            OFFSETS = [
                [MARGIN, MARGIN],
                [2 * MARGIN, MARGIN],
                [3 * MARGIN, MARGIN],
                [MARGIN, 2 * MARGIN],
                [MARGIN, 3 * MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)), MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN, MARGIN + MARGIN],
                [MARGIN, self.height - (MARGIN + (8 * self.block_size))],
                [MARGIN + MARGIN, self.height - (MARGIN + (8 * self.block_size)) - MARGIN],
                [self.width - (2 * MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))],
                [self.width - (3 * MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))],
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)],
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size)) - MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))]
            ]

        elif level == 7:
            SQUARE_SIZE = 1
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 2
            MARGIN = MARGIN * MUL_FACTOR
            OFFSETS = []
            for i in range(1, 10):
                for j in range(1, 7):
                    OFFSETS.append([i * MARGIN, j * MARGIN])

        elif level == 8:
            SQUARE_SIZE = 1
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 3
            MARGIN = MARGIN + MUL_FACTOR
            OFFSETS = []
            for i in range(7):
                for j in range(7):
                    if j > i:
                        OFFSETS.append([i * MARGIN, j * MARGIN])
            for i in range(7):
                for j in range(7):
                    if j < i:
                        OFFSETS.append([(i + 3) * MARGIN, j * MARGIN])

        elif level == 9:
            SQUARE_SIZE = 1
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 1
            MARGIN = MARGIN * MUL_FACTOR

            OFFSETS = [
                [1 * MARGIN, MARGIN],
                [2 * MARGIN, MARGIN],
                [3 * MARGIN, MARGIN],
                [4 * MARGIN, MARGIN],
                [6 * MARGIN, MARGIN],
                [7 * MARGIN, MARGIN],
                [7 * MARGIN, 2 * MARGIN],
                [8 * MARGIN, MARGIN],
                [9 * MARGIN, MARGIN],
                [10 * MARGIN, MARGIN],
                [11 * MARGIN, MARGIN],
                [12 * MARGIN, MARGIN],
                [13 * MARGIN, MARGIN],
                [14 * MARGIN, MARGIN],
                [14 * MARGIN, 2 * MARGIN],
                [14 * MARGIN, 3 * MARGIN],
                [14 * MARGIN, 4 * MARGIN],
                [14 * MARGIN, 5 * MARGIN],
                [14 * MARGIN, 6 * MARGIN],
                [12 * MARGIN, 5 * MARGIN],
                [13 * MARGIN, 7 * MARGIN],
                [14 * MARGIN, 7 * MARGIN],
                [14 * MARGIN, 8 * MARGIN],
                [14 * MARGIN, 9 * MARGIN],
                [13 * MARGIN, 9 * MARGIN],
                [11 * MARGIN, 8 * MARGIN],
                [10 * MARGIN, 8 * MARGIN],
                [10 * MARGIN, 9 * MARGIN],
                [9 * MARGIN, 8 * MARGIN],
                [11 * MARGIN, 7 * MARGIN],
                [11 * MARGIN, 6 * MARGIN],
                [11 * MARGIN, 5 * MARGIN],
                [11 * MARGIN, 4 * MARGIN],
                [11 * MARGIN, 3 * MARGIN],
                [10 * MARGIN, 3 * MARGIN],
                [19 * MARGIN, 3 * MARGIN],
                [MARGIN, 2 * MARGIN],
                [MARGIN, 3 * MARGIN],
                [MARGIN, 4 * MARGIN],
                [2 * MARGIN, 4 * MARGIN],
                [3 * MARGIN, 4 * MARGIN],
                [4 * MARGIN, 4 * MARGIN],
                [5 * MARGIN, 4 * MARGIN],
                [MARGIN, 5 * MARGIN],
                [MARGIN, 6 * MARGIN],
                [MARGIN, 7 * MARGIN],
                [MARGIN, 8 * MARGIN],
                [MARGIN, 9 * MARGIN],
                [2 * MARGIN, 9 * MARGIN],
                [3 * MARGIN, 9 * MARGIN],
                [4 * MARGIN, 9 * MARGIN],
                [6 * MARGIN, 9 * MARGIN],
                [7 * MARGIN, 9 * MARGIN],
                [7 * MARGIN, 8 * MARGIN],
                [7 * MARGIN, 7 * MARGIN],
                [7 * MARGIN, 6 * MARGIN],
                [7 * MARGIN, 4 * MARGIN]
            ]

        elif level == 10:
            SQUARE_SIZE = 1
            MARGIN = (int(0.1 * self.base_dim) // self.block_size) * self.block_size
            MUL_FACTOR = 2
            MARGIN = MARGIN * MUL_FACTOR

            OFFSETS = []
            for i in range(4, 23):
                for j in range(4, 26, 1):
                    OFFSETS.append([j * MARGIN, i * MARGIN])

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
