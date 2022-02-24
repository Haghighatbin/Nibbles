from collections import namedtuple

Coordinate = namedtuple('Coordinate', 'x, y')


class Levels:
    def __init__(self, block_size, width, height) -> None:
        self.block_size = block_size
        self.width = width
        self.height = height

    def _level_generator(self, level):
        coordinates, obstacles = [], []
        if level == 0:
            return obstacles

        if level == 1:
            SQUARE_SIZE, MARGIN = 10, 80
            OFFSETS = [[MARGIN, MARGIN], 
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (MARGIN + (SQUARE_SIZE * self.block_size))]] 

        if level == 2:
            SQUARE_SIZE, MARGIN = 5, 50
            OFFSETS = []
            for i in range(2, 10):
                OFFSETS.append((i * MARGIN, 2 * MARGIN))
            OFFSETS.append((2 * MARGIN, 3 * MARGIN))
            for i in range(2, 10):
                OFFSETS.append((i * MARGIN, 6 * MARGIN))
            OFFSETS.append((9 * MARGIN, 5 * MARGIN))

        if level == 3:
            SQUARE_SIZE, MARGIN = 6, 60
            OFFSETS = [[MARGIN, MARGIN], 
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (2 * SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (3 * SQUARE_SIZE * self.block_size)), MARGIN],

                [MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size))], 
                [MARGIN, self.height - (MARGIN + (2 * SQUARE_SIZE * self.block_size))], 
                [MARGIN, self.height - (MARGIN + (3 * SQUARE_SIZE * self.block_size))], 

                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (2 * MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (2 * MARGIN + (SQUARE_SIZE * self.block_size))],
                [self.width - (2 * MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (2 * MARGIN + (SQUARE_SIZE * self.block_size))]] 
        
        if level == 4:
            SQUARE_SIZE, MARGIN = 8, 60
            OFFSETS = [[MARGIN, MARGIN], 
                [MARGIN + MARGIN, MARGIN + MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), MARGIN],
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN, MARGIN + MARGIN],
                [MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size))], 
                [MARGIN + MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN], 
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN, self.height - (MARGIN + (SQUARE_SIZE * self.block_size)) - MARGIN], 
                [self.width - (MARGIN + (SQUARE_SIZE * self.block_size)), self.height - (MARGIN + (SQUARE_SIZE * self.block_size))]]
        
        if level == 5:
            SQUARE_SIZE, MARGIN = 5, 60
            OFFSETS = [[MARGIN, MARGIN], 
                [MARGIN + MARGIN, MARGIN + MARGIN],
                [MARGIN * 3, MARGIN * 3],

                [self.width - (MARGIN + (8 * self.block_size)), MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN, MARGIN + MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - (2 * MARGIN), 3 * MARGIN],

                [MARGIN, self.height - (MARGIN + (8 * self.block_size))], 
                [MARGIN + MARGIN, self.height - (MARGIN + (8 * self.block_size)) - MARGIN], 
                [MARGIN * 3, self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)], 

                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN, self.height - (MARGIN + (8 * self.block_size)) - MARGIN], 
                [self.width - (MARGIN + (8 * self.block_size)) - (2 * MARGIN), self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)], 
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))]] 

        if level == 6:
            SQUARE_SIZE, MARGIN = 8, 60
            OFFSETS = [[MARGIN, MARGIN], 
                [MARGIN * 2, MARGIN],
                [MARGIN * 3, MARGIN],

                [MARGIN, MARGIN * 2],
                [MARGIN, MARGIN * 3],

                [self.width - (MARGIN + (8 * self.block_size)), MARGIN],
                [self.width - (MARGIN + (8 * self.block_size)) - MARGIN, MARGIN + MARGIN],

                [MARGIN, self.height - (MARGIN + (8 * self.block_size))], 
                [MARGIN + MARGIN, self.height - (MARGIN + (8 * self.block_size)) - MARGIN], 

                [self.width - (2 * MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))], 
                [self.width - (3 * MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size))], 

                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size)) - (2 * MARGIN)], 
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size)) - MARGIN], 
                [self.width - (MARGIN + (8 * self.block_size)), self.height - (MARGIN + (8 * self.block_size)) ]] 

        if level == 7:
            SQUARE_SIZE, MARGIN = 5, 60
            OFFSETS = []
            for i in range(1, 10):
                for j in range(1, 7):
                    OFFSETS.append((i * MARGIN, j * MARGIN))

        if level == 8:
            SQUARE_SIZE, MARGIN = 5, 60
            OFFSETS = []
            for i in range(1, 7):
                for j in range(0, 7):
                    if j > i:
                        OFFSETS.append((i * MARGIN, j * MARGIN))
            
            for i in range(1, 7):
                for j in range(0, 7):
                    if j < i:
                        OFFSETS.append(((i + 3) * MARGIN, j * MARGIN))
          
        if level == 9:
            SQUARE_SIZE, MARGIN = 4, 40
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
                [7 * MARGIN, 4 * MARGIN]]

        if level == 10:
            SQUARE_SIZE, MARGIN = 1, 20
            OFFSETS = []
            for i in range(23):
                for j in range(5, 26, 5):
                    OFFSETS.append((j * MARGIN, i * MARGIN))

        for offset in OFFSETS:
            for i in range(1, SQUARE_SIZE + 1):
                for j in range(1, SQUARE_SIZE + 1):
                    coordinate = Coordinate(offset[0] + (i * self.block_size), offset[1] + (j * self.block_size))
                    coordinates.append(coordinate)
            obstacles.append(coordinates)
            coordinates = []
        return obstacles