import pygame
import pygame_gui
import random
import glob
from enum import Enum
from collections import namedtuple
from levels import Levels

MUSIC_VOLUME = 0.1
SOUND_VOLUME = 0.2
BLOCK_SIZE = 10
SPEED = 20
MAP_WIDTH = 640
MAP_HEIGHT = 480

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
pygame.init()

# Loaded sound files
ate_tracks = glob.glob('Sounds/wav_Tracks/ate/*')
burp_tracks = glob.glob('Sounds/wav_Tracks/burp/*')
fart_tracks = glob.glob('Sounds/wav_Tracks/fart/*')
failed_tracks = glob.glob('Sounds/wav_Tracks/failed/*')
start_tracks = glob.glob('Sounds/wav_Tracks/start/*')
yes_tracks = glob.glob('Sounds/wav_Tracks/yes/*')

# Loaded theme tracks
theme_tracks = glob.glob('Sounds/wav_Themes/*')

# Loaded ending track
ending_track = 'Sounds/ending/we-are-the-champions.wav'

font = pygame.font.Font(None, 24)

# Loaded ending stars
# stars_list = glob.glob('Images/stars/*')

# Loaded images and icons
apple_img = pygame.image.load('Images/apple.png')
betsy_img = pygame.image.load('Images/betsy_logo_final.png')
hole_img = pygame.image.load('Images/hole.png')
sound_img = pygame.image.load('Images/sound_icon.png')
sound_mute_img = pygame.image.load('Images/sound_mute_icon.png')
music_img = pygame.image.load('Images/music_icon.png')
music_mute_img = pygame.image.load('Images/music_mute_icon.png')

# Resized images and icons
transformed_apple_img = pygame.transform.scale(apple_img, (BLOCK_SIZE, BLOCK_SIZE))
transformed_betsy_img = pygame.transform.scale(betsy_img, (10 * BLOCK_SIZE, 10 * BLOCK_SIZE))
transformed_sound_img = pygame.transform.scale(sound_img, (2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
transformed_sound_mute_img = pygame.transform.scale(sound_mute_img, (2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
transformed_music_img = pygame.transform.scale(music_img, (2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
transformed_music_mute_img = pygame.transform.scale(music_mute_img, (2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
transformed_hole_img  = pygame.transform.scale(hole_img, (BLOCK_SIZE, BLOCK_SIZE))


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Coordinate = namedtuple('Coordinate', 'x, y')

class Nibbles:
    def __init__(self, width: int=MAP_WIDTH, height: int=MAP_HEIGHT) -> None:
        self.width, self.height = width, height
        self.music = True
        self.sound = True
        self.star = []
        self._restart(self.width, self.height)
    
    def _feed(self) -> Coordinate:
        x = random.randint(1, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(1, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.apple = Coordinate(x,y)

        if self.apple in self.nibble:
            self._feed()
        for set in self.obstacle:
            if self.apple in set:
                self._feed()
        return self.apple

    def _obstacle(self, level: int=0) -> list[Coordinate]:
        _level = Levels(block_size=BLOCK_SIZE, width=self.width, height=self.height)
        obstacles = _level._level_generator(level=level)        
        return obstacles
    
    def run_frames(self) -> list[bool, int, int, bool]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    if self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                if event.key == pygame.K_UP:
                    if self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                if event.key == pygame.K_DOWN:
                    if self.direction != Direction.UP:
                        self.direction = Direction.DOWN
        self._move(self.direction)
        self.nibble.insert(0, self.head)
        
        game_over = False
        if self._is_collision()[0]:
            pygame_gui.windows.UIConfirmationDialog(rect=pygame.Rect((self.width/2) - 120, (self.height/2) - 80, 260, 200),
            action_long_desc=f'Oops! Looks like you {self._is_collision()[1]}!<br>Wanna play again?',
            action_short_name='Yes',
            window_title='Game Over',
            manager=self.manager,
            blocking=True)

            time_delta = self.clock.tick(60)/1000.0

            while True:
                self.manager.draw_ui(self.display)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_object_id == '#confirmation_dialog.#close_button':
                            game_over = True
                            return game_over, self.score, self.level, False
                        if event.ui_object_id == '#confirmation_dialog.#cancel_button':
                            game_over = True
                            return game_over, self.score, self.level, False
                        if event.ui_object_id == '#confirmation_dialog.#confirm_button':
                            game_over = True
                            return game_over, self.score, self.level, True

                    self.manager.process_events(event)
                self.manager.update(time_delta)
            
        if self.head == self.apple:
            if self.sound:
                if self.score % 3 == 0 and self.score % 5 != 0:
                    self._sound(pygame.mixer.Sound(random.choice(yes_tracks)))

                if self.score % 5 == 0 and self.score != 0:
                    self._sound(pygame.mixer.Sound(random.choice(burp_tracks)))
                
                if self.score % 7 == 0 and self.score != 0:
                    self._sound(pygame.mixer.Sound(random.choice(fart_tracks)))
                
                else:
                    self._sound(pygame.mixer.Sound(random.choice(ate_tracks)))

            if self.score == 50:
                self._restart(level=self.level + 1)
            
            # Ending
            if self.level == 10 and self.score == 50:
                self._music(pygame.mixer.music.load(ending_track))

                ### I need to get my head around a proper ending...

                # for _ in range(100):
                #     x = random.randint(1, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
                #     y = random.randint(1, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
                #     self.star.append(Coordinate(x,y))
                # for star in self.star:
                #     star_img = pygame.image.load(random.choice(stars_list))
                #     transformed_star_img = pygame.transform.scale(star_img, (5 * BLOCK_SIZE, 5 * BLOCK_SIZE))
                #     self.display.blit(transformed_star_img, (star.x,  star.y))
                #     self.clock.tick(20)
                #     pygame.display.flip()
                # pygame.time.delay(5000)
                
                pygame_gui.windows.UIConfirmationDialog(rect=pygame.Rect((self.width/2) - 120, (self.height/2) - 80, 300, 240),
                    action_long_desc=f'Wow! Congratulations, you did it! I hope you enjoyed it {self._is_collision()[1]}!<br>Wanna play again?',
                    action_short_name='Yes',
                    window_title='You won the game!',
                    manager=self.manager,
                    blocking=True)

                time_delta = self.clock.tick(60)/1000.0

                while True:
                    self.manager.draw_ui(self.display)
                    pygame.display.update()
                    for event in pygame.event.get():
                        if event.type == pygame_gui.UI_BUTTON_PRESSED:
                            if event.ui_object_id == '#confirmation_dialog.#close_button':
                                game_over = True
                                return game_over, self.score, self.level, False
                            if event.ui_object_id == '#confirmation_dialog.#cancel_button':
                                game_over = True
                                return game_over, self.score, self.level, False
                            if event.ui_object_id == '#confirmation_dialog.#confirm_button':
                                game_over = True
                                return game_over, self.score, 0, True

                        self.manager.process_events(event)
                    self.manager.update(time_delta)

            self.score += 1
            self._feed()
        else:
            self.nibble.pop()
        
        self._update_ui()
        self.clock.tick(SPEED + (self.score + 1) * 0.2)

        return game_over, self.score, self.level, False
    
    def _is_collision(self) -> list[bool, str]:
        if self.head.x > self.width - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0:
            if self.sound:
                self._sound(pygame.mixer.Sound(random.choice(failed_tracks)))
            return True, 'hit the bounderies'

        if self.head in self.nibble[5:]:
            if self.sound:
                self._sound(pygame.mixer.Sound(random.choice(failed_tracks)))
            return True, 'hit yourself'
        
        for set in self.obstacle:
            if self.head in set:
                if self.sound:
                    self._sound(pygame.mixer.Sound(random.choice(failed_tracks)))
                return True, 'hit the obstacles'

        return False, ''
    
    def _update_ui(self) -> None:
        self.display.fill((0,0,0))

        if self.music:
            self.display.blit(transformed_music_img, (self.width - 6 * BLOCK_SIZE, BLOCK_SIZE))
        
        if not self.music:
            self.display.blit(transformed_music_mute_img, (self.width - 6 * BLOCK_SIZE, BLOCK_SIZE))
        
        if self.sound:
            self.display.blit(transformed_sound_img, (self.width - 3 * BLOCK_SIZE, BLOCK_SIZE))

        if not self.sound:
            self.display.blit(transformed_sound_mute_img, (self.width - 3 * BLOCK_SIZE, BLOCK_SIZE))

        mouse_pointer = pygame.mouse.get_pos()
        button_clicked = pygame.mouse.get_pressed()
        music_rect = pygame.Rect((self.width - 6 * BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
        sound_rect = pygame.Rect((self.width - 3 * BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE))
        
        music_button_hovered = music_rect.collidepoint(mouse_pointer)
        sound_button_hovered = sound_rect.collidepoint(mouse_pointer)

        if music_button_hovered:
            if button_clicked[0]:
                if self.music:
                    self.music = False
                    self._music(pygame.mixer.music.load(random.choice(theme_tracks)), music_mute=True)
                elif not self.music:
                    self.music = True
                    self._music(pygame.mixer.music.load(random.choice(theme_tracks)), music_mute=False)
                
        if sound_button_hovered:
            if button_clicked[0]:
                if self.sound:
                    self.sound = False
                elif not self.sound:
                    self.sound = True

        self.display.blit(transformed_betsy_img, (self.width - 100 , self.height - 80))

        for set in self.obstacle:
            for coordinate in set:
                pygame.draw.rect(self.display, (255, 100, 0), pygame.Rect(coordinate.x, coordinate.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(coordinate.x + 1, coordinate.y + 1, BLOCK_SIZE-2, BLOCK_SIZE-2))
        
        pygame.draw.rect(self.display, (255, 255, 255), pygame.Rect(self.nibble[0].x, self.nibble[0].y, BLOCK_SIZE, BLOCK_SIZE),6)
        pygame.draw.rect(self.display, (0, 100, 0), pygame.Rect(self.nibble[0].x + 1, self.nibble[0].y + 1, BLOCK_SIZE-2, BLOCK_SIZE-2))
        
        MARBLE = True
        for coordinate in self.nibble[1:]:
            pygame.draw.rect(self.display, (255, 255, 255), pygame.Rect(coordinate.x, coordinate.y, BLOCK_SIZE, BLOCK_SIZE))
            if not MARBLE:
                pygame.draw.rect(self.display, (255, 191, 0), pygame.Rect(coordinate.x + 1, coordinate.y + 1, BLOCK_SIZE-2, BLOCK_SIZE-2))
                MARBLE = True
            elif MARBLE:
                pygame.draw.rect(self.display, (0, 220, 0), pygame.Rect(coordinate.x + 1, coordinate.y + 1, BLOCK_SIZE-2, BLOCK_SIZE-2))
                MARBLE = False
            
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(transformed_apple_img, (self.apple.x, self.apple.y))
        if self.score == 30:
            self.display.blit(transformed_hole_img, (self.apple.x, self.apple.y))

        self.display.blit(text, [BLOCK_SIZE, BLOCK_SIZE])
        pygame.display.flip()
    
    def _move(self, direction: Direction) -> Coordinate:
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Coordinate(x, y)
        return self.head
    
    def _restart(self, width: int=MAP_WIDTH, height: int=MAP_HEIGHT, level: int=0) -> None:
        self.width = width
        self.height = height
        self.level = level
        
        self.display = pygame.display.set_mode((self.width, self.height))
        self.manager = pygame_gui.UIManager((self.width, self.height))
        pygame.display.set_caption("Betsy's Saga")
        self.clock = pygame.time.Clock()

        self.direction = Direction.RIGHT
        self.head = Coordinate(self.width/2, self.height/2)
        self.nibble = [self.head, Coordinate(self.head.x - BLOCK_SIZE, self.head.y), Coordinate(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.apple, self.obstacle = None, None
        self.obstacle = self._obstacle(level=self.level)
        self._feed()

    def _sound(self, track: pygame.mixer.Sound, sound_mute: bool=False, volume: float=SOUND_VOLUME) -> None:
        if sound_mute:
            track.stop()
        else:
            track.set_volume(volume)
            track.play()

    def _music(self, theme: pygame.mixer.music, music_mute: bool=False, volume: float=MUSIC_VOLUME, loops: int=-1, start: float=0.0, fade_ms: int=5000) -> None:
        if music_mute:
            pygame.mixer.music.stop()
        else:
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play(loops=loops, start=start, fade_ms=fade_ms)


if __name__ == '__main__':
    game = Nibbles()
    if game.sound:
        Nibbles()._sound(pygame.mixer.Sound(random.choice(start_tracks)))

    if game.music:
        Nibbles()._music(pygame.mixer.music.load(random.choice(theme_tracks)))

    while True:
        game_over, score, level, restart = game.run_frames()
        if game_over:
            if restart:
                if game.music:
                    Nibbles()._music(pygame.mixer.music.load(random.choice(theme_tracks)))

                if game.sound:
                    Nibbles()._sound(pygame.mixer.Sound(random.choice(start_tracks)))

                game._restart(level=level)
            else:
                break
    print(f'Level: {level}\nFinal Score: {score}')
    pygame.quit()