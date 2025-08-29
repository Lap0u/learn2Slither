import pygame
from src.snake import Snake
import src.globals as globals
import src.generator as gen
import numpy as np

# ENV : 1 = WALL; 0 = EMPTY; 2 = GREEN APPLE; 3 = RED APPLE 4 = SNAKE HEAD; 5 = SNAKE BODY


class Game:

    def __init__(self):
        self.environment = np.zeros((globals.WIDTH, globals.HEIGHT))
        self.is_done = False
        self.reward = 0
        self.snake = Snake()
        self.green_apple_1, self.green_apple_2, self.red_apple = gen.generate_apples(
            self.snake
        )
        self.update_environment()
        self.snake_view = self.update_snake_view()

    def agent_render(self, screen, bg_image, tile, my_font, steps):
        screen.fill((0, 0, 0))
        self.render_tiles(tile, screen)
        screen.blit(bg_image, (0, 0))
        self.green_apple_1.render(screen)
        self.green_apple_2.render(screen)
        self.red_apple.render(screen)
        self.snake.render(screen)
        # if you want to use this module.
        text_surface = my_font.render(
            f"Size: {self.snake.size} steps {steps}",
            False,
            (62, 69, 129),
            (166, 166, 166),
        )
        screen.blit(text_surface, (globals.TILE_SIZE // 4, globals.TILE_SIZE // 4))
        pygame.display.flip()

    def update_snake_view(self):
        x, y = self.snake.x_pos[0], self.snake.y_pos[0]
        x_view = np.zeros(globals.WIDTH, dtype=int)
        y_view = np.zeros(globals.HEIGHT, dtype=int)
        for i in range(globals.WIDTH):
            for j in range(globals.HEIGHT):
                if x == i:
                    x_view[j] = self.environment[i][j]
                if y == j:
                    y_view[i] = self.environment[i][j]
        return x_view, y_view

    def step(
        self,
        action,
        display=False,
        step=0,
        screen=None,
        bg_image=None,
        tile=None,
        my_font=None,
    ):
        self.snake.direction = action
        self.snake.check_apple_collision(
            self.green_apple_1, self.green_apple_2, self.red_apple
        )
        self.reward = self.snake.move()
        self.update_environment()
        self.snake_view = self.update_snake_view()
        if display is True and screen is not None and self.snake.is_dead is False:
            self.agent_render(screen, bg_image, tile, my_font, step)
        else:
            self.terminal_display()
        return self.snake_view, self.reward, self.snake.is_dead

    def terminal_display(self):
        # print(self.environment.T)
        pass

    def update_environment(self):
        # print("Apple 1", self.green_apple_1.x, self.green_apple_1.y)
        # print("Apple 2", self.green_apple_2.x, self.green_apple_2.y)
        # print("Red Apple", self.red_apple.x, self.red_apple.y)
        for i in range(globals.WIDTH):
            for j in range(globals.HEIGHT):
                if (
                    i == 0
                    or i == globals.WIDTH - 1
                    or j == 0
                    or j == globals.HEIGHT - 1
                ):
                    self.environment[i][j] = 1
                elif (i, j) == (self.green_apple_1.x, self.green_apple_1.y) or (
                    i,
                    j,
                ) == (self.green_apple_2.x, self.green_apple_2.y):
                    self.environment[i][j] = 2
                elif (i, j) == (self.red_apple.x, self.red_apple.y):
                    self.environment[i][j] = 3
                elif (i, j) == (self.snake.x_pos[0], self.snake.y_pos[0]):
                    self.environment[i][j] = 4
                elif (i, j) in zip(self.snake.x_pos[1:], self.snake.y_pos[1:]):
                    self.environment[i][j] = 5
                else:
                    self.environment[i][j] = 0
        # print("env", self.environment.T)

    def render_tiles(self, tile, screen):
        for i in range(0, globals.WIDTH * globals.TILE_SIZE, globals.TILE_SIZE):
            for j in range(0, globals.HEIGHT * globals.TILE_SIZE, globals.TILE_SIZE):
                if (
                    i == 0
                    or i == globals.WIDTH * globals.TILE_SIZE - globals.TILE_SIZE
                    or j == 0
                    or j == globals.HEIGHT * globals.TILE_SIZE - globals.TILE_SIZE
                ):
                    screen.blit(tile, (i, j))

    def launch_game(self):
        pygame.init()

        screen = pygame.display.set_mode((globals.WIDTH, globals.HEIGHT))
        pygame.display.set_caption("Snake Game")
        running = True
        clock = pygame.time.Clock()
        bg_image = pygame.image.load("assets/bg-tr.png")
        bg_image.set_alpha(128)
        tile = pygame.image.load("assets/tile.png")
        while running:
            paused = False
            clock.tick(globals.GAME_SPEED)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if (
                        event.key == pygame.K_UP
                        and self.snake.direction != "DOWN"
                        and not paused
                    ):
                        self.snake.direction = "UP"
                        paused = True
                    elif (
                        event.key == pygame.K_DOWN
                        and self.snake.direction != "UP"
                        and not paused
                    ):
                        self.snake.direction = "DOWN"
                        paused = True
                    elif (
                        event.key == pygame.K_LEFT
                        and self.snake.direction != "RIGHT"
                        and not paused
                    ):
                        self.snake.direction = "LEFT"
                        paused = True
                    elif (
                        event.key == pygame.K_RIGHT
                        and self.snake.direction != "LEFT"
                        and not paused
                    ):
                        self.snake.direction = "RIGHT"
                        paused = True
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            screen.fill((0, 0, 0))
            self.render_tiles(tile, screen)
            screen.blit(bg_image, (0, 0))
            self.green_apple_1.render(screen)
            self.green_apple_2.render(screen)
            self.red_apple.render(screen)
            self.terminal_display()
            self.snake.move()
            self.update_environment()
            self.snake.render(screen)
            pygame.display.flip()
            self.snake.check_apple_collision(
                self.green_apple_1, self.green_apple_2, self.red_apple
            )

        pygame.quit()
