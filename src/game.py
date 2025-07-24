import pygame
from src.snake import Snake
import src.globals as globals
import src.generator as gen


def render_tiles(tile, screen):
    for i in range(0, globals.WIDTH, globals.TILE_SIZE):
        for j in range(0, globals.HEIGHT, globals.TILE_SIZE):
            if (
                i == 0
                or i == globals.WIDTH - globals.TILE_SIZE
                or j == 0
                or j == globals.HEIGHT - globals.TILE_SIZE
            ):
                screen.blit(tile, (i, j))


def launch_game():
    pygame.init()

    screen = pygame.display.set_mode((globals.WIDTH, globals.HEIGHT))
    pygame.display.set_caption("Snake Game")
    running = True
    clock = pygame.time.Clock()
    snake = Snake()
    green_apple_1, green_apple_2, red_apple = gen.generate_apples(snake)
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
                    and snake.direction != "DOWN"
                    and not paused
                ):
                    snake.direction = "UP"
                    paused = True
                elif (
                    event.key == pygame.K_DOWN
                    and snake.direction != "UP"
                    and not paused
                ):
                    snake.direction = "DOWN"
                    paused = True
                elif (
                    event.key == pygame.K_LEFT
                    and snake.direction != "RIGHT"
                    and not paused
                ):
                    snake.direction = "LEFT"
                    paused = True
                elif (
                    event.key == pygame.K_RIGHT
                    and snake.direction != "LEFT"
                    and not paused
                ):
                    snake.direction = "RIGHT"
                    paused = True
                elif event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((0, 0, 0))
        render_tiles(tile, screen)
        screen.blit(bg_image, (0, 0))
        green_apple_1.render(screen)
        green_apple_2.render(screen)
        red_apple.render(screen)
        snake.move()
        snake.render(screen)
        pygame.display.flip()
        snake.check_apple_collision(green_apple_1, green_apple_2, red_apple)

    pygame.quit()
