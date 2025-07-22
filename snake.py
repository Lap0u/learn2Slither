import argparse
import random
import pygame

WIDTH = 816
HEIGHT = 612
TILE_SIZE = 34
DIRECTIONS = {
    "UP": [0, -1],
    "DOWN": [0, 1],
    "LEFT": [-1, 0],
    "RIGHT": [1, 0],
}


class Snake:
    def __init__(self):
        self.size = 3
        self.head = pygame.image.load("assets/snake_head.png")
        self.body = pygame.image.load("assets/snake_body.png")
        self.head_xx = pygame.image.load("assets/snake_head_xx.png")
        self.direction = random.choice(list(DIRECTIONS.keys()))
        self.x_pos = [
            WIDTH / TILE_SIZE / 2,
            WIDTH / TILE_SIZE / 2 - DIRECTIONS[self.direction][0],
            WIDTH / TILE_SIZE / 2 - DIRECTIONS[self.direction][0] * 2,
        ]
        self.y_pos = [
            HEIGHT / TILE_SIZE / 2,
            HEIGHT / TILE_SIZE / 2 - DIRECTIONS[self.direction][1],
            HEIGHT / TILE_SIZE / 2 - DIRECTIONS[self.direction][1] * 2,
        ]

    def check_collision(self, new_x, new_y):
        if (
            new_x < 0
            or new_x >= WIDTH / TILE_SIZE
            or new_y < 0
            or new_y >= HEIGHT / TILE_SIZE
        ):
            print("Game Over! Snake hit the wall.")
            pygame.quit()
            exit()

    def move(self):
        new_x = self.x_pos[0] + DIRECTIONS[self.direction][0]
        new_y = self.y_pos[0] + DIRECTIONS[self.direction][1]
        self.check_collision(new_x, new_y)
        self.x_pos.insert(0, new_x)
        self.y_pos.insert(0, new_y)
        self.x_pos.pop()
        self.y_pos.pop()


def render_tiles(tile, screen):
    for i in range(0, WIDTH, TILE_SIZE):
        for j in range(0, HEIGHT, TILE_SIZE):
            if i == 0 or i == WIDTH - TILE_SIZE or j == 0 or j == HEIGHT - TILE_SIZE:
                screen.blit(tile, (i, j))


def generate_apples():
    ga_1_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
    ga_1_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)
    ga_2_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
    ga_2_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)
    while ga_1_x == ga_2_x and ga_1_y == ga_2_y:
        ga_2_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
        ga_2_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)

    ra_1_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
    ra_1_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)
    while (
        ra_1_x == ga_1_x and ra_1_y == ga_1_y or ra_1_x == ga_2_x and ra_1_y == ga_2_y
    ):
        ra_1_x = random.randrange(0, WIDTH - TILE_SIZE * 2, TILE_SIZE)
        ra_1_y = random.randrange(0, HEIGHT - TILE_SIZE * 2, TILE_SIZE)

    return ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y


def render_apples(
    green_apple, red_apple, screen, ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y
):
    screen.blit(green_apple, (ga_1_x, ga_1_y))
    screen.blit(green_apple, (ga_2_x, ga_2_y))
    screen.blit(red_apple, (ra_1_x, ra_1_y))


def render_snake(snake, screen):
    head_x = snake.x_pos[0] * TILE_SIZE
    head_y = snake.y_pos[0] * TILE_SIZE
    screen.blit(snake.head, (head_x, head_y))
    for i in range(1, snake.size):
        screen.blit(
            snake.body, (snake.x_pos[i] * TILE_SIZE, snake.y_pos[i] * TILE_SIZE)
        )


def launch_game():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")
    running = True
    clock = pygame.time.Clock()
    ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y = generate_apples()
    bg_image = pygame.image.load("assets/bg-tr.png")
    green_apple = pygame.image.load("assets/apple_green_32.png")
    red_apple = pygame.image.load("assets/apple_red_32.png")
    bg_image.set_alpha(128)
    tile = pygame.image.load("assets/tile.png")
    snake = Snake()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != "DOWN":
                    snake.direction = "UP"
                elif event.key == pygame.K_DOWN and snake.direction != "UP":
                    snake.direction = "DOWN"
                elif event.key == pygame.K_LEFT and snake.direction != "RIGHT":
                    snake.direction = "LEFT"
                elif event.key == pygame.K_RIGHT and snake.direction != "LEFT":
                    snake.direction = "RIGHT"
        screen.fill((0, 0, 0))
        render_tiles(tile, screen)
        screen.blit(bg_image, (0, 0))
        render_apples(
            green_apple,
            red_apple,
            screen,
            ga_1_x,
            ga_1_y,
            ga_2_x,
            ga_2_y,
            ra_1_x,
            ra_1_y,
        )
        render_snake(snake, screen)
        snake.move()
        pygame.display.flip()
        clock.tick(5)

    pygame.quit()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Snake Game")
    # parser.add_argument("--size", type=int, default=10, help="Size of the game board")
    # parser.add_argument(
    #     "--model", type=str, help="Path to the trained model file", required=True
    # )
    # parser.add_argument("--green", type=int, default=2, help="Number of green apples")
    # parser.add_argument("--red", type=int, default=1, help="Number of red apples")
    # parser.add_argument(
    #     "--visual",
    #     type=bool,
    #     default=True,
    #     help="Show game interface and snake view",
    #     store=True,
    # )
    # parser.add_argument(
    #     "--nolearn",
    #     type=bool,
    #     default=True,
    #     help="Don't train the agent, just play",
    #     store=True,
    # )
    # parser.add_argument(
    #     "--step-by-step",
    #     type=bool,
    #     default=True,
    #     help="Run the game step by step",
    #     store=True,
    # )

    # parser.add_argument(
    #     "--session", type=int, default=10, help="Number of sessions to train the agent"
    # )
    # args = parser.parse_args()
    launch_game()
