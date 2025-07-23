import argparse
import random
import pygame

FPS = 5
WIDTH = 748
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
        self.grow = False
        self.reduce = False
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
            new_x < 1
            or new_x >= WIDTH / TILE_SIZE - 1
            or new_y < 1
            or new_y >= HEIGHT / TILE_SIZE - 1
        ):
            print("Game Over! Snake hit the wall.")
            pygame.quit()
            exit()
        if (
            len([pos for pos in zip(self.x_pos, self.y_pos) if pos == (new_x, new_y)])
            > 0
        ):
            print("Game Over! Snake collided with itself.")
            pygame.quit()
            exit()

    def check_apple_collision(self, ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y):
        # print(new_x, new_y, ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y)
        if (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (ga_1_x, ga_1_y):
            self.size += 1
            self.grow = True
            ga_1_x, ga_1_y = generate_new_apple(self, ga_2_x, ga_2_y, ra_1_x, ra_1_y)
        elif (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (ga_2_x, ga_2_y):
            self.size += 1
            self.grow = True
            ga_2_x, ga_2_y = generate_new_apple(self, ga_1_x, ga_1_y, ra_1_x, ra_1_y)

        elif (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (ra_1_x, ra_1_y):
            self.size -= 1
            self.reduce = True
            if self.size < 1:
                print("Game Over! Snake size reduced to zero.")
                pygame.quit()
                exit()
            ra_1_x, ra_1_y = generate_new_apple(self, ga_2_x, ga_2_y, ga_1_x, ga_1_y)
        return ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y

    def move(self):
        new_x = self.x_pos[0] + DIRECTIONS[self.direction][0]
        new_y = self.y_pos[0] + DIRECTIONS[self.direction][1]
        self.check_collision(new_x, new_y)
        self.x_pos.insert(0, new_x)
        self.y_pos.insert(0, new_y)

        if self.grow:
            self.grow = False
        elif self.reduce:
            self.reduce = False
            self.x_pos.pop()
            self.y_pos.pop()
            self.x_pos.pop()
            self.y_pos.pop()
        else:
            self.x_pos.pop()
            self.y_pos.pop()


def generate_new_apple(snake, x_1, y_1, x_2, y_2):
    ga_1_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
    ga_1_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)
    while (
        ga_1_x == x_1
        and ga_1_y == y_1
        or ga_1_x == x_2
        and ga_1_y == y_2
        or len(
            [pos for pos in zip(snake.x_pos, snake.y_pos) if pos == (ga_1_x, ga_1_y)]
        )
        > 0
    ):
        ga_1_x = random.randrange(TILE_SIZE, WIDTH - TILE_SIZE * 2, TILE_SIZE)
        ga_1_y = random.randrange(TILE_SIZE, HEIGHT - TILE_SIZE * 2, TILE_SIZE)
    return ga_1_x, ga_1_y


def render_tiles(tile, screen):
    for i in range(0, WIDTH, TILE_SIZE):
        for j in range(0, HEIGHT, TILE_SIZE):
            if i == 0 or i == WIDTH - TILE_SIZE or j == 0 or j == HEIGHT - TILE_SIZE:
                screen.blit(tile, (i, j))


def generate_apples(snake):
    ga_1_x, ga_1_y = generate_new_apple(snake, 500000, 500000, 500000, 500000)
    ga_2_x, ga_2_y = generate_new_apple(snake, ga_1_x, ga_1_y, 500000, 500000)
    ra_1_x, ra_1_y = generate_new_apple(snake, ga_1_x, ga_1_y, ga_2_x, ga_2_y)
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
    for i in range(1, len(snake.x_pos)):
        screen.blit(
            snake.body, (snake.x_pos[i] * TILE_SIZE, snake.y_pos[i] * TILE_SIZE)
        )


def launch_game():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")
    running = True
    clock = pygame.time.Clock()
    snake = Snake()
    ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y = generate_apples(snake)
    bg_image = pygame.image.load("assets/bg-tr.png")
    green_apple = pygame.image.load("assets/apple_green_32.png")
    red_apple = pygame.image.load("assets/apple_red_32.png")
    bg_image.set_alpha(128)
    tile = pygame.image.load("assets/tile.png")
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
                elif event.key == pygame.K_ESCAPE:
                    running = False

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
        snake.move()
        render_snake(snake, screen)
        pygame.display.flip()
        ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y = snake.check_apple_collision(
            ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y
        )
        clock.tick(FPS)

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
