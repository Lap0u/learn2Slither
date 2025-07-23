import argparse
import random
import pygame

GAME_SPEED = 7
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

    def render(self, screen):
        head_x = self.x_pos[0] * TILE_SIZE
        head_y = self.y_pos[0] * TILE_SIZE
        screen.blit(self.head, (head_x, head_y))
        for i in range(1, len(self.x_pos)):
            screen.blit(
                self.body, (self.x_pos[i] * TILE_SIZE, self.y_pos[i] * TILE_SIZE)
            )

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

    def check_apple_collision(self, green_apple_1, green_apple_2, red_apple):
        # print(new_x, new_y, ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y)
        if (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (
            green_apple_1.x,
            green_apple_1.y,
        ):
            self.size += 1
            self.grow = True
            green_apple_1.x, green_apple_1.y = generate_new_apple(
                self, green_apple_2.x, green_apple_2.y, red_apple.x, red_apple.y
            )
        elif (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (
            green_apple_2.x,
            green_apple_2.y,
        ):
            self.size += 1
            self.grow = True
            green_apple_2.x, green_apple_2.y = generate_new_apple(
                self, green_apple_1.x, green_apple_1.y, red_apple.x, red_apple.y
            )

        elif (self.x_pos[0] * TILE_SIZE, self.y_pos[0] * TILE_SIZE) == (
            red_apple.x,
            red_apple.y,
        ):
            self.size -= 1
            self.reduce = True
            if self.size < 1:
                print("Game Over! Snake size reduced to zero.")
                pygame.quit()
                exit()
            red_apple.x, red_apple.y = generate_new_apple(
                self, green_apple_2.x, green_apple_2.y, red_apple.x, red_apple.y
            )
        return green_apple_1, green_apple_2, red_apple

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


class Apple:
    def __init__(self, x, y, green, red, path):
        self.x = x
        self.y = y
        self.red = red
        self.green = green
        self.path = pygame.image.load(path)

    def render(self, screen):
        screen.blit(self.path, (self.x, self.y))


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
    green_apple_1 = Apple(ga_1_x, ga_1_y, True, False, "assets/apple_green_32.png")
    green_apple_2 = Apple(ga_2_x, ga_2_y, True, False, "assets/apple_green_32.png")
    red_apple = Apple(ra_1_x, ra_1_y, False, True, "assets/apple_red_32.png")
    return green_apple_1, green_apple_2, red_apple


def launch_game():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")
    running = True
    clock = pygame.time.Clock()
    snake = Snake()
    green_apple_1, green_apple_2, red_apple = generate_apples(snake)
    bg_image = pygame.image.load("assets/bg-tr.png")
    bg_image.set_alpha(128)
    tile = pygame.image.load("assets/tile.png")
    while running:
        paused = False
        clock.tick(GAME_SPEED)
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
