import random
import src.apple as Apple
import src.globals as globals


def generate_new_apple(snake, x_1, y_1, x_2, y_2):
    ga_1_x = random.randrange(0, globals.WIDTH - 2)
    ga_1_y = random.randrange(0, globals.HEIGHT - 2)
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
        ga_1_x = random.randrange(0, globals.WIDTH - 2)
        ga_1_y = random.randrange(0, globals.HEIGHT - 2)
    return ga_1_x, ga_1_y


def generate_apples(snake):
    ga_1_x, ga_1_y = generate_new_apple(snake, 500000, 500000, 500000, 500000)
    ga_2_x, ga_2_y = generate_new_apple(snake, ga_1_x, ga_1_y, 500000, 500000)
    ra_1_x, ra_1_y = generate_new_apple(snake, ga_1_x, ga_1_y, ga_2_x, ga_2_y)
    green_apple_1 = Apple.Apple(
        ga_1_x,
        ga_1_y,
        True,
        False,
        "assets/apple_green_32.png",
    )
    green_apple_2 = Apple.Apple(
        ga_2_x,
        ga_2_y,
        True,
        False,
        "assets/apple_green_32.png",
    )
    red_apple = Apple.Apple(
        ra_1_x,
        ra_1_y,
        False,
        True,
        "assets/apple_red_32.png",
    )
    return green_apple_1, green_apple_2, red_apple
