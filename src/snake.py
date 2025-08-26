import pygame
import random
import src.generator as gen
import src.globals as globals


class Snake:
    def __init__(self):
        self.size = 3
        self.grow = False
        self.reduce = False
        self.is_dead = False
        self.head = pygame.image.load("assets/snake_head.png")
        self.body = pygame.image.load("assets/snake_body.png")
        self.head_xx = pygame.image.load("assets/snake_head_xx.png")
        # self.direction = "RIGHT"
        self.direction = random.choice(list(globals.DIRECTIONS.keys()))
        self.x_pos = [
            globals.WIDTH / globals.TILE_SIZE / 2,
            globals.WIDTH / globals.TILE_SIZE / 2
            - globals.DIRECTIONS[self.direction][0],
            globals.WIDTH / globals.TILE_SIZE / 2
            - globals.DIRECTIONS[self.direction][0] * 2,
        ]
        self.y_pos = [
            globals.HEIGHT / globals.TILE_SIZE / 2,
            globals.HEIGHT / globals.TILE_SIZE / 2
            - globals.DIRECTIONS[self.direction][1],
            globals.HEIGHT / globals.TILE_SIZE / 2
            - globals.DIRECTIONS[self.direction][1] * 2,
        ]

    def render(self, screen):
        head_x = self.x_pos[0] * globals.TILE_SIZE
        head_y = self.y_pos[0] * globals.TILE_SIZE
        screen.blit(self.head, (head_x, head_y))
        for i in range(1, len(self.x_pos)):
            screen.blit(
                self.body,
                (self.x_pos[i] * globals.TILE_SIZE, self.y_pos[i] * globals.TILE_SIZE),
            )

    def check_collision(self, new_x, new_y):
        if (
            new_x < 1
            or new_x >= globals.WIDTH / globals.TILE_SIZE - 1
            or new_y < 1
            or new_y >= globals.HEIGHT / globals.TILE_SIZE - 1
        ):
            print("Game Over! Snake hit the wall. size : ", self.size)
            self.is_dead = True
            pygame.quit()
        if (
            len([pos for pos in zip(self.x_pos, self.y_pos) if pos == (new_x, new_y)])
            > 0
        ):
            print("Game Over! Snake collided with itself. size : ", self.size)
            self.is_dead = True
            pygame.quit()

    def check_apple_collision(self, green_apple_1, green_apple_2, red_apple):
        # print(new_x, new_y, ga_1_x, ga_1_y, ga_2_x, ga_2_y, ra_1_x, ra_1_y)
        if (self.x_pos[0], self.y_pos[0]) == (
            green_apple_1.x,
            green_apple_1.y,
        ):
            self.size += 1
            self.grow = True
            green_apple_1.x, green_apple_1.y = gen.generate_new_apple(
                self, green_apple_2.x, green_apple_2.y, red_apple.x, red_apple.y
            )
        elif (self.x_pos[0], self.y_pos[0]) == (
            green_apple_2.x,
            green_apple_2.y,
        ):
            self.size += 1
            self.grow = True
            green_apple_2.x, green_apple_2.y = gen.generate_new_apple(
                self, green_apple_1.x, green_apple_1.y, red_apple.x, red_apple.y
            )

        elif (self.x_pos[0], self.y_pos[0]) == (
            red_apple.x,
            red_apple.y,
        ):
            self.size -= 1
            self.reduce = True
            if self.size < 1:
                print("Game Over! Snake size reduced to zero.")
                self.is_dead = True
                pygame.quit()
            red_apple.x, red_apple.y = gen.generate_new_apple(
                self, green_apple_2.x, green_apple_2.y, red_apple.x, red_apple.y
            )
        return green_apple_1, green_apple_2, red_apple

    def move(self):
        new_x = self.x_pos[0] + globals.DIRECTIONS[self.direction][0]
        new_y = self.y_pos[0] + globals.DIRECTIONS[self.direction][1]
        self.check_collision(new_x, new_y)
        self.check_apple_collision
        if self.is_dead:
            return -100
        self.x_pos.insert(0, new_x)
        self.y_pos.insert(0, new_y)

        if self.grow:
            self.grow = False
            return 10
        elif self.reduce:
            self.reduce = False
            self.x_pos.pop()
            self.y_pos.pop()
            self.x_pos.pop()
            self.y_pos.pop()
            return -10
        else:
            self.x_pos.pop()
            self.y_pos.pop()
            return 1
        return 1
