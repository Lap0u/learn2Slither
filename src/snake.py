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
        self.direction = random.choice(list(globals.DIR.keys()))
        self.x_pos, self.y_pos = self.spawn_snake()

    def spawn_snake(self):
        x = random.randint(3, globals.WIDTH - 3)
        y = random.randint(3, globals.HEIGHT - 3)
        return ([
            x,
            x - globals.DIR[self.direction][0][0],
            x - globals.DIR[self.direction][0][0] * 2,
        ],
            [
                y,
                y - globals.DIR[self.direction][0][1],
                y - globals.DIR[self.direction][0][1] * 2,
            ]
        )

    def render(self, screen):
        head_x = self.x_pos[0] * globals.TILE
        head_y = self.y_pos[0] * globals.TILE
        screen.blit(self.head, (head_x, head_y))
        for i in range(1, len(self.x_pos)):
            screen.blit(
                self.body,
                (self.x_pos[i] * globals.TILE, self.y_pos[i] * globals.TILE),
            )

    def check_collision(self, new_x, new_y):
        if (
            new_x < 1
            or new_x >= globals.WIDTH - 1
            or new_y < 1
            or new_y >= globals.HEIGHT - 1
        ):
            print("Game Over! Snake hit the wall. size : ", self.size)
            self.is_dead = True
            pygame.quit()
        if (
            len([pos for pos in zip(self.x_pos, self.y_pos)
                 if pos == (new_x, new_y)])
            > 0
        ):
            print("Game Over! Snake collided with itself. size : ", self.size)
            self.is_dead = True
            pygame.quit()

    def check_apple_collision(self, g_apple_1, g_apple_2, r_apple):
        if (self.x_pos[0], self.y_pos[0]) == (
            g_apple_1.x,
            g_apple_1.y,
        ):
            self.size += 1
            self.grow = True
            g_apple_1.x, g_apple_1.y = gen.generate_new_apple(
                self, g_apple_2.x, g_apple_2.y, r_apple.x, r_apple.y
            )
        elif (self.x_pos[0], self.y_pos[0]) == (
            g_apple_2.x,
            g_apple_2.y,
        ):
            self.size += 1
            self.grow = True
            g_apple_2.x, g_apple_2.y = gen.generate_new_apple(
                self, g_apple_1.x, g_apple_1.y, r_apple.x, r_apple.y
            )

        elif (self.x_pos[0], self.y_pos[0]) == (
            r_apple.x,
            r_apple.y,
        ):
            self.size -= 1
            self.reduce = True
            if self.size < 1:
                print("Game Over! Snake size reduced to zero.")
                self.is_dead = True
                pygame.quit()
            r_apple.x, r_apple.y = gen.generate_new_apple(
                self, g_apple_2.x, g_apple_2.y, g_apple_1.x, g_apple_1.y
            )
        return g_apple_1, g_apple_2, r_apple

    def move(self):
        new_x = self.x_pos[0] + globals.DIR[self.direction][0][0]
        new_y = self.y_pos[0] + globals.DIR[self.direction][0][1]
        self.check_collision(new_x, new_y)
        self.check_apple_collision
        if self.is_dead:
            return -100
        self.x_pos.insert(0, new_x)
        self.y_pos.insert(0, new_y)

        if self.grow:
            self.grow = False
            return 20
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
            return -1
        return -1
