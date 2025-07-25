import pygame
import src.globals as globals


class Apple:
    def __init__(self, x, y, green, red, path):
        self.x = x
        self.y = y
        self.red = red
        self.green = green
        self.path = pygame.image.load(path)

    def render(self, screen):
        screen.blit(self.path, (self.x * globals.TILE_SIZE, self.y * globals.TILE_SIZE))
