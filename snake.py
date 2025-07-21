import argparse
import pygame


def launch_game():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake Game")
    running = True
    bg_image = pygame.image.load("sprites/bg.png")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # screen.fill((0, 0, 0))
        screen.blit(bg_image, (0, 0))
        pygame.display.update()

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
