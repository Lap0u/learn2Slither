# python3 play.py --mode multiplay --num_games 500 --path
#  ./3000_green_rew_dqn_snake_model.pth -> max size 13
# python3 play.py --mode multiplay --num_games 500 --path
#  ./low_epsilon_3000_full_rew.pth -> max size 15
# python3 play.py --mode multiplay --num_games 500 --path
#  ./high_epsilon_3000_full_rew.pth -> max size 11 se bloque un peu
# python3 play.py --mode multiplay --num_games 5000 --path
#  ./low_epsilon_3000_full_rew.pth -> max size 17
# python3 play.py --mode multiplay --num_games 500 --path
#  ./3000_new_dqn_snake_model.pth -> max size 9
# python3 play.py --mode multiplay --num_games 500 --path
#  ./3000_no_reward_dqn_snake_model.pth -> max size 9 se bloque beaucoup
#  python3 play.py --mode multiplay --num_games 500 --path
#  ./3000_reward_full_dqn_snake_model.pth -> max size 11 se bloque un peu
# python3 play.py --mode multiplay --num_games 500 --path
#  ./5000_basic_reward_dqn_snake_model.pth -> max size 12
# python3 play.py --mode multiplay --num_games 500 --path
#  ./5000_green_rew_dqn_snake_model.pth -> max size 11
# python3 play.py --mode multiplay --num_games 5000 --path
#  ./3000_lowlr_dqn_snake_model.pth -> max size 13

from src.agentQ import AgentQ
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["play", "multiplay"],
    )
    parser.add_argument('--speed', type=int, default=50)
    parser.add_argument("--path", type=str, default="dqn_snake_model.pth")
    parser.add_argument("--num_games", type=int, default=50)
    args = parser.parse_args()
    agent = AgentQ()
    if args.mode == "play":
        agent.play(model_path=args.path, speed=args.speed)
    elif args.mode == "multiplay":
        agent.multiplay(model_path=args.path, num_games=args.num_games)
