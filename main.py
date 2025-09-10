import argparse


from src.agentQ import AgentQ
from src.game import Game

if __name__ == "__main__":
    agent = AgentQ()
    agent.train()
