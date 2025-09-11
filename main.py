import argparse
from src.agentQ import AgentQ

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=3000)
    args = parser.parse_args()
    agent = AgentQ()
    agent.train(sessions=args.session)
