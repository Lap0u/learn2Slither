import argparse
from src.agentQ import AgentQ

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=int, default=3000)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    agent = AgentQ()
    agent.train(sessions=args.session, model=args.model)
