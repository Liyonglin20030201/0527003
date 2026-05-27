"""Train all driving styles sequentially."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_dqn import train


def train_all(timesteps=50000):
    styles = ["aggressive", "conservative", "balanced"]
    for style in styles:
        print(f"\n{'#'*60}")
        print(f"# Training style: {style}")
        print(f"{'#'*60}")
        train(style=style, total_timesteps=timesteps)
    print("\n\nAll models trained successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()
    train_all(args.timesteps)
