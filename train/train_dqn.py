"""
DQN Training module for the Highway Driving environment.

Supports different driving styles via reward shaping:
- aggressive: rewards higher speed, more overtakes, less penalty for lane changes
- conservative: rewards lane keeping, penalizes high speed and frequent lane changes
- balanced: default balanced rewards
"""

import os
import argparse
import json
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import env  # noqa: F401 (triggers gymnasium registration)
import gymnasium as gym


STYLE_CONFIGS = {
    "aggressive": {
        "learning_rate": 1e-3,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.02,
        "gamma": 0.95,
        "batch_size": 128,
        "description": "Aggressive driver: prioritizes speed and overtaking",
    },
    "conservative": {
        "learning_rate": 5e-4,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
        "batch_size": 64,
        "description": "Conservative driver: prioritizes safety and lane keeping",
    },
    "balanced": {
        "learning_rate": 7e-4,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.03,
        "gamma": 0.97,
        "batch_size": 96,
        "description": "Balanced driver: moderate risk and speed",
    },
}


class TrainingLogger(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self):
        self.current_rewards += self.locals["rewards"][0]
        self.current_length += 1

        dones = self.locals.get("dones", self.locals.get("terminateds", [False]))
        if dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            self.current_rewards = 0
            self.current_length = 0

            if len(self.episode_rewards) % 10 == 0:
                recent = self.episode_rewards[-10:]
                print(
                    f"  Episode {len(self.episode_rewards)} | "
                    f"Avg Reward: {np.mean(recent):.1f} | "
                    f"Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}"
                )
        return True

    def _on_training_end(self):
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
            }, f)
        print(f"Training log saved to {log_path}")


def train(style="balanced", total_timesteps=50000, save_dir="models"):
    assert style in STYLE_CONFIGS, f"Unknown style: {style}. Choose from {list(STYLE_CONFIGS.keys())}"

    config = STYLE_CONFIGS[style]
    print(f"\n{'='*60}")
    print(f"Training DQN agent - Style: {style}")
    print(f"Description: {config['description']}")
    print(f"Timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    env_instance = gym.make("HighwayDriving-v0", style=style)

    model_dir = os.path.join(save_dir, style)
    os.makedirs(model_dir, exist_ok=True)

    logger = TrainingLogger(log_dir=model_dir)

    model = DQN(
        "MlpPolicy",
        env_instance,
        learning_rate=config["learning_rate"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        buffer_size=50000,
        learning_starts=1000,
        target_update_interval=500,
        train_freq=4,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=logger)

    model_path = os.path.join(model_dir, "dqn_highway")
    model.save(model_path)

    meta = {
        "style": style,
        "description": config["description"],
        "total_timesteps": total_timesteps,
        "hyperparams": {k: v for k, v in config.items() if k != "description"},
        "episodes_trained": len(logger.episode_rewards),
        "final_avg_reward": float(np.mean(logger.episode_rewards[-20:])) if logger.episode_rewards else 0,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to: {model_path}.zip")
    print(f"Metadata saved to: {model_dir}/metadata.json")
    env_instance.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Highway Driving Agent")
    parser.add_argument("--style", type=str, default="balanced",
                        choices=["aggressive", "conservative", "balanced"],
                        help="Driving style to train")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    train(style=args.style, total_timesteps=args.timesteps, save_dir=args.save_dir)
