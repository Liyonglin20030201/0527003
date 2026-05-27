"""
Flask backend service for the autonomous driving simulator.

Provides REST APIs for:
- Listing available models
- Running simulation episodes with CarRacing environment
- Streaming simulation state step-by-step
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import gymnasium as gym
import numpy as np

import env  # noqa: F401

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ACTION_NAMES = ["keep", "left", "right", "accelerate", "brake"]


def load_model(style):
    model_path = os.path.join(MODELS_DIR, style, "dqn_highway.zip")
    if not os.path.exists(model_path):
        return None
    return DQN.load(model_path)


def make_env(style):
    def _init():
        return gym.make("HighwayDriving-v0", style=style)
    return _init


def create_vec_env(style):
    """Create vectorized env matching training setup (frame stacking)."""
    vec_env = DummyVecEnv([make_env(style)])
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    models = []
    if not os.path.exists(MODELS_DIR):
        return jsonify({"models": models})

    for style in os.listdir(MODELS_DIR):
        meta_path = os.path.join(MODELS_DIR, style, "metadata.json")
        model_path = os.path.join(MODELS_DIR, style, "dqn_highway.zip")
        if os.path.exists(model_path):
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            models.append({
                "style": style,
                "description": meta.get("description", style),
                "final_avg_reward": meta.get("final_avg_reward", 0),
                "episodes_trained": meta.get("episodes_trained", 0),
            })

    return jsonify({"models": models})


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run a full simulation episode and return all states."""
    data = request.get_json() or {}
    style = data.get("style", "balanced")
    max_steps = data.get("max_steps", 300)

    model = load_model(style)
    if model is None:
        return jsonify({"error": f"Model '{style}' not found. Train it first."}), 404

    # Use vectorized env with frame stacking (same as training)
    vec_env = create_vec_env(style)
    # Also keep a reference to the raw env for get_state_for_render
    raw_env = vec_env.envs[0].unwrapped

    obs = vec_env.reset()

    states = []
    rewards = []
    cumulative_reward = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_arr, done_arr, info_arr = vec_env.step(action)

        reward = float(reward_arr[0])
        cumulative_reward += reward

        state = raw_env.get_state_for_render()
        state["reward"] = reward
        state["cumulative_reward"] = float(cumulative_reward)
        state["action"] = int(action[0])
        state["action_name"] = ACTION_NAMES[int(action[0])]

        states.append(state)
        rewards.append(float(cumulative_reward))

        if done_arr[0]:
            break

    vec_env.close()

    return jsonify({
        "style": style,
        "total_steps": len(states),
        "total_reward": float(cumulative_reward),
        "overtakes": states[-1]["overtakes"] if states else 0,
        "collisions": states[-1]["collisions"] if states else 0,
        "states": states,
        "rewards": rewards,
    })


@app.route("/api/training-log/<style>", methods=["GET"])
def get_training_log(style):
    """Return training episode rewards for visualization."""
    log_path = os.path.join(MODELS_DIR, style, "training_log.json")
    if not os.path.exists(log_path):
        return jsonify({"error": "Training log not found"}), 404

    with open(log_path) as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/api/compare", methods=["POST"])
def compare_styles():
    """Run simulations for multiple styles and return comparison data."""
    data = request.get_json() or {}
    styles = data.get("styles", ["aggressive", "conservative", "balanced"])
    max_steps = data.get("max_steps", 300)

    results = {}
    for style in styles:
        model = load_model(style)
        if model is None:
            continue

        vec_env = create_vec_env(style)
        raw_env = vec_env.envs[0].unwrapped
        obs = vec_env.reset()

        cumulative_reward = 0
        rewards_over_time = []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward_arr, done_arr, _ = vec_env.step(action)
            cumulative_reward += float(reward_arr[0])
            rewards_over_time.append(float(cumulative_reward))

            if done_arr[0]:
                break

        state = raw_env.get_state_for_render()
        vec_env.close()

        results[style] = {
            "total_reward": float(cumulative_reward),
            "steps": len(rewards_over_time),
            "overtakes": state["overtakes"],
            "collisions": state["collisions"],
            "avg_speed": state["speed"],
            "rewards_over_time": rewards_over_time,
        }

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
