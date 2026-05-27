"""
Flask backend service for the autonomous driving simulator.

Provides REST APIs for:
- Listing available models
- Running simulation episodes
- Streaming simulation state step-by-step
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from stable_baselines3 import DQN
import gymnasium as gym
import numpy as np

import env  # noqa: F401

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_model(style):
    model_path = os.path.join(MODELS_DIR, style, "dqn_highway.zip")
    if not os.path.exists(model_path):
        return None
    return DQN.load(model_path)


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

    env_instance = gym.make("HighwayDriving-v0", style=style)
    obs, _ = env_instance.reset()

    states = []
    rewards = []
    actions = []
    cumulative_reward = 0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env_instance.step(action)
        cumulative_reward += reward

        state = env_instance.unwrapped.get_state_for_render()
        state["reward"] = float(reward)
        state["cumulative_reward"] = float(cumulative_reward)
        state["action"] = action
        state["action_name"] = ["keep", "accelerate", "brake", "left", "right"][action]

        states.append(state)
        rewards.append(float(cumulative_reward))
        actions.append(action)

        if terminated or truncated:
            break

    env_instance.close()

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

        env_instance = gym.make("HighwayDriving-v0", style=style)
        obs, _ = env_instance.reset(seed=42)

        cumulative_reward = 0
        rewards_over_time = []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_instance.step(int(action))
            cumulative_reward += reward
            rewards_over_time.append(float(cumulative_reward))

            if terminated or truncated:
                break

        state = env_instance.unwrapped.get_state_for_render()
        env_instance.close()

        results[style] = {
            "total_reward": float(cumulative_reward),
            "steps": len(rewards_over_time),
            "overtakes": state["overtakes"],
            "collisions": state["collisions"],
            "avg_speed": state["ego"]["speed"],
            "rewards_over_time": rewards_over_time,
        }

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
