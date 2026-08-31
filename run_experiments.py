from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from fishrl import AgentConfig, DQNAgent, FishEnv, RewardConfig


def evaluate(agent: DQNAgent, reward_name: str, seed: int, episodes: int,
             max_steps: int) -> dict[str, float]:
    rows = []
    for episode in range(episodes):
        env = FishEnv(reward_name, seed=seed + 100_000 + episode, max_steps=max_steps)
        state, done = env.reset(), False
        training_return = 0.0
        while not done:
            state, reward, done, info = env.step(agent.act(state, training=False))
            training_return += reward
        rows.append((info["objective_return"], info["fish_eaten"], info["death"],
                     env.steps, training_return))
    values = np.asarray(rows)
    return {
        "objective_return": float(values[:, 0].mean()),
        "fish_eaten": float(values[:, 1].mean()),
        "success_rate": float((values[:, 1] > 0).mean()),
        "death_rate": float(values[:, 2].mean()),
        "episode_length": float(values[:, 3].mean()),
        "training_return": float(values[:, 4].mean()),
    }


def train_one(args, reward_name: str, exploration: str, architecture: str,
              seed: int, output: Path) -> dict[str, float]:
    env = FishEnv(reward_name, seed=seed, max_steps=args.max_steps)
    cfg = AgentConfig(architecture=architecture, exploration=exploration,
                      warmup=args.warmup, decay_steps=args.decay_steps)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg, seed=seed, device=args.device)
    started = time.time()
    for episode in range(args.episodes):
        state, done = env.reset(seed + episode), False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if args.verbose and (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"{reward_name}/{exploration}/{architecture}/seed={seed}: "
                  f"{episode+1}/{args.episodes}")
    metrics = evaluate(agent, reward_name, seed, args.eval_episodes, args.max_steps)
    run_id = f"reward={reward_name}__explore={exploration}__net={architecture}__seed={seed}"
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_dir / f"{run_id}.pt"))
    return {"run_id": run_id, "reward": reward_name, "exploration": exploration,
            "architecture": architecture, "seed": seed, "episodes": args.episodes,
            "seconds": time.time()-started, **metrics}


def parse_list(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible FishRL experiment matrix")
    parser.add_argument("--rewards", default="sparse,full")
    parser.add_argument("--explorations", default="epsilon,softmax,dynamic_softmax")
    parser.add_argument("--architectures", default="dueling")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=30000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for name in ["episodes", "eval_episodes", "max_steps", "warmup", "decay_steps"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name, allowed in [("rewards", {"sparse", "approach", "approach_danger", "full"}),
                          ("explorations", {"epsilon", "softmax", "dynamic_softmax"}),
                          ("architectures", {"vanilla", "dueling"})]:
        values = parse_list(getattr(args, name))
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            parser.error(f"invalid or duplicate --{name}")
    try:
        seeds = parse_list(args.seeds, int)
    except ValueError:
        parser.error("--seeds must contain integers")
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        parser.error("--seeds must be unique nonnegative integers")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    rows = []
    for reward in parse_list(args.rewards):
        RewardConfig.preset(reward)
        for exploration in parse_list(args.explorations):
            for architecture in parse_list(args.architectures):
                for seed in parse_list(args.seeds, int):
                    row = train_one(args, reward, exploration, architecture, seed, output)
                    rows.append(row)
                    print(row)
                    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
