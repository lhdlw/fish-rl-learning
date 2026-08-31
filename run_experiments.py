"""Fixed-step experiment: random baseline versus sparse/full Double DQN."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from fishrl import AgentConfig, DQNAgent, FishEnv, RewardConfig

METRICS = ['objective_return', 'fish_eaten', 'success_rate', 'death_rate', 'episode_length', 'training_return']

def stream_seed(seed, stream, episode=0):
    return int(np.random.SeedSequence([seed, stream, episode]).generate_state(1)[0])

class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
    def act(self, state, training=False):
        return int(self.rng.integers(0, 4))

def evaluate(agent, reward_name, seed, episodes, max_steps, raw_rows=None):
    rows = []
    for episode in range(episodes):
        env_seed = stream_seed(seed, 1, episode)
        env = FishEnv(reward_name, seed=env_seed, max_steps=max_steps)
        state, done = env.reset(seed=env_seed), False
        total = 0.0
        while not done:
            state, reward, done, info = env.step(agent.act(state, training=False))
            total += reward
        rows.append(dict(episode=episode, eval_env_seed=env_seed,
            objective_return=info['objective_return'], fish_eaten=info['fish_eaten'],
            success_rate=float(info['fish_eaten'] > 0), death_rate=info['death'],
            episode_length=env.steps, training_return=total, time_limit=info['time_limit']))
    if raw_rows is not None:
        raw_rows.extend(rows)
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def provenance():
    root = Path(__file__).resolve().parent
    sources = [root/'run_experiments.py', root/'analyze_results.py', *sorted((root/'fishrl').glob('*.py'))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    def git(*args):
        result = subprocess.run(['git', '-c', 'core.excludesFile=', *args], cwd=root,
                                capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    return {'code_commit': git('rev-parse', 'HEAD'), 'working_tree': git('status', '--porcelain'),
            'source_sha256': hashes, 'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'torch', 'matplotlib']}}

def train_one(args, method, seed, output):
    reward_name = 'full' if method == 'dqn_full' else 'sparse'
    run_id = f'{method}__seed={seed}'
    folder = output/'runs'/run_id
    folder.mkdir(parents=True)
    env_seed, policy_seed = stream_seed(seed, 0), stream_seed(seed, 2)
    cfg = AgentConfig(warmup=args.warmup, decay_steps=args.decay_steps)
    config = {'method': method, 'seed': seed, 'train_env_seed': env_seed, 'policy_seed': policy_seed,
        'agent_config': asdict(cfg), 'reward_config': asdict(RewardConfig.preset(reward_name)),
        'environment_version': FishEnv.version, 'state_dim': FishEnv.state_dim, 'horizon': args.max_steps,
        'training_steps': 0 if method == 'random' else args.steps, 'eval_episodes': args.eval_episodes,
        'device': 'cpu', 'threads': args.threads,
        'seed_scheme': 'SeedSequence([seed, stream, episode]); train=0,eval=1,policy=2'}
    (folder/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    started, history = time.time(), []
    if method == 'random':
        agent, updates = RandomPolicy(policy_seed), 0
    else:
        agent = DQNAgent(FishEnv.state_dim, 4, cfg, seed=policy_seed, device='cpu')
        env = FishEnv(reward_name, seed=env_seed, max_steps=args.max_steps)
        state = env.reset(seed=env_seed)
        total, episode, episode_steps, losses = 0.0, 0, 0, []
        for step in range(1, args.steps + 1):
            action = agent.act(state)
            ns, reward, done, info = env.step(action)
            agent.observe(state, action, reward, ns, bool(info['terminated']))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            state = ns
            total += reward
            episode_steps += 1
            if done or step == args.steps:
                history.append(dict(episode=episode, end_step=step, steps=episode_steps,
                    training_return=total, objective_return=info['objective_return'],
                    fish_eaten=info['fish_eaten'], death=info['death'], time_limit=info['time_limit'],
                    budget_cutoff=int(not done and step == args.steps),
                    mean_loss=float(np.mean(losses)) if losses else '', epsilon=agent.exploration_value()))
                if done and step < args.steps:
                    state = env.reset()
                    total, episode_steps, losses = 0.0, 0, []
                    episode += 1
            if args.verbose and step % max(1, args.steps//5) == 0:
                print(f'{run_id}: {step}/{args.steps} interactions', flush=True)
        write_csv(folder/'training_episodes.csv', history)
        agent.save(str(folder/'model.pt'))
        updates = agent.updates
    raw = []
    metrics = evaluate(agent, reward_name, seed, args.eval_episodes, args.max_steps, raw)
    write_csv(folder/'evaluation_episodes.csv', raw)
    summary = {'run_id': run_id, 'method': method, 'seed': seed,
        'train_steps': 0 if method == 'random' else args.steps, 'updates': updates,
        'eval_episodes': args.eval_episodes, **metrics, 'seconds': time.time()-started}
    (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"{run_id}: fish={metrics['fish_eaten']:.3f}, death={metrics['death_rate']:.3f}", flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', default='random,dqn_sparse,dqn_full')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--decay-steps', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', default='results_v2')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    methods = [x.strip() for x in args.methods.split(',') if x.strip()]
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {'random','dqn_sparse','dqn_full'}:
        parser.error('invalid or duplicate --methods')
    try:
        seeds = [int(x) for x in args.seeds.split(',')]
    except ValueError:
        parser.error('--seeds must be integers')
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error('--seeds must be unique nonnegative integers')
    for key in ['steps', 'eval_episodes', 'max_steps', 'warmup', 'decay_steps', 'threads']:
        if getattr(args,key) <= 0:
            parser.error(f'{key} must be positive')
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('choose a new, empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = {**vars(args), 'environment_version': FishEnv.version, 'provenance': provenance()}
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    for method in methods:
        for seed in seeds:
            rows.append(train_one(args, method, seed, output))
            write_csv(output/'runs.csv', rows)

if __name__ == '__main__':
    main()
