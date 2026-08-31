"""Render a fixed v2 evaluation episode; never select the best-looking episode."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np  # Load NumPy runtime before Torch on the tested Windows setup.
from PIL import Image, ImageDraw
import torch
from fishrl import AgentConfig, DQNAgent, FishEnv, RewardConfig
from run_experiments import stream_seed


def frame(env, reward_total):
    image = Image.new('RGB', (800, 650), '#eff7fa')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 600, 800, 650), fill='#15354a')
    for x, y, size, velocity in env.npc:
        color = '#c24c4c' if size >= env.size else '#357eaf'
        radius = size / 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        direction = 1 if velocity > 0 else -1
        draw.line((x, y, x+direction*radius, y), fill='white', width=2)
    radius = env.size / 2
    draw.ellipse((env.x-radius, env.y-radius, env.x+radius, env.y+radius), fill='#249c68')
    draw.text((14, 609), f'v2 | green=agent, blue=smaller, red=larger | step {env.steps}/{env.max_steps}', fill='white')
    draw.text((14, 629), f'fish={env.fish_eaten} death={env.deaths} training reward={reward_total:.2f}', fill='white')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default='media/v2_demo.gif')
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    folder, output = Path(args.run_dir), Path(args.output)
    metadata_path = output.with_suffix('.json')
    if output.exists() or metadata_path.exists():
        parser.error('output already exists; choose a new path')
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config['environment_version'] != FishEnv.version or config['state_dim'] != FishEnv.state_dim:
        parser.error('checkpoint environment version does not match current source')
    torch.set_num_threads(1)
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    agent = DQNAgent(FishEnv.state_dim, 4, AgentConfig(**config['agent_config']), device='cpu')
    agent.online.load_state_dict(checkpoint['model'])
    agent.online.eval()
    seed = stream_seed(config['seed'], 1, 0)
    env = FishEnv(RewardConfig(**config['reward_config']), seed=seed, max_steps=config['horizon'])
    state, done = env.reset(seed=seed), False
    total, frames = 0.0, [frame(env, 0.0)]
    while not done:
        state, reward, done, info = env.step(agent.act(state, training=False))
        total += reward
        if env.steps % args.stride == 0 or done:
            frames.append(frame(env, total))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0)
    metadata = {'selection_rule': 'first evaluation episode; no search for best trajectory',
        'method': config['method'], 'seed': config['seed'], 'eval_env_seed': seed,
        'environment_version': FishEnv.version, 'steps': env.steps,
        'fish_eaten': env.fish_eaten, 'death': env.deaths, 'training_return': total,
        'stride': args.stride, 'frame_duration_ms': 100,
        'model_sha256': hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment_sha256': hashlib.sha256((Path(__file__).parent/'fishrl/environment.py').read_bytes()).hexdigest()}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
