import copy
import numpy as np
import pytest
import torch
from fishrl import FishEnv, DQNAgent, AgentConfig
from run_experiments import RandomPolicy, evaluate, stream_seed

def test_remaining_time_and_timeout():
    env = FishEnv(seed=0, max_steps=2)
    assert env.reset()[-1] == 1.0
    state, _, done, info = env.step(0)
    assert state[-1] == .5 and not done and info['terminated'] == 0
    state, _, done, info = env.step(0)
    assert state[-1] == 0 and done and info['time_limit'] == 1
    assert info['death'] == 0 and info['terminated'] == 1 and info['truncated'] == 0
    with pytest.raises(RuntimeError):
        env.step(0)

def test_death_distinct_from_time_limit():
    env = FishEnv(seed=0, max_steps=500)
    env.reset()
    env.npc = [[400., 294., 55., 0.]] + [[0., 50., 25., 0.]]*3
    _, _, done, info = env.step(0)
    assert done and info['death'] == 1 and info['time_limit'] == 0
    assert info['objective_return'] == -5

def test_seed_streams_and_random_reproducibility():
    seeds = [stream_seed(s, stream, ep) for s in range(3) for stream in range(3) for ep in range(50)]
    assert len(seeds) == len(set(seeds))
    a = evaluate(RandomPolicy(3), 'sparse', 1, 3, 20)
    b = evaluate(RandomPolicy(3), 'sparse', 1, 3, 20)
    assert a == b

def test_evaluation_does_not_mutate_agent_or_rng():
    agent = DQNAgent(28, 4, AgentConfig(), seed=4, device='cpu')
    params = [p.detach().clone() for p in agent.online.parameters()]
    rng = copy.deepcopy(agent.rng.bit_generator.state)
    replay_rng = agent.buffer.rng.getstate()
    raw = []
    evaluate(agent, 'full', 3, 3, 20, raw)
    assert len(raw) == 3 and len(agent.buffer) == 0 and agent.updates == 0
    assert rng == agent.rng.bit_generator.state and replay_rng == agent.buffer.rng.getstate()
    assert all(torch.equal(p, q) for p,q in zip(params, agent.online.parameters()))
