import numpy as np
import pytest
from fishrl import AgentConfig, DQNAgent, FishEnv
from run_experiments import evaluate


def test_invalid_action():
    with pytest.raises(ValueError):
        FishEnv().step(4)


def test_reward_presets_do_not_change_task_dynamics():
    a, b = FishEnv('sparse', seed=8), FishEnv('full', seed=8)
    assert np.array_equal(a.reset(), b.reset())
    for action in [0, 1, 2, 3] * 30:
        sa, _, da, ia = a.step(action)
        sb, _, db, ib = b.step(action)
        assert np.array_equal(sa, sb)
        assert da == db
        for key in ['fish_eaten', 'death', 'objective_return']:
            assert ia[key] == ib[key]
        if da:
            break


@pytest.mark.parametrize('architecture', ['vanilla', 'dueling'])
def test_training_step_and_evaluation_isolation(architecture):
    env = FishEnv(seed=3, max_steps=20)
    agent = DQNAgent(27, 4, AgentConfig(architecture=architecture, warmup=8, batch_size=8), seed=3, device='cpu')
    state = env.reset()
    for _ in range(10):
        action = agent.act(state)
        ns, reward, done, _ = env.step(action)
        agent.observe(state, action, reward, ns, done)
        state = ns
    loss = agent.update()
    assert loss is not None and np.isfinite(loss)
    before = (len(agent.buffer), agent.updates)
    metrics = evaluate(agent, 'full', seed=100, episodes=2, max_steps=10)
    assert before == (len(agent.buffer), agent.updates)
    assert all(np.isfinite(v) for v in metrics.values())
    assert 0 <= metrics['success_rate'] <= 1
