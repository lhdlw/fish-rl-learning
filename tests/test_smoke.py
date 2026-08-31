import numpy as np

from fishrl import AgentConfig, DQNAgent, FishEnv


def test_seeded_environment_is_reproducible():
    a, b = FishEnv(seed=7), FishEnv(seed=7)
    sa, sb = a.reset(), b.reset()
    assert np.array_equal(sa, sb)
    for action in [0, 3, 1, 2] * 10:
        sa, ra, da, ia = a.step(action)
        sb, rb, db, ib = b.step(action)
        assert np.array_equal(sa, sb)
        assert (ra, da, ia) == (rb, db, ib)


def test_observation_and_all_reward_presets():
    for reward in ["sparse", "approach", "approach_danger", "full"]:
        env = FishEnv(reward=reward, seed=0, max_steps=5)
        state = env.reset()
        assert state.shape == (27,)
        for _ in range(5):
            state, value, done, info = env.step(0)
        assert done and info["truncated"] == 1.0


def test_every_exploration_policy_returns_valid_action():
    env = FishEnv(seed=0)
    for exploration in ["epsilon", "softmax", "dynamic_softmax"]:
        agent = DQNAgent(env.state_dim, env.action_dim,
                         AgentConfig(exploration=exploration, warmup=2), seed=0,
                         device="cpu")
        assert agent.act(env.reset()) in range(4)
