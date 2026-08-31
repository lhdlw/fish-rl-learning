import csv
from types import SimpleNamespace
import numpy as np
import torch
from run_experiments import train_one


def test_short_training_exact_repeat_and_budget(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args = SimpleNamespace(warmup=64, decay_steps=100, max_steps=50,
                           steps=75, eval_episodes=2, threads=1, verbose=False)
    outputs = [tmp_path/'first', tmp_path/'second']
    summaries = [train_one(args, 'dqn_full', 0, output) for output in outputs]
    for summary in summaries:
        assert summary['train_steps'] == 75
        assert summary['updates'] == 12
    assert {k:v for k,v in summaries[0].items() if k != 'seconds'} == {
        k:v for k,v in summaries[1].items() if k != 'seconds'}
    folders = [p/'runs/dqn_full__seed=0' for p in outputs]
    for name in ['evaluation_episodes.csv', 'training_episodes.csv']:
        assert (folders[0]/name).read_bytes() == (folders[1]/name).read_bytes()
    with (folders[0]/'training_episodes.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert sum(int(r['steps']) for r in rows) == 75
    assert rows[-1]['budget_cutoff'] == '1'
    assert rows[-1]['time_limit'] == '0.0'
    models = [torch.load(p/'model.pt', map_location='cpu', weights_only=True)['model'] for p in folders]
    assert all(torch.equal(models[0][k], models[1][k]) for k in models[0])
