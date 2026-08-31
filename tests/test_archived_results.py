import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_archived_counts_metrics_and_demo():
    folder = ROOT/'data/v2_small'
    runs = rows(folder/'runs.csv')
    assert len(runs) == 9
    assert {(r['method'], r['seed']) for r in runs} == {
        (m,str(s)) for m in ['random','dqn_sparse','dqn_full'] for s in range(3)}
    for run in runs:
        run_folder = folder/'runs'/run['run_id']
        evaluation = rows(run_folder/'evaluation_episodes.csv')
        assert len(evaluation) == 50
        for key in ['fish_eaten','death_rate','success_rate','objective_return','training_return','episode_length']:
            assert np.isclose(np.mean([float(r[key]) for r in evaluation]), float(run[key]))
        for episode in evaluation:
            assert float(episode['objective_return']) == 5*(float(episode['fish_eaten'])-float(episode['death_rate']))
        if run['method'] != 'random':
            training = rows(run_folder/'training_episodes.csv')
            assert sum(int(r['steps']) for r in training) == 15000
            assert int(run['train_steps']) == 15000
    metadata = json.loads((ROOT/'media/v2_demo.json').read_text())
    first = rows(folder/'runs/dqn_full__seed=0/evaluation_episodes.csv')[0]
    assert metadata['eval_env_seed'] == int(first['eval_env_seed'])
    assert metadata['fish_eaten'] == float(first['fish_eaten'])
    assert metadata['death'] == float(first['death_rate'])
    assert np.isclose(metadata['training_return'], float(first['training_return']))
    model = folder/'runs/dqn_full__seed=0/model.pt'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == metadata['model_sha256']
