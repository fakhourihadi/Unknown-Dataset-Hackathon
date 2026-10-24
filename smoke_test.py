import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
n = 100
x = pd.DataFrame({'age': rng.integers(18, 70, n), 'income': rng.normal(50000, 15000, n), 'city': rng.choice(['A','B','C'], n), 'constant': 1})
x.loc[3:8, 'income'] = np.nan
cases = [
    ('classification', x.assign(target=np.where(x['age'] > 40, 'yes', 'no')), ['--target', 'target']),
    ('regression', x.assign(target=x['age'] * 2.5 + rng.normal(0, 2, n)), ['--target', 'target']),
    ('clustering', x, ['--clusters', '3']),
]
for task, data, extra in cases:
    train_path = folder / f'_test_{task}.csv'
    output = folder / f'_pred_{task}.csv'
    profile = folder / f'_profile_{task}.json'
    data.to_csv(train_path, index=False)
    cmd = [sys.executable, str(folder / 'main.py'), '--train', str(train_path), '--task', task, '--output', str(output), '--profile', str(profile)] + extra
    subprocess.run(cmd, check=True)
    assert len(pd.read_csv(output)) == n
    assert profile.exists()
    print(f'{task}: PASS')
    for p in [train_path, output, profile]:
        p.unlink()
