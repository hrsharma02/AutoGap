"""Regenerate Fig 4 (Sobol indices) from the d=10 checkpoint."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, ROOT)

import pickle
from src import visualize as viz

CKPT = os.path.join(ROOT, 'results', 'main_checkpoint_d10_v2.pkl')
OUT  = os.path.join(ROOT, 'experiments1', 'fig3_sobol_d10.png')

with open(CKPT, 'rb') as f:
    all_results = pickle.load(f)

sb = all_results[0]['sobol']
print(f'Generating Fig 4 from {CKPT}')
print(f'  Parameters: {sb["names"]}')
print(f'  Max ST:     {max(sb["ST"]):.4f}')
assert not any(n.startswith('dummy_') for n in sb['names']), \
    'ERROR: checkpoint contains dummies'

viz.fig_sobol_indices(sb['names'], sb['S1'], sb['ST'], OUT)
print(f'Saved: {OUT}')