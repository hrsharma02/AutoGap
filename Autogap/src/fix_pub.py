import pickle
import numpy as np
from src import statistical_validation as sv
from src import visualize as viz

with open('results/main_checkpoint_d10_v2.pkl', 'rb') as f:
    all_results = pickle.load(f)

all_gaps, all_alignments = [], []
for r in all_results:
    for gk, ak in [('gap_autogap', 'al_autogap'),
                   ('gap_autogap_simple', 'al_autogap_simple'),
                   ('gap_rs10', 'al_rs10'),
                   ('gap_rs30', 'al_rs30'),
                   ('gap_dr', 'al_dr'),
                   ('gap_direct', 'al_direct')]:
        all_gaps.append(r[gk])
        all_alignments.append(r[ak])
all_gaps = np.array(all_gaps)
all_alignments = np.array(all_alignments)

print('N =', len(all_gaps))

corr = sv.correlation_with_ci(all_gaps, all_alignments, method='pearson')
print('r =', corr['r'], 'p =', corr['p'], 'CI =', [corr['ci_lo'], corr['ci_hi']])

viz.fig_correlation_scatter(all_gaps, all_alignments, corr,
                            'experiments1/fig4_correlation.png')
print('Saved experiments1/fig4_correlation.png')