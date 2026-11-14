"""Plot paired validation outcomes; training seeds are not independent datasets."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    report = json.loads((ROOT/'outputs'/'baseline_check_results.json').read_text(encoding='utf-8'))
    rows = report['runs']
    seeds = report['plan']['seeds']
    metrics = [('macro_f1', 'Clean Macro-F1'), ('plastic_recall', 'Plastic recall'),
               ('dark_02_f1', 'Dark (brightness 0.2) Macro-F1'), ('blur_4_f1', 'Blur (radius 4) Macro-F1')]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), layout='constrained')
    for axis, (key, title) in zip(axes, metrics):
        values = []
        for seed in seeds:
            scores = [next(row[key] for row in rows if row['seed'] == seed and row['recipe'] == recipe)
                      for recipe in ['baseline', 'candidate']]
            values.extend(scores)
            axis.plot([0, 1], scores, marker='o', linewidth=1.6, label=f'Seed {seed}')
        axis.set(xticks=[0, 1], xticklabels=['Baseline', 'Candidate'], ylabel='Validation score', title=title,
                 xlim=(-.2, 1.2), ylim=(max(0, min(values)-.04), min(1, max(values)+.04)))
        axis.grid(axis='y', alpha=.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Fixed split, paired training seeds; 375 validation images\nCandidate adds brightness/blur augmentation and class weighting. Real-world accuracy remains unknown.', fontsize=10)
    fig.savefig(ROOT/'outputs'/'baseline_check_comparison.png', dpi=160)
    plt.close(fig)
    print('Saved outputs/baseline_check_comparison.png', flush=True)


if __name__ == '__main__':
    main()
