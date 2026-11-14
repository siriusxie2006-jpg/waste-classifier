"""Plot the principal training-method ablations on identical validation conditions."""
import csv
from train_baseline import ROOT, plt


def main():
    with (ROOT/'outputs'/'merged_validation_comparison.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    runs = [('baseline_resnet18_seed42', 'Baseline'),
            ('resnet18_both_seed42', 'Light + blur'),
            ('merged_weighted_seed42', 'Class weighting'),
            ('merged_both_weighted_seed42', 'Weighting + light + blur')]
    panels = [('Darkening', ['clean', 'brightness_0.7', 'brightness_0.4', 'brightness_0.2']),
              ('Blur', ['clean', 'blur_1', 'blur_2', 'blur_4']),
              ('Brightening', ['clean', 'brightness_1.3', 'brightness_1.7'])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    for axis, (title, conditions) in zip(axes, panels):
        for run, label in runs:
            scores = [float(next(row['macro_f1'] for row in rows
                                 if row['run_name'] == run and row['condition'] == condition))
                      for condition in conditions]
            axis.plot(range(len(conditions)), scores, marker='o', label=label)
        axis.set(title=title, xticks=range(len(conditions)), xticklabels=conditions,
                 ylabel='Validation Macro-F1', ylim=(.5, 1.))
        axis.tick_params(axis='x', rotation=30)
        axis.grid(alpha=.2)
    axes[1].legend(loc='lower left', fontsize=8)
    fig.suptitle('Merged experiments: seed 42 exploratory validation results')
    fig.tight_layout()
    fig.savefig(ROOT/'outputs'/'merged_validation_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
