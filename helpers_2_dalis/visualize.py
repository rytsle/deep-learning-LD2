import os
import matplotlib.pyplot as plt


VARIANT_STYLES = {
    'scratch_noaug':    {'color': '#1f77b4', 'linestyle': '-',  'label': 'Scratch, No Aug'},
    'scratch_aug':      {'color': '#aec7e8', 'linestyle': '--', 'label': 'Scratch, Aug'},
    'pretrained_noaug': {'color': '#d62728', 'linestyle': '-',  'label': 'Pretrained, No Aug'},
    'pretrained_aug':   {'color': '#ff9896', 'linestyle': '--', 'label': 'Pretrained, Aug'},
}


def plot_experiment_results(all_results, save_dir, filename):
    percentages = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(percentages), figsize=(20, 5), sharey=True)

    legend_handles = []

    for ax, pct in zip(axes, percentages):
        variants = all_results[pct]
        for variant_key, history in variants.items():
            style = VARIANT_STYLES.get(variant_key, {})
            color = style.get('color', 'black')
            linestyle = style.get('linestyle', '-')
            label = style.get('label', variant_key)

            epochs = list(range(1, len(history['train_f1']) + 1))

            # Train line: thinner/lighter
            ax.plot(epochs, history['train_f1'],
                    color=color, linestyle=linestyle, linewidth=1, alpha=0.4)

            # Val/test line: bolder
            val_line, = ax.plot(epochs, history['test_f1'],
                                color=color, linestyle=linestyle, linewidth=2, label=label)

        ax.set_title(f'{pct}% of data')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('F1 Macro')

    # Build legend from last axes (all variants should be plotted there)
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {os.path.join(save_dir, filename)}")
