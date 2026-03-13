import json
import os
import pandas as pd
import torch
from sklearn.model_selection import KFold

from helpers_1_dalis.architekturos import IndividualiArchitektura
from helpers_1_dalis.train_evaluate import train_model, test_model

TARGET_F1 = 0.91
MAX_ITERATIONS = 10
N_SPLITS = 5


def evaluate_slice(labels_data, fraction, device):
    """KFold cross-validation on labels_data.iloc[:int(n*fraction)].
    For each fold: trains a fresh model on 4/5, tests on 1/5. Returns mean f1_macro."""
    n = len(labels_data)
    subset = labels_data.iloc[:int(n * fraction)].reset_index(drop=True)

    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    f1_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(subset)):
        train_df = subset.iloc[train_idx].reset_index(drop=True)
        test_df = subset.iloc[test_idx].reset_index(drop=True)

        model = IndividualiArchitektura(num_classes=3)
        model, _ = train_model(
            model=model,
            model_name=f"kfold_fold{fold}",
            train_df=train_df,
            val_df=test_df,
            img_dir='LD2_dataset/images',
            epochs=10,
            batch_size=64,
            lr=0.001,
            device=device,
            save_history=False
        )

        metrics = test_model(
            model=model,
            model_name=f"kfold_fold{fold}",
            metrics_save_dir=None,
            plots_save_dir=None,
            test_df=test_df,
            img_dir='LD2_dataset/images',
            device=device,
            batch_size=64,
        )
        f1_scores.append(metrics['f1_macro'])

    return sum(f1_scores) / len(f1_scores)


def main():
    # Preprocess data
    labels_data = pd.read_csv('LD2_dataset/labels.csv', dtype={'Image': str, 'class_label': int})
    labels_data = labels_data[~labels_data['class_label'].isin([5, 8])]
    label_map = {0: 0, 2: 0, 4: 0, 3: 1, 6: 1, 1: 2, 7: 2, 9: 2}
    labels_data['class_label'] = labels_data['class_label'].map(label_map)
    labels_data = labels_data.reset_index(drop=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = {}
    iterations = 0
    first_above = None
    last_below = 0.0

    # Phase 1: linear scan 10% -> 100%
    for step in range(1, 11):
        if iterations >= MAX_ITERATIONS:
            break
        p = step / 10
        print(f"\n[Iter {iterations + 1}/{MAX_ITERATIONS}] Evaluating {int(p * 100)}% of data ({int(len(labels_data) * p)} rows)...")
        avg_f1 = evaluate_slice(labels_data, p, device)
        key = f"{int(p * 100)}_percent_f1_score"
        results[key] = avg_f1
        iterations += 1
        print(f"  avg F1: {avg_f1:.4f}")

        if avg_f1 >= TARGET_F1:
            first_above = p
            print(f"  >> Threshold {TARGET_F1} reached at {int(p * 100)}%")
            break
        last_below = p

    # Phase 2: binary search
    if first_above is not None and iterations < MAX_ITERATIONS:
        low, high = last_below, first_above
        print(f"\nBinary search between {low * 100:.1f}% and {high * 100:.1f}%")

        while iterations < MAX_ITERATIONS:
            mid = (low + high) / 2
            print(f"\n[Iter {iterations + 1}/{MAX_ITERATIONS}] Evaluating {mid * 100:.1f}% ({int(len(labels_data) * mid)} rows)...")
            avg_f1 = evaluate_slice(labels_data, mid, device)
            key = f"{mid * 100:.1f}_percent_f1_score"
            results[key] = avg_f1
            iterations += 1
            print(f"  avg F1: {avg_f1:.4f}")

            if avg_f1 >= TARGET_F1:
                high = mid
            else:
                low = mid

    # Summary
    print("\n" + "=" * 50)
    above = {k: v for k, v in results.items() if v >= TARGET_F1}
    if above:
        best_key = min(above, key=lambda k: float(k.replace('_percent_f1_score', '')))
        print(f"Smallest slice exceeding F1={TARGET_F1}: {best_key} -> {above[best_key]:.4f}")
    else:
        print(f"Threshold F1={TARGET_F1} not reached within {MAX_ITERATIONS} iterations.")

    os.makedirs('model_params', exist_ok=True)
    out_path = 'model_params/kfold_f1_search_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
