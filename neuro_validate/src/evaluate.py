from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import balanced_accuracy_score, f1_score

from .models import build_classifier
from .schema import ModelEvaluation


def evaluate_feature_family(
    feature_family: str,
    X: list[list[float]],
    y: list[int],
    groups: list[str],
    model_family: str = "logistic_regression",
) -> ModelEvaluation:
    folds = _leave_one_group_out_splits(groups)
    balanced_accuracies = []
    macro_f1s = []
    for train_indices, test_indices in folds:
        model = build_classifier(model_family).fit(
            [X[index] for index in train_indices],
            [y[index] for index in train_indices],
        )
        predictions = model.predict([X[index] for index in test_indices])
        truth = [y[index] for index in test_indices]
        balanced_accuracies.append(float(balanced_accuracy_score(truth, predictions)))
        macro_f1s.append(float(f1_score(truth, predictions, average="macro")))

    return ModelEvaluation(
        feature_family=feature_family,
        mean_balanced_accuracy=(sum(balanced_accuracies) / len(balanced_accuracies)) if balanced_accuracies else 0.0,
        mean_macro_f1=(sum(macro_f1s) / len(macro_f1s)) if macro_f1s else 0.0,
        per_fold_balanced_accuracy=tuple(balanced_accuracies),
        per_fold_macro_f1=tuple(macro_f1s),
        sample_count=len(X),
        subject_count=len(set(groups)),
    )


def _leave_one_group_out_splits(groups: list[str]) -> list[tuple[list[int], list[int]]]:
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[group].append(index)
    splits = []
    all_indices = set(range(len(groups)))
    for held_out_group, test_indices in sorted(group_to_indices.items()):
        train_indices = sorted(all_indices - set(test_indices))
        if not train_indices or not test_indices:
            continue
        splits.append((train_indices, list(test_indices)))
    return splits
