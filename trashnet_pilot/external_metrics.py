"""Score five observed materials without discarding the sixth model output."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
OBSERVED = list(range(5))


def material_metrics(truth, probabilities):
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or len(truth) == 0 or not np.isin(truth, OBSERVED).all():
        raise ValueError('External truth must contain the five mapped materials')
    if scores.shape != (len(truth), 6) or not np.isfinite(scores).all():
        raise ValueError('Expected six finite scores for every image')
    if (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(1), 1, atol=1e-5):
        raise ValueError('Expected probabilities over all six outputs')
    pred = scores.argmax(1)
    ranked = np.sort(scores, axis=1)
    needs_review = (ranked[:, -1] < .70) | (ranked[:, -1]-ranked[:, -2] < .15)
    wrong = pred != truth
    plastic, glass = truth == 4, truth == 1
    result = dict(count=len(truth), accuracy=float(accuracy_score(truth, pred)),
                  macro_f1_five_materials=float(f1_score(truth, pred, labels=OBSERVED,
                                                       average='macro', zero_division=0)),
                  classification_report=classification_report(truth, pred, labels=list(range(6)),
                      target_names=CLASSES, output_dict=True, zero_division=0),
                  confusion_matrix=confusion_matrix(truth, pred, labels=list(range(6))).tolist(),
                  predicted_trash_count=int((pred == 5).sum()),
                  predicted_trash_rate=float((pred == 5).mean()),
                  plastic_as_glass_count=int(((pred == 1) & plastic).sum()),
                  glass_as_plastic_count=int(((pred == 4) & glass).sum()),
                  plastic_as_glass_rate=float((pred[plastic] == 1).mean()) if plastic.any() else None,
                  glass_as_plastic_rate=float((pred[glass] == 4).mean()) if glass.any() else None,
                  needs_review_count=int(needs_review.sum()),
                  unflagged_error_count=int((wrong & ~needs_review).sum()),
                  unflagged_count=int((~needs_review).sum()),
                  unflagged_error_rate=float(wrong[~needs_review].mean()) if (~needs_review).any() else None,
                  score_note='Uncalibrated softmax and existing display heuristic; not unknown detection')
    return result, pred, needs_review
