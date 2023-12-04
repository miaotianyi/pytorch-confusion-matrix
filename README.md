# pytorch-confusion-matrix
A self-contained PyTorch library for differentiable precision, recall,
F-beta score (including F1 score), and dice coefficient.

The only dependency is PyTorch.

These scores are "the bigger, the better",
so `1 - score` can be used as a loss function.

Our contribution:
1. Both `y_true` and `y_pred` are of shape `[N, C, ...]`, where `N` is batch size
and `C` is the number of channels. They must be float tensors.
We allow both input tensors to be real-valued probabilities,
which generalize 0-1 hard labels.

2. We formally separate different averaging methods
for these metrics, such as `macro`, `micro`, `samples`,
according to [sklearn conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) 

You can just copy the code without the fuss of importing from this repository.

# Understanding different averaging methods
The following numpy code computes accuracy, precision, recall, and F1-score from a confusion matrix
in a multi-class classification setting. It can help us understand what different averaging methods do.

```python
import numpy as np
from sklearn import metrics


class ConfusionMatrix:
    def __init__(self, cm):
        """
        This helper class computes various scikit-learn metrics
        (accuracy, precision, recall, f1-score)
        given a confusion matrix C of counts.
        C[i, j] is the integer number of points in ground-truth group i and predicted group j.
        
        - This helper is intended to be a reference table for understanding how
            different averaging methods work.
        - "binary" for binary classification and "samples" for multilabel classification
            are not supported yet.
        - We don't deal with division by 0.
        """
        self.cm = cm
    
    def accuracy_score(self):
        return self.cm.diagonal().sum() / self.cm.sum()
    
    def precision_score(self, average=None):
        if average is None:
            return self.cm.diagonal() / self.cm.sum(axis=0)
        elif average == "micro":
            return self.accuracy_score()
        elif average == "macro":
            return self.precision_score(average=None).mean()
        elif average == "weighted":
            return self.precision_score(average=None) @ self.cm.sum(axis=1) / self.cm.sum()
        raise ValueError(f"Uknown average method [{average}]")
    
    def recall_score(self, average=None):
        if average is None:
            return self.cm.diagonal() / self.cm.sum(axis=1)
        elif average == "micro":
            return self.accuracy_score()
        elif average == "macro":
            return self.recall_score(average=None).mean()
        elif average == "weighted":
            return self.accuracy_score()
        raise ValueError(f"Uknown average method [{average}]")
    
    def f1_score(self, average=None):
        if average is None:
            return 2 * self.cm.diagonal() / (self.cm.sum(axis=0) + self.cm.sum(axis=1))
        elif average == "micro":
            # surprisingly, my method is actually more numerically stable than scikit-learn
            return self.accuracy_score()
        elif average == "macro":
            return self.f1_score(average=None).mean()
        elif average == "weighted":
            return self.f1_score(average=None) @ self.cm.sum(axis=1) / self.cm.sum()
        raise ValueError(f"Uknown average method [{average}]")


def test_my_confusion_matrix():
    n_classes = 5
    n_samples = 100

    y1 = np.random.randint(n_classes, size=n_samples)
    y2 = np.random.randint(n_classes, size=n_samples)

    my_cm = metrics.confusion_matrix(y_true=y1, y_pred=y2, labels=np.arange(n_classes), normalize=None)
    cm_helper = ConfusionMatrix(my_cm)

    # accuracy
    assert cm_helper.accuracy_score() == metrics.accuracy_score(y_true=y1, y_pred=y2, normalize=True)

    # precision
    assert np.allclose(metrics.precision_score(y_true=y1, y_pred=y2, average=None), cm_helper.precision_score(average=None))
    assert metrics.precision_score(y_true=y1, y_pred=y2, average="micro") == cm_helper.precision_score(average="micro")
    assert metrics.precision_score(y_true=y1, y_pred=y2, average="macro") == cm_helper.precision_score(average="macro")
    assert metrics.precision_score(y_true=y1, y_pred=y2, average="weighted") == cm_helper.precision_score(average="weighted")

    # recall
    assert np.allclose(metrics.recall_score(y_true=y1, y_pred=y2, average=None), cm_helper.recall_score(average=None))
    assert metrics.recall_score(y_true=y1, y_pred=y2, average="micro") == cm_helper.recall_score(average="micro")
    assert metrics.recall_score(y_true=y1, y_pred=y2, average="macro") == cm_helper.recall_score(average="macro")
    assert metrics.recall_score(y_true=y1, y_pred=y2, average="weighted") == cm_helper.recall_score(average="weighted")

    # f1 score
    # allclose is used more often due to numerical instability
    assert np.allclose(metrics.f1_score(y_true=y1, y_pred=y2, average=None), cm_helper.f1_score(average=None))
    assert np.allclose(metrics.f1_score(y_true=y1, y_pred=y2, average="micro"), cm_helper.f1_score(average="micro"))
    assert np.allclose(metrics.f1_score(y_true=y1, y_pred=y2, average="macro"), cm_helper.f1_score(average="macro"))
    assert np.allclose(metrics.f1_score(y_true=y1, y_pred=y2, average="weighted"), cm_helper.f1_score(average="weighted"))
```
