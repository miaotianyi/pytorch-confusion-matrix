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
