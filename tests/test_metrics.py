import unittest

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_score, recall_score

from metrics import confusion_matrix_loss


class MyTestCase(unittest.TestCase):
    def test_precision(self):
        # test equivalence of my continuous metrics and sklearn metrics
        # sklearn metrics only support binary input
        a = torch.rand(10, 5) > 0.5
        b = torch.rand(10, 5) > 0.5
        for average in "micro", "samples", "macro":
            expected = 1 - precision_score(y_true=a.numpy(), y_pred=b.numpy(), average=average)
            actual = confusion_matrix_loss(a, b, metric="precision", average=average).item()
            self.assertAlmostEqual(expected, actual)

    def test_recall(self):
        a = torch.rand(10, 5) > 0.5
        b = torch.rand(10, 5) > 0.5
        for average in "micro", "samples", "macro":
            expected = 1 - recall_score(y_true=a.numpy(), y_pred=b.numpy(), average=average)
            actual = confusion_matrix_loss(a, b, metric="recall", average=average).item()
            self.assertAlmostEqual(expected, actual)

    def test_f_beta(self):
        a = torch.rand(10, 5) > 0.5
        b = torch.rand(10, 5) > 0.5
        for average in "micro", "samples", "macro":
            for beta in np.random.rand(10) * 5:
                expected = 1 - fbeta_score(y_true=a.numpy(), y_pred=b.numpy(), beta=beta, average=average)
                actual_1 = confusion_matrix_loss(a, b, metric="f_beta", beta=beta, average=average).item()
                # dice should be equivalent to beta when is binary
                actual_2 = confusion_matrix_loss(a, b, metric="dice", beta=beta, average=average).item()
                self.assertAlmostEqual(expected, actual_1, delta=1e-6)
                self.assertAlmostEqual(expected, actual_2, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
