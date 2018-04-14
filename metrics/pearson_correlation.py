from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
import scipy.stats as stats
import torch


class PearsonCorrelation(Metric):
    """
    Calculates the Pearson correlation.
    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._predictions = []
        self._gold = []

    def update(self, output):
        y_pred, y = output
        self._predictions.append(y_pred)
        self._gold.append(y)

    def compute(self):
        if len(self._predictions) == 0:
            raise NotComputableError('Pearson correlation must have at least one example before it can be computed')

        predicted_scores = torch.cat(self._predictions).data.cpu().numpy()
        gold_scores = torch.cat(self._gold).data.cpu().numpy()

        pearson_score = stats.pearsonr(predicted_scores, gold_scores)[0]
        return pearson_score
