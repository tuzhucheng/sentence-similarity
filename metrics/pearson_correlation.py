from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class PearsonCorrelation(Metric):
    """
    Calculates the correlation.
    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._predictions = []
        self._gold = []

    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.sum(squared_errors)
        self._num_examples += y.shape[0]

    def compute(self):
        if len(self._predictions) == 0:
            raise NotComputableError('Pearson correlation must have at least one example before it can be computed')
        return self._sum_of_squared_errors / self._num_examples
