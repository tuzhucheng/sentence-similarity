"""
Implementation of "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
https://arxiv.org/abs/1603.06560
Adapted from code on blog post: https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
"""
import logging
from math import ceil, log
import pprint
from time import time

import numpy as np


class Hyperband(object):

    def __init__(self, get_random_hyperparameter_configuration, run_then_return_val_loss, max_iter=81, eta=3):
        self.get_random_hyperparameter_configuration = get_random_hyperparameter_configuration
        self.run_then_return_val_loss = run_then_return_val_loss
        self.max_iter = max_iter  # max epochs per configuration
        self.eta = eta  # down-sampling rate
        # total number of unique executions of Successive Halving (minus 1), just log_{eta} (max_iter)
        self.s_max = int(log(self.max_iter) / log(self.eta))
        # total number of iterations per execution of Successive Halving
        self.B = (self.s_max + 1) * self.max_iter

        self.best_loss = np.inf
        self.search_results = []

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.logger = logger

    def run(self):
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s+1) * (self.eta**s)))
            # initial number of iterations to run configurations for
            r = self.max_iter * self.eta**(-s)

            # Successive Halving with (n, r)
            T = [self.get_random_hyperparameter_configuration() for _ in range(n)]
            for i in range(s+1):
                # Run each of the n_i configs for r_i iters and keep best n_i / eta
                n_i = n * self.eta**(-i)
                r_i = int(r * self.eta**i)

                self.logger.info('%s configurations, %s iterations each', len(T), r_i)

                val_losses = []

                for t in T:
                    start_time = time()
                    val_loss = self.run_then_return_val_loss(r_i, t)
                    elapsed_time = time() - start_time

                    val_losses.append(val_loss)

                    config_results = {
                        'iters': r_i,
                        'params': t,
                        'loss': val_loss,
                        'time': elapsed_time
                    }
                    self.search_results.append(config_results)

                    self.logger.info('Completed new config run: %s', pprint.pformat(config_results, indent=4))

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.logger.info('This is a new best...')

                T = [T[i] for i in np.argsort(val_losses)[:int(n_i) // self.eta]]

        return self.search_results
