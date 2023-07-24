import json
import numpy as np
from queue import Queue
from collections import defaultdict

from utils import avg, ENDPOINTS, MAX_CONCURRENT_TRANSFERS

from Caws.predictors import RuntimePredictor

class InputLength(RuntimePredictor):

    LEARNING_THRESH = 3

    def __init__(self, endpoints, train_every=1, *args, **kwargs):
        # TODO: ensure that the number of data points stored stays under some
        # threshold, to guarantee low memory usage and fast training
        super().__init__(endpoints)
        self.lengths = defaultdict(lambda: defaultdict(list))
        self.runtimes = defaultdict(lambda: defaultdict(list))
        self.weights = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))

        self.train_every = train_every
        self.updates_since_train = defaultdict(lambda: defaultdict(int))

    def predict(self, func, group, payload, *args, **kwargs):
        pred = self.weights[func][group].T.dot(self._preprocess(len(payload)))
        return pred.item()

    def update(self, task_info, new_runtime):
        func = task_info['function_id']
        end = task_info['endpoint_id']
        group = self.endpoints[end]['group']

        self.lengths[func][group].append(len(task_info['payload']))
        self.runtimes[func][group].append(new_runtime)

        self.updates_since_train[func][group] += 1
        if self.updates_since_train[func][group] >= self.train_every:
            self._train(func, group)
            self.updates_since_train[func][group] = 0

    def has_learned(self, func, endpoint):
        group = self.endpoints[endpoint]['group']
        return len(self.runtimes[func][group]) > self.LEARNING_THRESH

    def _train(self, func, group):
        lengths = np.array([self._preprocess(x)
                            for x in self.lengths[func][group]])
        lengths = lengths.reshape((-1, 3))
        runtimes = np.array([self.runtimes[func][group]]).reshape((-1, 1))
        self.weights[func][group] = np.linalg.pinv(lengths).dot(runtimes)

    def _preprocess(self, x):
        '''Create features that are easy to learn from.'''
        return np.array([1, x, x ** 2])


def init_runtime_predictor(predictor, *args, **kwargs):
    predictor = predictor.strip().lower()
    if predictor.endswith('average') or predictor.endswith('avg'):
        return RollingAverage(*args, **kwargs)
    elif predictor.endswith('length') or predictor.endswith('size'):
        return InputLength(*args, **kwargs)
    else:
        raise NotImplementedError("Predictor: {}".format(predictor))