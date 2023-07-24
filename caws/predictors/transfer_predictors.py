import json
import numpy as np
from queue import Queue
from collections import defaultdict

class TransferPredictor(object):

    def __init__(self, endpoints=None, train_every=1, state_file=None):
        self.endpoints = endpoints or ENDPOINTS
        self.sizes = defaultdict(lambda: defaultdict(list))
        self.times = defaultdict(lambda: defaultdict(list))
        self.weights = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))

        self.train_every = train_every
        self.updates_since_train = defaultdict(lambda: defaultdict(int))

        if state_file is not None:
            self._load_state_from_file(state_file)

    def predict_one(self, src, dst, size):
        if src == dst:
            return 0.0

        src_grp = self.endpoints[src]['transfer_group']
        dst_grp = self.endpoints[dst]['transfer_group']

        pred = self.weights[src_grp][dst_grp].T.dot(self._preprocess(size))
        return pred.item()

    def predict(self, files_by_src, dst):
        '''Predict the time for transfers from each source, and return
        the maximum. Assumption: all transfers will happen concurrently.'''

        assert(len(files_by_src) <= MAX_CONCURRENT_TRANSFERS)

        if len(files_by_src) == 0:
            return 0.0

        times = []
        for src, pairs in files_by_src.items():
            _, sizes = zip(*pairs)
            times.append(self.predict_one(src, dst, sum(sizes)))

        return max(times)

    def update(self, src, dst, size, transfer_time):
        src_grp = self.endpoints[src]['transfer_group']
        dst_grp = self.endpoints[dst]['transfer_group']

        self.sizes[src_grp][dst_grp].append(size)
        self.times[src_grp][dst_grp].append(transfer_time)

        self.updates_since_train[src_grp][dst_grp] += 1
        if self.updates_since_train[src_grp][dst_grp] >= self.train_every:
            self._train(src_grp, dst_grp)
            self.updates_since_train[src_grp][dst_grp] = 0

    def _train(self, src_grp, dst_grp):
        sizes = np.array([self._preprocess(x)
                          for x in self.sizes[src_grp][dst_grp]])
        sizes = sizes.reshape((-1, 3))
        times = np.array([self.times[src_grp][dst_grp]]).reshape((-1, 1))
        self.weights[src_grp][dst_grp] = np.linalg.pinv(sizes).dot(times)

    def _preprocess(self, x):
        '''Create features that are easy to learn from.'''
        return np.array([1, x, np.log(x)])

    def to_file(self, file_name):
        sizes = {k: dict(vs) for (k, vs) in self.sizes.items()}
        times = {k: dict(vs) for (k, vs) in self.times.items()}
        weights = {s: {d: w.tolist() for (d, w) in vs.items()}
                   for (s, vs) in self.weights.items()}

        state = {
            'sizes': sizes,
            'times': times,
            'weights': weights,
        }

        with open(file_name, 'w') as fh:
            json.dump(state, fh)

    def _load_state_from_file(self, file_name):
        with open(file_name) as fh:
            state = json.load(fh)

        for s, vs in state['sizes'].items():
            for d, xs in vs.items():
                self.sizes[s][d] = xs

        for s, vs in state['times'].items():
            for d, xs in vs.items():
                self.times[s][d] = xs

        for s, vs in state['weights'].items():
            for d, xs in vs.items():
                self.weights[s][d] = np.array(xs)

        return self

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __str__(self):
        return type(self).__name__