class RuntimePredictor(object):

    def __init__(self, endpoint_file):
        self.endpoints = endpoints

    def predict(self, func, group, payload):
        raise NotImplementedError

    def update(self, task_info, new_runtime):
        raise NotImplementedError

    def has_learned(self, func, endpoint):
        '''Whether some learning has happened for this (func, endpoint) pair,
        or whether we are still guessing.'''
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __str__(self):
        return type(self).__name__