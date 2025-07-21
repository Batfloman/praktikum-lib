import copy

class CopyManager:
    def __call__(self):
        return copy.deepcopy(self._super)

    def __init__(self, super):
        self._super = super;