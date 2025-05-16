from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock

@dataclass
class Person:
    ID:int
    InTime:int
    OutTime:int
    IsWaitting:bool


class ThreadSafeOrderedDict:
    def __init__(self):
        self.od = OrderedDict()
        self.lock = Lock()

    def set(self, key, value:Person):
        with self.lock:
            self.od[key] = value

    def get(self, key):
        with self.lock:
            return self.od.get(key)

    def remove(self, key):
        with self.lock:
            self.od.pop(key, None)

    def pop_first(self):
        with self.lock:
            return self.od.popitem(last=False)

    def __contains__(self, key):
        with self.lock:
            return key in self.od
    
    def __len__(self):
        with self.lock:
            return len(self.od)

    def keys(self):
        with self.lock:
            return list(self.od.keys())
        


