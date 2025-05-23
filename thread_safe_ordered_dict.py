from dataclasses import dataclass
# from collections import OrderedDict
from threading import Lock
import numpy as np

@dataclass
class TrackInfo:
    id:int
    view_id:int
    in_view_time:int
    out_view_time:int
    feature:np.ndarray = None

@dataclass
class Person:
    id:int
    in_line_time:int
    out_line_time:int
    is_waitting:bool

    track_infos:dict[int, TrackInfo] = None


class ThreadSafeOrderedDict:
    def __init__(self):
        self.od = dict()
        self.lock = Lock()

    def set(self, key, value:object):
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

    def keys(self):
        with self.lock:
            return list(self.od.keys())
    
    def __contains__(self, key):
        with self.lock:
            return key in self.od
    
    def __len__(self):
        with self.lock:
            return len(self.od)
        

