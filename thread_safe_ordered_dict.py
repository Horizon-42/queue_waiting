from dataclasses import dataclass
# from collections import OrderedDict
from threading import Lock
import numpy as np
import time

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

    global_feature:np.ndarray = None


class ThreadSafeOrderedDict:
    def __init__(self):
        self.od = dict()
        self.lock = Lock()
        self.next_id = 0

    def add(self, person_feature:np.ndarray):
        with self.lock:
            self.od[self.next_id] = Person(id=self.next_id, in_line_time=time.thread_time(), is_waitting=True, global_feature=person_feature)
    
    def update(self, id, person_feature:np.ndarray):
        with self.lock:
            if id in self.od:
                # ??? time? need a global time?
                self.od[id].global_feature = person_feature
            

    def get(self, id):
        with self.lock:
            return self.od.get(id)

    def remove(self, id):
        with self.lock:
            self.od.pop(id, None)

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
        

