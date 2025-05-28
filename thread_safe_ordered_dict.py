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
    id:int = -1  # unique identifier for the person
    in_line_time:int = 0  # time when the person enters the line
    out_line_time:int = 0  # time when the person leaves the line
    detected_times:int = 0
    is_waitting:bool = True

    global_feature:np.ndarray = None


class ThreadSafeOrderedDict:
    def __init__(self):
        self.od:dict[int, Person] = {}
        self.lock = Lock()
        self.next_id = 0

    def add(self, person_feature:np.ndarray):
        with self.lock:
            self.od[self.next_id] = Person(id=self.next_id, detected_times=1, is_waitting=True, global_feature=person_feature)
            self.next_id += 1
    
    def update(self, id, person_feature:np.ndarray):
        with self.lock:
            if id in self.od:
                # ??? time? need a global time?
                self.od[id].detected_times += 1
                self.od[id].global_feature += (person_feature - self.od[id].global_feature) / self.od[id].detected_times
            

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
    def values(self):
        with self.lock:
            return list(self.od.values())
    
    def __contains__(self, key):
        with self.lock:
            return key in self.od
    
    def __len__(self):
        with self.lock:
            return len(self.od)
        

